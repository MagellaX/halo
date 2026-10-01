"""Reference storage remains bounded, mapped, durable and filesystem-aware."""

import datetime
import os
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

import src.trainers.grpo.reference_cache as cache_module
from src.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from src.trainers.grpo.reference_cache import ReferenceScoreCache, reference_cache_writers
from tests.common.distributed import shared_output_dir
from tests.common.gloo import run_gloo_ranks
from tests.cpu.grpo.test_offline_grpo_reference_logps import SETTINGS, _dataset, _scores, _Trainer


@pytest.mark.parametrize(
    "shared,local_world,expected", [(True, 2, (0,)), (False, 2, (0, 2)), (False, 1, (0, 1, 2, 3))]
)
def test_cache_writer_election_follows_output_filesystem(monkeypatch, shared, local_world, expected):
    monkeypatch.setattr(cache_module, "is_output_shared_filesystem", lambda: shared)
    monkeypatch.setattr(cache_module, "get_global_world_size", lambda: 4)
    monkeypatch.setattr(cache_module, "get_local_world_size", lambda: local_world)
    assert reference_cache_writers() == expected


def _map_shared_cache_from_rank_local_scratch(rank: int, root: str, broadcast_output: bool) -> None:
    rank_directory = os.path.join(root, f"rank-{rank}")
    os.makedirs(rank_directory)
    context = SimpleNamespace(rank=rank, output_dir=rank_directory)
    output_dir = shared_output_dir(context) if broadcast_output else context.output_dir
    dataset = Dataset.from_dict({"completion_input_ids": [[1], [2, 3]]})
    scores = [torch.tensor([-0.25]), torch.tensor([-0.75, -1.5])]
    cache = ReferenceScoreCache(output_dir, dp_size=2)
    cache.collect_batch([scores[rank]], {0: 0, 1: 1})
    mapped = cache.finish(dataset)
    assert mapped.column().to_pylist() == [[-0.25], [-0.75, -1.5]]
    assert mapped.lengths.tolist() == [1, 2]


def test_shared_reference_cache_maps_from_distinct_rank_local_scratch_directories(tmp_path):
    run_gloo_ranks(
        _map_shared_cache_from_rank_local_scratch,
        2,
        str(tmp_path),
        True,
        env={"DIST_OUTPUT_SHARED_FILESYSTEM": "1"},
        pg_timeout=datetime.timedelta(seconds=30),
    )


def test_rank_local_output_negative_control_cannot_map_the_shared_reference_cache(tmp_path):
    with pytest.raises(torch.multiprocessing.ProcessRaisedException, match="No such file or directory.*complete"):
        run_gloo_ranks(
            _map_shared_cache_from_rank_local_scratch,
            2,
            str(tmp_path),
            False,
            env={"DIST_OUTPUT_SHARED_FILESYSTEM": "1"},
            pg_timeout=datetime.timedelta(seconds=30),
        )


def test_reference_cache_maps_one_arrow_token_buffer_and_serializes_it_without_repacking(tmp_path, monkeypatch):
    trainer = _Trainer(tmp_path)
    allocated = []
    original_empty = torch.empty

    def record_empty(*args, **kwargs):
        if kwargs.get("dtype") is torch.float32:
            allocated.append(args)
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", record_empty)
    attached = trainer._attach_scored_reference_logps(_dataset(), "train", _scores(), settings=SETTINGS)
    owner = trainer._reference_storage_by_split["train"]
    arrow = attached.data.column(REF_PER_TOKEN_LOGPS_COLUMN).chunk(0)
    assert arrow.values.buffers()[1].address == owner.values.data_ptr()
    payload = trainer._reference_checkpoint_payload()["train"]
    assert payload["values"].data_ptr() == owner.values.data_ptr()
    assert not allocated, "attachment or save repacked the whole completion-token table"
    with open(next(tmp_path.rglob("complete.values")), "r+b") as backing_file:
        backing_file.write(torch.tensor([-9.25]).numpy().tobytes())
    assert owner.values[0] == -9.25, "the cache owner detached its values from the mapped file"
    assert attached[REF_PER_TOKEN_LOGPS_COLUMN][0][0] == -9.25


@pytest.mark.parametrize("damage", ["missing_rows", "wrong_lengths", "nan", "truncated"])
def test_incomplete_or_corrupt_cache_is_not_published(tmp_path, damage):
    dataset = Dataset.from_dict({"completion_input_ids": [[1, 2], [3]]})
    cache = ReferenceScoreCache(tmp_path, dp_size=1)
    if damage == "missing_rows":
        cache.append_rows(0, [torch.tensor([-1.0, -2.0])])
    elif damage == "wrong_lengths":
        cache.append_rows(0, [torch.tensor([-1.0]), torch.tensor([-2.0, -3.0])])
    else:
        cache.append_rows(0, [torch.tensor([-1.0, -2.0]), torch.tensor([-3.0])])
        with open(cache._path(0, "values"), "ab" if damage == "truncated" else "r+b") as output:
            output.write(b"!" if damage == "truncated" else torch.tensor([float("nan")]).numpy().tobytes())
    with pytest.raises(ValueError, match="Incomplete|truncated"):
        cache.finish(dataset)
    assert not any(path.name.startswith("complete.") for path in tmp_path.rglob("*"))


def test_buffered_validation_detects_corruption_after_the_first_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "REFERENCE_BUFFER_VALUES", 2)
    dataset = Dataset.from_dict({"completion_input_ids": [[1, 2, 3, 4]]})
    cache = ReferenceScoreCache(tmp_path, dp_size=1)
    cache.append_rows(0, [torch.tensor([-1.0, -2.0, -3.0, -4.0])])
    with open(cache._path(0, "values"), "r+b") as output:
        output.seek(3 * 4)
        output.write(torch.tensor([float("nan")]).numpy().tobytes())
    with pytest.raises(ValueError, match="non-finite"):
        cache.finish(dataset)
    assert not list(tmp_path.rglob("*.values")), "failed validation left an unreachable token cache"


def test_failed_cache_write_is_cleaned_up_before_checkpointing(tmp_path, monkeypatch):
    trainer = _Trainer(tmp_path)

    def fail_write(*args):
        raise OSError("reference cache disk full")

    monkeypatch.setattr(ReferenceScoreCache, "_append", fail_write)
    with pytest.raises(ValueError, match="reference cache disk full"):
        trainer._attach_scored_reference_logps(_dataset(), "train", _scores(), settings=SETTINGS)
    assert not list((tmp_path / ".reference-cache").iterdir())
    assert not trainer._reference_logps_by_split


def test_resume_reads_mmap_and_reuses_the_mapped_checkpoint_storage(tmp_path, monkeypatch):
    first = _Trainer(tmp_path)
    first._attach_scored_reference_logps(_dataset(), "train", _scores(), settings=SETTINGS)
    first.save_checkpoint()
    resumed = _Trainer(tmp_path, checkpoint=str(tmp_path / "checkpoint-1"))
    calls = []
    original_load = torch.load

    def record_load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", record_load)
    attached = resumed._restore_reference_logps_or_none(_dataset(), "train", settings=SETTINGS)
    assert calls == [{"map_location": "cpu", "weights_only": True, "mmap": True}]
    owner = resumed._reference_storage_by_split["train"]
    assert (
        attached.data.column(REF_PER_TOKEN_LOGPS_COLUMN).chunk(0).values.buffers()[1].address
        == owner.values.data_ptr()
    )
    assert resumed._reference_checkpoint_payload()["train"]["values"].data_ptr() == owner.values.data_ptr()


def test_training_dropout_setting_does_not_change_the_eval_mode_anchor(tmp_path):
    trainer = _Trainer(tmp_path)
    trainer._attach_scored_reference_logps(
        _dataset(), "train", _scores(), settings={**SETTINGS, "disable_dropout": True}
    )
    trainer.save_checkpoint()
    resumed = _Trainer(tmp_path, checkpoint=str(tmp_path / "checkpoint-1"))
    restored = resumed._restore_reference_logps_or_none(_dataset(), "train", settings=SETTINGS)
    assert restored[REF_PER_TOKEN_LOGPS_COLUMN] == [[-0.25, -1.5], [-0.75], []]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
