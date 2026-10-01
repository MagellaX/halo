"""Dynamic KL evaluation never turns a trained policy into its own reference."""

from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from torch import nn

from src.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from src.trainers.grpo.reference_logps import OfflineGRPOReferenceLogpsMixin
from tests.common.gloo import run_gloo_ranks

_SETTINGS = {"model_type": "fixture", "temperature": 1.0}


def _dataset():
    return Dataset.from_dict(
        {"prompt_input_ids": [[1], [2], [3]], "completion_input_ids": [[4, 5], [6], []], "group_id": [0, 1, 2]}
    )


class _Base:
    def evaluate(self, dataset=None, *, ignore_keys=None, metric_key_prefix="eval"):
        self.evaluated = dataset
        self.eval_options = (ignore_keys, metric_key_prefix)
        if isinstance(dataset, dict):
            return {
                f"{metric_key_prefix}_{name}": self.evaluate(data, metric_key_prefix=f"{metric_key_prefix}_{name}")
                for name, data in dataset.items()
            }
        return {f"{metric_key_prefix}_loss": 42.0}


class _Trainer(OfflineGRPOReferenceLogpsMixin, _Base):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self._init_reference_logps(resume_checkpoint=None)
        self._precompute_reference = True
        self.model = nn.Linear(1, 1)
        self.ref_model = None
        self.parallelism_config = SimpleNamespace(is_cp_mode=False)
        self._pp_runtime = None
        self._reference_settings = lambda: dict(_SETTINGS)
        self.train_dataset = self._attach_scored_reference_logps(
            _dataset(), "train", [[-0.25, -1.5], [-0.75], []], settings=_SETTINGS
        )


def test_reordered_subset_and_duplicate_rows_reuse_exact_original_scores(tmp_path):
    trainer = _Trainer(tmp_path)
    copied = _dataset().select([1, 0, 1, 2])
    trainer._sweep_reference_logps = lambda *args: pytest.fail("reuse must not sweep the trained model")
    result = trainer.evaluate(copied, ignore_keys=["hidden_states"], metric_key_prefix="heldout")
    assert result == {"heldout_loss": 42.0}
    assert trainer.eval_options == (["hidden_states"], "heldout")
    assert trainer.evaluated[REF_PER_TOKEN_LOGPS_COLUMN] == [[-0.75], [-0.25, -1.5], [-0.75], []]
    assert not trainer._reference_evaluation_datasets


def test_named_eval_splits_preserve_standard_recursive_evaluation(tmp_path):
    trainer = _Trainer(tmp_path)
    result = trainer.evaluate({"first": _dataset().select([0]), "second": _dataset().select([1])})
    assert result == {"eval_first": {"eval_first_loss": 42.0}, "eval_second": {"eval_second_loss": 42.0}}


def test_later_named_split_failure_releases_earlier_temporary_reference_cache(tmp_path):
    trainer = _Trainer(tmp_path)
    unseen = _dataset().select([0]).remove_columns("prompt_input_ids").add_column("prompt_input_ids", [[99]])
    with pytest.raises(ValueError, match="unseen token rows"):
        trainer.evaluate({"known": _dataset().select([0]), "unseen": unseen})
    assert not trainer._reference_evaluation_datasets
    assert trainer._reference_evaluation_depth == 0
    assert len(list((tmp_path / ".reference-cache").iterdir())) == 1, "failed evaluation left a temporary cache"


def test_unseen_rows_fail_before_any_live_policy_forward_with_a_recovery_path(tmp_path):
    trainer = _Trainer(tmp_path)
    unseen = _dataset().select([0]).remove_columns("prompt_input_ids").add_column("prompt_input_ids", [[99]])
    trainer._sweep_reference_logps = lambda *args: pytest.fail("trained policy cannot recover unseen scores")
    with pytest.raises(ValueError, match="original_reference_model=original_frozen_policy"):
        trainer.evaluate(unseen)


def test_changed_reference_settings_do_not_reuse_old_token_scores(tmp_path):
    trainer = _Trainer(tmp_path)
    trainer._reference_settings = lambda: {**_SETTINGS, "temperature": 2.0}
    with pytest.raises(ValueError, match="unseen token rows"):
        trainer.evaluate(trainer.train_dataset)


@pytest.mark.parametrize("kind", ["live", "trainable", "training", "pp"])
def test_invalid_original_reference_is_rejected_before_scoring(tmp_path, kind):
    trainer = _Trainer(tmp_path)
    reference = nn.Linear(1, 1).requires_grad_(False).eval()
    if kind == "live":
        reference = trainer.model
    elif kind == "trainable":
        reference.requires_grad_(True)
    elif kind == "training":
        reference.train()
    else:
        trainer._pp_runtime = object()
    with pytest.raises(ValueError, match="trained/live|frozen|PP"):
        trainer.evaluate(_dataset(), original_reference_model=reference)


@pytest.mark.parametrize("failure", [False, True])
def test_explicit_original_reference_scores_unseen_rows_and_restores_ownership(tmp_path, failure):
    trainer = _Trainer(tmp_path)
    reference = nn.Linear(1, 1).requires_grad_(False).eval()
    original_weights = reference.weight.clone()
    previous_reference = trainer.ref_model = object()
    device = reference.weight.device
    moves = []
    reference_to = reference.to

    def record_to(target):
        moves.append(target)
        return reference_to(target)

    reference.to = record_to
    unseen = Dataset.from_dict({"prompt_input_ids": [[99]], "completion_input_ids": [[7, 8]], "group_id": [0]})

    def sweep(dataset, split):
        assert trainer.ref_model is reference and not reference.training
        assert dataset["prompt_input_ids"] == [[99]]
        if failure:
            raise RuntimeError("scoring failed")
        return trainer._attach_scored_reference_logps(
            dataset, "fixture-explicit", [[-4.25, -5.75]], settings=_SETTINGS
        )._reference_storage_owner

    trainer._sweep_reference_logps = sweep
    if failure:
        with pytest.raises(RuntimeError, match="scoring failed"):
            trainer.evaluate(unseen, original_reference_model=reference)
    else:
        assert trainer.evaluate(unseen, original_reference_model=reference) == {"eval_loss": 42.0}
        assert trainer.evaluated[REF_PER_TOKEN_LOGPS_COLUMN] == [[-4.25, -5.75]]
    assert trainer.ref_model is previous_reference
    assert reference.weight.device == device
    assert moves == [trainer.model.weight.device, device]
    torch.testing.assert_close(reference.weight, original_weights, rtol=0, atol=0)
    assert not reference.training and not reference.weight.requires_grad


def _ranked_object_identity(rank, root):
    trainer = _Trainer(root)
    dataset = trainer.train_dataset if rank == 0 else _dataset()
    trainer.evaluate(dataset)
    assert trainer.evaluated[REF_PER_TOKEN_LOGPS_COLUMN] == [[-0.25, -1.5], [-0.75], []]


def test_rank_local_object_identity_does_not_skip_collectives(tmp_path):
    run_gloo_ranks(_ranked_object_identity, 2, str(tmp_path))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
