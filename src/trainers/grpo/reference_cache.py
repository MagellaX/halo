"""Bounded, filesystem-aware storage of offline GRPO's ragged frozen-reference scores."""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Iterable
from dataclasses import dataclass

import pyarrow as pa
import torch
import torch.distributed as dist
from datasets import Dataset

from src.distributed.filesystem import store_reject_across_ranks
from src.distributed.runtime import (
    DeferredRankFailure,
    broadcast_from_rank0,
    collective_device,
    fs_aware_save_rank,
    get_global_rank,
    get_global_world_size,
    get_local_world_size,
    is_output_shared_filesystem,
)
from src.trainers.mixins.reference_logps import fsync_reference_directory

REFERENCE_BUFFER_VALUES = 1 << 18
REFERENCE_BATCH_ROWS = 256


@dataclass
class MappedReferenceScores:
    """Keep the mapped tensor owners alive for Arrow and checkpoint serialization."""

    lengths: torch.Tensor
    values: torch.Tensor
    offsets: torch.Tensor
    cache_directory: str | None = None

    def __len__(self) -> int:
        return self.lengths.numel()

    def __iter__(self):
        for index in range(len(self)):
            yield self.values[int(self.offsets[index]) : int(self.offsets[index + 1])]

    def column(self) -> pa.LargeListArray:
        return pa.LargeListArray.from_arrays(
            pa.array(self.offsets.numpy()), pa.array(self.values.numpy(), type=pa.float32())
        )

    def discard_cache(self) -> None:
        """Mappings stay valid after unlink; evaluation needs no persistent run-local files."""
        if self.cache_directory is not None and fs_aware_save_rank() and os.path.isdir(self.cache_directory):
            shutil.rmtree(self.cache_directory)


def reference_cache_writers() -> tuple[int, ...]:
    """Checkpoint filesystem ownership also owns the run-local reference cache."""
    if is_output_shared_filesystem():
        return (0,)
    return tuple(range(0, get_global_world_size(), get_local_world_size()))


def reference_payload_mismatch(lengths, values, dataset: Dataset) -> str | None:
    """Validate ragged score storage without allocating a full-token validation plane."""
    if not isinstance(lengths, torch.Tensor) or lengths.dtype != torch.int64 or lengths.shape != (len(dataset),):
        return "its row lengths are malformed"
    if not isinstance(values, torch.Tensor) or values.dtype != torch.float32 or values.ndim != 1:
        return "its flat float32 reference values are malformed"
    total = 0
    for start, batch in zip(
        range(0, len(dataset), REFERENCE_BATCH_ROWS),
        dataset.select_columns(["completion_input_ids"]).iter(batch_size=REFERENCE_BATCH_ROWS),
        strict=True,
    ):
        expected = torch.tensor([len(tokens) for tokens in batch["completion_input_ids"]], dtype=torch.int64)
        actual = lengths[start : start + expected.numel()]
        if not torch.equal(actual, expected):
            return "its reference lengths do not match the completions"
        total += int(actual.sum())
    if total != values.numel():
        return "its row lengths do not cover its reference values"
    for start in range(0, values.numel(), REFERENCE_BUFFER_VALUES):
        if not bool(torch.isfinite(values[start : start + REFERENCE_BUFFER_VALUES]).all()):
            return "its reference values contain non-finite log-probabilities"
    return None


def mapped_reference_scores(lengths: torch.Tensor, values: torch.Tensor) -> MappedReferenceScores:
    """Checkpoint tensors stay mmap-backed; only row-sized offsets need materializing."""
    offsets = torch.empty(lengths.numel() + 1, dtype=torch.int64)
    offsets[0] = 0
    torch.cumsum(lengths, dim=0, out=offsets[1:])
    return MappedReferenceScores(lengths, values, offsets)


class ReferenceScoreCache:
    """Stream current batches to output-FS writers, then map the ordered token table."""

    def __init__(self, output_dir: str, *, dp_size: int):
        identifier = broadcast_from_rank0(uuid.uuid4().hex if get_global_rank() == 0 else None)
        self.directory = os.path.join(os.fspath(output_dir), ".reference-cache", identifier)
        self.dp_size = dp_size
        guard = DeferredRankFailure("Creating the offline GRPO reference cache")
        if fs_aware_save_rank():
            guard.run(lambda: os.makedirs(self.directory))
        guard.reject()

    def _path(self, shard: int | str, kind: str) -> str:
        return os.path.join(self.directory, f"{shard}.{kind}")

    def discard(self) -> None:
        """Remove only this unpublished UUID cache, without requiring a healthy process group."""
        if fs_aware_save_rank() and os.path.isdir(self.directory):
            shutil.rmtree(self.directory)

    def _append(self, shard: int, kind: str, values: torch.Tensor) -> None:
        with open(self._path(shard, kind), "ab") as destination:
            destination.write(memoryview(values.detach().cpu().contiguous().numpy()).cast("B"))

    def append_rows(self, shard: int, rows: Iterable[torch.Tensor]) -> None:
        """Local writer path, also used for exact-token evaluation reuse."""
        for row in rows:
            values = torch.as_tensor(row, dtype=torch.float32).detach().cpu()
            if values.ndim != 1 or not bool(torch.isfinite(values).all()):
                raise ValueError("Reference rows must contain finite, one-dimensional float32 scores")
            self._append(shard, "lengths", torch.tensor([values.numel()], dtype=torch.int64))
            for start in range(0, values.numel(), REFERENCE_BUFFER_VALUES):
                self._append(shard, "values", values[start : start + REFERENCE_BUFFER_VALUES])

    def collect_batch(self, rows: list[torch.Tensor] | None, representatives: dict[int, int]) -> None:
        """Only a DP representative transmits, and only filesystem writers receive score values."""
        rank, world = get_global_rank(), get_global_world_size()
        device = collective_device()
        guard = DeferredRankFailure("Packing an offline GRPO reference batch")

        def pack():
            lengths = torch.tensor([value.numel() for value in rows], dtype=torch.int64)
            values = torch.cat(rows).float().cpu() if rows else torch.empty(0, dtype=torch.float32)
            if any(value.ndim != 1 for value in rows) or not bool(torch.isfinite(values).all()):
                raise ValueError("Reference batch contains malformed or non-finite scores")
            return lengths, values

        packed = guard.run(pack) if rows is not None else None
        guard.reject()
        writers = reference_cache_writers()
        for shard, source in representatives.items():
            header = torch.tensor(
                [packed[0].numel(), packed[1].numel()] if rank == source and packed is not None else [0, 0],
                dtype=torch.int64,
                device=device,
            )
            if world > 1:
                dist.broadcast(header, src=source)
            row_count, value_count = header.cpu().tolist()
            if not row_count:
                continue
            guard = DeferredRankFailure("Writing an offline GRPO reference batch")
            for writer in writers:
                for kind, count, dtype in (
                    ("lengths", row_count, torch.int64),
                    ("values", value_count, torch.float32),
                ):
                    for start in range(0, count, REFERENCE_BUFFER_VALUES):
                        size = min(REFERENCE_BUFFER_VALUES, count - start)
                        buffer = None
                        transfer = DeferredRankFailure("Preparing a bounded reference transfer")
                        if rank == source:
                            tensor = packed[0] if kind == "lengths" else packed[1]
                            buffer = transfer.run(
                                lambda tensor=tensor, start=start, size=size: tensor[start : start + size].to(device)
                            )
                        elif rank == writer:
                            buffer = transfer.run(
                                lambda size=size, dtype=dtype: torch.empty(size, dtype=dtype, device=device)
                            )
                        transfer.reject()
                        if rank == source:
                            if writer != source:
                                dist.send(buffer, dst=writer)
                        elif rank == writer:
                            dist.recv(buffer, src=source)
                        if rank == writer:
                            guard.run(lambda buffer=buffer, kind=kind, shard=shard: self._append(shard, kind, buffer))
            guard.reject()

    def _map(self, shard: int | str) -> MappedReferenceScores:
        tensors = []
        for kind, dtype in (("lengths", torch.int64), ("values", torch.float32)):
            path = self._path(shard, kind)
            itemsize = torch.tensor([], dtype=dtype).element_size()
            if os.path.getsize(path) % itemsize:
                raise ValueError(f"Malformed reference cache '{path}': truncated {kind}")
            tensors.append(torch.from_file(path, shared=False, size=os.path.getsize(path) // itemsize, dtype=dtype))
        scores = mapped_reference_scores(*tensors)
        scores.cache_directory = self.directory
        return scores

    def finish(self, dataset: Dataset) -> MappedReferenceScores:
        try:
            return self._finish(dataset)
        except BaseException:
            self.discard()
            raise

    def _finish(self, dataset: Dataset) -> MappedReferenceScores:
        guard = DeferredRankFailure("Completing the offline GRPO reference cache", exc_type=ValueError)

        def merge():
            for kind in ("lengths", "values"):
                with open(self._path("partial", kind), "wb") as destination:
                    for shard in range(self.dp_size):
                        path = self._path(shard, kind)
                        if os.path.exists(path):
                            with open(path, "rb") as source:
                                shutil.copyfileobj(source, destination, length=REFERENCE_BUFFER_VALUES * 4)
                    destination.flush()
                    os.fsync(destination.fileno())
            mapped = self._map("partial")
            mismatch = reference_payload_mismatch(mapped.lengths, mapped.values, dataset)
            if mismatch:
                raise ValueError(f"Incomplete offline GRPO reference cache: {mismatch}")

        if fs_aware_save_rank():
            guard.run(merge)
        store_reject_across_ranks(
            f"reference-cache/{os.path.basename(self.directory)}/merge", guard.reason, guard.what, exc_type=ValueError
        )
        guard = DeferredRankFailure("Publishing the offline GRPO reference cache")
        if fs_aware_save_rank():
            for kind in ("lengths", "values"):
                guard.run(lambda kind=kind: os.replace(self._path("partial", kind), self._path("complete", kind)))
            for shard in range(self.dp_size):
                for kind in ("lengths", "values"):
                    path = self._path(shard, kind)
                    if os.path.exists(path):
                        guard.run(lambda path=path: os.unlink(path))
            guard.run(lambda: fsync_reference_directory(self.directory))
        guard.reject()
        guard = DeferredRankFailure("Mapping the offline GRPO reference cache", exc_type=ValueError)
        mapped = guard.run(lambda: self._map("complete"))
        guard.reject()
        return mapped
