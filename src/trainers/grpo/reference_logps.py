"""Ragged completion-token payload hooks for offline GRPO's frozen-reference sidecar."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence

import pyarrow as pa
import torch
from datasets import Dataset

from src.checkpoint.format import REFERENCE_LOGPS_FILE
from src.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from src.distributed.runtime import (
    DeferredRankFailure,
    fs_aware_save_rank,
    rank_consensus,
    reject_across_ranks,
    reject_divergent_settings,
)
from src.models.structure import base_transformers_model
from src.trainers.grpo.reference_cache import (
    REFERENCE_BATCH_ROWS,
    REFERENCE_BUFFER_VALUES,
    MappedReferenceScores,
    ReferenceScoreCache,
    mapped_reference_scores,
    reference_payload_mismatch,
)
from src.trainers.mixins.reference_logps import ReferenceLogpsCheckpointMixin, token_digest

_TOKEN_COLUMNS = ("prompt_input_ids", "completion_input_ids")


def reject_unsupported_reference_input(
    train_dataset: Dataset, eval_dataset: Dataset | None, *, presharded: bool = False
) -> None:
    """Supplied raw scores are not the run-start reference this implementation checkpoints."""
    supplied = REF_PER_TOKEN_LOGPS_COLUMN in train_dataset.column_names or (
        eval_dataset is not None and REF_PER_TOKEN_LOGPS_COLUMN in eval_dataset.column_names
    )
    reject_across_ranks(
        "Offline GRPO reference precompute is not supported with a pre-sharded dataset; "
        "load the same unsharded dataset on every rank."
        if presharded
        else "Supplied ref_per_token_logps are not supported by offline GRPO; load an unsharded "
        "dataset and let the trainer prepare its checkpointed run-start reference."
        if supplied
        else None,
        "Validating supplied offline GRPO references",
        exc_type=ValueError,
    )


def _rows_digest(dataset: Dataset, rows: Sequence[Sequence[float] | torch.Tensor]) -> str:
    """Validate sequence callers in bounded row buffers without a second flat token table."""
    if len(rows) != len(dataset):
        raise ValueError(f"Reference sweep returned {len(rows)} rows for a split with {len(dataset)} rows")
    digest = hashlib.sha256()
    for batch in dataset.select_columns(["completion_input_ids"]).iter(batch_size=REFERENCE_BATCH_ROWS):
        digest.update(torch.tensor([len(tokens) for tokens in batch["completion_input_ids"]]).numpy().tobytes())
    for index, row in enumerate(rows):
        expected_length = len(dataset[index]["completion_input_ids"])
        values = torch.as_tensor(row, dtype=torch.float32).detach().cpu()
        if values.ndim != 1 or values.numel() != expected_length:
            raise ValueError(
                f"Reference row {index} has shape {tuple(values.shape)} for a completion of {expected_length} tokens"
            )
        if not torch.isfinite(values).all():
            raise ValueError(f"Reference row {index} contains a non-finite log-probability")
        digest.update(memoryview(values.contiguous().numpy()).cast("B"))
    return digest.hexdigest()


def _scores_digest(lengths: torch.Tensor, values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in (lengths, values):
        for start in range(0, tensor.numel(), REFERENCE_BUFFER_VALUES):
            digest.update(memoryview(tensor[start : start + REFERENCE_BUFFER_VALUES].contiguous().numpy()).cast("B"))
    return digest.hexdigest()


def _assert_replicated_scores(split: str, lengths: torch.Tensor, values: torch.Tensor) -> str:
    digest = _scores_digest(lengths, values)
    reject_divergent_settings(
        {"split": split, "reference_digest": digest},
        "Offline GRPO reference values",
        "The same unsharded rows must carry the same raw reference scores on every rank.",
    )
    return digest


def _attach_reference_column(dataset: Dataset, scores: MappedReferenceScores, digest: str | None = None) -> Dataset:
    digest = _scores_digest(scores.lengths, scores.values) if digest is None else digest
    fingerprint = hashlib.sha256(f"{dataset._fingerprint}/{digest}".encode()).hexdigest()
    attached = dataset.add_column(REF_PER_TOKEN_LOGPS_COLUMN, scores.column(), new_fingerprint=fingerprint)
    attached._reference_storage_owner = scores
    return attached


def _token_row_keys(dataset: Dataset):
    for batch in dataset.select_columns(list(_TOKEN_COLUMNS)).iter(batch_size=REFERENCE_BATCH_ROWS):
        for prompt, completion in zip(batch["prompt_input_ids"], batch["completion_input_ids"], strict=True):
            digest = hashlib.sha256()
            for tokens in (prompt, completion):
                digest.update(torch.tensor([len(tokens), *tokens], dtype=torch.int64).numpy().tobytes())
            yield digest.digest()


class OfflineGRPOReferenceLogpsMixin(ReferenceLogpsCheckpointMixin):
    """Restore frozen token scores using the DPO/KTO checkpoint lifecycle."""

    def _reference_resume_required(self) -> bool:
        return self._policy_from_checkpoint

    def _init_reference_logps(self, *, resume_checkpoint: str | None, resume_context_given: bool = True) -> None:
        self._init_reference_state(
            checkpoint=resume_checkpoint,
            given=resume_context_given,
            policy_from_checkpoint=resume_checkpoint is not None,
        )
        self._reference_dataset_by_split: dict[str, Dataset] = {}
        self._reference_storage_by_split: dict[str, MappedReferenceScores] = {}
        self._reference_evaluation_datasets: list[Dataset] = []
        self._reference_evaluation_depth = 0

    def _reference_cache_output_dir(self) -> str:
        args = getattr(self, "args", None)
        return os.fspath(getattr(args, "output_dir", None) or self.output_dir)

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        prepared = bool(self._reference_logps_by_split) or getattr(self, "_precompute_reference", False)
        if prepared and self.beta != 0.0:
            declared = self._reference_resume_checkpoint
            requested = None if resume_from_checkpoint is False else resume_from_checkpoint
            reason = None
            if requested is True or (requested is not None and declared is None):
                reason = "Resolve the offline GRPO resume checkpoint before constructing the trainer"
            elif declared is not None:
                same_checkpoint = isinstance(requested, (str, os.PathLike)) and os.path.realpath(
                    os.fspath(requested)
                ) == os.path.realpath(os.fspath(declared))
                if not same_checkpoint:
                    reason = "Offline GRPO must train from the same checkpoint used to restore its reference"
            reject_across_ranks(reason, "Validating the offline GRPO training checkpoint", exc_type=ValueError)
        return super().train(resume_from_checkpoint, *args, **kwargs)

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
        *,
        original_reference_model=None,
    ):
        """Reuse the run-start anchor, or score unseen rows with a caller-owned original policy."""
        outermost = self._reference_evaluation_depth == 0
        self._reference_evaluation_depth += 1
        try:
            if getattr(self, "_precompute_reference", False):
                reject_divergent_settings(
                    {
                        "reference_model_supplied": original_reference_model is not None,
                        "evaluation_kind": "default"
                        if eval_dataset is None
                        else "named"
                        if isinstance(eval_dataset, dict)
                        else "split",
                    },
                    "Offline GRPO evaluation call",
                    "Every rank must supply the same evaluation/reference call layout.",
                )
            if not getattr(self, "_precompute_reference", False):
                if original_reference_model is not None:
                    raise ValueError("original_reference_model is only used by full-finetuning KL evaluation")
            elif eval_dataset is not None:
                if isinstance(eval_dataset, dict):
                    reject_divergent_settings(
                        {"eval_split_names": list(eval_dataset)},
                        "Offline GRPO evaluation splits",
                        "Every rank must evaluate the same named splits in the same order.",
                    )
                    eval_dataset = {
                        name: self._prepare_evaluation_reference(dataset, original_reference_model)
                        for name, dataset in eval_dataset.items()
                    }
                else:
                    eval_dataset = self._prepare_evaluation_reference(eval_dataset, original_reference_model)
            elif original_reference_model is not None:
                raise ValueError("Pass an evaluation dataset when supplying original_reference_model")
            return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        finally:
            self._reference_evaluation_depth -= 1
            if outermost:
                try:
                    for dataset in self._reference_evaluation_datasets:
                        dataset._reference_storage_owner.discard_cache()
                finally:
                    self._reference_evaluation_datasets.clear()

    def _validate_evaluation_reference_model(self, model) -> None:
        reason = None
        if not isinstance(model, torch.nn.Module):
            reason = "original_reference_model must be the exact original frozen policy model"
        elif model is self.model or base_transformers_model(model) is base_transformers_model(self.model):
            reason = "The trained/live policy cannot be original_reference_model"
        elif model.training or any(parameter.requires_grad for parameter in model.parameters()):
            reason = "original_reference_model must be frozen (requires_grad=False) and in eval mode"
        elif getattr(self, "_pp_runtime", None) is not None:
            reason = (
                "An external original_reference_model is not supported for PP evaluation. "
                "Declare the evaluation split before training so the original policy precomputes it."
            )
        reject_across_ranks(reason, "Validating the original evaluation reference", exc_type=ValueError)

    def _prepare_evaluation_reference(self, dataset, original_reference_model) -> Dataset:
        if original_reference_model is not None:
            self._validate_evaluation_reference_model(original_reference_model)
        reject_across_ranks(
            None if isinstance(dataset, Dataset) else "KL evaluation requires a finite tokenized datasets.Dataset",
            "Preparing offline GRPO evaluation references",
            exc_type=ValueError,
        )
        original_dataset = dataset
        if REF_PER_TOKEN_LOGPS_COLUMN in dataset.column_names:
            dataset = dataset.remove_columns(REF_PER_TOKEN_LOGPS_COLUMN)
        identity = self._reference_split_identity(dataset, "evaluation")
        reject_divergent_settings(identity, "Offline GRPO evaluation rows", "Every rank must evaluate the same rows.")
        known = any(
            original_dataset is stored and self._reference_logps_by_split[name]["settings"] == identity["settings"]
            for name, stored in self._reference_dataset_by_split.items()
        ) or any(
            original_dataset is stored and getattr(stored, "_reference_settings", None) == identity["settings"]
            for stored in self._reference_evaluation_datasets
        )
        if rank_consensus(known)[0]:
            return original_dataset
        guard = DeferredRankFailure("Matching evaluation rows to the original reference", exc_type=ValueError)

        def match_rows():
            known = {}
            for name, stored in self._reference_dataset_by_split.items():
                if self._reference_logps_by_split[name]["settings"] != identity["settings"]:
                    continue
                storage = self._reference_storage_by_split[name]
                for index, key in enumerate(_token_row_keys(stored)):
                    known.setdefault(key, (storage, index))
            return [known.get(key) for key in _token_row_keys(dataset)]

        matches = guard.run(match_rows)
        guard.reject()
        missing = any(match is None for match in matches)
        reject_across_ranks(
            "Evaluation contains unseen token rows whose original KL reference was not precomputed. "
            "Declare this eval split before training, or outside PP call "
            "evaluate(new_dataset, original_reference_model=original_frozen_policy). "
            "Use the exact run-start weights, not the trained checkpoint or a different base revision."
            if missing and original_reference_model is None
            else None,
            "Preparing offline GRPO evaluation references",
            exc_type=ValueError,
        )
        if missing:
            model = original_reference_model
            devices = {tensor.device for tensor in (*model.parameters(), *model.buffers())}
            reject_across_ranks(
                "original_reference_model must reside on one device before temporary scoring placement"
                if len(devices) != 1
                else None,
                "Placing the original evaluation reference",
                exc_type=ValueError,
            )
            original_device = next(iter(devices))
            previous_reference = self.ref_model
            try:
                guard = DeferredRankFailure("Placing the original evaluation reference")
                guard.run(lambda: model.to(next(self.model.parameters()).device))
                guard.reject()
                self.ref_model = model
                scores = self._sweep_reference_logps(dataset, "evaluation")
            finally:
                self.ref_model = previous_reference
                guard = DeferredRankFailure("Restoring the caller-owned evaluation reference")
                guard.run(lambda: model.to(original_device))
                guard.reject()
        else:
            cache = ReferenceScoreCache(self._reference_cache_output_dir(), dp_size=1)
            guard = DeferredRankFailure("Reusing original evaluation reference scores")
            try:
                if fs_aware_save_rank():
                    guard.run(
                        lambda: cache.append_rows(
                            0,
                            (
                                storage.values[int(storage.offsets[index]) : int(storage.offsets[index + 1])]
                                for storage, index in matches
                            ),
                        )
                    )
                guard.reject()
                scores = cache.finish(dataset)
            except BaseException:
                cache.discard()
                raise
        digest = _assert_replicated_scores("evaluation", scores.lengths, scores.values)
        guard = DeferredRankFailure("Attaching original evaluation reference scores", exc_type=ValueError)
        attached = guard.run(lambda: _attach_reference_column(dataset, scores, digest))
        guard.reject()
        attached._reference_settings = identity["settings"]
        self._reference_evaluation_datasets.append(attached)
        return attached

    def _reference_input_digests(self, dataset: Dataset, name: str) -> dict[str, str]:
        if not isinstance(dataset, Dataset):
            raise TypeError(f"'{name}' must be a finite datasets.Dataset for a reference sweep")
        for column in _TOKEN_COLUMNS:
            arrow_type = dataset.features.arrow_schema.field(column).type
            if not (pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type)) or not pa.types.is_integer(
                arrow_type.value_type
            ):
                raise ValueError(f"'{column}' must be a list of integer token IDs, got {arrow_type}")
        return {column: token_digest(dataset, column) for column in _TOKEN_COLUMNS}

    def _restore_reference_logps_or_none(
        self, dataset: Dataset, split: str, *, settings: Mapping[str, object]
    ) -> Dataset | None:
        reject_across_ranks(
            None if isinstance(dataset, Dataset) else f"'{split}' must be a finite datasets.Dataset",
            "Validating offline GRPO reference dataset",
            exc_type=ValueError,
        )
        reject_unsupported_reference_input(dataset, None, presharded=getattr(self, "_dataset_presharded", False))
        self._check_reference_resume_context()
        identity = self._reference_split_identity(dataset, split, settings)
        reject_divergent_settings(
            {"split": split, **identity},
            "Offline GRPO reference inputs",
            "The same unsharded tokenized split and settings must reach every rank.",
        )
        return self._restore_reference_split(dataset, split, (REF_PER_TOKEN_LOGPS_COLUMN,), identity)

    def _attach_scored_reference_logps(
        self,
        dataset: Dataset,
        split: str,
        rows: MappedReferenceScores | Sequence[Sequence[float] | torch.Tensor],
        *,
        settings: Mapping[str, object],
    ) -> Dataset:
        reject_across_ranks(
            f"Reference split '{split}' was already attached" if split in self._reference_logps_by_split else None,
            f"Recording the '{split}' GRPO reference",
            exc_type=ValueError,
        )
        identity = self._reference_split_identity(dataset, split, settings)
        guard = DeferredRankFailure(f"Packing the '{split}' GRPO reference", exc_type=ValueError)
        if not isinstance(rows, MappedReferenceScores):
            digest = guard.run(lambda: _rows_digest(dataset, rows))
            guard.reject()
            reject_divergent_settings(
                {"split": split, "reference_digest": digest},
                "Offline GRPO reference values",
                "The same unsharded rows must carry the same raw reference scores on every rank.",
            )
            cache = ReferenceScoreCache(self._reference_cache_output_dir(), dp_size=1)
            try:
                if fs_aware_save_rank():
                    guard.run(lambda: cache.append_rows(0, rows))
                guard.reject()
                rows = cache.finish(dataset)
            except BaseException:
                cache.discard()
                raise
        digest = _assert_replicated_scores(split, rows.lengths, rows.values)
        guard = DeferredRankFailure(f"Attaching the '{split}' GRPO reference", exc_type=ValueError)
        attached = guard.run(lambda: _attach_reference_column(dataset, rows, digest))
        guard.reject()
        self._reference_storage_by_split[split] = rows
        self._remember_reference_split(split, identity, {}, attached)
        return attached

    def _reference_payload_mismatch(self, entry: Mapping, dataset: Dataset, needed: Sequence[str]) -> str | None:
        del needed
        return reference_payload_mismatch(entry.get("lengths"), entry.get("values"), dataset)

    def _attach_reference_payload(self, dataset: Dataset, entry: Mapping, needed: Sequence[str]) -> Dataset:
        del needed
        scores = mapped_reference_scores(entry["lengths"], entry["values"])
        return _attach_reference_column(dataset, scores)

    def _validate_restored_reference_payload(self, name: str, entry: Mapping) -> None:
        _assert_replicated_scores(name, entry["lengths"], entry["values"])

    def _remember_reference_split(self, name: str, identity: Mapping, payload: Mapping, dataset: Dataset) -> None:
        if payload:
            self._reference_storage_by_split[name] = mapped_reference_scores(payload["lengths"], payload["values"])
        self._reference_logps_by_split[name] = dict(identity)
        self._reference_dataset_by_split[name] = dataset
        self._resumed_reference_logps.pop(name, None)
        self._reference_state_generation += 1

    def _reference_checkpoint_payload(self) -> dict:
        return {
            **self._resumed_reference_logps,
            **{
                name: {
                    **identity,
                    "lengths": self._reference_storage_by_split[name].lengths,
                    "values": self._reference_storage_by_split[name].values,
                }
                for name, identity in self._reference_logps_by_split.items()
            },
        }

    def _missing_reference_split(self, checkpoint: str, name: str, needed: Sequence[str]) -> str:
        del needed
        return (
            f"Cannot resume offline GRPO from {checkpoint}: its {REFERENCE_LOGPS_FILE} lacks '{name}'. "
            "Recomputing it would score the TRAINED checkpoint policy as its own reference. "
            "Restore the original sidecar from a complete checkpoint. If none exists, run the same "
            "config from the exact original model/revision and tokenized train/eval data into a "
            "separate scratch output (--resume_from_checkpoint=null --max_steps=1 "
            "--save_strategy=steps --save_steps=1 --save_only_model=true), then copy its "
            f"checkpoint-1/{REFERENCE_LOGPS_FILE} here on every node with node-local checkpoints. "
            "Do not use the trained checkpoint as that recovery run's model source."
        )

    def _read_reference_checkpoint(self, path: str):
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)

    def _reference_entry_mismatch(self, entry, dataset, needed, identity) -> str | None:
        if isinstance(entry, Mapping) and isinstance(entry.get("settings"), Mapping):
            # Reference forwards always use eval mode, independent of training dropout settings.
            settings = {key: value for key, value in entry["settings"].items() if key != "disable_dropout"}
            entry = {**entry, "settings": settings}
        return super()._reference_entry_mismatch(entry, dataset, needed, identity)
