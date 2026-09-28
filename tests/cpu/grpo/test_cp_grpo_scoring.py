#!/usr/bin/env python
"""GRPO's CP chunked head scores the same targets as an unsplit HF forward."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.distributed.context_parallel.wrapper import UlyssesCPModelWrapper, find_cp_wrapper
from src.trainers.grpo.mixins.chunked_logprobs import ChunkedLogprobsCore
from tests.common.models import TINY_QWEN3_CONFIG

SEQ = 12
VOCAB = 64
CONFIG = TINY_QWEN3_CONFIG | {
    "vocab_size": VOCAB,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
}


def _inputs():
    torch.manual_seed(51)
    ids = torch.randint(1, VOCAB, (2, SEQ))
    mask = torch.ones_like(ids)
    mask[1, -2:] = 0
    labels = ids.clone()
    labels[0, :2] = -100
    labels[1, :5] = -100
    labels[1, -2:] = -100
    return ids, mask, labels


def _model():
    torch.manual_seed(19)
    config = Qwen3Config(**CONFIG, attn_implementation="eager")
    config.use_cache = False
    return Qwen3ForCausalLM(config).float().train()


def _scorer():
    scorer = ChunkedLogprobsCore()
    scorer.temperature = 1.0
    return scorer


def _scored_shards(model, hidden, ids, mask, labels, cp_size):
    scorer = _scorer()
    logps, shifted = [], []
    for rank in range(cp_size):
        cp_config = SimpleNamespace(cp_size=cp_size, cp_rank=rank)
        wrapper = UlyssesCPModelWrapper.__new__(UlyssesCPModelWrapper)
        nn.Module.__init__(wrapper)
        wrapper.model = model
        wrapper.cp_size = cp_size
        wrapper.cp_rank = rank
        wrapper.cp_config = cp_config
        start = rank * SEQ // cp_size
        end = (rank + 1) * SEQ // cp_size
        # CPU has no Ulysses attention kernel. The actual wrapper split is pinned separately in
        # test_cp_grpo_hidden.py; inject the full-backbone hidden shard here to isolate the real
        # boundary shift, head transform, ignore-index handling and chunked scorer.
        wrapper.forward_hidden_states = lambda **kwargs: hidden[:, start:end]
        local_logps, local_labels = scorer._cp_chunked_logps_impl(wrapper, ids, mask, labels)
        logps.append(local_logps)
        shifted.append(local_labels)
    return torch.cat(logps, dim=1), torch.cat(shifted, dim=1)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_scored_targets_and_gradients_match_full_qwen3(cp_size):
    ids, mask, labels = _inputs()
    baseline_model = _model()
    model = copy.deepcopy(baseline_model)
    baseline_logits = baseline_model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1]
    safe_labels = labels[:, 1:].clamp(min=0)
    baseline_logps = F.log_softmax(baseline_logits.float(), dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

    full_hidden = model.base_model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    got_logps, got_labels = _scored_shards(model, full_hidden, ids, mask, labels, cp_size)
    assert got_labels.shape == labels[:, 1:].shape
    torch.testing.assert_close(got_labels, labels[:, 1:], atol=0, rtol=0)
    supervised = got_labels != -100
    if cp_size > 1:
        assert supervised[:, SEQ // cp_size - 1].any(), "fixture never supervises the shard boundary"
    torch.testing.assert_close(got_logps[supervised], baseline_logps[supervised], atol=1e-4, rtol=1e-4)

    (-baseline_logps[supervised].sum()).backward()
    (-got_logps[supervised].sum()).backward()
    baseline_params = dict(baseline_model.named_parameters())
    current_params = dict(model.named_parameters())
    for name in (
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "lm_head.weight",
    ):
        torch.testing.assert_close(current_params[name].grad, baseline_params[name].grad, atol=1e-4, rtol=1e-4)


def test_removing_a_boundary_target_changes_the_supervised_sequence():
    ids, mask, labels = _inputs()
    model = _model()
    hidden = model.base_model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    _, correct_labels = _scored_shards(model, hidden, ids, mask, labels, cp_size=4)
    chunk = SEQ // 4
    assert labels[0, chunk] != -100
    assert correct_labels[0, chunk - 1] == labels[0, chunk]
    # Dropping boundary supervision must differ from the full-row target sequence.
    bad_labels = torch.cat([labels[:, rank * chunk + 1 : (rank + 1) * chunk] for rank in range(4)], dim=1)
    assert bad_labels.shape[1] == SEQ - 4
    assert correct_labels.shape[1] == SEQ - 1
    assert (correct_labels != -100).sum() > (bad_labels != -100).sum()


def test_scorer_rejects_misaligned_labels():
    ids, mask, labels = _inputs()
    model = _model()
    hidden = model.base_model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    with pytest.raises(ValueError, match="aligned with the full input row"):
        _scored_shards(model, hidden, ids, mask, labels[:, :-1], cp_size=2)


@pytest.mark.parametrize("peft_outside", [False, True], ids=["cp-over-peft", "peft-over-cp"])
def test_cp_scorer_finds_real_peft_layout_and_preserves_adapter_gradients(peft_outside):
    ids, mask, labels = _inputs()
    wrapper = UlyssesCPModelWrapper.__new__(UlyssesCPModelWrapper)
    nn.Module.__init__(wrapper)
    wrapper.model = _model()
    wrapper.cp_size = 1
    wrapper.cp_rank = 0
    wrapper.cp_config = SimpleNamespace(cp_size=1, cp_rank=0, process_group=None)
    wrapper._attention_layers = []
    config = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], init_lora_weights=False, task_type="CAUSAL_LM")
    if peft_outside:
        model = get_peft_model(wrapper, config)
    else:
        wrapper.model = get_peft_model(wrapper.model, config)
        model = wrapper
    assert find_cp_wrapper(model) is wrapper
    scorer = _scorer()
    scorer.accelerator = SimpleNamespace(unwrap_model=lambda value: value)
    expected_logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1]
    expected = F.log_softmax(expected_logits, dim=-1).gather(-1, labels[:, 1:].clamp(min=0).unsqueeze(-1))[..., 0]
    actual, shifted = scorer._cp_chunked_logps(model, ids, mask, labels)
    supervised = shifted != -100
    torch.testing.assert_close(actual[supervised], expected[supervised], atol=1e-4, rtol=1e-4)
    params = [parameter for name, parameter in model.named_parameters() if "lora_" in name]
    expected_grad = torch.autograd.grad(expected[supervised].sum(), params)
    actual_grad = torch.autograd.grad(actual[supervised].sum(), params)
    assert any(gradient.abs().max() > 0 for gradient in actual_grad)
    for got, want in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_zero_width_final_shard_backprops_zero_through_head_and_hidden():
    ids = torch.tensor([[2, 5]])
    labels = ids.clone()
    model = _model()
    hidden = torch.randn(1, 1, CONFIG["hidden_size"], requires_grad=True)
    wrapper = UlyssesCPModelWrapper.__new__(UlyssesCPModelWrapper)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    wrapper.cp_size, wrapper.cp_rank = 2, 1
    wrapper.forward_hidden_states = lambda **kwargs: hidden
    logps, shifted = _scorer()._cp_chunked_logps_impl(wrapper, ids, torch.ones_like(ids), labels)
    assert logps.shape == shifted.shape == (1, 0)
    logps.sum().backward()
    assert hidden.grad is not None and model.lm_head.weight.grad is not None
    torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden))
    torch.testing.assert_close(model.lm_head.weight.grad, torch.zeros_like(model.lm_head.weight))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
