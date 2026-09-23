#!/usr/bin/env python
"""TP=2 CUDA equivalence and optimizer coverage for the LoRA compatibility bridge.

The trainer gate intentionally remains closed in this PR. This test covers the complete bridge
boundary on CUDA: PEFT injection after HF-native TP, DTensor placement, both required collectives,
gradient checkpointing, clipping, fused AdamWBF16 updates, and no-grad evaluation.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 tests/gpu/parallelism/tp/test_tp_lora_bridge.py
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from src.distributed.tensor_parallel.lora import apply_tp_to_lora
from src.optimizers.adamw_bf16 import AdamWBF16
from tests.common.harness import gpu_test_main
from tests.common.tp_lora_bridge import (
    IN_FEATURES,
    INPUT_SEED,
    align_reference_from_bridge,
    assert_replicated_factors_equal,
    full_grad,
    lora_layers,
    reference,
    sync_plain_replicated_grads,
    tp_peft,
)

FORWARD_RTOL = 3e-2
FORWARD_ATOL = 3e-3
GRAD_RTOL = 5e-2
GRAD_ATOL = 5e-3
STEPS = 3


def _global_grad_norm(model: torch.nn.Module) -> torch.Tensor:
    total = torch.zeros((), device=torch.cuda.current_device(), dtype=torch.float32)
    for param in model.parameters():
        if param.grad is None:
            continue
        total.add_(full_grad(param.grad).float().square().sum())
    return total.sqrt()


def _scale_grads(model: torch.nn.Module, coefficient: torch.Tensor) -> None:
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is None:
                continue
            grad = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
            grad.mul_(coefficient)


def _snapshot_trainable(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().to_local().clone() if isinstance(param, DTensor) else param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _assert_first_step_matches_reference(
    model: torch.nn.Module,
    unsharded: torch.nn.Module,
    batch: torch.Tensor,
) -> None:
    x = batch.detach().clone().requires_grad_(True)
    ref_x = batch.detach().clone().requires_grad_(True)
    output = model(x)
    ref_output = unsharded(ref_x)
    torch.testing.assert_close(output, ref_output, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)

    probe = torch.linspace(-0.7, 0.9, output.numel(), device=output.device).reshape_as(output)
    (output.float() * probe).sum().backward()
    (ref_output.float() * probe).sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=GRAD_RTOL, atol=GRAD_ATOL)

    reference_layers = lora_layers(unsharded)
    for name, layer in lora_layers(model).items():
        reference_layer = reference_layers[name]
        for factor_name in ("lora_A", "lora_B"):
            grad = getattr(layer, factor_name)["default"].weight.grad
            reference_grad = getattr(reference_layer, factor_name)["default"].weight.grad
            assert grad is not None and reference_grad is not None
            torch.testing.assert_close(
                full_grad(grad).float(),
                reference_grad.float(),
                rtol=GRAD_RTOL,
                atol=GRAD_ATOL,
            )

    torch.testing.assert_close(
        _global_grad_norm(model),
        _global_grad_norm(unsharded),
        rtol=GRAD_RTOL,
        atol=GRAD_ATOL,
    )


def _run_mode(ctx, *, checkpointing: bool, max_grad_norm: float) -> tuple[bool, bool, float]:
    model = tp_peft(
        ctx.world_size,
        device=ctx.device,
        dtype=torch.bfloat16,
        checkpointing=checkpointing,
    )
    unsharded = reference(device=ctx.device, dtype=torch.bfloat16, checkpointing=checkpointing)
    align_reference_from_bridge(model, unsharded)
    before = _snapshot_trainable(model)
    optimizer = AdamWBF16(
        [param for param in model.parameters() if param.requires_grad],
        lr=2e-2,
        weight_decay=0.0,
    )

    generator = torch.Generator(device=ctx.device).manual_seed(INPUT_SEED)
    batches = [
        torch.randn((4, IN_FEATURES), device=ctx.device, dtype=torch.bfloat16, generator=generator)
        for _ in range(STEPS)
    ]

    model.train()
    unsharded.train()
    clipping_ok = True
    optimizer.zero_grad(set_to_none=True)
    _assert_first_step_matches_reference(model, unsharded, batches[0])
    sync_plain_replicated_grads(model)
    if max_grad_norm > 0:
        norm = _global_grad_norm(model)
        _scale_grads(model, (max_grad_norm / norm).clamp(max=1.0))
        clipping_ok &= bool((_global_grad_norm(model) <= max_grad_norm + 1e-6).item())
    optimizer.step()
    assert_replicated_factors_equal(model, ctx.world_size)

    for batch in batches[1:]:
        optimizer.zero_grad(set_to_none=True)
        model(batch).float().square().mean().backward()
        sync_plain_replicated_grads(model)
        if max_grad_norm > 0:
            norm = _global_grad_norm(model)
            _scale_grads(model, (max_grad_norm / norm).clamp(max=1.0))
            clipping_ok &= bool((_global_grad_norm(model) <= max_grad_norm + 1e-6).item())
        optimizer.step()
        assert_replicated_factors_equal(model, ctx.world_size)

    all_changed = True
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        local = param.detach().to_local() if isinstance(param, DTensor) else param.detach()
        all_changed &= not torch.equal(local, before[name])

    model.eval()
    with torch.no_grad():
        eval_output = model(batches[0])
    peers = [torch.empty_like(eval_output) for _ in range(ctx.world_size)]
    dist.all_gather(peers, eval_output)
    rank_spread = max((peer.float() - peers[0].float()).abs().max().item() for peer in peers)
    assert torch.isfinite(eval_output).all()
    assert eval_output.grad_fn is None
    assert apply_tp_to_lora(model) == 0
    return all_changed, bool(clipping_ok), rank_spread


def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    for checkpointing, max_grad_norm in ((False, 0.0), (True, 0.2)):
        label = "checkpointed_clipped" if checkpointing else "plain_unclipped"
        changed, clipping_ok, rank_spread = _run_mode(
            ctx,
            checkpointing=checkpointing,
            max_grad_norm=max_grad_norm,
        )
        checks[f"{label}_all_factors_updated"] = changed
        if max_grad_norm > 0:
            checks[f"{label}_clipping_valid"] = clipping_ok
        checks[f"{label}_ranks_match"] = rank_spread == 0.0
        metrics[f"{label}_rank_spread"] = rank_spread
    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(exact_world_size=2, prefix="tp_lora_bridge")(run)

if __name__ == "__main__":
    main()
