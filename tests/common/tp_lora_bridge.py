"""Shared tiny-model fixtures for TP-aware LoRA bridge tests."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint
from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES

from src.distributed.tensor_parallel.lora import apply_tp_to_lora

IN_FEATURES = 16
HIDDEN_FEATURES = 16
OUT_FEATURES = 8
LORA_RANK = 4
BASE_SEED = 117
INPUT_SEED = 902


class TinyTPModel(nn.Module):
    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        checkpointing: bool = False,
    ):
        super().__init__()
        self.q_proj = nn.Linear(IN_FEATURES, HIDDEN_FEATURES, bias=False, dtype=dtype, device=device)
        self.o_proj = nn.Linear(HIDDEN_FEATURES, OUT_FEATURES, bias=False, dtype=dtype, device=device)
        self.checkpointing = checkpointing

    def _block(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(torch.tanh(self.q_proj(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.checkpointing and self.training:
            return checkpoint(self._block, x, use_reentrant=False)
        return self._block(x)


def lora_config(*, targets: Sequence[str] = ("q_proj", "o_proj"), initialization: bool | str = False) -> LoraConfig:
    return LoraConfig(
        r=LORA_RANK,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=list(targets),
        init_lora_weights=initialization,
        bias="none",
        task_type=None,
    )


def tp_base(
    world_size: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    checkpointing: bool = False,
) -> nn.Module:
    torch.manual_seed(BASE_SEED)
    model = TinyTPModel(dtype=dtype, device=device, checkpointing=checkpointing)
    mesh = init_device_mesh(torch.device(device).type, (world_size,), mesh_dim_names=("tp",))
    model._tp_plan = {"q_proj": "colwise", "o_proj": "rowwise"}
    model._device_mesh = mesh
    for name, style_name in model._tp_plan.items():
        module = model.get_submodule(name)
        style = ALL_PARALLEL_STYLES[style_name]
        style.shard_param(module, "weight", mesh)
        style.install_forward(module, mesh)
    return model


def tp_peft(
    world_size: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    checkpointing: bool = False,
    targets: Sequence[str] = ("q_proj", "o_proj"),
    initialization: bool | str = False,
    bridge: bool = True,
) -> nn.Module:
    model = get_peft_model(
        tp_base(world_size, device=device, dtype=dtype, checkpointing=checkpointing),
        lora_config(targets=targets, initialization=initialization),
    )
    if bridge:
        assert apply_tp_to_lora(model) == len(targets)
    return model


def reference(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    checkpointing: bool = False,
    targets: Sequence[str] = ("q_proj", "o_proj"),
    initialization: bool | str = False,
) -> nn.Module:
    torch.manual_seed(BASE_SEED)
    return get_peft_model(
        TinyTPModel(dtype=dtype, device=device, checkpointing=checkpointing),
        lora_config(targets=targets, initialization=initialization),
    )


def lora_layers(model: nn.Module) -> dict[str, LoraLayer]:
    base = model.get_base_model()
    return {name: module for name, module in base.named_modules() if isinstance(module, LoraLayer)}


def full(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor.detach().clone()


def full_grad(grad: torch.Tensor) -> torch.Tensor:
    return grad.full_tensor() if isinstance(grad, DTensor) else grad.detach().clone()


def align_reference_from_bridge(tp_model: nn.Module, unsharded: nn.Module) -> None:
    unsharded_layers = lora_layers(unsharded)
    with torch.no_grad():
        for name, tp_layer in lora_layers(tp_model).items():
            unsharded_layer = unsharded_layers[name]
            for factor_name in ("lora_A", "lora_B"):
                tp_factor = getattr(tp_layer, factor_name)["default"]
                unsharded_factor = getattr(unsharded_layer, factor_name)["default"]
                unsharded_factor.weight.copy_(full(tp_factor.weight))


def sync_plain_replicated_grads(model: nn.Module) -> None:
    for name, layer in lora_layers(model).items():
        factor = layer.lora_A["default"] if name == "q_proj" else layer.lora_B["default"]
        assert factor.weight.grad is not None
        dist.all_reduce(factor.weight.grad, op=dist.ReduceOp.AVG)


def assert_replicated_factors_equal(model: nn.Module, world_size: int) -> None:
    for name, layer in lora_layers(model).items():
        factor = layer.lora_A["default"] if name == "q_proj" else layer.lora_B["default"]
        assert not isinstance(factor.weight, DTensor)
        bits = factor.weight.detach().contiguous().view(torch.uint8)
        peers = [torch.empty_like(bits) for _ in range(world_size)]
        dist.all_gather(peers, bits)
        assert all(torch.equal(peer, peers[0]) for peer in peers[1:]), f"{name} replica drifted"
