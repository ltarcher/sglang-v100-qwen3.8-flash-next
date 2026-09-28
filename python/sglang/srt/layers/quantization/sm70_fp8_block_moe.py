"""V100 fallback for FP8 block-128 MoE with block-misaligned TP shards.

The EAGLE/MTP draft layer of Qwen3.8-Flash-Next-NVFP4 stores routed experts as
FP8 block-128 with moe_intermediate 640; TP4 shards them into 160 columns,
which block scales cannot follow (1.25 blocks per rank). Every rank therefore
keeps the whole per-expert scale tensors (the weight_loader_v2 flag
``load_scale_full_per_expert`` routes loading around TP slicing), dequantizes
its shard to FP16 after loading, and runs the TurboMind FP16 MoE kernels
instead of Fp8MoEMethod.
"""

from __future__ import annotations

import logging

import torch
from torch.nn.parameter import Parameter

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.sm70_fp16_moe import SM70FP16MoEMethod
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)

# Experts dequantized per chunk to bound the fp32 intermediate during load
# (512 experts x 128 rows x 2560 cols x 4B would be ~670 MB unchunked).
_DEQUANT_EXPERT_CHUNK = 128


def can_use_sm70_fp8_block_moe_dequant(layer, quant_config) -> bool:
    if envs.SGLANG_DISABLE_SM70_FP8_BLOCK_MOE_DEQUANT.get():
        return False
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 7:
        return False
    if layer.params_dtype != torch.float16:
        return False
    block = quant_config.weight_block_size
    if block is None or tuple(block) != (128, 128):
        return False
    # Only for shards the standard Fp8MoEMethod path rejects.
    shard = layer.intermediate_size_per_partition
    return shard % block[0] != 0 or shard % block[1] != 0


def _dequant_shard(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_n: int,
    block_k: int,
    row_start: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize a TP shard of block-128 FP8 rows to ``dtype``.

    weight: [E, R, K] fp8 shard covering global rows [row_start, row_start+R).
    scale:  [E, full_blocks, K // block_k] fp32, whole per expert.
    """
    rows = weight.shape[-2]
    k = weight.shape[-1]
    out = torch.empty(weight.shape, dtype=dtype, device=weight.device)
    for b in range(row_start // block_n, (row_start + rows - 1) // block_n + 1):
        c0 = max(row_start, b * block_n) - row_start
        c1 = min(row_start + rows, (b + 1) * block_n) - row_start
        chunk = weight[..., c0:c1, :].to(torch.float32)
        chunk = chunk.reshape(*chunk.shape[:-1], k // block_k, block_k)
        # scale[..., b, kb] broadcasts over the block_k columns it covers.
        scale_b = (
            scale[..., b, :].to(torch.float32).unsqueeze(-1).unsqueeze(-3)
        )
        out[..., c0:c1, :] = (
            (chunk * scale_b).reshape(*chunk.shape[:-2], k)
        ).to(dtype)
    return out


class SM70FP8BlockMoEDequantMethod(SM70FP16MoEMethod):
    """FP8 block-128 routed experts, dequantized to FP16 for the SM70 runner."""

    def __init__(self, quant_config):
        super().__init__()  # loads the TurboMind ops, raises if absent
        self.quant_config = quant_config
        self.weight_block_size = tuple(quant_config.weight_block_size)
        # Consumed by FusedMoE.weight_loader_v2: load block scales whole per
        # expert because the per-rank shard is not a block multiple.
        self.load_scale_full_per_expert = True

    def create_weights(
        self,
        layer,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import (
            FusedMoeWeightScaleSupported,
        )

        block_n, block_k = self.weight_block_size
        tp_size = get_parallel().tp_size
        num_shards = 2 if layer.moe_runner_config.is_gated else 1
        full_blocks = (intermediate_size_per_partition * tp_size) // block_n

        w13_weight = Parameter(
            torch.empty(
                num_experts,
                num_shards * intermediate_size_per_partition,
                hidden_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        w2_weight = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Whole per-expert scales, duplicated on every rank; TP windowing
        # happens at dequant time in process_weights_after_loading. Gate rows
        # sit in [0, full_blocks), up rows in [full_blocks, 2 * full_blocks),
        # matching the w1/w3 halves the loader copies into.
        w13_scale = Parameter(
            torch.zeros(
                num_experts,
                num_shards * full_blocks,
                hidden_size // block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_scale = Parameter(
            torch.zeros(
                num_experts,
                hidden_size // block_n,
                full_blocks,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_inv", w13_scale)
        layer.register_parameter("w2_weight_scale_inv", w2_scale)
        scale_attrs = dict(extra_weight_attrs)
        scale_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
        )
        set_weight_attrs(w13_scale, scale_attrs)
        set_weight_attrs(w2_scale, scale_attrs)

    def process_weights_after_loading(self, layer) -> None:
        block_n, block_k = self.weight_block_size
        tp_size = get_parallel().tp_size
        tp_rank = get_parallel().tp_rank
        shard = layer.intermediate_size_per_partition
        num_shards = layer.w13_weight.shape[1] // shard
        full_blocks = (shard * tp_size) // block_n
        row_start = shard * tp_rank
        dtype = layer.params_dtype

        s13 = layer.w13_weight_scale_inv.data
        # down_proj scales are [k_blocks, n_blocks]; flip to the common layout.
        s2 = layer.w2_weight_scale_inv.data.transpose(1, 2).contiguous()

        w13 = torch.empty(
            layer.w13_weight.shape, dtype=dtype, device=layer.w13_weight.device
        )
        w2 = torch.empty(
            layer.w2_weight.shape, dtype=dtype, device=layer.w2_weight.device
        )
        num_experts = layer.w13_weight.shape[0]
        for e0 in range(0, num_experts, _DEQUANT_EXPERT_CHUNK):
            e1 = min(e0 + _DEQUANT_EXPERT_CHUNK, num_experts)
            for shard_id in range(num_shards):
                w13[e0:e1, shard_id * shard : (shard_id + 1) * shard] = (
                    _dequant_shard(
                        layer.w13_weight.data[
                            e0:e1,
                            shard_id * shard : (shard_id + 1) * shard,
                        ],
                        s13[
                            e0:e1,
                            shard_id * full_blocks : (shard_id + 1) * full_blocks,
                        ],
                        block_n,
                        block_k,
                        row_start,
                        dtype,
                    )
                )
            w2t = _dequant_shard(
                layer.w2_weight.data[e0:e1].transpose(1, 2),
                s2[e0:e1],
                block_n,
                block_k,
                row_start,
                dtype,
            )
            w2[e0:e1] = w2t.transpose(1, 2)

        layer.w13_weight = Parameter(w13, requires_grad=False)
        layer.w2_weight = Parameter(w2, requires_grad=False)
        layer.w13_weight_scale_inv = None
        layer.w2_weight_scale_inv = None
        del s2, s13

        super().process_weights_after_loading(layer)
        logger.info(
            "SM70 FP8 block-128 MoE dequantized to FP16 "
            "(shard=%d cols at TP%d, %d experts, TurboMind batched GEMM)",
            shard,
            tp_size,
            num_experts,
        )
