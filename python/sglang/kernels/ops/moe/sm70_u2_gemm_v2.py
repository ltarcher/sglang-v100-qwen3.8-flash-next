# SPDX-License-Identifier: Apache-2.0
"""SM70 u2b2 v2 GEMM op: transposed-word register-direct MoE kernel.

Lives in the same marlin_v100 .so as moe_wna16_marlin_gemm
(csrc/moe/marlin_moe_wna16/sm70_u2_gemm_v2.cu) and is dispatched from
fused_marlin_moe under SGLANG_USE_SM70_U2_GEMM_V2 for prefill shapes
(moe_block_size == 32). Weights must be in the transposed layout produced by
sm70_u2_pool.u2_packed_to_T; callers signal that via u2_v2_words and the
binding site (convert_moe_layer_to_u2) hard-fails at boot if this op is
missing, so a fallback here can never read T-layout bytes with the marlin
kernel.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_op = False


def sm70_u2_gemm_v2_available() -> bool:
    """True iff the running marlin_v100 .so registers sm70_u2_gemm_v2."""
    return _resolve_op() is not None


def _resolve_op():
    global _op
    if _op is False:
        from sglang.kernels.ops.moe.moe_wna16_marlin import _load_marlin_v100_op

        # Loads the shared .so (registering every op in it); None on
        # non-SM70 or when the library is absent entirely.
        _load_marlin_v100_op()
        try:
            _op = getattr(torch.ops._moe_C, "sm70_u2_gemm_v2", None)
        except AttributeError:
            _op = None
        if _op is None:
            logger.info(
                "SM70 u2 gemm v2 op not present in the loaded marlin_v100 "
                ".so; marlin u2 stays the only u2 GEMM path"
            )
    return _op


def sm70_u2_gemm_v2(
    a: torch.Tensor,
    c: torch.Tensor,
    b_qweight_t: torch.Tensor,
    b_scales: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    mul_topk_weights: bool,
    size_m: int,
    size_n: int,
    size_k: int,
    group_size: int,
) -> torch.Tensor:
    op = _resolve_op()
    assert op is not None, (
        "sm70_u2_gemm_v2 unavailable: SGLANG_USE_SM70_U2_GEMM_V2 bound "
        "T-layout weights but the marlin_v100 .so lacks the op"
    )
    # The epilogue reads topk_weights as float32 (matches moe_wna16_marlin).
    if topk_weights.dtype != torch.float32:
        topk_weights = topk_weights.float()
    op(
        a,
        c,
        b_qweight_t,
        b_scales,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size,
        top_k,
        mul_topk_weights,
        size_m,
        size_n,
        size_k,
        group_size,
    )
    return c
