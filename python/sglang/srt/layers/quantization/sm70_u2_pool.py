# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash P4: requantize modelopt NVFP4 routed experts to 2-bit.

Converts each MoE layer's checkpoint NVFP4 expert tensors into the uint2b2
SM70 Marlin format so the full expert pool stays resident in VRAM (~20.7
GB/rank for GLM-5.3 on TP4) and no expert ever spills to host RAM or pages
in during decode.

``create_weights`` stages the checkpoint tensors on host RAM at their normal
NVFP4 shapes (the full NVFP4 pool is 43.85 GB/rank and cannot sit in VRAM
even transiently); this module converts them layer by layer inside
``process_weights_after_loading``:

    host NVFP4 staging -> H2D chunk -> dequantize_nvfp4 (fp32, per-half
    weight_scale_2 folded in) -> RTN u2 -> repack_u2_sm70 -> GPU pool

The u2b2 contract consumed by ``fused_marlin_moe(num_bits=2)``:

    w13_weight        [E, H/16, 2I] int32  macro-N packed codes
    w2_weight         [E, I/16, H]  int32
    w13_weight_scale  [E, H/g, 2I]  fp16   (k-group major, transposed vs w13)
    w2_weight_scale   [E, I/g, H]   fp16

The requantization folds each expert's NVFP4 ``weight_scale_2`` into the
dequantized fp32 weights, so the u2 group scales are the only scale at
apply time (no per-expert global scale, unlike the u4 Marlin path).
"""

import logging
import os
import tempfile
import time

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.utils import copy_or_rebind_param

logger = logging.getLogger(__name__)

# Kernel decode grid: value = (code - bias) * scale with the fp16 bias the
# SM70 u2 dequant subtracts (1.45, encoded 0x3dcd) -- see requant_u2. The
# (bias, scale) pair is the uniform-grid MSE optimum measured on real
# GLM-5.3 experts: end-to-end MoE cosine 0.8295 vs NVFP4 truth, vs 0.7777
# for the original bias-2/0.375 grid and 0.830 for the non-uniform
# Lloyd-Max bound. Bias and multiplier must move together with the kernel.
_U2_CODE_BIAS = 1.4501953125
_U2_SCALE_AMAX_MULT = 0.36

_ANNOUNCED = False


def alloc_u2_staging(layer: torch.nn.Module, shape, dtype: torch.dtype) -> torch.Tensor:
    """Allocate a u2 staging tensor backed by a per-layer disk memmap.

    The staged NVFP4 checkpoint tensors are ~43 GB/rank across all layers
    and four ranks share one ~125 GB host: anonymous CPU tensors would OOM
    the box. A file-backed memmap keeps the pages in the (evictable,dirty-
    throttled) page cache instead -- the loader writes NVFP4 bytes in, the
    layer's conversion reads them back once, then the file is unlinked
    (see _free_u2_staging). SGLANG_SM70_U2_STAGE_DIR must be real disk,
    never tmpfs.
    """
    import numpy as np

    stage_dir = envs.SGLANG_SM70_U2_STAGE_DIR.get() or tempfile.gettempdir()
    if not hasattr(layer, "_sm70_u2_staging"):
        layer._sm70_u2_staging = []
    # Per-allocation file: a second w+ open of the same path would ftruncate
    # the first mapping's backing store and SIGBUS its pages.
    path = os.path.join(
        stage_dir,
        f"sglang_u2_stage_{os.getpid()}_{id(layer)}_{len(layer._sm70_u2_staging)}.bin",
    )
    nbytes = 1
    for dim in shape:
        nbytes *= dim
    mm = np.memmap(path, mode="w+", dtype=np.uint8, shape=(nbytes,))
    layer._sm70_u2_staging.append((mm, path))
    return torch.from_numpy(mm).view(dtype).view(*shape)


def _free_u2_staging(layer: torch.nn.Module) -> None:
    """Close and unlink this layer's staging memmaps (post-conversion)."""
    for mm, path in layer._sm70_u2_staging:
        try:
            mm._m.close()
        except Exception:  # noqa: BLE001 -- best-effort reclaim
            pass
        try:
            os.unlink(path)
        except OSError:
            pass
    layer._sm70_u2_staging = []


def sm70_u2_expert_pool_enabled() -> bool:
    """Whether the u2b2 resident-expert pool is requested (SM70 only)."""
    if not envs.SGLANG_SM70_U2_EXPERT_POOL.get():
        return False
    if torch.cuda.get_device_capability()[0] != 7:
        raise ValueError(
            "SGLANG_SM70_U2_EXPERT_POOL=1 requires SM70 (V100); the u2b2 "
            f"pool kernel is SM70-only, got cc{torch.cuda.get_device_capability()}"
        )
    return True


def resolve_u2_group_size(hidden_size: int, intermediate_size: int) -> int:
    """Pick the u2 group size: env knob first, then 128/64/32.

    The SM70 u2 kernel supports group sizes 32/64/128, and the group must
    divide both contraction dims (hidden for gemm1, moe_intermediate/TP
    for gemm2). GLM-5.3 on TP4 divides at 128; the fallbacks cover draft
    (MTP) or differently-sharded layers.
    """
    requested = envs.SGLANG_SM70_U2_GROUP.get()
    for candidate in (requested, 128, 64, 32):
        if hidden_size % candidate == 0 and intermediate_size % candidate == 0:
            return candidate
    raise ValueError(
        "SGLANG_SM70_U2_EXPERT_POOL: no supported group size "
        f"({requested}/128/64/32) divides hidden={hidden_size} and "
        f"intermediate={intermediate_size}"
    )


def requant_u2(w: torch.Tensor, group_size: int):
    """RTN onto the fixed uint(2, 2) decode grid.

    Code c decodes to (c - _U2_CODE_BIAS) * scale, a symmetric midrise grid
    {-1.45, -0.45, +0.45, +1.45} * scale with no exact-zero level. The bias
    lives as an fp16 constant inside the SM70 u2 dequant
    (marlin-v100-u2-experts.patch) and cannot be changed without rebuilding
    that kernel; scale = 0.36 * max|w| is the MSE-optimal multiplier for
    THIS bias (joint bias x scale sweep on real GLM-5.3 experts, g=128).

    w: [..., R, C] float32 (R = gemm N, C = gemm K). Returns codes
    [..., R, C] uint8 and scales [..., C // group_size, R] float16 -- the
    layout the SM70 op derives group_size from (num_groups = scales' dim -2).
    """
    *lead, r, c = w.shape
    wg = w.view(*lead, r, c // group_size, group_size)
    s = (wg.abs().amax(dim=-1, keepdim=True) * _U2_SCALE_AMAX_MULT).clamp_min(1e-12)
    codes = torch.round(wg / s + _U2_CODE_BIAS).clamp_(0, 3).to(torch.uint8)
    scales = s.squeeze(-1).permute(*range(len(lead)), len(lead) + 1, len(lead))
    scales = scales.contiguous().to(torch.float16)
    return codes.view(*lead, r, c), scales


def repack_u2_sm70(codes_u8: torch.Tensor, packed_macro_n: int) -> torch.Tensor:
    """codes_u8 [E, R(n), C(k)] values 0..3 -> b_q_weight [E, C/16, R] int32.

    Executable port of the packing contract documented in
    sm70_marlin_u2_gemm.cu: one uint32 word holds the 16 codes of one
    (k, 16-column) row, and within every 8-column run bit-pair p holds the
    run's column (p % 4) * 2 + p / 4 (low halfword = columns 0..7, high =
    8..15). ``packed_macro_n`` must match sm70_marlin_auto_packed_macro_n.
    """
    e, r, c = codes_u8.shape
    assert c % 16 == 0 and r % 64 == 0 and 64 <= packed_macro_n <= 256
    k_group_tiles = packed_macro_n // 64
    dev = codes_u8.device

    j = torch.arange(16, device=dev)
    pos = (j % 8 >> 1) + ((j & 1) << 2) + (j // 8) * 8
    codes_kn = codes_u8.permute(0, 2, 1).contiguous()  # [E, K, N]
    pos_ordered = codes_kn.reshape(e, c, r // 16, 16)[..., torch.argsort(pos)].int()
    words = (pos_ordered << (2 * j)).sum(-1).to(torch.int32)  # [E, C, R/16]

    # addr(k, n16) = (k/16)*R + group*kG*64 + (k%16*4 + n16%4)*kG
    #               + ((n16/4) % kG)
    kk = torch.arange(c, device=dev)
    m = torch.arange(r // 16, device=dev)
    n_tile = m // 4
    addr = (
        (kk // 16).view(-1, 1) * r
        + ((kk % 16).view(-1, 1) * 4 + (m % 4).view(1, -1)) * k_group_tiles
        + (n_tile // k_group_tiles).view(1, -1) * k_group_tiles * 64
        + (n_tile % k_group_tiles).view(1, -1)
    )  # [C, R/16]
    out = torch.zeros(e, (c // 16) * r, dtype=torch.int32, device=dev)
    flat_idx = addr.reshape(1, -1).expand(e, -1)
    out.scatter_(1, flat_idx, words.reshape(e, -1))
    return out.view(e, c // 16, r)


def _macro_for_n(n: int) -> int:
    """Mirror of sm70_marlin_auto_packed_macro_n for the host packer."""
    if n % 256 == 0:
        return 256
    if n % 128 == 0:
        return 128
    return 64


_U2_CHUNK_EXPERTS = 16


def convert_moe_layer_to_u2(layer: torch.nn.Module) -> None:
    """Requantize one FusedMoE layer's staged NVFP4 experts into the u2 pool.

    Runs inside process_weights_after_loading, before any CUDA graph
    capture. Frees the host staging by rebinding the layer parameters.
    """
    global _ANNOUNCED

    from sglang.srt.layers.quantization.dequantization import dequantize_nvfp4

    is_gated = layer.moe_runner_config.is_gated
    t0 = time.perf_counter()
    # Staging shapes: w13 [E, 2I, H/2] u8 / w2 [E, H, I/2] u8.
    num_experts, n13, packed_hidden = layer.w13_weight.shape
    hidden = packed_hidden * 2
    inter = layer.w2_weight.shape[2] * 2
    assert (n13 == 2 * inter) == is_gated, "staging/gated mismatch"
    group_size = resolve_u2_group_size(hidden, inter)
    macro13 = _macro_for_n(n13)
    macro2 = _macro_for_n(hidden)
    device = layer.w13_weight_scale_2.device

    pool_w13 = torch.empty(
        num_experts, hidden // 16, n13, dtype=torch.int32, device=device
    )
    pool_w2 = torch.empty(
        num_experts, inter // 16, hidden, dtype=torch.int32, device=device
    )
    pool_s13 = torch.empty(
        num_experts, hidden // group_size, n13, dtype=torch.float16, device=device
    )
    pool_s2 = torch.empty(
        num_experts, inter // group_size, hidden, dtype=torch.float16, device=device
    )

    # Per-half weight_scale_2 for the fused W13: gate rows x s2[:, 0], up
    # rows x s2[:, 1]. Folded here (not at apply) so the u2 group scales
    # carry the full weight scale.
    s2_13 = layer.w13_weight_scale_2.data.to(torch.float32)
    if is_gated and s2_13.dim() == 2 and s2_13.shape[1] >= 2:
        gate_s2 = s2_13[:, 0]
        up_s2 = s2_13[:, 1]
    else:
        gate_s2 = s2_13.reshape(num_experts)
        up_s2 = gate_s2
    s2_2 = layer.w2_weight_scale_2.data.to(torch.float32).reshape(num_experts)

    for lo in range(0, num_experts, _U2_CHUNK_EXPERTS):
        hi = min(lo + _U2_CHUNK_EXPERTS, num_experts)

        w13_q = layer.w13_weight.data[lo:hi].to(device, non_blocking=True)
        half_rows = gate_s2[lo:hi].view(-1, 1).expand(-1, inter)
        up_rows = up_s2[lo:hi].view(-1, 1).expand(-1, inter)
        row_s2 = torch.stack([half_rows, up_rows], dim=1).reshape(-1, 1)
        w13_s = layer.w13_weight_scale.data[lo:hi].to(device).float() * row_s2.view(
            hi - lo, n13, 1
        )
        w13_fp32 = dequantize_nvfp4(w13_q, w13_s, None, torch.float32)
        codes, scales = requant_u2(w13_fp32, group_size)
        pool_w13[lo:hi].copy_(repack_u2_sm70(codes, macro13))
        pool_s13[lo:hi].copy_(scales)
        del w13_q, w13_s, w13_fp32, codes, scales, half_rows, up_rows, row_s2

        w2_q = layer.w2_weight.data[lo:hi].to(device, non_blocking=True)
        w2_s = layer.w2_weight_scale.data[lo:hi].to(device).float() * s2_2[
            lo:hi
        ].view(-1, 1, 1)
        w2_fp32 = dequantize_nvfp4(w2_q, w2_s, None, torch.float32)
        codes, scales = requant_u2(w2_fp32, group_size)
        pool_w2[lo:hi].copy_(repack_u2_sm70(codes, macro2))
        pool_s2[lo:hi].copy_(scales)
        del w2_q, w2_s, w2_fp32, codes, scales

    copy_or_rebind_param(layer, "w13_weight", pool_w13)
    copy_or_rebind_param(layer, "w2_weight", pool_w2)
    copy_or_rebind_param(layer, "w13_weight_scale", pool_s13)
    copy_or_rebind_param(layer, "w2_weight_scale", pool_s2)
    # Last tensor references died with the rebinds; release the page cache
    # and disk now rather than at process exit.
    _free_u2_staging(layer)

    # The u4 Marlin path's Triton fallback reads w13_scale2/w2_scale2; the
    # u2 pool has no fallback and no global scale, so none are set.
    elapsed = time.perf_counter() - t0
    if not _ANNOUNCED:
        _ANNOUNCED = True
        pool_gib = (
            pool_w13.nbytes + pool_w2.nbytes + pool_s13.nbytes + pool_s2.nbytes
        ) / 2**30
        logger.info(
            "SM70 u2 expert pool: requantized first layer (E=%d H=%d I=%d "
            "g=%d macro=%d/%d, %.2f GiB/rank per layer) in %.2fs",
            num_experts,
            hidden,
            inter,
            group_size,
            macro13,
            macro2,
            pool_gib,
            elapsed,
        )
    else:
        logger.debug(
            "SM70 u2 expert pool: layer converted in %.2fs (E=%d)",
            elapsed,
            num_experts,
        )
