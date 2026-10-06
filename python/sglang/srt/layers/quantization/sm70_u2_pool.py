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

With SGLANG_USE_SM70_U2_GEMM_V2 set (and the shape gate met), the two
weight pools are instead bound in the sm70_u2_gemm_v2 transposed layout
(same shapes, permuted words -- see u2_packed_to_T) and
``layer._u2_v2_words`` marks the mode for the apply path. The persistent
cache always stores the marlin format, so the env only decides the final
GPU binding and flipping it just re-derives on the next boot.

The requantization folds each expert's NVFP4 ``weight_scale_2`` into the
dequantized fp32 weights, so the u2 group scales are the only scale at
apply time (no per-expert global scale, unlike the u4 Marlin path).
"""

import hashlib
import logging
import math
import os
import re
import shutil
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
_STALE_STAGING_SWEPT = False

# Layer ids whose checkpoint expert bytes were skipped because the
# persistent cache promised a hit (see u2_checkpoint_reader). Empty unless
# this boot armed the skip; convert_moe_layer_to_u2 hard-fails on a broken
# promise instead of converting empty staging into a zero expert pool.
_U2_SKIP_ARMED_LAYER_IDS: frozenset = frozenset()


def _sweep_stale_staging(stage_dir: str) -> None:
    """Delete stage files whose writer pid no longer exists.

    A boot killed mid-load (crash, OOM, container rm) leaves its per-layer
    staging memmaps behind -- tens of GB per rank -- and nothing else ever
    reclaims them. Runs once per process, before the first staging alloc.
    The pid in the file name is interpreted in this container's pid
    namespace, same as the writer's; a pid that is alive but recycled can
    only make the sweep skip a file, never delete a live one.
    """
    freed = kept = 0
    try:
        names = os.listdir(stage_dir)
    except OSError as e:
        logger.warning("u2 staging sweep: cannot list %s: %s", stage_dir, e)
        return
    for name in names:
        if not name.startswith("sglang_u2_stage_"):
            continue
        try:
            pid = int(name.split("_")[3])
        except (IndexError, ValueError):
            continue
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            try:
                os.unlink(os.path.join(stage_dir, name))
                freed += 1
            except OSError:
                kept += 1
        except OSError:
            # Alive (or unreclaimable); another process's problem.
            kept += 1
    if freed:
        logger.info(
            "u2 staging sweep: removed %d stale stage file(s) from %s "
            "(%d skipped, writer pid alive)",
            freed,
            stage_dir,
            kept,
        )


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

    global _STALE_STAGING_SWEPT
    stage_dir = envs.SGLANG_SM70_U2_STAGE_DIR.get() or tempfile.gettempdir()
    if not _STALE_STAGING_SWEPT:
        _STALE_STAGING_SWEPT = True
        _sweep_stale_staging(stage_dir)
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


def repack_u2_sm70T(codes_u8: torch.Tensor) -> torch.Tensor:
    """codes_u8 [E, R(n), C(k)] values 0..3 -> words [E, C/16, R] int32.

    Direct builder for the sm70_u2_gemm_v2 transposed layout: one uint32
    holds 16 consecutive-k codes of ONE column; code j = k %% 16 sits at BIT
    position (j even ? j : j + 15) -- even-j codes in the low halfword at
    nibble positions, odd-j codes in the high halfword (sm70_u2_gemm_v2.cu
    u2_deq). Production never calls this (weights are derived from the cached
    marlin words via u2_packed_to_T); it is the independent reference the
    layout unit test cross-checks that inverse against.
    """
    e, r, c = codes_u8.shape
    assert c % 16 == 0
    dev = codes_u8.device
    codes = codes_u8.view(e, r, c // 16, 16).to(torch.int32)
    j = torch.arange(16, device=dev)
    pos = torch.where(j % 2 == 0, j, j + 15)
    words = (codes << pos.view(1, 1, 1, 16)).sum(-1, dtype=torch.int32)
    return words.permute(0, 2, 1).contiguous()


def u2_packed_to_T(
    packed: torch.Tensor, r: int, c: int, packed_macro_n: int
) -> torch.Tensor:
    """marlin packed u2 words [E, C/16, R] -> sm70_u2_gemm_v2 words [E, C/16, R].

    Exact inverse chain of repack_u2_sm70: undo the macro-N scatter (gather
    with the same address matrix), unpack the 16 run-column codes of each
    (k, 16-column) word (code of run-column q lives at bit-pair pos[q]), and
    repack k-major -- one uint32 = 16 consecutive-k codes of ONE column, code
    j = k %% 16 at BIT position (j even ? j : j + 15), matching the T layout
    the v2 kernel documents. Same shape as the input; verified bit-exact
    against repack_u2_sm70T(requant codes) in test_sm70_u2_gemm_v2_layout.py.
    """
    e = packed.shape[0]
    assert c % 16 == 0 and r % 64 == 0 and 64 <= packed_macro_n <= 256
    k_group_tiles = packed_macro_n // 64
    dev = packed.device

    kk = torch.arange(c, device=dev)
    m = torch.arange(r // 16, device=dev)
    n_tile = m // 4
    addr = (
        (kk // 16).view(-1, 1) * r
        + ((kk % 16).view(-1, 1) * 4 + (m % 4).view(1, -1)) * k_group_tiles
        + (n_tile // k_group_tiles).view(1, -1) * k_group_tiles * 64
        + (n_tile % k_group_tiles).view(1, -1)
    )  # [C, R/16]
    words = (packed.reshape(e, -1).gather(1, addr.reshape(1, -1).expand(e, -1))).view(
        e, c, r // 16
    )  # logical [k, n16] word grid

    # codes[e, k, n16, q] = run-column q of the (k, n16) word.
    j = torch.arange(16, device=dev)
    pos = (j % 8 >> 1) + ((j & 1) << 2) + (j // 8) * 8
    codes = (words.unsqueeze(-1) >> (2 * pos).view(1, 1, 1, 16)) & 3

    # T word (kw, n): code j = k % 16 at BIT position pos_T[j] (even j in the
    # low halfword at nibble positions, odd j in the high halfword).
    codes = codes.view(e, c // 16, 16, r // 16, 16)  # [e, kw, j, n16, q]
    codes = codes.permute(0, 1, 3, 4, 2).reshape(e, c // 16, r, 16)
    pos_t = torch.where(j % 2 == 0, j, j + 15)
    return (codes.int() << pos_t.view(1, 1, 1, 16)).sum(-1, dtype=torch.int32)


def _u2_v2_layout_ok(layout: dict) -> bool:
    """Shape gate for sm70_u2_gemm_v2 (kernel TORCH_CHECKs mirrored)."""
    hidden = layout["hidden"]
    inter = layout["inter"]
    n13 = layout["n13"]
    return (
        hidden % 64 == 0
        and inter % 64 == 0
        and n13 % 128 == 0
        and hidden % 128 == 0
        and layout["group_size"] in (32, 64, 128)
    )


def _repack_pools_to_v2(
    layout: dict, w13_words: torch.Tensor, w2_words: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked marlin -> v2 transposed repack of both weight pools."""
    w13_t = torch.empty_like(w13_words)
    w2_t = torch.empty_like(w2_words)
    for lo in range(0, layout["num_experts"], _U2_CHUNK_EXPERTS):
        hi = min(lo + _U2_CHUNK_EXPERTS, layout["num_experts"])
        w13_t[lo:hi] = u2_packed_to_T(
            w13_words[lo:hi], layout["n13"], layout["hidden"], layout["macro13"]
        )
        w2_t[lo:hi] = u2_packed_to_T(
            w2_words[lo:hi], layout["hidden"], layout["inter"], layout["macro2"]
        )
    return w13_t, w2_t


_V2_SELFTEST_DONE = False


def _u2_v2_binding_selftest(
    layer: torch.nn.Module,
    w13_m: torch.Tensor,
    w2_m: torch.Tensor,
    w13_t: torch.Tensor,
    w2_t: torch.Tensor,
    s13: torch.Tensor,
    s2: torch.Tensor,
    layout: dict,
) -> None:
    """One-shot plumbing guard for the v2 word binding (U2B2).

    Runs fused_marlin_moe twice on identical tiny inputs -- once on the
    marlin-format words with u2_v2_words=False, once on the v2 words with
    True -- and requires bit equality. If the flag fails to reach the GEMM
    dispatch (a dropped kwarg on some call path, a new call site, ...), the
    first call's marlin kernel reads the v2 words as marlin bytes and the
    outputs diverge. That failure mode is silent garbage in serve, so it
    must fail the boot instead.
    """
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
        fused_marlin_moe,
    )

    device = w13_t.device
    # M=1024 reproduces the production prefill metadata (both arms pick
    # moe_block_size 32); tiny M sends the fused path through metadata whose
    # uninitialized intermediates leak NaN into BOTH arms and would make the
    # guard flaky.
    m, top_k = 1024, layer.moe_runner_config.top_k
    gen = torch.Generator(device="cpu").manual_seed(20261005)
    h = (
        torch.randn(m, layout["hidden"], generator=gen).to(
            device=device, dtype=torch.float16
        )
        * 0.1
    )
    g = torch.randn(m, layout["num_experts"], generator=gen).to(
        device=device, dtype=torch.float16
    )
    ti = torch.randint(
        0, layout["num_experts"], (m, top_k), generator=gen, dtype=torch.int32
    ).to(device)
    tw = torch.rand(m, top_k, generator=gen).to(device=device, dtype=torch.float32)

    def run(w1, w2, v2: bool):
        return fused_marlin_moe(
            hidden_states=h,
            w1=w1,
            w2=w2,
            w1_scale=s13,
            w2_scale=s2,
            gating_output=g,
            topk_weights=tw,
            topk_ids=ti,
            workspace=torch.zeros(4096, dtype=torch.int32, device=device),
            num_bits=2,
            activation=layer.moe_runner_config.activation,
            is_gated=layout["is_gated"],
            clamp_limit=(
                layer.moe_runner_config.gemm1_clamp_limit
                if layer.moe_runner_config.gemm1_alpha is not None
                else layer.moe_runner_config.swiglu_limit
            ),
            gemm1_alpha=layer.moe_runner_config.gemm1_alpha,
            u2_v2_words=v2,
        )

    y_ref = run(w13_m, w2_m, v2=False)
    y_v2 = run(w13_t, w2_t, v2=True)
    if not torch.isclose(
        y_ref.float(), y_v2.float(), rtol=0, atol=0, equal_nan=True
    ).all():
        raise RuntimeError(
            "SM70 u2 v2 binding self-test failed: fused_marlin_moe output "
            "differs between the marlin-format and v2-transposed words; the "
            "u2_v2_words flag is not reaching the GEMM dispatch, so the "
            "marlin kernel would read the v2 bytes as silent garbage"
        )
    logger.info(
        "SM70 u2 v2 binding self-test passed (layer %s, words=v2-transposed)",
        layer.layer_id,
    )


_U2_CHUNK_EXPERTS = 16


# Bump when anything that determines the cached u2 bytes changes: requant_u2,
# repack_u2_sm70, the decode grid (_U2_CODE_BIAS / _U2_SCALE_AMAX_MULT), the
# group-size resolution, or the marlin-v100-u2-experts kernel's baked-in bias.
_U2_CACHE_VERSION = 1

_U2_CACHE_HANDLE = None


def _u2_cache_fingerprint(model_path: str, tp_size: int) -> str:
    """Fingerprint of everything that determines the converted u2 bytes.

    Weight-file identity is (name, size, mtime_ns) -- the build-system style
    staleness check: cheap, and it re-fires on any re-quantized checkpoint
    or model swap. Content hashes would be bulletproof but read ~180 GB.
    """
    h = hashlib.sha256()
    h.update(f"u2-cache-v{_U2_CACHE_VERSION}\0tp={tp_size}\0".encode())
    for name in sorted(os.listdir(model_path)):
        if not name.endswith((".safetensors", ".json")):
            continue
        # Sampling-only configs cannot change the converted u2 bytes; hashing
        # their mtime colds the cache on any generation_config.json edit, and a
        # cold requant OOMs at 230k (fixed ~31.1GB staging footprint).
        if name == "generation_config.json":
            continue
        st = os.stat(os.path.join(model_path, name))
        h.update(f"{name}:{st.st_size}:{st.st_mtime_ns}\0".encode())
    return h.hexdigest()[:16]


class _U2LayerCache:
    """Per-boot handle on the persistent converted-u2-pool cache.

    One directory per fingerprint under
    ``$SGLANG_SM70_U2_STAGE_DIR/u2_cache/<fp>/``, one self-describing
    torch.save payload per (rank, layer). A crashed boot leaves a partial
    cache: every stored layer is individually validated on load, so a
    partial cache yields partial hits, never wrong bytes.

    Any failure disables the handle for the rest of the boot (the plain
    conversion path runs); the cache must never break a load.
    """

    def __init__(self, cache_dir: str, rank: int) -> None:
        self.cache_dir = cache_dir
        self.rank = rank
        self.dead = False
        self.hits = 0
        self.stores = 0

    def _disable(self, what: str, err: Exception) -> None:
        self.dead = True
        logger.warning("u2 cache disabled after %s failure: %s", what, err)

    def _layer_path(self, layer: torch.nn.Module) -> str:
        return os.path.join(
            self.cache_dir, f"r{self.rank}_layer{int(layer.layer_id)}.pt"
        )

    def load_layer(
        self,
        layer: torch.nn.Module,
        group_size: int,
        macro13: int,
        macro2: int,
        device: torch.device,
    ):
        """Return the cached pool tensors, or None on miss/mismatch."""
        if self.dead:
            return None
        path = self._layer_path(layer)
        if not os.path.exists(path):
            return None
        try:
            payload = torch.load(path, map_location=device, weights_only=True)
            meta = payload["meta"]
            if (
                meta["group_size"] != group_size
                or meta["macro13"] != macro13
                or meta["macro2"] != macro2
            ):
                logger.info(
                    "u2 cache: layer %s meta mismatch, requantizing",
                    layer.layer_id,
                )
                return None
            pools = {k: payload[k] for k in ("w13", "w2", "s13", "s2")}
            self.hits += 1
            return pools
        except Exception as e:  # noqa: BLE001 -- cache must never break a load
            self._disable(f"load layer {layer.layer_id}", e)
            return None

    def store_layer(
        self,
        layer: torch.nn.Module,
        group_size: int,
        macro13: int,
        macro2: int,
        pools: dict,
    ) -> None:
        if self.dead:
            return
        path = self._layer_path(layer)
        tmp = path + f".tmp{os.getpid()}"
        try:
            payload = {
                "meta": {
                    "layer_id": int(layer.layer_id),
                    "group_size": group_size,
                    "macro13": macro13,
                    "macro2": macro2,
                },
                **pools,
            }
            torch.save(payload, tmp)
            # Atomic publish: a reader never sees a half-written payload.
            os.replace(tmp, path)
            self.stores += 1
        except Exception as e:  # noqa: BLE001
            try:
                os.unlink(tmp)
            except OSError:
                pass
            self._disable(f"store layer {layer.layer_id}", e)


def _u2_cache_open() -> _U2LayerCache | None:
    """Bind the boot's cache handle once; None when disabled or unusable."""
    global _U2_CACHE_HANDLE
    if not envs.SGLANG_SM70_U2_CACHE.get():
        return None
    if _U2_CACHE_HANDLE is not None:
        return None if _U2_CACHE_HANDLE.dead else _U2_CACHE_HANDLE
    try:
        from sglang.srt.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        model_path = _u2_cache_model_path()
        stage_dir = envs.SGLANG_SM70_U2_STAGE_DIR.get() or tempfile.gettempdir()
        root = os.path.join(stage_dir, "u2_cache")
        os.makedirs(root, exist_ok=True)
        fp = _u2_cache_fingerprint(model_path, tp_size)
        cache_dir = os.path.join(root, fp)
        os.makedirs(cache_dir, exist_ok=True)
        _u2_cache_prune(root, cache_dir)
        _U2_CACHE_HANDLE = _U2LayerCache(cache_dir=cache_dir, rank=rank)
        logger.info(
            "u2 cache: %s (fp=%s rank=%d dir=%s)",
            "cold" if not os.listdir(cache_dir) else "open",
            fp,
            rank,
            cache_dir,
        )
        return _U2_CACHE_HANDLE
    except Exception as e:  # noqa: BLE001
        if _U2_CACHE_HANDLE is None:
            _U2_CACHE_HANDLE = _U2LayerCache(cache_dir="", rank=-1)
        _U2_CACHE_HANDLE._disable("open", e)
        return None


def _u2_cache_model_path() -> str:
    from sglang.srt.runtime_context import get_server_args

    return get_server_args().model_path


def _u2_pool_layout_ok(pools: dict, pool_shapes: dict) -> bool:
    """Shape+dtype check of a cached payload against this layer's layout."""
    dtypes = {
        "w13": torch.int32,
        "w2": torch.int32,
        "s13": torch.float16,
        "s2": torch.float16,
    }
    return all(
        key in pools
        and tuple(pools[key].shape) == pool_shapes[key]
        and pools[key].dtype == dtypes[key]
        for key in dtypes
    )


def _u2_cache_prune(root: str, keep: str) -> None:
    """Keep the current fingerprint dir plus the newest other one.

    Older generations are dead weight once their model is gone (~76 GB per
    GLM-5.3 TP4 cache), so switching models auto-reclaims after the next
    successful open.
    """
    try:
        others = [
            os.path.join(root, e.name)
            for e in os.scandir(root)
            if e.is_dir() and os.path.join(root, e.name) != keep
        ]
        others.sort(key=lambda p: os.stat(p).st_mtime, reverse=True)
        for path in others[1:]:
            shutil.rmtree(path)
            logger.info("u2 cache: pruned stale generation %s", path)
    except OSError as e:
        logger.warning("u2 cache: prune skipped (%s)", e)


def _layer_pool_layout(layer: torch.nn.Module) -> dict:
    """Pool shapes + requant params derived from one layer's staged shapes.

    Staging shapes: w13 [E, 2I, H/2] u8 / w2 [E, H, I/2] u8.
    """
    is_gated = layer.moe_runner_config.is_gated
    num_experts, n13, packed_hidden = layer.w13_weight.shape
    hidden = packed_hidden * 2
    inter = layer.w2_weight.shape[2] * 2
    assert (n13 == 2 * inter) == is_gated, "staging/gated mismatch"
    group_size = resolve_u2_group_size(hidden, inter)
    macro13 = _macro_for_n(n13)
    macro2 = _macro_for_n(hidden)
    return {
        "is_gated": is_gated,
        "num_experts": num_experts,
        "n13": n13,
        "hidden": hidden,
        "inter": inter,
        "group_size": group_size,
        "macro13": macro13,
        "macro2": macro2,
        "pool_shapes": {
            "w13": (num_experts, hidden // 16, n13),
            "w2": (num_experts, inter // 16, hidden),
            "s13": (num_experts, hidden // group_size, n13),
            "s2": (num_experts, inter // group_size, hidden),
        },
    }


def convert_moe_layer_to_u2(layer: torch.nn.Module) -> None:
    """Requantize one FusedMoE layer's staged NVFP4 experts into the u2 pool.

    Runs inside process_weights_after_loading, before any CUDA graph
    capture. Frees the host staging by rebinding the layer parameters.
    """
    global _ANNOUNCED, _V2_SELFTEST_DONE

    from sglang.srt.layers.quantization.dequantization import dequantize_nvfp4

    t0 = time.perf_counter()
    layout = _layer_pool_layout(layer)
    pool_shapes = layout["pool_shapes"]
    device = layer.w13_weight_scale_2.device
    # Boot-time contract: the env decides the on-GPU word layout (marlin vs
    # v2 transposed); fused_marlin_moe reads the same env plus layer._u2_v2_words.
    use_v2 = _u2_v2_layout_ok(layout) and envs.SGLANG_USE_SM70_U2_GEMM_V2.get()
    if use_v2:
        from sglang.kernels.ops.moe.sm70_u2_gemm_v2 import (
            sm70_u2_gemm_v2_available,
        )

        # No fallback: marlin reading T-layout bytes is silent garbage, so a
        # stale .so must fail the load instead of serving.
        if not sm70_u2_gemm_v2_available():
            raise RuntimeError(
                "SGLANG_USE_SM70_U2_GEMM_V2 is set but the loaded "
                "marlin_v100 .so does not register sm70_u2_gemm_v2; rebuild "
                "it (scripts/setup_v100_marlin.sh) or unset the env"
            )

    cache = _u2_cache_open()
    if cache is not None:
        pools = cache.load_layer(
            layer,
            group_size=layout["group_size"],
            macro13=layout["macro13"],
            macro2=layout["macro2"],
            device=device,
        )
        if pools is not None and _u2_pool_layout_ok(pools, pool_shapes):
            if use_v2:
                w13_t, w2_t = _repack_pools_to_v2(layout, pools["w13"], pools["w2"])
                if not _V2_SELFTEST_DONE:
                    _u2_v2_binding_selftest(
                        layer,
                        pools["w13"],
                        pools["w2"],
                        w13_t,
                        w2_t,
                        pools["s13"],
                        pools["s2"],
                        layout,
                    )
                    _V2_SELFTEST_DONE = True
                copy_or_rebind_param(layer, "w13_weight", w13_t)
                copy_or_rebind_param(layer, "w2_weight", w2_t)
                layer._u2_v2_words = True
            else:
                copy_or_rebind_param(layer, "w13_weight", pools["w13"])
                copy_or_rebind_param(layer, "w2_weight", pools["w2"])
                layer._u2_v2_words = False
            copy_or_rebind_param(layer, "w13_weight_scale", pools["s13"])
            copy_or_rebind_param(layer, "w2_weight_scale", pools["s2"])
            # The staged NVFP4 bytes are pure garbage on a hit; reclaim the
            # disk now instead of at process exit.
            _free_u2_staging(layer)
            elapsed = time.perf_counter() - t0
            if not _ANNOUNCED:
                _ANNOUNCED = True
                logger.info(
                    "SM70 u2 expert pool: layer %s restored from persistent "
                    "cache in %.2fs (hit, words=%s)",
                    layer.layer_id,
                    elapsed,
                    "v2-transposed" if use_v2 else "marlin",
                )
            else:
                logger.debug(
                    "SM70 u2 expert pool: layer %s restored from cache in %.2fs",
                    layer.layer_id,
                    elapsed,
                )
            return
        if pools is not None:
            # Layout drifted (shape/dtype): treat as a miss and overwrite.
            logger.info(
                "u2 cache: layer %s layout mismatch, requantizing", layer.layer_id
            )

    # Checked after the restore attempt, not inside it: the handle can go
    # dead between the reader's promise (load time) and here, and the
    # armed check must still fire. The checkpoint reader skipped this
    # layer's expert bytes on a promised hit, so the staging is empty and
    # converting it would dequantize into a silent all-zero expert pool
    # (the failure mode smoke_v100.sh exists for) -- fail the load instead.
    if int(layer.layer_id) in _U2_SKIP_ARMED_LAYER_IDS:
        raise RuntimeError(
            f"u2 cache: layer {layer.layer_id} skipped checkpoint expert "
            "reads on a promised cache hit, but the cache missed at "
            "process time. Delete the stale u2_cache generation under "
            "$SGLANG_SM70_U2_STAGE_DIR and reboot."
        )

    pool_w13 = torch.empty(*pool_shapes["w13"], dtype=torch.int32, device=device)
    pool_w2 = torch.empty(*pool_shapes["w2"], dtype=torch.int32, device=device)
    pool_s13 = torch.empty(*pool_shapes["s13"], dtype=torch.float16, device=device)
    pool_s2 = torch.empty(*pool_shapes["s2"], dtype=torch.float16, device=device)

    # Per-half weight_scale_2 for the fused W13: gate rows x s2[:, 0], up
    # rows x s2[:, 1]. Folded here (not at apply) so the u2 group scales
    # carry the full weight scale.
    s2_13 = layer.w13_weight_scale_2.data.to(torch.float32)
    if layout["is_gated"] and s2_13.dim() == 2 and s2_13.shape[1] >= 2:
        gate_s2 = s2_13[:, 0]
        up_s2 = s2_13[:, 1]
    else:
        gate_s2 = s2_13.reshape(layout["num_experts"])
        up_s2 = gate_s2
    s2_2 = layer.w2_weight_scale_2.data.to(torch.float32).reshape(layout["num_experts"])

    for lo in range(0, layout["num_experts"], _U2_CHUNK_EXPERTS):
        hi = min(lo + _U2_CHUNK_EXPERTS, layout["num_experts"])

        w13_q = layer.w13_weight.data[lo:hi].to(device, non_blocking=True)
        half_rows = gate_s2[lo:hi].view(-1, 1).expand(-1, layout["inter"])
        up_rows = up_s2[lo:hi].view(-1, 1).expand(-1, layout["inter"])
        row_s2 = torch.stack([half_rows, up_rows], dim=1).reshape(-1, 1)
        w13_s = layer.w13_weight_scale.data[lo:hi].to(device).float() * row_s2.view(
            hi - lo, layout["n13"], 1
        )
        w13_fp32 = dequantize_nvfp4(w13_q, w13_s, None, torch.float32)
        codes, scales = requant_u2(w13_fp32, layout["group_size"])
        pool_w13[lo:hi].copy_(repack_u2_sm70(codes, layout["macro13"]))
        pool_s13[lo:hi].copy_(scales)
        del w13_q, w13_s, w13_fp32, codes, scales, half_rows, up_rows, row_s2

        w2_q = layer.w2_weight.data[lo:hi].to(device, non_blocking=True)
        w2_s = layer.w2_weight_scale.data[lo:hi].to(device).float() * s2_2[lo:hi].view(
            -1, 1, 1
        )
        w2_fp32 = dequantize_nvfp4(w2_q, w2_s, None, torch.float32)
        codes, scales = requant_u2(w2_fp32, layout["group_size"])
        pool_w2[lo:hi].copy_(repack_u2_sm70(codes, layout["macro2"]))
        pool_s2[lo:hi].copy_(scales)
        del w2_q, w2_s, w2_fp32, codes, scales

    if cache is not None:
        cache.store_layer(
            layer,
            group_size=layout["group_size"],
            macro13=layout["macro13"],
            macro2=layout["macro2"],
            pools={"w13": pool_w13, "w2": pool_w2, "s13": pool_s13, "s2": pool_s2},
        )

    if use_v2:
        w13_t, w2_t = _repack_pools_to_v2(layout, pool_w13, pool_w2)
        if not _V2_SELFTEST_DONE:
            _u2_v2_binding_selftest(
                layer,
                pool_w13,
                pool_w2,
                w13_t,
                w2_t,
                pool_s13,
                pool_s2,
                layout,
            )
            _V2_SELFTEST_DONE = True
        copy_or_rebind_param(layer, "w13_weight", w13_t)
        copy_or_rebind_param(layer, "w2_weight", w2_t)
        layer._u2_v2_words = True
    else:
        copy_or_rebind_param(layer, "w13_weight", pool_w13)
        copy_or_rebind_param(layer, "w2_weight", pool_w2)
        layer._u2_v2_words = False
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
            "g=%d macro=%d/%d, %.2f GiB/rank per layer, words=%s) in %.2fs",
            layout["num_experts"],
            layout["hidden"],
            layout["inter"],
            layout["group_size"],
            layout["macro13"],
            layout["macro2"],
            pool_gib,
            "v2-transposed" if use_v2 else "marlin",
            elapsed,
        )
    else:
        logger.debug(
            "SM70 u2 expert pool: layer converted in %.2fs (E=%d)",
            elapsed,
            layout["num_experts"],
        )


# Checkpoint names that feed the four pool params of one FusedMoE layer.
# Only the big suffixes are skipped: weight (packed fp4) and weight_scale
# (fp8 block scales). weight_scale_2 / input_scale are a few KB per layer
# and keep loading normally.
_U2_EXPERT_SKIP_NAME = re.compile(
    r"\.mlp\.(experts\.\d+|shared_experts)\.(gate_proj|up_proj|down_proj)"
    r"\.(weight|weight_scale)$"
)
_U2_LAYER_IN_NAME = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.")

_POOL_ITEM_BYTES = {"w13": 4, "w2": 4, "s13": 2, "s2": 2}


def u2_checkpoint_reader(model: torch.nn.Module):
    """checkpoint_tensor_reader hook: skip expert bytes the cache rebinds.

    Returns a ``(name, safetensors_handle) -> tensor | None`` callable, or
    None when this boot must read the checkpoint normally (cache off, no
    cached layer present). A layer's expert ``weight``/``weight_scale``
    bytes are skipped only when this rank's persistent cache payload for
    that layer exists with at least the expected byte size; if the promise
    breaks at process time, convert_moe_layer_to_u2 hard-fails rather than
    dequantizing empty staging into a silent all-zero expert pool.
    """
    global _U2_SKIP_ARMED_LAYER_IDS
    if not envs.SGLANG_SM70_U2_CACHE.get() or not sm70_u2_expert_pool_enabled():
        return None
    cache = _u2_cache_open()
    if cache is None or cache.dead:
        return None

    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptNvFp4FusedMoEMethod,
    )

    verified = set()
    for _, mod in model.named_modules():
        if not isinstance(mod, FusedMoE) or not isinstance(
            mod.quant_method, ModelOptNvFp4FusedMoEMethod
        ):
            continue
        if not mod.quant_method.sm70_u2_pool:
            continue
        shapes = _layer_pool_layout(mod)["pool_shapes"]
        expected = sum(
            math.prod(shapes[key]) * item_bytes
            for key, item_bytes in _POOL_ITEM_BYTES.items()
        )
        try:
            if os.path.getsize(cache._layer_path(mod)) >= expected:
                verified.add(int(mod.layer_id))
        except OSError:
            continue
    if not verified:
        return None

    fuse_shared = int(model.num_fused_shared_experts) > 0
    _U2_SKIP_ARMED_LAYER_IDS = frozenset(verified)
    logger.info(
        "u2 cache: skipping checkpoint expert reads for %d cached layers "
        "(r%d, shared_experts %s)",
        len(verified),
        cache.rank,
        "skipped" if fuse_shared else "kept",
    )

    def read(name: str, handle):
        match = _U2_LAYER_IN_NAME.search(name)
        if match is None or int(match.group(1)) not in verified:
            return handle.get_tensor(name)
        if not _U2_EXPERT_SKIP_NAME.search(name):
            return handle.get_tensor(name)
        if "shared_experts" in name and not fuse_shared:
            return handle.get_tensor(name)
        return None

    return read
