"""FP16 MQA logits for the DSA indexer on SM70 (Volta).

Upstream computes the indexer logits with FP8 DeepGEMM
(``fp8_mqa_logits`` / ``fp8_paged_mqa_logits``) or the TileLang FP8
paged-MQA kernel; all of them need SM90+ tensor cores, and Triton on
sm70 has no MMA path at all (``tl.dot`` lowers to scalar FMA there --
verified in the P5-a probe). Volta therefore takes a different,
mathematically identical route built on linearity of the head sum:

    logits[i, j] = sum_h gate[i, h] * (q[i, h] . k[p(i, j)])
                 = (sum_h gate[i, h] * q[i, h]) . k[p(i, j)]
                 = q_eff[i] . k[p(i, j)]

``gate`` carries ``weights_proj(q_lora) * n_heads**-0.5 * q_scale *
softmax_scale`` (mirroring ``Indexer._get_logits_head_gate``). The
head-fold runs in FP32 (the llama-glm5 reference accumulates the
indexer weights in FP32); the collapsed GEMM is one rounding to FP16 --
still ~2^8 more precise than upstream's per-head FP8 quantization --
with FP32 accumulation inside cuBLAS / the paged kernel.

Cost drops 32x against the per-head formulation (H collapses before
the GEMM instead of after): prefill logits are one cuBLAS FP16 GEMM
with FP32 output; decode / target-verify is a memory-bound Triton
paged kernel with no tensor-core requirement.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


def gate_fold(q: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Collapse heads with the per-(row, head) gate.

    q: [n, H, D] fp16 (fp32 also accepted); gate: [n, H] fp32.
    Returns q_eff: [n, D] fp16.
    """
    qf = q.float()
    q_eff = torch.einsum("nh,nhd->nd", gate, qf)
    return q_eff.to(q.dtype if q.dtype.is_floating_point else torch.float16)


def fp16_ragged_mqa_logits(
    q_eff: torch.Tensor,
    k: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Ragged logits for the extend path, mirroring deep_gemm.fp8_mqa_logits.

    q_eff: [n, D] fp16 from :func:`gate_fold`; k: [total_k, D] fp16;
    ks/ke: [n] int32 inclusive/exclusive per-row k ranges. Returns
    [n, max(ke-ks)] FP32 where column j holds k row ``ks[i] + j``
    (zeros outside the row's range; the caller passes ``row_starts=ks``
    to the top-k transform, which restores absolute positions).

    Rows are grouped by their range start (one group per sequence in a
    chunked-prefill batch); each group runs one dense cuBLAS FP16 GEMM
    with FP32 output over its own k slice, then masks the causal tail.
    The hot single-sequence case is exactly one GEMM with no copies.
    The grouping reads ks/ke on host (.tolist()); callers on the
    per-layer hot path should hoist that sync to once per forward batch.
    """
    n, _ = q_eff.shape
    span = int((ke - ks).max().item()) if n else 0
    if out is None:
        out = torch.zeros((n, span), dtype=torch.float32, device=q_eff.device)
    else:
        assert out.dtype == torch.float32 and out.shape[1] >= span
    if n == 0 or span == 0:
        return out

    ks_l = ks.tolist()
    ke_l = ke.tolist()
    groups: dict = {}
    for i in range(n):
        groups.setdefault(ks_l[i], []).append(i)

    cols_buf = None
    zero = torch.zeros((), device=q_eff.device)
    for ks_v, rows in groups.items():
        ke_max = max(ke_l[i] for i in rows)
        rows_t = None
        if len(rows) == n and ks_v == 0:
            q_g = q_eff  # hot path: one sequence, no gather
            ke_t = ke
        else:
            rows_t = torch.tensor(rows, dtype=torch.long, device=q_eff.device)
            q_g = q_eff.index_select(0, rows_t)
            ke_t = ke.index_select(0, rows_t)
        logits = torch.mm(q_g, k[ks_v:ke_max].t(), out_dtype=torch.float32)
        width = logits.shape[1]
        if cols_buf is None or cols_buf.numel() < width:
            cols_buf = torch.arange(width, device=q_eff.device, dtype=torch.int32)
        valid = cols_buf[None, :width] < (ke_t - ks_v)[:, None]
        logits = torch.where(valid, logits, zero)
        if rows_t is None:
            out[:, :width] = logits
        else:
            padded = torch.zeros(
                (len(rows), span), dtype=torch.float32, device=q_eff.device
            )
            padded[:, :width] = logits
            out.index_copy_(0, rows_t, padded)
    return out


@triton.jit
def _fp16_paged_mqa_logits_kernel(
    qeff_ptr,  # [n, D] fp16
    kcache_ptr,  # [num_pages, PAGE, D] fp16 pooled K cache
    seqlens_ptr,  # [n] int32, pooled lengths per row
    bt_ptr,  # [n, max_pages] int32 pooled page table
    out_ptr,  # [n, out_stride] fp32, col = pooled position
    stride_qn,
    stride_kp,
    stride_kt,
    stride_bt,
    stride_on,
    D: tl.constexpr,
    PAGE: tl.constexpr,
):
    """One program: one row x one pooled page. Memory-bound elementwise
    reduce -- no tl.dot, so no dependence on sm70 MMA support."""
    pid_q = tl.program_id(0)
    page_slot = tl.program_id(1)

    seq_len = tl.load(seqlens_ptr + pid_q)
    if page_slot * PAGE >= seq_len:
        return
    page = tl.load(bt_ptr + pid_q * stride_bt + page_slot)

    offs_d = tl.arange(0, D)
    offs_t = tl.arange(0, PAGE)
    t_mask = (page_slot * PAGE + offs_t) < seq_len

    q_eff = tl.load(qeff_ptr + pid_q * stride_qn + offs_d).to(tl.float32)
    k = tl.load(
        kcache_ptr + page * stride_kp + offs_t[:, None] * stride_kt + offs_d[None, :],
        mask=t_mask[:, None],
        other=0.0,
    ).to(tl.float32)  # [PAGE, D]
    logits = tl.sum(q_eff[None, :] * k, axis=1)  # [PAGE]

    tl.store(
        out_ptr + pid_q * stride_on + page_slot * PAGE + offs_t,
        logits,
        mask=t_mask,
    )


def fp16_paged_mqa_logits(
    q_eff: torch.Tensor,
    kcache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    max_len: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pooled-paged logits for decode / target-verify, mirroring
    deep_gemm.fp8_paged_mqa_logits (kpool layout, next_n == 1).

    q_eff: [n, D] fp16 from :func:`gate_fold`; kcache: [num_pages, PAGE, D]
    fp16 pooled K cache; seqlens: [n] int32 pooled lengths; block_tables:
    [n, max_pages] int32. Returns [n, stride] FP32 with column = pooled
    position; ``stride`` is 256-aligned like the DeepGEMM layout the fused
    top-k kernel expects.
    """
    n, d = q_eff.shape
    _, page, d2 = kcache.shape
    assert d == d2 and page in (32, 64, 128), (page, d)
    assert block_tables.shape[1] >= triton.cdiv(max_len, page)
    stride = (max_len + 255) // 256 * 256
    if out is None:
        out = torch.zeros((n, stride), dtype=torch.float32, device=q_eff.device)
    else:
        assert out.stride(1) == 1 and out.shape[1] >= stride
    grid = (n, triton.cdiv(max_len, page))
    if grid[0] and grid[1]:
        _fp16_paged_mqa_logits_kernel[grid](
            q_eff,
            kcache,
            seqlens,
            block_tables,
            out,
            q_eff.stride(0),
            kcache.stride(0),
            kcache.stride(1),
            block_tables.stride(0),
            out.stride(0),
            D=d,
            PAGE=page,
            num_warps=2,
        )
    return out


@triton.jit
def _e4m3_to_f32(x_u8):
    """Decode fp8 e4m3 by bit math: Volta Triton has no fp8e4nv loads.

    NaN payloads (0x7f / 0xff) decode as 480; the cache writer clamps to
    +-448 and never emits NaN, so the case is unreachable.
    """
    u = x_u8.to(tl.int32)
    sign = tl.where((u & 128) != 0, -1.0, 1.0)
    exp = ((u >> 3) & 15).to(tl.float32)
    man = (u & 7).to(tl.float32)
    normal = tl.exp2(exp - 7.0) * (1.0 + man * 0.125)
    sub = man * 0.001953125  # 2**-9
    return sign * tl.where(exp == 0.0, sub, normal)


@triton.jit
def _fp16_q_fp8k_paged_mqa_logits_kernel(
    qeff_ptr,  # [n, D] fp16
    kbytes_ptr,  # [num_pages, PAGE_BYTES] uint8 pooled fp8 K cache
    scale_ptr,  # [num_pages, PAGE] fp32 view of the per-page scale area
    seqlens_ptr,  # [n] int32 pooled lengths per row
    bt_ptr,  # [n_bt, max_pages] int32 pooled page table
    out_ptr,  # [n, out_stride] fp32, col = pooled position
    stride_qn,
    stride_kp,
    stride_sp,
    stride_bt,
    stride_on,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    NEXT_N: tl.constexpr,
):
    """Same contract as :func:`_fp16_paged_mqa_logits_kernel` but reading the
    upstream fp8 pooled cache layout directly (per page: PAGE*D key bytes
    then PAGE*4 scale bytes, see index_buf_accessor.GetK/GetS) and
    dequantizing in-kernel: k_fp32 = e4m3(k) * scale. The write pipeline
    stays byte-identical to upstream. NEXT_N is the MTP draft width: q rows
    are n_bt * NEXT_N and draft rows of one sequence share its pooled page
    table row (the deep_gemm scheduler folds this mapping; here it is one
    integer divide)."""
    pid_q = tl.program_id(0)
    page_slot = tl.program_id(1)

    seq_len = tl.load(seqlens_ptr + pid_q)
    if page_slot * PAGE >= seq_len:
        return
    page = tl.load(bt_ptr + (pid_q // NEXT_N) * stride_bt + page_slot)

    offs_d = tl.arange(0, D)
    offs_t = tl.arange(0, PAGE)
    t_mask = (page_slot * PAGE + offs_t) < seq_len

    q_eff = tl.load(qeff_ptr + pid_q * stride_qn + offs_d).to(tl.float32)
    kb = tl.load(
        kbytes_ptr + page * stride_kp + offs_t[:, None] * D + offs_d[None, :],
        mask=t_mask[:, None],
        other=0,
    )
    k = _e4m3_to_f32(kb)  # [PAGE, D]
    scale = tl.load(
        scale_ptr + page * stride_sp + offs_t, mask=t_mask, other=0.0
    )  # [PAGE]
    logits = tl.sum(q_eff[None, :] * k, axis=1) * scale  # [PAGE]

    tl.store(
        out_ptr + pid_q * stride_on + page_slot * PAGE + offs_t,
        logits,
        mask=t_mask,
    )


def fp16_paged_mqa_logits_fp8kcache(
    q_eff: torch.Tensor,
    kcache_bytes: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    max_len: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """:func:`fp16_paged_mqa_logits` against the upstream fp8 pooled cache
    buffer (uint8 [num_pages, PAGE*D + PAGE*4], k-major with a trailing
    per-page scale area). Dequantization happens in-kernel, so the pooled
    fp8 cache and its whole write pipeline stay untouched. q rows may be
    block-table rows * MTP draft width; draft rows share the table row.
    """
    n, d = q_eff.shape
    assert kcache_bytes.dtype == torch.uint8 and kcache_bytes.dim() == 2
    _, page_bytes = kcache_bytes.shape
    page = page_bytes // (d + 4)
    assert page in (32, 64, 128) and page * (d + 4) == page_bytes, page_bytes
    assert block_tables.shape[1] >= triton.cdiv(max_len, page)
    n_bt = block_tables.shape[0]
    assert n % n_bt == 0, (n, n_bt)
    next_n = n // n_bt
    scale = kcache_bytes[:, page * d :].view(torch.float32)
    stride = (max_len + 255) // 256 * 256
    if out is None:
        out = torch.zeros((n, stride), dtype=torch.float32, device=q_eff.device)
    else:
        assert out.stride(1) == 1 and out.shape[1] >= stride
    grid = (n, triton.cdiv(max_len, page))
    if grid[0] and grid[1]:
        _fp16_q_fp8k_paged_mqa_logits_kernel[grid](
            q_eff,
            kcache_bytes,
            scale,
            seqlens,
            block_tables,
            out,
            q_eff.stride(0),
            kcache_bytes.stride(0),
            scale.stride(0),
            block_tables.stride(0),
            out.stride(0),
            D=d,
            PAGE=page,
            NEXT_N=next_n,
            num_warps=2,
        )
    return out
