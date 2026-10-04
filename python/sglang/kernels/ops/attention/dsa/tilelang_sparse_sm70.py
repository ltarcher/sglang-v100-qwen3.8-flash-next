"""SM70 (Volta) TileLang sparse MLA for the DSA layers.

Upstream's ``tilelang_kernel.py`` sparse kernels are sized for Hopper:
``sparse_attention_fwd_kernel_v2`` is warp-specialized over 384 threads
with ~170KB of shared memory, and even the simpler v1 double-buffers its
[64, 512] KV tile (~140KB at num_stages=2). V100 offers 96KB per block,
so neither kernel can launch there.

This file keeps v1's algorithm and grid shape (one program per query
token, all heads inside the program, online softmax in exp2 domain,
index gather straight from the page-size-1 KV pool) and resizes it to
the Volta budget:

- 128 threads (4 warps): the sm70 gemm emitter requires every warp to
  own a full 16x16 MMA tile, so with M=16 (the GLM TP4 head count)
  FullCol only partitions under 4 warps -- 256 threads would split M
  and trip the emitter's warp_row_tiles assert.
- dtype fp16: Volta WMMA has no bf16 path, and the whole machine runs
  the fp16 pipeline (SGLANG_SM70_FORCE_FP16=1). Q/K/S are fp16, the
  accumulation stays fp32.
- K and V live in separate buffers: on sm70 the QK gemm (B transposed)
  and the PV gemm (B non-transposed) want different Volta swizzle
  layouts, and one physical buffer cannot satisfy both -- the same
  "Get different layout" TVM failure the M4 probe hit on the d512 paged
  MLA kernel. Both buffers are half-width [BI, D/2] and each D half is
  gathered into them in turn, which keeps the tile at 32KB + 32KB.
- no O_shared: the fp16 [H, D] staging tile is dropped and the fp32
  accumulator is written to global memory directly (v1 does the same in
  its final copy).

Shared-memory total at the GLM shape (H=16, D=512, tail=64, BI=64):
Q 16KB + K/V 64KB + K_tail 8KB + S 2KB = 90KB, against the 96KB Volta
per-block ceiling.

Because V == K_nope in absorbed MLA, the PV half could in principle be
fed from a shared-to-shared copy of the K halves instead of a second
gather; that needs V_l and V_r resident simultaneously after the
softmax and does not fit the budget, so each KV row is simply read
twice from L2 (once per role).

``sparse_attention_fwd_fp8_sm70`` is the same algorithm against the fp8
block-128 pool store (``--kv-cache-dtype fp8_e4m3``; GLM's ropeless
latent rows of 512 e4m3 bytes + 4 f32 group scales): the gather decodes
e4m3 via a shared LUT and multiplies the row's group scale, which halves
the gathered bytes per KV row. ``tilelang_sparse_fwd_sm70`` dispatches
on the pool dtype so call sites stay unchanged.
"""

import functools

import tilelang
import tilelang.language as T
import torch

DTYPE = "float16"


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_attention_fwd_fp8_sm70(
    num_heads,
    dim,
    topk,
    *,
    sm_scale: float | None = None,
    block_I=64,
    threads=128,
):
    """fp8-store variant: pool rows are the block-128 e4m3 layout written by
    ``quantize_k_cache_separate`` — ``dim`` KV bytes + ``(dim//128)`` f32 group
    scales (GLM: 512 + 16 = 528 B; no decoupled rope, tail_dim is unsupported).

    The gather decodes e4m3 in-kernel through a 256-entry fp16 LUT (512 B
    shared, filled once per program with exact power-of-two arithmetic) and
    multiplies the per-128-group f32 scale, so the pool stays the single
    storage and the gather bytes halve vs the fp16 pool. Byte 0x7F/0xFF
    (e4m3 NaN) decodes to ±480 instead of NaN; the quantizer clamps to ±448
    and never emits it.
    """
    assert dim % 128 == 0, "the kernel gathers K/V in two D/2 halves"
    assert topk % block_I == 0, (
        "otherwise will load some index=0 thus causing wrong kv to be loaded"
    )
    # log2(e) = 1.44269504, folded in so the kernel runs exp2
    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5 * 1.44269504
    else:
        sm_scale = sm_scale * 1.44269504

    ROW_BYTES = dim + (dim // 128) * 4
    F32_PER_ROW = ROW_BYTES // 4
    F32_SCALES_AT = dim // 4

    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    seq_len_kv = T.symbolic("seq_len_kv")

    q_shape = [batch, seq_len, num_heads, dim]
    o_shape = [batch, seq_len, num_heads, dim]
    indices_shape = [batch, seq_len, 1, topk]
    indices_dtype = "int32"

    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_half = dim // 2

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, DTYPE),  # type: ignore
        KVBytes: T.Tensor([batch, seq_len_kv * ROW_BYTES], "uint8"),  # type: ignore
        GroupScales: T.Tensor([batch, seq_len_kv * F32_PER_ROW], "float32"),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        Output: T.Tensor(o_shape, DTYPE),  # type: ignore
    ):
        with T.Kernel(seq_len, batch, threads=threads) as (bx, by):
            LUT_shared = T.alloc_shared([256], DTYPE)
            Q_shared_l = T.alloc_shared([num_heads, D_half], DTYPE)
            Q_shared_r = T.alloc_shared([num_heads, D_half], DTYPE)
            K_shared = T.alloc_shared([BI, D_half], DTYPE)
            V_shared = T.alloc_shared([BI, D_half], DTYPE)
            Scale_shared = T.alloc_shared([BI, 4], "float")
            mask = T.alloc_fragment([BI], "bool")

            acc_o_l = T.alloc_fragment([num_heads, D_half], "float")
            acc_o_r = T.alloc_fragment([num_heads, D_half], "float")
            acc_s = T.alloc_fragment([num_heads, BI], "float")
            S_shared = T.alloc_shared([num_heads, BI], DTYPE)
            sumexp = T.alloc_fragment([num_heads], "float")
            sumexp_i = T.alloc_fragment([num_heads], "float")
            alpha = T.alloc_fragment([num_heads], "float")
            m_i = T.alloc_fragment([num_heads], "float")
            m_i_prev = T.alloc_fragment([num_heads], "float")

            T.fill(acc_o_l, 0)
            T.fill(acc_o_r, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan

            b_i, s_i = by, bx

            for i_i in T.Parallel(256):
                e = (i_i >> 3) & 15
                m = i_i & 7
                mag = T.if_then_else(
                    e == 0,
                    # denorm: m * 2^-9, and m == 0 gives the exact +0/-0
                    T.Cast("float", m) * 0.001953125,
                    (1.0 + T.Cast("float", m) * 0.125) * T.exp2(T.Cast("float", e - 7)),
                )
                LUT_shared[i_i] = T.if_then_else(i_i >= 128, -mag, mag)

            T.copy(Q[b_i, s_i, :, :D_half], Q_shared_l)
            T.copy(Q[b_i, s_i, :, D_half:D], Q_shared_r)

            for i_i in T.serial(NI):
                for bi_i in T.Parallel(BI):
                    mask[bi_i] = Indices[b_i, s_i, 0, i_i * BI + bi_i] >= 0

                for h_i, bi_i in T.Parallel(num_heads, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(
                        mask[bi_i], 0, -T.infinity(acc_s.dtype)
                    )

                # Stage the per-row group scales once; the gather then reads
                # them from shared. row is clamped so masked (-1) lanes read
                # the zeroed padding slot deterministically instead of
                # walking off the front of the buffer.
                for bi_i, g_i in T.Parallel(BI, 4):
                    row_s = T.max(Indices[b_i, s_i, 0, i_i * BI + bi_i], 0)
                    Scale_shared[bi_i, g_i] = GroupScales[
                        b_i, row_s * F32_PER_ROW + F32_SCALES_AT + g_i
                    ]

                # K halves: one buffer, two gathers, both reads are the
                # B-transposed gemm role so the swizzle stays consistent.
                for bi_i, d_i in T.Parallel(BI, D_half):
                    row = T.max(Indices[b_i, s_i, 0, i_i * BI + bi_i], 0)
                    K_shared[bi_i, d_i] = (
                        LUT_shared[KVBytes[b_i, row * ROW_BYTES + d_i]]
                        * Scale_shared[bi_i, d_i // 128]
                    )
                T.gemm(
                    Q_shared_l,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    clear_accum=False,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                for bi_i, d_i in T.Parallel(BI, D_half):
                    row = T.max(Indices[b_i, s_i, 0, i_i * BI + bi_i], 0)
                    K_shared[bi_i, d_i] = (
                        LUT_shared[KVBytes[b_i, row * ROW_BYTES + D_half + d_i]]
                        * Scale_shared[bi_i, (D_half + d_i) // 128]
                    )
                T.gemm(
                    Q_shared_r,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    clear_accum=False,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                # The gemm ran over every lane, and a masked lane's clamped
                # row-0 gather carries whatever the pool's padding slot holds
                # (NaN scales included); re-assert the mask so -inf wins over
                # NaN before the max reduction consumes the row.
                for h_i, bi_i in T.Parallel(num_heads, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(
                        mask[bi_i], acc_s[h_i, bi_i], -T.infinity(acc_s.dtype)
                    )

                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(num_heads):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(num_heads, BI):
                    acc_s[h_i, bi_i] = T.exp2(
                        acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                    )
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(num_heads):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(num_heads, D_half):
                    acc_o_l[h_i, d_i] = acc_o_l[h_i, d_i] * alpha[h_i]
                    acc_o_r[h_i, d_i] = acc_o_r[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)

                # V halves: same two gathers into the other buffer; both
                # reads are the non-transposed B role. Masked lanes are
                # zeroed because their clamped row-0 dequant can be NaN,
                # and S_shared is exactly 0 there -- 0 * NaN would
                # re-poison the accumulator the mask just cleaned.
                for bi_i, d_i in T.Parallel(BI, D_half):
                    row = T.max(Indices[b_i, s_i, 0, i_i * BI + bi_i], 0)
                    V_shared[bi_i, d_i] = T.if_then_else(
                        mask[bi_i],
                        LUT_shared[KVBytes[b_i, row * ROW_BYTES + d_i]]
                        * Scale_shared[bi_i, d_i // 128],
                        0,
                    )
                T.gemm(
                    S_shared,
                    V_shared,
                    acc_o_l,
                    clear_accum=False,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                for bi_i, d_i in T.Parallel(BI, D_half):
                    row = T.max(Indices[b_i, s_i, 0, i_i * BI + bi_i], 0)
                    V_shared[bi_i, d_i] = T.if_then_else(
                        mask[bi_i],
                        LUT_shared[KVBytes[b_i, row * ROW_BYTES + D_half + d_i]]
                        * Scale_shared[bi_i, (D_half + d_i) // 128],
                        0,
                    )
                T.gemm(
                    S_shared,
                    V_shared,
                    acc_o_r,
                    clear_accum=False,
                    policy=T.GemmWarpPolicy.FullCol,
                )

            # Rescale
            for h_i, d_i in T.Parallel(num_heads, D_half):
                acc_o_l[h_i, d_i] /= sumexp[h_i]
                acc_o_r[h_i, d_i] /= sumexp[h_i]

            T.copy(acc_o_l, Output[b_i, s_i, :, :D_half])
            T.copy(acc_o_r, Output[b_i, s_i, :, D_half:D])

    return main


@functools.lru_cache(maxsize=32)
def is_sm70(device_index: int = 0) -> bool:
    props = torch.cuda.get_device_properties(device_index)
    return props.major == 7


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_attention_fwd_kernel_sm70(
    num_heads,
    dim,
    tail_dim,
    topk,
    *,
    kv_group=1,
    sm_scale: float | None = None,
    block_I=64,
    threads=128,
):
    assert dim == tilelang.math.next_power_of_2(dim) or dim % 64 == 0, (
        f"dim={dim} must be a power of 2 or a multiple of 64"
    )
    assert tail_dim == 0 or tail_dim == tilelang.math.next_power_of_2(tail_dim), (
        f"tail_dim={tail_dim} must be 0 or a power of 2"
    )
    assert dim % 128 == 0, "the kernel gathers K/V in two D/2 halves"
    assert topk % block_I == 0, (
        "otherwise will load some index=0 thus causing wrong kv to be loaded"
    )
    # log2(e) = 1.44269504, folded in so the kernel runs exp2
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504
    else:
        sm_scale = sm_scale * 1.44269504

    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    seq_len_kv = T.symbolic("seq_len_kv")

    head_kv = num_heads // kv_group
    q_shape = [batch, seq_len, num_heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, num_heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    indices_dtype = "int32"

    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_half = dim // 2
    D_tail = tail_dim

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, DTYPE),  # type: ignore
        KV: T.Tensor(kv_shape, DTYPE),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        Output: T.Tensor(o_shape, DTYPE),  # type: ignore
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (
            bx,
            by,
            bz,
        ):
            Q_shared_l = T.alloc_shared([H_per_block, D_half], DTYPE)
            Q_shared_r = T.alloc_shared([H_per_block, D_half], DTYPE)
            K_shared = T.alloc_shared([BI, D_half], DTYPE)
            V_shared = T.alloc_shared([BI, D_half], DTYPE)
            # GLM-5.3-Flash runs DSA with qk_rope_head_dim=0, so D_tail can be
            # 0: trace-time guard (D_tail is a host constexpr). Zero-extent
            # T.alloc_shared buffers make TVM layout inference fail with
            # "no available layout found".
            if D_tail > 0:
                Q_tail_shared = T.alloc_shared([H_per_block, D_tail], DTYPE)
                K_tail_shared = T.alloc_shared([BI, D_tail], DTYPE)
            mask = T.alloc_fragment([BI], "bool")

            acc_o_l = T.alloc_fragment([H_per_block, D_half], "float")
            acc_o_r = T.alloc_fragment([H_per_block, D_half], "float")
            acc_s = T.alloc_fragment([H_per_block, BI], "float")
            S_shared = T.alloc_shared([H_per_block, BI], DTYPE)
            sumexp = T.alloc_fragment([H_per_block], "float")
            sumexp_i = T.alloc_fragment([H_per_block], "float")
            alpha = T.alloc_fragment([H_per_block], "float")
            m_i = T.alloc_fragment([H_per_block], "float")
            m_i_prev = T.alloc_fragment([H_per_block], "float")

            T.fill(acc_o_l, 0)
            T.fill(acc_o_r, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D_half], Q_shared_l)
            T.copy(Q[b_i, s_i, H0:H1, D_half:D], Q_shared_r)
            if D_tail > 0:
                T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

            for i_i in T.serial(NI):
                for bi_i in T.Parallel(BI):
                    mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] >= 0

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(
                        mask[bi_i], 0, -T.infinity(acc_s.dtype)
                    )

                # K halves: one buffer, two gathers, both reads are the
                # B-transposed gemm role so the swizzle stays consistent.
                for bi_i, d_i in T.Parallel(BI, D_half):
                    K_shared[bi_i, d_i] = KV[
                        b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i
                    ]
                T.gemm(
                    Q_shared_l,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    clear_accum=False,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                for bi_i, d_i in T.Parallel(BI, D_half):
                    K_shared[bi_i, d_i] = KV[
                        b_i,
                        Indices[b_i, s_i, g_i, i_i * BI + bi_i],
                        g_i,
                        D_half + d_i,
                    ]
                T.gemm(
                    Q_shared_r,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    clear_accum=False,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                if D_tail > 0:
                    for bi_i, d_i in T.Parallel(BI, D_tail):
                        K_tail_shared[bi_i, d_i] = KV[
                            b_i,
                            Indices[b_i, s_i, g_i, i_i * BI + bi_i],
                            g_i,
                            D + d_i,
                        ]
                    T.gemm(
                        Q_tail_shared,
                        K_tail_shared,
                        acc_s,
                        transpose_B=True,
                        clear_accum=False,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(
                        acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                    )
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D_half):
                    acc_o_l[h_i, d_i] = acc_o_l[h_i, d_i] * alpha[h_i]
                    acc_o_r[h_i, d_i] = acc_o_r[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)

                # V halves: same two gathers into the other buffer; both
                # reads are the non-transposed B role.
                for bi_i, d_i in T.Parallel(BI, D_half):
                    V_shared[bi_i, d_i] = KV[
                        b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i
                    ]
                T.gemm(
                    S_shared,
                    V_shared,
                    acc_o_l,
                    clear_accum=False,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                for bi_i, d_i in T.Parallel(BI, D_half):
                    V_shared[bi_i, d_i] = KV[
                        b_i,
                        Indices[b_i, s_i, g_i, i_i * BI + bi_i],
                        g_i,
                        D_half + d_i,
                    ]
                T.gemm(
                    S_shared,
                    V_shared,
                    acc_o_r,
                    clear_accum=False,
                    policy=T.GemmWarpPolicy.FullCol,
                )

            # Rescale
            for h_i, d_i in T.Parallel(H_per_block, D_half):
                acc_o_l[h_i, d_i] /= sumexp[h_i]
                acc_o_r[h_i, d_i] /= sumexp[h_i]

            T.copy(acc_o_l, Output[b_i, s_i, H0:H1, :D_half])
            T.copy(acc_o_r, Output[b_i, s_i, H0:H1, D_half:D])

    return main


_probe_calls = 0


def _runtime_probe_enabled() -> bool:
    import os

    return os.environ.get("SGLANG_SM70_SPARSE_PROBE", "0") == "1"


def _runtime_probe(q, kv_bytes, indices, sm_scale, d_v, out) -> None:
    """Env-gated (SGLANG_SM70_SPARSE_PROBE=1) first-calls numeric audit: run
    the fp16 kernel on dequantized gathered pool rows and compare, so a
    kernel-vs-pool mismatch inside the live engine is localized."""
    global _probe_calls
    if _probe_calls >= 4:
        return
    if torch.cuda.is_current_stream_capturing():
        return  # graph-capture dummy inputs are garbage by construction
    _probe_calls += 1
    try:
        with torch.no_grad():
            flat = kv_bytes.view(-1)
            row_bytes = kv_bytes.shape[-1]
            rows = indices[0, 0, :].long().clamp(min=0)
            gathered = flat.view(-1, row_bytes)[rows]  # [topk, row_bytes]
            deq = gathered[:, :d_v].view(torch.float8_e4m3fn).to(torch.float16).float()
            scales = gathered[:, d_v:].view(torch.float32)
            deq = (deq.view(-1, 4, 128) * scales[:, :4, None]).reshape(-1, d_v).half()
            ref_kv = torch.zeros(
                indices.shape[-1], d_v, dtype=torch.float16, device=q.device
            )
            valid = indices[0, 0, :] >= 0
            ref_kv[valid] = deq[valid]
            ref_out = sparse_attention_fwd_kernel_sm70(
                q.shape[1], d_v, 0, indices.shape[-1], sm_scale=sm_scale
            )(
                q.unsqueeze(0),
                ref_kv.unsqueeze(1).unsqueeze(0),
                indices.unsqueeze(0),
            )
            d = (out[0].float() - ref_out[0].float()).abs().max().item()
            idx_raw = indices[0, 0]
            pool_rows = kv_bytes.shape[0]
            print(
                f"[sm70-fp8-probe #{_probe_calls}] shape q={tuple(q.shape)} "
                f"kv={tuple(kv_bytes.shape)} idx={tuple(indices.shape)} "
                f"scale={sm_scale:.5f} fp8-vs-dequant-fp16 max|d|={d:.5f} "
                f"deq_absmean={deq.abs().mean().item():.4f} "
                f"out_absmean={out.abs().mean().item():.4f} "
                f"q_absmean={q.float().abs().mean().item():.4f} "
                f"q_nan={int(torch.isnan(q.float()).sum().item())} "
                f"idx_min={int(idx_raw.min().item())} "
                f"idx_max={int(idx_raw.max().item())} "
                f"idx_neg={int((idx_raw < 0).sum().item())} "
                f"pool_rows={pool_rows} "
                f"sample_bytes={list(gathered[0, :8].tolist())} "
                f"sample_scales={list(scales[0, :4].tolist())}",
                flush=True,
            )
            if _probe_calls == 1:
                torch.save(
                    {
                        "q": q.cpu(),
                        "kv_bytes": kv_bytes.cpu(),
                        "indices": indices.cpu(),
                        "sm_scale": sm_scale,
                        "d_v": d_v,
                        "out": out.cpu(),
                    },
                    "/tmp/sm70_fp8_firstcall.pt",
                )
                print("[sm70-fp8-probe] first real call dumped to /tmp", flush=True)
    except Exception as exc:  # probe must never break serving
        print(f"[sm70-fp8-probe #{_probe_calls}] failed: {exc}", flush=True)


def tilelang_sparse_fwd_sm70(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    """Sparse attention over top-k pool indices, SM70 fp16 port of the CUDA
    branch of upstream ``tilelang_sparse_fwd``.

    q: [M, H, d_v + tail] fp16; kv: the page-size-1 MLA KV pool buffer —
    either the fp16 pool [pool_size, 1, d_v + tail] or the fp8 block-128
    store [pool_size, 1, d_v + (d_v//128)*4] viewed float8_e4m3fn (ropeless
    latent, e.g. GLM); indices: [M, 1, topk] int32 pool slots with -1
    padding (any multiple-of-64 topk; every 64-column block must keep at
    least one valid index so no block goes all-masked, which the %64 tail
    padding guarantees). Returns [1, M, H, d_v] fp16. Argument shapes
    mirror upstream ``tilelang_sparse_fwd`` so call sites dispatch without
    reshaping.
    """
    assert q.dim() == 3 and kv.dim() == 3 and indices.dim() == 3
    num_heads = q.shape[1]
    tail_dim = q.shape[2] - d_v
    topk = indices.shape[-1]
    assert topk % 64 == 0, "topk must be padded to a multiple of 64"
    assert q.dtype == torch.float16

    if kv.dtype != torch.float16:
        # fp8 block-128 store; the scales tail lives in the same rows, so
        # both kernel arguments are flat views of the one pool allocation.
        assert kv.dtype == torch.float8_e4m3fn, kv.dtype
        assert tail_dim == 0, "the fp8 sm70 kernel only covers ropeless latent"
        kv_bytes = kv.view(torch.uint8)
        assert kv_bytes.shape[-1] == d_v + (d_v // 128) * 4, (
            kv_bytes.shape,
            d_v,
        )
        flat_bytes = kv_bytes.view(-1)
        out = sparse_attention_fwd_fp8_sm70(num_heads, d_v, topk, sm_scale=sm_scale)(
            q.unsqueeze(0),
            flat_bytes.unsqueeze(0),
            flat_bytes.view(torch.float32).unsqueeze(0),
            indices.unsqueeze(0),
        )
        if _runtime_probe_enabled():
            _runtime_probe(q, kv_bytes, indices, sm_scale, d_v, out)
        return out

    kernel = sparse_attention_fwd_kernel_sm70(
        num_heads, d_v, tail_dim, topk, sm_scale=sm_scale
    )
    return kernel(
        q.unsqueeze(0),
        kv.unsqueeze(0),
        indices.unsqueeze(0),
    )
