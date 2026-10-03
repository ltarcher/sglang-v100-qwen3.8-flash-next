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
"""

import functools

import tilelang
import tilelang.language as T
import torch

DTYPE = "float16"


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


def tilelang_sparse_fwd_sm70(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    """Sparse attention over top-k pool indices, SM70 fp16 port of the CUDA
    branch of upstream ``tilelang_sparse_fwd``.

    q: [M, H, d_v + tail] fp16; kv: the page-size-1 MLA KV pool buffer
    [pool_size, 1, d_v + tail] fp16; indices: [M, 1, topk] int32 pool
    slots with -1 padding (any multiple-of-64 topk; every 64-column
    block must keep at least one valid index so no block goes
    all-masked, which the %64 tail padding guarantees). Returns
    [1, M, H, d_v] fp16. Argument shapes mirror upstream
    ``tilelang_sparse_fwd`` so call sites dispatch without reshaping.
    """
    assert q.dim() == 3 and kv.dim() == 3 and indices.dim() == 3
    num_heads = q.shape[1]
    tail_dim = q.shape[2] - d_v
    topk = indices.shape[-1]
    assert topk % 64 == 0, "topk must be padded to a multiple of 64"
    assert q.dtype == torch.float16 and kv.dtype == torch.float16

    kernel = sparse_attention_fwd_kernel_sm70(
        num_heads, d_v, tail_dim, topk, sm_scale=sm_scale
    )
    return kernel(
        q.unsqueeze(0),
        kv.unsqueeze(0),
        indices.unsqueeze(0),
    )
