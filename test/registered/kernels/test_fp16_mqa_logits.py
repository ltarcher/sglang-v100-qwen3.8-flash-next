"""Numerics gate for the SM70 FP16 DSA-indexer logits path.

The FP16 route must reproduce the FP8 DeepGEMM logit semantics:
logits[i, j] = sum_h gate[i, h] * (q[i, h] . k[pos]). The torch
reference computes that reduction in FP32 end to end; the kernel path
folds the gate into q_eff in FP32 (llama-glm5 red line: indexer weights
accumulate FP32), rounds once to FP16, and accumulates the GEMM in
FP32. Agreement within fp16-input tolerance (~1e-2 relative) is the
pass bar -- upstream's per-head FP8 quantization is ~8x coarser.
"""

import unittest

import torch
from sglang.kernels.ops.attention.dsa.fp16_mqa_logits import (
    fp16_paged_mqa_logits,
    fp16_paged_mqa_logits_fp8kcache,
    fp16_ragged_mqa_logits,
    gate_fold,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu")


def _reference_logits(q, k, gate, pos_k):
    """logits[i, j] = sum_h gate[i, h] * (q[i, h] . k[pos_k[i, j]]) in FP32.

    ``pos_k`` is [n, span] int64 holding the k row each output column
    reads (built from ks/ke or a page table by the caller).
    """
    qf, kf = q.float(), k.float()
    dots = torch.einsum("nhd,jd->nhj", qf, kf)
    return torch.einsum("nh,nhj->nj", gate, dots).gather(1, pos_k.clamp(min=0)) * (
        pos_k >= 0
    )


@unittest.skipUnless(torch.cuda.is_available(), "Test requires CUDA")
class TestFp16MqaLogits(CustomTestCase):
    H, D = 32, 128

    def _gen(self, n_rows, device):
        gen = torch.Generator(device="cpu").manual_seed(7 + n_rows)
        q = (
            (torch.randn((n_rows, self.H, self.D), generator=gen) * 0.5)
            .to(torch.float16)
            .to(device)
        )
        gate = torch.randn((n_rows, self.H), generator=gen).to(device)
        return q, gate

    def test_gate_fold(self):
        device = torch.device("cuda")
        q, gate = self._gen(9, device)
        q_eff = gate_fold(q, gate)
        ref = torch.einsum("nh,nhd->nd", gate, q.float())
        torch.testing.assert_close(q_eff.float(), ref, rtol=1e-2, atol=1e-2)

    def test_ragged_shared_range(self):
        device = torch.device("cuda")
        n_rows, total_k = 16, 512
        q, gate = self._gen(n_rows, device)
        k = (torch.randn((total_k, self.D)) * 0.5).to(torch.float16).to(device)
        # all rows share one range (single-sequence chunked prefill)
        ks = torch.zeros(n_rows, dtype=torch.int32, device=device)
        ke = torch.full((n_rows,), total_k, dtype=torch.int32, device=device)
        q_eff = gate_fold(q, gate)
        out = fp16_ragged_mqa_logits(q_eff, k, ks, ke)
        pos = torch.arange(total_k, device=device).unsqueeze(0).expand(n_rows, -1)
        ref = _reference_logits(q, k, gate, pos)
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=5e-1)

    def test_ragged_union_ranges(self):
        device = torch.device("cuda")
        n_rows, total_k = 8, 256
        q, gate = self._gen(n_rows, device)
        k = (torch.randn((total_k, self.D)) * 0.5).to(torch.float16).to(device)
        # rows start at staggered offsets (multi-sequence ragged buffer)
        ks = torch.arange(n_rows, dtype=torch.int32, device=device) * 16
        ke = ks + 128
        q_eff = gate_fold(q, gate)
        out = fp16_ragged_mqa_logits(q_eff, k, ks, ke)
        pos = (ks.unsqueeze(1) + torch.arange(128, device=device).unsqueeze(0)).long()
        ref = _reference_logits(q, k, gate, pos)
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=5e-1)

    def test_paged(self):
        device = torch.device("cuda")
        n_rows, page, num_pages = 5, 64, 7
        q, gate = self._gen(n_rows, device)
        kcache = (
            (torch.randn((num_pages, page, self.D)) * 0.5).to(torch.float16).to(device)
        )
        seqlens = torch.tensor([64, 129, 300, 1, 448], dtype=torch.int32, device=device)
        block_tables = torch.zeros(n_rows, 7, dtype=torch.int32, device=device)
        for i in range(n_rows):
            n_p = (int(seqlens[i]) + page - 1) // page
            block_tables[i, :n_p] = (torch.arange(n_p) + i) % num_pages
        max_len = int(seqlens.max().item())
        q_eff = gate_fold(q, gate)
        out = fp16_paged_mqa_logits(q_eff, kcache, seqlens, block_tables, max_len)
        pos = torch.full((n_rows, max_len), -1, dtype=torch.int64, device=device)
        for i in range(n_rows):
            n_p = (int(seqlens[i]) + page - 1) // page
            for p in range(n_p):
                lo, hi = p * page, min((p + 1) * page, int(seqlens[i]))
                pos[i, lo:hi] = block_tables[i, p].long() * page + torch.arange(
                    lo - p * page, hi - p * page, device=device
                )
        ref = _reference_logits(q, kcache.reshape(-1, self.D), gate, pos)
        torch.testing.assert_close(out[:, :max_len], ref, rtol=2e-2, atol=5e-1)

    def test_paged_alignment_stride(self):
        # Output row stride must stay 256-aligned (fused top-k ABI).
        device = torch.device("cuda")
        q_eff = torch.randn((2, self.D), device=device, dtype=torch.float16)
        kcache = torch.zeros((2, 64, self.D), dtype=torch.float16, device=device)
        seqlens = torch.tensor([100, 200], dtype=torch.int32, device=device)
        block_tables = torch.tensor(
            [[0, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.int32, device=device
        )
        out = fp16_paged_mqa_logits(q_eff, kcache, seqlens, block_tables, 200)
        self.assertEqual(out.shape[1] % 256, 0)
        self.assertEqual(out.stride(0) % 4, 0)

    def test_paged_fp8kcache(self):
        # The kernel must decode the upstream pooled fp8 layout directly
        # (per page: PAGE*D e4m3 key bytes then PAGE*4 fp32 scale bytes,
        # index_buf_accessor.GetK/GetS) without any repack pass.
        device = torch.device("cuda")
        page = 64
        n_rows, num_pages = 5, 9
        q, gate = self._gen(n_rows, device)
        k_f16 = (
            (torch.randn((num_pages, page, self.D)) * 0.5).to(torch.float16).to(device)
        )
        amax = k_f16.float().abs().amax(dim=-1).clamp(min=1e-4)
        scale = amax / 448.0
        k_fp8 = (
            (k_f16.float() / scale[:, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        )
        buf = torch.zeros(
            num_pages, page * (self.D + 4), dtype=torch.uint8, device=device
        )
        buf[:, : page * self.D] = k_fp8.view(torch.uint8).reshape(
            num_pages, page * self.D
        )
        buf[:, page * self.D :] = (
            scale.reshape(num_pages, page, 1)
            .expand(num_pages, page, 4)
            .reshape(num_pages, page * 4)
            .contiguous()
        )
        k_deq = (
            k_fp8.to(torch.float16).float()
            * buf[:, page * self.D :].view(torch.float32)[:, :, None]
        )

        seqlens = torch.tensor([64, 129, 300, 1, 448], dtype=torch.int32, device=device)
        block_tables = torch.zeros(n_rows, 7, dtype=torch.int32, device=device)
        for i in range(n_rows):
            n_p = (int(seqlens[i]) + page - 1) // page
            block_tables[i, :n_p] = (torch.arange(n_p) + i * 3) % num_pages
        max_len = int(seqlens.max().item())
        q_eff = gate_fold(q, gate)
        out = fp16_paged_mqa_logits_fp8kcache(
            q_eff, buf, seqlens, block_tables, max_len
        )
        pos = torch.full((n_rows, max_len), -1, dtype=torch.int64, device=device)
        for i in range(n_rows):
            n_p = (int(seqlens[i]) + page - 1) // page
            for p in range(n_p):
                lo, hi = p * page, min((p + 1) * page, int(seqlens[i]))
                pos[i, lo:hi] = block_tables[i, p].long() * page + torch.arange(
                    lo - p * page, hi - p * page, device=device
                )
        ref = _reference_logits(q, k_deq.reshape(-1, self.D), gate, pos)
        torch.testing.assert_close(out[:, :max_len], ref, rtol=2e-2, atol=5e-1)

    def test_paged_fp8kcache_stride(self):
        # Same 256-aligned fused top-k ABI on the fp8-cache variant.
        device = torch.device("cuda")
        q_eff = torch.randn((2, self.D), device=device, dtype=torch.float16)
        buf = torch.zeros(2, 64 * (self.D + 4), dtype=torch.uint8, device=device)
        seqlens = torch.tensor([100, 200], dtype=torch.int32, device=device)
        block_tables = torch.tensor(
            [[0, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.int32, device=device
        )
        out = fp16_paged_mqa_logits_fp8kcache(q_eff, buf, seqlens, block_tables, 200)
        self.assertEqual(out.shape[1] % 256, 0)
        self.assertEqual(out.stride(0) % 4, 0)

    def test_paged_fp8kcache_mtp_next_n(self):
        # MTP verify: q rows are bs * draft width while the pooled page
        # table stays per sequence; draft rows of one sequence must read
        # its table row and produce their own logits row.
        device = torch.device("cuda")
        page = 64
        bs, next_n, num_pages = 2, 3, 6
        gen = torch.Generator(device="cpu").manual_seed(11)
        k_f16 = (
            (torch.randn((num_pages, page, self.D), generator=gen) * 0.5)
            .to(torch.float16)
            .to(device)
        )
        amax = k_f16.float().abs().amax(dim=-1).clamp(min=1e-4)
        scale = amax / 448.0
        k_fp8 = (
            (k_f16.float() / scale[:, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        )
        buf = torch.zeros(
            num_pages, page * (self.D + 4), dtype=torch.uint8, device=device
        )
        buf[:, : page * self.D] = k_fp8.view(torch.uint8).reshape(
            num_pages, page * self.D
        )
        buf[:, page * self.D :] = (
            scale.reshape(num_pages, page, 1)
            .expand(num_pages, page, 4)
            .reshape(num_pages, page * 4)
            .contiguous()
        )
        k_deq = (
            k_fp8.to(torch.float16).float()
            * buf[:, page * self.D :].view(torch.float32)[:, :, None]
        )

        seq_lens = [200, 130]
        q = (
            (torch.randn((bs * next_n, self.H, self.D), generator=gen) * 0.5)
            .to(torch.float16)
            .to(device)
        )
        gate = torch.randn((bs * next_n, self.H), generator=gen).to(device)
        q_eff = gate_fold(q, gate)
        seqlens = torch.tensor(
            [l for l in seq_lens for _ in range(next_n)],
            dtype=torch.int32,
            device=device,
        )
        block_tables = torch.tensor(
            [[1, 4, 2, 0], [3, 2, 5, 0]], dtype=torch.int32, device=device
        )
        max_len = max(seq_lens)
        out = fp16_paged_mqa_logits_fp8kcache(
            q_eff, buf, seqlens, block_tables, max_len
        )
        self.assertEqual(out.shape[0], bs * next_n)
        for i in range(bs * next_n):
            b = i // next_n
            length = seq_lens[b]
            rows = []
            for p in range((length + page - 1) // page):
                lo, hi = p * page, min((p + 1) * page, length)
                rows.append(k_deq[block_tables[b, p], : hi - lo])
            ref = torch.cat(rows, dim=0) @ q_eff[i].float()
            torch.testing.assert_close(out[i, :length], ref, rtol=2e-2, atol=5e-1)


if __name__ == "__main__":
    unittest.main()
