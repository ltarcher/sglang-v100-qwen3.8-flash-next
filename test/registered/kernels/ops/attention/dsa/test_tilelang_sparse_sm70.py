"""Correctness of the sm70 fp8 sparse-attention TileLang kernel.

Regression: a -1-padded index lane is gathered from pool row 0 via the
in-kernel address clamp; if that padding slot ever holds NaN group scales
(observed after a boot where the pool's row 0 was written with NaN fp32
scales), the fp8 dequant re-introduced NaN into the QK gemm faster than
the seed-time -inf mask could suppress it, and every query row returned
NaN. The kernel now re-asserts the mask after the QK gemms and zeroes
masked lanes in the V gather.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

D_V = 512
GROUP = 128
N_GROUPS = D_V // GROUP
ROW_BYTES = D_V + N_GROUPS * 4
TOPK = 128  # one block, so the all-masked-block guarantee stays trivial
M, H = 4, 16


def _sm70():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() == (7, 0)


def _build_pool(num_rows, gen, poison_row=None):
    """Random finite fp8 block-128 pool as the kernel sees it."""
    lat = torch.randn(num_rows, D_V, generator=gen).to(torch.float16)
    lat = lat.clamp(-8, 8).to(torch.float8_e4m3fn).view(torch.uint8)
    scale = (
        torch.rand(num_rows, N_GROUPS, generator=gen) * 0.008 + 0.001
    ).float()
    rows = torch.cat([lat, scale.view(torch.uint8)], dim=-1)
    pool = rows.view(torch.float8_e4m3fn).reshape(num_rows, 1, ROW_BYTES).cuda()
    if poison_row is not None:
        raw = pool.view(torch.uint8).reshape(num_rows, ROW_BYTES)
        nan_bytes = torch.tensor([torch.nan], dtype=torch.float32).view(torch.uint8)
        raw[poison_row, D_V:] = nan_bytes.repeat(N_GROUPS)
        pool = raw.view(torch.float8_e4m3fn).reshape(num_rows, 1, ROW_BYTES)
    return pool


def _reference(q, pool, indices, sm_scale):
    """Masked torch reference over the valid lanes only."""
    raw = pool.view(torch.uint8).reshape(-1, ROW_BYTES)
    lat = raw[:, :D_V].view(torch.float8_e4m3fn).to(torch.float32)
    sc = raw[:, D_V:].view(torch.float32)
    out = torch.full(
        (1, M, H, D_V), float("nan"), dtype=torch.float16, device=q.device
    )
    for b in range(M):
        idx = indices[b, 0]
        valid = idx >= 0
        rows = idx[valid].long()
        k = (lat[rows] * sc[rows].repeat_interleave(GROUP, dim=-1)).to(torch.float16)
        v = k
        s = torch.einsum("hd,nd->hn", q[b].float(), k.float()) * sm_scale
        p = torch.softmax(s, dim=-1).to(torch.float16)
        out[0, b] = (p.float() @ v.float()).to(torch.float16)
    return out


@unittest.skipUnless(_sm70(), "sm70 tilelang fp8 sparse kernel")
class TestTilelangSparseFp8Sm70(CustomTestCase):
    def setUp(self):
        from sglang.kernels.ops.attention.dsa.tilelang_sparse_sm70 import (
            tilelang_sparse_fwd_sm70,
        )

        self.fwd = tilelang_sparse_fwd_sm70
        self.gen = torch.Generator().manual_seed(7)

    def _indices(self, num_rows, valid_per_row):
        idx = torch.full((M, 1, TOPK), -1, dtype=torch.int32, device="cuda")
        for b in range(M):
            slots = torch.randperm(num_rows - 1, generator=self.gen)[: valid_per_row]
            idx[b, 0, :valid_per_row] = (slots + 1).to(torch.int32)
        return idx

    def test_poisoned_padding_slot_stays_finite(self):
        # Row 0 is the clamp target of every -1 lane; poison its scales.
        pool = _build_pool(4096, self.gen, poison_row=0)
        q = torch.randn(M, H, D_V, generator=self.gen).to('cuda', torch.float16)
        idx = self._indices(4096, valid_per_row=48)
        out = self.fwd(q=q, kv=pool, indices=idx, sm_scale=0.0625, d_v=D_V)
        torch.cuda.synchronize()
        self.assertFalse(torch.isnan(out).any(), "NaN leaked from the padding slot")
        ref = _reference(q, pool, idx, 0.0625)
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    def test_matches_reference_on_clean_pool(self):
        pool = _build_pool(4096, self.gen)
        q = torch.randn(M, H, D_V, generator=self.gen).to('cuda', torch.float16)
        idx = self._indices(4096, valid_per_row=64)
        out = self.fwd(q=q, kv=pool, indices=idx, sm_scale=0.0625, d_v=D_V)
        torch.cuda.synchronize()
        self.assertFalse(torch.isnan(out).any())
        ref = _reference(q, pool, idx, 0.0625)
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    def test_all_valid_indices(self):
        pool = _build_pool(TOPK, self.gen)
        q = torch.randn(M, H, D_V, generator=self.gen).to('cuda', torch.float16)
        idx = (
            torch.arange(TOPK, dtype=torch.int32, device="cuda")
            .repeat(M, 1)
            .unsqueeze(1)
        )
        out = self.fwd(q=q, kv=pool, indices=idx, sm_scale=0.0625, d_v=D_V)
        torch.cuda.synchronize()
        self.assertFalse(torch.isnan(out).any())
        ref = _reference(q, pool, idx, 0.0625)
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main(verbosity=3)
