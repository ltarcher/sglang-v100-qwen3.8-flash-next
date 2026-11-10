"""CPU unit tests for ``build_grouped_union_plan`` (review m23,
docs/v100/CODE_REVIEW_TODO_20261110.md).

The planner's int32/int64 key-width decision used to read the actual last
group slot (``int(group_slot[-1])``), a second hidden D2H sync on top of the
documented single U_pad readback. It now derives a host-side bound from
shapes only (g <= (R-1)//group_rows, batch <= bs-1), which can only
overestimate — falling back to int64 keys one regime early, never corrupting
a plan.

Fixtures force both key regimes:
  - fixture A: bs=3, max slot 2 -> int32 keys (2 + 1 < 2**(31-20));
  - fixture B: bs=2048, slot bound 2047 -> 2048 is NOT < 2**11 -> int64 keys.
Both plans must match a pure-python reference for uidx/mask.
"""

import torch

from sglang.srt.layers.attention.tilelang_fa_v100._kernels_grouped_sparse_prefill import (
    build_grouped_union_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.test.test_utils import CustomTestCase


def _fixture_a():
    """bs=3, R=8, one group of 8 rows across 3 requests (int32 regime).

    topk entries are logical positions inside each request (< max_ctx);
    req_to_token maps them to disjoint pool rows per request (0..63 /
    64..127 / 128..135) so the union spans 11 distinct slots.
    """
    max_ctx, topk = 64, 4
    cu = torch.tensor([0, 4, 6, 8], dtype=torch.int64)
    seq_lens = torch.tensor([10, 8, 3], dtype=torch.int64)
    req_pool_indices = torch.tensor([0, 1, 2], dtype=torch.int32)
    req_to_token = torch.zeros(3, max_ctx, dtype=torch.int64)
    req_to_token[0] = torch.arange(max_ctx)
    req_to_token[1] = 64 + torch.arange(max_ctx)
    req_to_token[2, :8] = 128 + torch.arange(8)
    ti = torch.full((8, topk), -1, dtype=torch.int32)
    for r in range(4):  # continuation, visible 7..10 -> keep 4
        ti[r] = torch.tensor([0, 5, 9, 20])
    for r in range(4, 6):  # continuation, visible 7..8 -> keep 4
        ti[r] = torch.tensor([0, 5, 9, 15])  # pool rows 64,69,73,79
    ti[6, :2] = torch.tensor([0, 2])  # fresh prefill, visible 2 -> pool 128,130
    ti[7, :3] = torch.tensor([0, 2, 3])  # fresh prefill, visible 3
    return ti, cu, seq_lens, req_pool_indices, req_to_token, max_ctx


def _fixture_b():
    """bs=2048 forces the int64 key regime (slot bound 2047, 2048 >= 2**11)."""
    bs, max_ctx, topk, R = 2048, 8, 4, 8
    cu = torch.zeros(bs + 1, dtype=torch.int64)
    cu[1:] = R  # every row belongs to "request 0" (batch = 0)
    seq_lens = torch.full((bs,), R + 10, dtype=torch.int64)
    req_pool_indices = torch.zeros(bs, dtype=torch.int32)
    req_to_token = torch.zeros(bs, max_ctx, dtype=torch.int64)
    req_to_token[0] = torch.arange(max_ctx) * 3
    ti = torch.arange(topk, dtype=torch.int32).expand(R, topk).contiguous()
    return ti, cu, seq_lens, req_pool_indices, req_to_token, max_ctx


def _plan_reference(ti, cu, seq_lens, req_pool_indices, req_to_token, u_pad):
    """Pure-python planner reference: per-group sorted union of row slots.

    Mask counts duplicate entries per (row, column) -- the planner's
    ``index_add_`` accumulates them the same way and the kernel only reads
    ``mask > 0``, but the pinned contract is the exact count.
    """
    R, topk = ti.shape
    group_rows = 8
    G = (R + group_rows - 1) // group_rows
    cu64 = cu.to(torch.int64)
    seq64 = seq_lens.to(torch.int64)
    uidx_ref = torch.zeros(G, u_pad, dtype=torch.int64)
    mask_ref = torch.zeros(G, group_rows, u_pad, dtype=torch.int64)
    unions = []
    for g in range(G):
        slots = set()
        row_slots = {}
        for r in range(g * group_rows, min((g + 1) * group_rows, R)):
            batch = int((cu64[1:] <= r).sum())
            rows_in_req = r - int(cu64[batch])
            extend_len = int(cu64[batch + 1] - cu64[batch])
            visible = int(seq64[batch]) - extend_len + rows_in_req + 1
            keep = min(visible, topk)
            mine = ti[r][:keep]
            mine = [int(e) for e in mine if int(e) >= 0]
            req = int(req_pool_indices[batch])
            mine_slots = [int(req_to_token[req, e]) for e in mine]
            row_slots[r] = mine_slots
            slots.update(mine_slots)
        union = sorted(slots)
        unions.append(union)
        for u, slot in enumerate(union):
            uidx_ref[g, u] = slot
            for r, mine_slots in row_slots.items():
                mask_ref[g, r % group_rows, u] += mine_slots.count(slot)
    return uidx_ref, mask_ref, unions


class TestGroupedUnionPlanNoHiddenReadback(CustomTestCase):
    def _build(self, fixture):
        ti, cu, seq_lens, req_pool_indices, req_to_token, max_ctx = fixture
        return build_grouped_union_plan(
            ti, cu, seq_lens, req_pool_indices, req_to_token, max_ctx
        )

    def test_no_implicit_int_readback(self):
        # The planner may keep exactly one scalar readback (U_pad, an explicit
        # .item()); an implicit int(tensor) is a second D2H sync and must be
        # gone (review m23).
        fixture = _fixture_a()
        orig = torch.Tensor.__int__

        def boom(self):
            raise AssertionError("implicit int(tensor) readback in planner")

        torch.Tensor.__int__ = boom
        try:
            plan = self._build(fixture)
        finally:
            torch.Tensor.__int__ = orig
        self.assertIsNotNone(plan)

    def _assert_plan(self, fixture):
        ti, cu, seq_lens, req_pool_indices, req_to_token, max_ctx = fixture
        plan = self._build(fixture)
        self.assertIsNotNone(plan)
        uidx, mask, u_pad = plan
        uidx_ref, mask_ref, unions = _plan_reference(
            ti, cu, seq_lens, req_pool_indices, req_to_token, u_pad
        )
        self.assertTrue(torch.equal(uidx.to(torch.int64), uidx_ref))
        self.assertTrue(torch.equal(mask.to(torch.int64), mask_ref))
        for g, union in enumerate(unions):
            for r in range(g * 8, min((g + 1) * 8, ti.shape[0])):
                self.assertGreaterEqual(
                    int(mask[g, r % 8].sum()), 1, f"row {r} fully masked"
                )

    def test_int32_regime_plan_matches_reference(self):
        self._assert_plan(_fixture_a())

    def test_int64_regime_plan_matches_reference(self):
        # bs=2048 pushes the host-side slot bound past the int32 key regime;
        # the planner must fall back to int64 keys and stay correct.
        fixture = _fixture_b()
        _, _, _, _, _, max_ctx = fixture
        ti, cu, seq_lens, req_pool_indices, req_to_token = fixture[:5]
        bs = cu.numel() - 1
        max_slot = ((ti.shape[0] - 1) // 8) * bs + (bs - 1)
        self.assertLessEqual(max_slot + 1, 1 << 40)  # sanity
        self.assertGreaterEqual(max_slot + 1, 1 << (31 - 20))  # int64 regime
        self._assert_plan(fixture)


if __name__ == "__main__":
    import unittest

    unittest.main()
