"""GPU unit tests: SM70 grouped sparse prefill online-softmax guard (C1).

Regression guard for docs/v100/CODE_REVIEW_TODO_20261110.md C1. The
kernel's online softmax is NaN-safe on fully-masked 64-column tiles
only because ``m_i`` starts at the *finite* ``-(2**30)`` and tilelang
0.1.8 ``reduce_max(clear=False)`` accumulates (``m_i = max(m_i,
tile_max)``): a fully-masked tile then leaves ``m_i`` unchanged
(alpha=1, numeric no-op). A ``-inf`` init would instead produce
``alpha = exp2(-inf - (-inf)) = NaN`` on any fully-masked tile (the
paged kernel survives a ``-inf`` init only via its explicit
``m_i == -inf`` clamp). Negative control: flipping the init to
``-inf`` makes rows 4-7 of the fixture below emit NaN on every
element -- this test pins the safe side of that line.

Fixture shape classes covered:
  - rows 4-7 (group 0): first tile fully masked (their request's
    union block sorts behind 128 lower-slot entries);
  - rows 8-11 (group 1): pad tail tiles 2-7 fully masked;
  - rows 12-15 (group 2): fresh prefill, row_keep=1..4, 4-entry
    union -- tiles 1-7 fully masked;
  - rows 0-3 (group 0): the ordinary continuation case.

Output must stay finite and match a torch reference over the same
union plan.
"""

import math

import pytest
import torch

from sglang.srt.layers.attention.tilelang_fa_v100._kernels_grouped_sparse_prefill import (
    build_grouped_union_plan,
    run_grouped_sparse_prefill,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

DEV = "cuda:0"
TOPK = 128
HEADS, DIM = 6, 256
GROUP_ROWS = 8


def _torch_reference(q, k_pool, v_pool, uidx, mask, sm_scale):
    """Per-row masked softmax attention over the union panel (fp32)."""
    R = q.shape[0]
    G = uidx.shape[0]
    out = torch.empty_like(q)
    for g in range(G):
        for m in range(min(GROUP_ROWS, R - g * GROUP_ROWS)):
            row = g * GROUP_ROWS + m
            vis = mask[g, m].bool()
            assert vis.any(), f"row {row} fully masked in union (planner contract)"
            idx = uidx[g, vis].long()
            k = k_pool[idx].float()
            v = v_pool[idx].float()
            s = (q[row].float() @ k.T) * sm_scale
            p = torch.softmax(s, dim=-1)
            out[row] = (p @ v).half()
    return out


def _build_mixed_group_fixture():
    """Three requests share the panel; union sorted by (group_slot, logical).

    req 0 (rows 0-3) and req 1 (rows 4-15) are chunk continuations whose
    128-entry windows sit at disjoint logical ranges; req 1's block
    therefore lands at union columns 128-255 of group 0, leaving rows
    4-7 with their first two tiles fully masked -- the NaN-critical
    shape. req 2 (rows 12-15) is a fresh 4-token prefill (visible 1..4)
    exercising the row_keep=1 edge.
    """
    bs, R, max_ctx = 3, 16, 2048
    cu = torch.tensor([0, 4, 12, 16], dtype=torch.int32, device=DEV)
    seq_lens = torch.tensor([2048, 2048, 4], dtype=torch.int32, device=DEV)
    req_pool_indices = torch.tensor([0, 1, 2], dtype=torch.int32, device=DEV)
    req_to_token = torch.zeros(bs, max_ctx, dtype=torch.int32)
    req_to_token[0, :2048] = torch.arange(2048)
    req_to_token[1, :2048] = 2048 + torch.arange(2048)
    req_to_token[2, :4] = 4096 + torch.arange(4)
    req_to_token = req_to_token.to(DEV)
    pool_rows = 8192

    topk_indices = torch.full((R, TOPK), -1, dtype=torch.int32)
    early = torch.arange(128, dtype=torch.int32)  # req 0 window
    late = torch.arange(128, dtype=torch.int32) + 1900  # req 1 window
    for r in range(4):
        topk_indices[r] = early
    for r in range(4, 12):
        topk_indices[r] = late
    for r in range(12, 16):  # fresh prefill: own tokens only
        topk_indices[r, : r - 11] = torch.arange(r - 11)
    topk_indices = topk_indices.to(DEV)

    q = torch.randn(R, HEADS, DIM, dtype=torch.float16, device=DEV) * 0.5
    k_pool = torch.randn(pool_rows, DIM, dtype=torch.float16, device=DEV) * 0.5
    v_pool = torch.randn(pool_rows, DIM, dtype=torch.float16, device=DEV) * 0.5

    plan = build_grouped_union_plan(
        topk_indices, cu, seq_lens, req_pool_indices, req_to_token, max_ctx
    )
    return q, k_pool, v_pool, plan


def test_grouped_fully_masked_tiles_stay_finite():
    q, k_pool, v_pool, plan = _build_mixed_group_fixture()
    assert plan is not None, "fixture must stay inside the planner envelope"
    uidx, mask, u_pad = plan
    # 128 + 128 distinct group-0 entries bucketed to the 512 panel step
    assert u_pad == 512
    # NaN-critical shape: rows 4-7 (group-0 m=4..7) see no visible column
    # in either of the first two tiles
    assert int(mask[0, 4, :128].sum()) == 0
    # pad tail: group-1 rows use only columns 0-127 of the 512 panel
    assert int(mask[1, 0, :128].sum()) == TOPK
    assert int(mask[1, 0, 128:].sum()) == 0
    # row_keep edge: first fresh row keeps exactly its single token
    assert int(mask[2, 0].sum()) == 1
    assert int(mask[2, 3].sum()) == 4

    sm_scale = 1.0 / math.sqrt(DIM)
    out = run_grouped_sparse_prefill(q, k_pool, v_pool, plan, sm_scale)

    assert torch.isfinite(out.float()).all(), "NaN/Inf leaked from fully-masked tiles"
    ref = _torch_reference(q, k_pool, v_pool, uidx, mask, sm_scale)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
