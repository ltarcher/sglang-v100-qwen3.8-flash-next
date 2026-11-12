"""Eligibility gate for the SM70 grouped TileLang sparse prefill (SGLANG_SM70_QSA_GROUPED_PREFILL).

The env flag (checked at the extend-path call site) only enables the attempt;
`_forward_sm70_grouped_sparse_prefill` itself must still vet every batch shape
before handing rows to the grouped planner, because the planner assumes the
uniform valid-first -1-padding layout of pure extend rows. Anything off — a
breakable CUDA graph window, a mixed-chunk tail, a wrong head dim, a
non-contiguous pool, no req_to_token — must fall through to the Triton packed
path by returning None. These are CPU fp16 tensors; the planner and the
TileLang kernel are mocked.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import ANY, patch

import torch

from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    ForwardMode,
    QwenSparseAttnBackend,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

BACKEND = "sglang.srt.layers.attention.qwen_sparse_attn_backend"
GROUPED_KERNELS = (
    "sglang.srt.layers.attention.tilelang_fa_v100._kernels_grouped_sparse_prefill"
)

SCALING = 1.0


class _Harness(CustomTestCase):
    def _backend(self, req_to_token="req_to_token"):
        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        backend.req_to_token = req_to_token
        backend.max_context_len = 2048
        backend._sm70_grouped_prefill_logged = False
        # No KV host tier (production-less stub); the tier3 gate reads this
        # attribute on every path that survives the mixed-batch guard.
        backend._kv_host_tier = None
        return backend

    def _batch(self, forward_mode=ForwardMode.EXTEND, original_mode=None):
        return SimpleNamespace(
            forward_mode=forward_mode,
            _original_forward_mode=original_mode,
            seq_lens=torch.tensor([256], dtype=torch.int64),
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
        )

    def _eligible_inputs(self):
        rows = 256
        q = torch.zeros(rows, 6, 256, dtype=torch.float16)
        topk = torch.zeros(rows, 16, dtype=torch.int32)
        k = torch.zeros(rows, 1, 256, dtype=torch.float16)
        v = torch.zeros(rows, 1, 256, dtype=torch.float16)
        return q, k, v, topk

    def _call(self, backend, q, k, v, topk, batch):
        cu_seqlens_q = torch.tensor([0, q.shape[0]], dtype=torch.int32)
        return backend._forward_sm70_grouped_sparse_prefill(
            q, k, v, batch, topk, cu_seqlens_q, SCALING
        )


class TestGroupedPrefillEligibility(_Harness):
    def _assert_ineligible(self, **kwargs):
        """The batch must fall through (None) without reaching the planner."""
        q, k, v, topk = self._eligible_inputs()
        for key, value in kwargs.items():
            if key == "q":
                q = value
            elif key == "k":
                k = value
            elif key == "v":
                v = value
            elif key == "topk":
                topk = value
        backend = self._backend()
        batch = self._batch(
            forward_mode=kwargs.get("forward_mode", ForwardMode.EXTEND),
            original_mode=kwargs.get("original_mode"),
        )
        with (
            patch(
                f"{BACKEND}.is_in_breakable_cuda_graph",
                return_value=kwargs.get("breakable", False),
            ),
            patch(f"{GROUPED_KERNELS}.build_grouped_union_plan") as build_plan,
            patch(f"{GROUPED_KERNELS}.run_grouped_sparse_prefill") as run_kernel,
        ):
            result = self._call(backend, q, k, v, topk, batch)
        self.assertIsNone(result)
        build_plan.assert_not_called()
        run_kernel.assert_not_called()

    def test_breakable_graph_window_falls_through(self):
        q, k, v, topk = self._eligible_inputs()
        backend = self._backend()
        batch = self._batch()
        with (
            patch(f"{BACKEND}.is_in_breakable_cuda_graph", return_value=True),
            patch(f"{GROUPED_KERNELS}.build_grouped_union_plan") as build_plan,
            patch(f"{GROUPED_KERNELS}.run_grouped_sparse_prefill") as run_kernel,
        ):
            self.assertIsNone(self._call(backend, q, k, v, topk, batch))
        build_plan.assert_not_called()
        run_kernel.assert_not_called()

    def test_mixed_forward_mode_falls_through(self):
        self._assert_ineligible(forward_mode=ForwardMode.MIXED)

    def test_mixed_original_mode_falls_through(self):
        self._assert_ineligible(original_mode=ForwardMode.MIXED)

    def test_rows_below_the_tile_floor_fall_through(self):
        _, _, _, topk = self._eligible_inputs()
        self._assert_ineligible(
            q=torch.zeros(128, 6, 256, dtype=torch.float16), topk=topk[:128]
        )

    def test_bf16_query_falls_through(self):
        q, k, v, topk = self._eligible_inputs()
        self._assert_ineligible(q=q.to(torch.bfloat16))

    def test_wrong_query_rank_falls_through(self):
        q, k, v, topk = self._eligible_inputs()
        self._assert_ineligible(q=q.reshape(256, 6 * 256))

    def test_topk_indices_must_be_int32(self):
        q, k, v, topk = self._eligible_inputs()
        self._assert_ineligible(topk=topk.to(torch.int64))

    def test_wrong_kv_head_shape_falls_through(self):
        q, k, v, topk = self._eligible_inputs()
        self._assert_ineligible(k=torch.zeros(256, 1, 128, dtype=torch.float16))

    def test_v_pool_must_match_k_pool(self):
        q, k, v, topk = self._eligible_inputs()
        self._assert_ineligible(v=torch.zeros(256, 1, 128, dtype=torch.float16))

    def test_non_contiguous_pool_falls_through(self):
        q, k, v, topk = self._eligible_inputs()
        padded = torch.zeros(256, 1, 512, dtype=torch.float16)
        k = padded[:, :, :256]
        self._assert_ineligible(k=k)

    def test_missing_req_to_token_falls_through(self):
        q, k, v, topk = self._eligible_inputs()
        backend = self._backend(req_to_token=None)
        batch = self._batch()
        with (
            patch(f"{BACKEND}.is_in_breakable_cuda_graph", return_value=False),
            patch(f"{GROUPED_KERNELS}.build_grouped_union_plan") as build_plan,
            patch(f"{GROUPED_KERNELS}.run_grouped_sparse_prefill") as run_kernel,
        ):
            self.assertIsNone(self._call(backend, q, k, v, topk, batch))
        build_plan.assert_not_called()
        run_kernel.assert_not_called()


class TestGroupedPrefillHappyPath(_Harness):
    def test_eligible_batch_reaches_the_grouped_kernel(self):
        q, k, v, topk = self._eligible_inputs()
        backend = self._backend()
        batch = self._batch()
        # Realistic plan tuple: the one-shot log line reads uidx.shape[0].
        plan = (
            torch.zeros(4, dtype=torch.int32),
            torch.zeros(4, 16, dtype=torch.int32),
            64,
            torch.zeros(4, dtype=torch.uint8),
        )
        kernel_output = torch.zeros(256, 6, 256, dtype=torch.float16)
        with (
            patch(f"{BACKEND}.is_in_breakable_cuda_graph", return_value=False),
            patch(
                f"{GROUPED_KERNELS}.build_grouped_union_plan", return_value=plan
            ) as build_plan,
            patch(
                f"{GROUPED_KERNELS}.run_grouped_sparse_prefill",
                return_value=kernel_output,
            ) as run_kernel,
        ):
            result = self._call(backend, q, k, v, topk, batch)
        self.assertIs(result, kernel_output)
        build_plan.assert_called_once_with(
            topk,
            ANY,
            batch.seq_lens,
            batch.req_pool_indices,
            backend.req_to_token,
            backend.max_context_len,
            slot_table=None,
            owner_mask=None,
        )
        run_kernel.assert_called_once_with(q, k, v, plan, SCALING, peer_kv=None)


if __name__ == "__main__":
    unittest.main()
