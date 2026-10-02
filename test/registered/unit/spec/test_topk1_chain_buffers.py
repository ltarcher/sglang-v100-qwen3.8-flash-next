"""Regression: topk=1 chain buffers must be stable per width across rebuilds.

draft_forward returns the prealloc parent_list / top_scores_index buffers and
the draft CUDA graph capture records them as graph outputs, so each width's
captured graphs keep their data pointers permanently. Rebuilding used to rebind
fresh tensors, orphaning every previously captured width's pointer; the first
replay after an adaptive width switch then read recycled memory as parent_list
and wedged the engine with "invalid eagle tree" warnings (GLM-5.3-Flash,
4xV100, 2026-10-01).
"""

import unittest
from unittest import mock

from sglang.srt.speculative import base_spec_worker


class _Bag:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _rebuild(worker, num_steps, max_bs=8):
    """Run _rebuild_topk1_chain_buffers under stubbed context bags."""

    graph = _Bag(cuda_graph_config=_Bag(decode=_Bag(max_bs=max_bs)))
    with (
        mock.patch.object(base_spec_worker, "get_exec", lambda: _Bag(graph=graph)),
        mock.patch.object(
            base_spec_worker, "get_schedule", lambda: _Bag(max_running_requests=4)
        ),
    ):
        worker.speculative_num_steps = num_steps
        worker.speculative_num_draft_tokens = num_steps + 1
        worker._rebuild_topk1_chain_buffers()


class TestTopk1ChainBuffers(unittest.TestCase):
    def _worker(self):
        class _W(base_spec_worker.EagleDraftWorkerBase):
            def draft(self):
                pass

            def draft_extend(self):
                pass

        worker = _W()
        worker.topk = 1
        worker.device = "cpu"
        worker.server_args = None
        return worker

    def test_buffers_survive_interleaved_rebuilds(self):
        worker = self._worker()
        _rebuild(worker, 3)
        parents, score_indices = (
            worker._topk1_parents_prealloc,
            worker._topk1_score_indices_prealloc,
        )
        _rebuild(worker, 5)
        _rebuild(worker, 3)
        # The width-3 graphs captured against the first rebuild must still
        # address live storage holding the width-3 values.
        self.assertIs(worker._topk1_parents_prealloc, parents)
        self.assertIs(worker._topk1_score_indices_prealloc, score_indices)
        self.assertEqual(parents[0, :2].tolist(), [-1, 0])
        self.assertEqual(score_indices[0, :3].tolist(), [0, 1, 2])

    def test_values_match_width_after_refill(self):
        worker = self._worker()
        _rebuild(worker, 5)
        self.assertEqual(worker._topk1_parents_prealloc[0, :4].tolist(), [-1, 0, 1, 2])
        self.assertEqual(worker._topk1_score_indices_prealloc.shape[1], 5)
        self.assertEqual(
            worker._topk1_score_indices_prealloc[0].tolist(), [0, 1, 2, 3, 4]
        )
        _rebuild(worker, 1)
        # A single-step chain has no parent entries.
        self.assertEqual(worker._topk1_parents_prealloc.shape[1], 0)
        self.assertEqual(worker._topk1_score_indices_prealloc[0].tolist(), [0])
        _rebuild(worker, 5)
        self.assertEqual(worker._topk1_parents_prealloc[0, :4].tolist(), [-1, 0, 1, 2])

    def test_zero_steps_adaptive_candidate(self):
        worker = self._worker()
        _rebuild(worker, 0)
        self.assertEqual(worker._topk1_parents_prealloc.shape, (8, 0))
        self.assertEqual(worker._topk1_score_indices_prealloc.shape, (8, 0))


if __name__ == "__main__":
    unittest.main()
