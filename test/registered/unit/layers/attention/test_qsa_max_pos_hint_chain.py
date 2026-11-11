"""Unit test (CPU): the QSA ``_MAX_POS_HINT`` global rope-bound chain.

The indexer sizes ``rotary_emb.cos_sin_cache`` from the module-global
``_MAX_POS_HINT`` (``qsa/metadata.py``) instead of syncing on
``positions.max().item()`` every layer. The hint is safe only while the
whole chain upholds two invariants:

1. Every indexer forward must see a hint >= the largest position that
   forward touches. ``QwenSparseAttnBackend.init_forward_metadata`` sets
   it from ``seq_lens_cpu``; the MTP draft multistep backend pre-sets it
   with ``_speculative_max_row_length`` (host lengths + draft window) and
   then each step backend overwrites it with that step's host maximum.
   The overwrite stays safe only because per-step lengths grow
   monotonically (``_make_step_forward_batch`` adds ``step + 1``), so the
   last step's overwrite is the global maximum.
2. ``_extend_yarn_cache`` must only grow the cache and must cover the
   inclusive index ``needed_max_pos - 1``; a shrink would let a later
   smaller-but-stale hint under-allocate against live positions.

ff52c061e8 audit residual: these boundaries were previously implicit.
"""

from types import SimpleNamespace

import torch

import sglang.srt.layers.attention.qwen_sparse_attn_backend as backend_mod
from sglang.srt.layers.attention.qsa.metadata import (
    get_qsa_max_pos_hint,
    set_qsa_max_pos_hint,
)
from sglang.srt.layers.rotary_embedding.yarn import _extend_yarn_cache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from sglang.test.test_utils import CustomTestCase


class TestExtendYarnCacheBound(CustomTestCase):
    def _cache(self, rows=4, width=8):
        return torch.zeros(rows, width)

    @staticmethod
    def _inv_freq():
        return torch.ones(4)

    def test_grow_covers_inclusive_boundary_aligned(self):
        cache, _ = _extend_yarn_cache(
            self._cache(), self._inv_freq, mscale=1.0, needed_max_pos=6
        )
        # Position ``needed_max_pos - 1`` must be a valid cache row and the
        # length is aligned up (SGLANG_ROPE_CACHE_ALIGN, default 128).
        self.assertGreaterEqual(cache.shape[0], 6)

    def test_never_shrinks_and_noop_is_identity(self):
        cache6, _ = _extend_yarn_cache(
            self._cache(), self._inv_freq, mscale=1.0, needed_max_pos=6
        )
        smaller, _ = _extend_yarn_cache(
            cache6, self._inv_freq, mscale=1.0, needed_max_pos=3
        )
        self.assertIs(smaller, cache6)
        again, _ = _extend_yarn_cache(
            cache6, self._inv_freq, mscale=1.0, needed_max_pos=6
        )
        self.assertIs(again, cache6)

    def test_grow_keeps_prefix_and_appends_valid_rows(self):
        cache, _ = _extend_yarn_cache(
            self._cache(), self._inv_freq, mscale=1.0, needed_max_pos=130
        )
        self.assertGreaterEqual(cache.shape[0], 130)
        self.assertEqual(cache.shape[1], 8)
        # Original rows must survive bit-exact.
        self.assertTrue(torch.equal(cache[:4], self._cache()))


class TestSpeculativeRowLengthBound(CustomTestCase):
    def test_target_verify_window_covers_last_draft_position(self):
        # Target verify appends ``draft_token_num`` rows per request; the
        # last draft row's position is ``max(seq_lens) + draft_token_num - 1``.
        forward_batch = SimpleNamespace(
            seq_lens_cpu=torch.tensor([100, 500]),
            spec_info=SimpleNamespace(draft_token_num=4),
        )
        bound = backend_mod.QwenSparseAttnBackend._speculative_max_row_length(
            forward_batch, forward_batch.seq_lens_cpu
        )
        self.assertEqual(bound, 500 + 4)
        self.assertGreater(bound, 500 + 4 - 1)

    def test_draft_extend_window_uses_num_tokens_per_req(self):
        forward_batch = SimpleNamespace(
            seq_lens_cpu=torch.tensor([100, 500]),
            spec_info=SimpleNamespace(num_tokens_per_req=4),
        )
        bound = backend_mod.QwenSparseAttnBackend._speculative_max_row_length(
            forward_batch, forward_batch.seq_lens_cpu
        )
        self.assertEqual(bound, 504)

    def test_missing_cpu_mirror_falls_back_to_readback(self):
        forward_batch = SimpleNamespace(seq_lens_cpu=None, spec_info=None)
        bound = backend_mod.QwenSparseAttnBackend._speculative_max_row_length(
            forward_batch, torch.tensor([100, 500])
        )
        self.assertEqual(bound, 500)


def _draft_backend_stub(steps, topk=1):
    """Bare multistep backend instance driving the real step-batch builder."""
    backend = object.__new__(backend_mod.QwenSparseMultiStepDraftBackend)
    backend.speculative_num_steps = steps
    backend.topk = topk
    return backend


def _apply_backend_hint(step_batch):
    """Mirror ``QwenSparseAttnBackend.init_forward_metadata``'s hint update.

    The real init cannot run on CPU (it builds full device metadata), so the
    test reproduces exactly its hint expression -- the contract under test is
    the chain of values, not that expression itself.
    """
    set_qsa_max_pos_hint(int(step_batch.seq_lens_cpu[: step_batch.batch_size].max()))


class TestDraftStepOverwriteChain(CustomTestCase):
    def setUp(self):
        set_qsa_max_pos_hint(None)

    def tearDown(self):
        set_qsa_max_pos_hint(None)

    def _forward_batch(self, seq_lens, steps, topk=1):
        batch = len(seq_lens)
        return SimpleNamespace(
            forward_mode=SimpleNamespace(),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
            seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
            batch_size=batch,
            out_cache_loc=torch.arange(batch * topk * steps),
            spec_info=SimpleNamespace(draft_token_num=steps),
        )

    def test_chain_bounds_every_step_position(self):
        steps, topk = 4, 1
        forward_batch = self._forward_batch([100, 500], steps, topk)
        backend = _draft_backend_stub(steps, topk)

        # Multistep pre-set (real expression used by init_forward_metadata).
        set_qsa_max_pos_hint(
            backend_mod.QwenSparseAttnBackend._speculative_max_row_length(
                forward_batch, forward_batch.seq_lens
            )
        )
        multistep_bound = get_qsa_max_pos_hint()

        # Per-step overwrites + per-step true max positions, from the real
        # step-batch builder.
        step_max_positions = []
        for step in range(steps):
            step_batch = backend._make_step_forward_batch(forward_batch, step)
            _apply_backend_hint(step_batch)
            step_max_positions.append(int(step_batch.seq_lens.max()) - 1)

        final_hint = get_qsa_max_pos_hint()
        self.assertEqual(final_hint, max(step_max_positions) + 1)
        self.assertGreaterEqual(final_hint, multistep_bound)
        for step, max_pos in enumerate(step_max_positions):
            self.assertLessEqual(
                max_pos,
                final_hint,
                f"step {step} position {max_pos} exceeds hint {final_hint}",
            )

    def test_step_lengths_grow_monotonically(self):
        # The overwrite chain is safe because the LAST step's overwrite is
        # the global maximum; pin the monotonicity of the real builder's
        # per-step lengths so a future refactor cannot silently reverse it.
        steps, topk = 4, 1
        forward_batch = self._forward_batch([100, 500], steps, topk)
        backend = _draft_backend_stub(steps, topk)
        lengths = [
            backend._make_step_forward_batch(forward_batch, step).seq_lens_cpu
            for step in range(steps)
        ]
        for prev, curr in zip(lengths, lengths[1:]):
            self.assertTrue(torch.ge(curr, prev).all())

    def test_padding_rows_cannot_lower_the_bound(self):
        # Padding semantics of ``_make_step_forward_batch``: the batch already
        # contains the padding slots as its tail rows (with garbage lengths);
        # the builder overwrites exactly those tail rows with length 1. The
        # bound must come from the real rows and the garbage tail must not
        # leak into it (either direction).
        steps, topk = 4, 1
        forward_batch = self._forward_batch(
            [100, 500, 999], steps, topk
        )  # 999: garbage padding slot
        backend = _draft_backend_stub(steps, topk)
        padded = backend._make_step_forward_batch(
            forward_batch, steps - 1, num_padding=1
        )
        _apply_backend_hint(padded)
        self.assertEqual(get_qsa_max_pos_hint(), 500 + steps)


if __name__ == "__main__":
    import unittest

    unittest.main()
