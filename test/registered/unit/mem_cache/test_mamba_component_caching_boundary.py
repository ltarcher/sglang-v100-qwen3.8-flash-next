"""TDD boundary coverage for MambaComponent's extra_buffer caching handoff.

The extra_buffer (ping-pong) strategy donates the request's tracked mamba
slot to the radix tree and swaps in a fresh slot; on finish, the tracked
state may overshoot the insert key, in which case the previous ping-pong
slot is donated instead (`_select_finished_checkpoint`). These paths had
zero direct unit coverage: the SWA sibling methods were tested, the int8
variant was tested, but the mamba donate/checkpoint selection was only
exercised indirectly through e2e KL tests that reuse the SAME prefix.
The 2026-11 testbench found the poison sequence this review targets
(32k request cached -> later unrelated short prompts degrade -> flush
restores), which lives exactly in this handoff, so the invariants here
pin the slot-ownership bookkeeping that must never cross-contaminate:

- a donated slot is owned by the tree, never freed by cleanup on success,
  and never handed to a new request (realloc requires a fresh slot);
- every cleanup path's freed-slot delta matches the insert outcome;
- eviction that frees donated slots must make them re-allocatable while
  the tree no longer references them;
- flush (reset) returns every mamba slot to the allocator.

Fixture note: reuses `build_fixture` from the sibling unittest module so
the cache/pool wiring is identical to the rest of the suite. Device
follows `get_device()` (cpu in a CPU container, cuda on a GPU runner).
"""

import unittest
from array import array
from unittest import mock

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from test_unified_radix_cache_unittest import CacheConfig, build_fixture

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-small")


def _slot(tensor_1d, idx=0) -> int:
    return int(tensor_1d.reshape(-1)[idx])


def _flatten_slots(cache) -> set[int]:
    flat = cache.all_mamba_values_flatten()
    if flat is None or flat.numel() == 0:
        return set()
    return {int(v) for v in flat.reshape(-1).tolist()}


def _make_req(pool, rid: str) -> Req:
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q"),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )
    pool.alloc([req])
    return req


def _set_track(req, last_seqlen, last_idx=1, prev_seqlen=None) -> None:
    req.kv.mamba_last_track_seqlen = last_seqlen
    req.kv.mamba_last_track_idx = last_idx
    req.kv.mamba_prev_track_seqlen = prev_seqlen


class TestMambaExtraBufferCachingHandoff(CustomTestCase):
    """Component-level: prepare_for_caching_req / cleanup_after_caching_req.

    Slot ledger per fresh request (extra_buffer, overlap=1, buffer size 2):
    alloc() takes 3 slots (1 state + 2 ping-pong). Donate allocates 1
    replacement. The assertions below pin each cleanup branch's delta.
    """

    cfg = CacheConfig(
        page_size=1,
        components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,
        mamba_cache_size=32,
        kv_size=1024,
        max_context_len=2048,
    )

    def setUp(self):
        self.cache, self.allocator, self.pool = build_fixture(self.cfg)
        self.mamba: MambaComponent = self.cache.components[ComponentType.MAMBA]
        self.alloc = self.pool.mamba_allocator
        self.baseline = self.alloc.available_size()
        self._rid = 0

    def _req(self) -> Req:
        self._rid += 1
        return _make_req(self.pool, f"mamba-handoff-{self._rid}")

    def test_alloc_contract_is_three_slots_per_request(self):
        req = self._req()
        self.assertEqual(self.baseline - self.alloc.available_size(), 3)
        self.assertTrue(req.kv.holds_mamba)
        self.assertTrue(req.kv.mamba_needs_clear)
        self.assertEqual(req.kv.mamba_ping_pong_track_buffer.numel(), 2)
        self.assertNotEqual(
            _slot(req.kv.mamba_ping_pong_track_buffer, 0),
            _slot(req.kv.mamba_ping_pong_track_buffer, 1),
        )

    def test_finished_normal_path_donates_tracked_slot(self):
        req = self._req()
        avail_after_alloc = self.alloc.available_size()
        _set_track(req, last_seqlen=64)

        params = InsertParams()
        ret = self.mamba.prepare_for_caching_req(req, params, 100, is_finished=True)
        self.assertEqual(ret, 64)
        self.assertEqual(params.mamba_keep_idx, req.kv.mamba_last_track_idx)
        self.assertIsNotNone(params.mamba_value)
        # The donated value is the tracked ping-pong slot itself (no copy).
        self.assertEqual(
            _slot(params.mamba_value),
            _slot(req.kv.mamba_ping_pong_track_buffer, params.mamba_keep_idx),
        )
        # Finished selection takes the slot in place: no new allocation.
        self.assertEqual(self.alloc.available_size(), avail_after_alloc)

        result = InsertResult(prefix_len=0, total_len=100)
        self.mamba.cleanup_after_caching_req(
            req, is_finished=True, insert_result=result, insert_params=params
        )
        # Frees the main state slot + the other ping-pong slot; the donated
        # slot stays tree-owned.
        self.assertEqual(self.alloc.available_size(), avail_after_alloc + 2)
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertIsNone(req.kv.mamba_ping_pong_track_buffer)
        self.assertEqual(self.baseline - self.alloc.available_size(), 1)

    def test_finished_overshoot_picks_previous_slot(self):
        req = self._req()
        avail_after_alloc = self.alloc.available_size()
        _set_track(req, last_seqlen=150, last_idx=1, prev_seqlen=72)

        params = InsertParams()
        ret = self.mamba.prepare_for_caching_req(req, params, 100, is_finished=True)
        self.assertEqual(ret, 72)
        self.assertEqual(params.mamba_keep_idx, 0)
        self.assertEqual(
            _slot(params.mamba_value), _slot(req.kv.mamba_ping_pong_track_buffer, 0)
        )

        result = InsertResult(prefix_len=0, total_len=100)
        self.mamba.cleanup_after_caching_req(
            req, is_finished=True, insert_result=result, insert_params=params
        )
        # keep=0: the previous slot is tree-owned, the tracked slot is freed.
        self.assertEqual(self.alloc.available_size(), avail_after_alloc + 2)
        self.assertEqual(self.baseline - self.alloc.available_size(), 1)

    def test_finished_overshoot_without_previous_donates_nothing(self):
        req = self._req()
        _set_track(req, last_seqlen=150, last_idx=1, prev_seqlen=None)

        params = InsertParams()
        ret = self.mamba.prepare_for_caching_req(req, params, 100, is_finished=True)
        self.assertEqual(ret, 0)
        self.assertIsNone(params.mamba_value)
        self.assertIsNone(params.mamba_keep_idx)

        result = InsertResult(prefix_len=0, total_len=100)
        self.mamba.cleanup_after_caching_req(
            req, is_finished=True, insert_result=result, insert_params=params
        )
        # Nothing cacheable: every slot the request held goes back.
        self.assertEqual(self.alloc.available_size(), self.baseline)

    def test_finished_overshoot_previous_beyond_key_donates_nothing(self):
        req = self._req()
        _set_track(req, last_seqlen=150, last_idx=1, prev_seqlen=120)

        params = InsertParams()
        ret = self.mamba.prepare_for_caching_req(req, params, 100, is_finished=True)
        self.assertEqual(ret, 0)

        result = InsertResult(prefix_len=0, total_len=100)
        self.mamba.cleanup_after_caching_req(
            req, is_finished=True, insert_result=result, insert_params=params
        )
        self.assertEqual(self.alloc.available_size(), self.baseline)

    def test_finished_without_track_state_returns_zero_and_recovers_all_slots(self):
        """cache_len=0 must make the insert early-exit and cleanup free all.

        Production shape (UnifiedRadixCache.checkpoint L1127-1133): an
        effective_cache_len of 0 skips the insert and calls
        cleanup_after_caching_req WITHOUT an insert_result. Note the current
        contract still hands out the tracked slot as mamba_value even with no
        track state (`_select_finished_checkpoint` maps None -> 0 <= key len);
        that value is dead weight here — if the early-exit ever stops firing,
        a never-written slot would land in the tree as a state. The assertions
        pin both the return value and the full recovery.
        """
        req = self._req()
        params = InsertParams()
        ret = self.mamba.prepare_for_caching_req(req, params, 100, is_finished=True)
        self.assertEqual(ret, 0)
        # Current contract: the tracked slot is still handed out, dead weight.
        self.assertIsNotNone(params.mamba_value)

        # Insert early-exit path: no insert_result argument at all.
        self.mamba.cleanup_after_caching_req(
            req, is_finished=True, insert_params=params
        )
        self.assertEqual(self.alloc.available_size(), self.baseline)

    def test_finished_dedup_frees_everything(self):
        req = self._req()
        _set_track(req, last_seqlen=64)

        params = InsertParams()
        ret = self.mamba.prepare_for_caching_req(req, params, 100, is_finished=True)
        self.assertEqual(ret, 64)

        # The tree already holds a state for this key: nothing was inserted.
        result = InsertResult(prefix_len=0, total_len=100, mamba_exist=True)
        self.mamba.cleanup_after_caching_req(
            req, is_finished=True, insert_result=result, insert_params=params
        )
        self.assertEqual(self.alloc.available_size(), self.baseline)

    def test_finished_size1_buffer_decision_semantics(self):
        """Spec-decode shape (buffer size 1): overshoot has no other slot."""
        req = self._req()
        _set_track(req, last_seqlen=150, last_idx=0, prev_seqlen=72)
        with mock.patch.object(self.pool, "mamba_ping_pong_track_buffer_size", 1):
            # Overshoot cannot fall back to a non-existent other slot.
            self.assertIsNone(self.mamba._select_finished_checkpoint(req, 100))
            # Normal path donates the only slot under its own index.
            _set_track(req, last_seqlen=64, last_idx=0)
            cache_len, keep_idx = self.mamba._select_finished_checkpoint(req, 100)
            self.assertEqual(cache_len, 64)
            self.assertEqual(keep_idx, 0)
            self.assertEqual(
                _slot(req.kv.mamba_ping_pong_track_buffer, keep_idx),
                _slot(req.kv.mamba_ping_pong_track_buffer, 0),
            )

    def test_unfinished_donate_swaps_slot_and_syncs_mapping(self):
        req = self._req()
        avail_after_alloc = self.alloc.available_size()
        _set_track(req, last_seqlen=64, last_idx=1)
        buffer_before = req.kv.mamba_ping_pong_track_buffer.clone()

        params = InsertParams()
        ret = self.mamba.prepare_for_caching_req(req, params, 100, is_finished=False)
        self.assertEqual(ret, 64)
        self.assertIsNotNone(params.mamba_value)
        # Replacement slot allocated, donated old slot returned.
        self.assertEqual(self.alloc.available_size(), avail_after_alloc - 1)
        donated = _slot(params.mamba_value)
        buffer = req.kv.mamba_ping_pong_track_buffer
        self.assertIn(donated, {_slot(buffer_before, 0), _slot(buffer_before, 1)})
        self.assertNotEqual(_slot(buffer, 1), donated)
        # Device-side mapping mirrors the authoritative req buffer.
        self.assertTrue(
            torch.equal(
                self.pool.req_index_to_mamba_ping_pong_track_buffer_mapping[
                    req.kv.req_pool_idx
                ],
                buffer,
            )
        )

        result = InsertResult(prefix_len=0, total_len=100)
        self.mamba.cleanup_after_caching_req(
            req, is_finished=False, insert_result=result, insert_params=params
        )
        # Success keeps the donated slot (tree-owned) and the request alive.
        self.assertEqual(self.alloc.available_size(), avail_after_alloc - 1)
        self.assertIsNone(req.kv.mamba_last_track_seqlen)
        self.assertIsNone(req.kv.mamba_prev_track_seqlen)

    def test_unfinished_dedup_frees_donated_slot(self):
        req = self._req()
        avail_after_alloc = self.alloc.available_size()
        _set_track(req, last_seqlen=64)

        params = InsertParams()
        self.mamba.prepare_for_caching_req(req, params, 100, is_finished=False)
        result = InsertResult(prefix_len=0, total_len=100, mamba_exist=True)
        self.mamba.cleanup_after_caching_req(
            req, is_finished=False, insert_result=result, insert_params=params
        )
        self.assertEqual(self.alloc.available_size(), avail_after_alloc)

    def test_unfinished_insert_skipped_frees_donated_slot(self):
        req = self._req()
        avail_after_alloc = self.alloc.available_size()
        _set_track(req, last_seqlen=64)

        params = InsertParams()
        self.mamba.prepare_for_caching_req(req, params, 100, is_finished=False)
        self.mamba.cleanup_after_caching_req(
            req, is_finished=False, insert_result=None, insert_params=params
        )
        self.assertEqual(self.alloc.available_size(), avail_after_alloc)


class TestMambaDonateEvictReallocIntegration(CustomTestCase):
    """Cache-level invariants: donate -> evict -> realloc stays clean.

    This is the registered-test shape the testbench poison sequence needs:
    the tree's mamba slots must be releasable, must never leak into a new
    request's allocation while still referenced, and a fresh slot must be
    flagged for clearing.
    """

    cfg = CacheConfig(
        page_size=1,
        components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,
        mamba_cache_size=16,
        kv_size=1024,
        max_context_len=2048,
    )

    def setUp(self):
        self.cache, self.allocator, self.pool = build_fixture(self.cfg)
        self.mamba: MambaComponent = self.cache.components[ComponentType.MAMBA]
        self.alloc = self.pool.mamba_allocator
        self.baseline = self.alloc.available_size()
        self._rid = 0

    def _req(self) -> Req:
        self._rid += 1
        return _make_req(self.pool, f"mamba-integ-{self._rid}")

    def _insert_donated(self, tokens) -> tuple[Req, InsertParams]:
        req = self._req()
        avail_after_alloc = self.alloc.available_size()
        _set_track(req, last_seqlen=len(tokens))
        value = self.allocator.alloc(len(tokens))
        self.assertIsNotNone(value)
        params = InsertParams(
            key=RadixKey(array("q", tokens)),
            value=value[: len(tokens)],
        )
        ret = self.mamba.prepare_for_caching_req(req, params, len(tokens), False)
        self.assertEqual(ret, len(tokens))
        self.assertEqual(self.alloc.available_size(), avail_after_alloc - 1)
        result = self.cache.insert(params)
        return req, params, result

    def test_donated_slots_evicted_then_reallocated_are_clean(self):
        _, params1, _ = self._insert_donated(list(range(1, 129)))
        _, params2, _ = self._insert_donated(list(range(200, 292)))

        tree_slots = _flatten_slots(self.cache)
        self.assertEqual(len(tree_slots), 2)
        self.assertIn(_slot(params1.mamba_value), tree_slots)
        self.assertIn(_slot(params2.mamba_value), tree_slots)

        result = self.cache.evict_for_alloc(EvictParams(num_tokens=0, mamba_num=2))
        self.assertEqual(result.mamba_num_evicted, 2)
        self.assertEqual(_flatten_slots(self.cache), set())
        self.assertEqual(self.cache.mamba_evictable_size(), 0)

        # Freed donated slots are back in the allocator for reuse.
        self.assertEqual(self.baseline - self.alloc.available_size(), 6)  # 2 reqs live
        req3 = self._req()
        self.assertTrue(req3.kv.mamba_needs_clear)
        self.assertNotIn(_slot(req3.kv.mamba_pool_idx), _flatten_slots(self.cache))
        self.cache.sanity_check()

    def test_match_cow_never_hands_tree_slot_to_request(self):
        self._insert_donated(list(range(1, 65)))
        tree_slots = _flatten_slots(self.cache)
        self.assertEqual(len(tree_slots), 1)
        tree_slot = next(iter(tree_slots))

        req = self._req()
        own_slot = _slot(req.kv.mamba_pool_idx)
        match = self.cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", list(range(1, 65)))),
                cow_mamba=True,
                req=req,
            )
        )
        self.assertEqual(len(match.device_indices), 64)
        self.assertIsNotNone(req.kv.mamba_cow_src_index)
        self.assertEqual(_slot(req.kv.mamba_cow_src_index), tree_slot)
        self.assertNotEqual(own_slot, tree_slot)
        self.assertFalse(req.kv.mamba_needs_clear)
        self.assertEqual(_flatten_slots(self.cache), tree_slots)
        self.cache.sanity_check()

    def test_flush_resets_every_mamba_slot(self):
        req1, _, _ = self._insert_donated(list(range(1, 65)))
        req2, _, _ = self._insert_donated(list(range(200, 232)))
        live = self.baseline - self.alloc.available_size()
        self.assertGreater(live, 0)

        # Production flush_cache runs with an empty running batch: every
        # request has finished and its non-donated slots are already freed.
        # Free them exactly the way the finished handoff does (main state +
        # the whole ping-pong buffer).
        self.pool.free_mamba_cache(req1)
        self.pool.free_mamba_cache(req2)

        # Recorded contract: cache.reset() ALONE drops the tree but does not
        # return the donated mamba slots (this fixture's allocator still sees
        # them held). The scheduler always pairs tree_cache.reset() with
        # token_to_kv_pool_allocator.clear() (scheduler.flush_cache), which
        # reclaims the mamba sub-allocator wholesale. If either side of that
        # pairing changes, this assertion is the tripwire.
        self.cache.reset()
        self.assertEqual(_flatten_slots(self.cache), set())
        self.assertEqual(
            self.alloc.available_size(), self.baseline - 2, "reset freed tree slots"
        )
        self.alloc.clear()
        self.assertEqual(
            self.alloc.available_size(), self.baseline, "flush leaked mamba slots"
        )
        match = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1])))
        )
        self.assertEqual(len(match.device_indices), 0)

    def test_finished_insert_lands_in_tree_and_survives_cleanup(self):
        req = self._req()
        _set_track(req, last_seqlen=48)
        value = self.allocator.alloc(64)
        params = InsertParams(
            key=RadixKey(array("q", list(range(1, 65)))),
            value=value[:64],
        )
        ret = self.mamba.prepare_for_caching_req(req, params, 64, is_finished=True)
        self.assertEqual(ret, 48)
        # The cache layer truncates the insert key to the components' effective
        # length before calling insert (UnifiedRadixCache.cache_finished_req),
        # so a donated state always sits on a node whose depth it covers.
        params.key = RadixKey(array("q", list(range(1, 49))))
        params.value = params.value[:48]
        result = self.cache.insert(params)
        self.assertFalse(result.mamba_exist)
        donated = _slot(params.mamba_value)
        self.assertIn(donated, _flatten_slots(self.cache))

        self.mamba.cleanup_after_caching_req(
            req, is_finished=True, insert_result=result, insert_params=params
        )
        self.assertIn(donated, _flatten_slots(self.cache))
        match = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", list(range(1, 49)))))
        )
        self.assertEqual(len(match.device_indices), 48)
        self.cache.sanity_check()


class TestMambaMatchClipsToReusableState(CustomTestCase):
    """A match must never serve Full KV longer than the reusable Mamba state.

    The device-only tree accepts a Full boundary without a Mamba state (a
    divergence split keeps the state on the deeper child, and a donated state
    can be evicted). Serving that prefix would start the request's state from
    zero at a non-zero position, so the match is clipped to the deepest
    state-bearing ancestor, or to zero when there is none.
    """

    cfg = CacheConfig(
        page_size=1,
        components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,
        mamba_cache_size=32,
        kv_size=4096,
        max_context_len=4096,
    )

    def setUp(self):
        self.cache, self.allocator, self.pool = build_fixture(
            self.cfg, mamba_cache_chunk_size=64
        )
        self.mamba: MambaComponent = self.cache.components[ComponentType.MAMBA]
        self._rid = 0

    def _req(self) -> Req:
        self._rid += 1
        return _make_req(self.pool, f"mamba-clip-{self._rid}")

    def _insert(self, tokens, donate_len) -> InsertParams:
        req = self._req()
        _set_track(req, last_seqlen=donate_len)
        value = self.allocator.alloc(len(tokens))
        self.assertIsNotNone(value)
        params = InsertParams(
            key=RadixKey(array("q", tokens)),
            value=value[: len(tokens)],
        )
        ret = self.mamba.prepare_for_caching_req(req, params, len(tokens), True)
        self.assertEqual(ret, donate_len)
        self.cache.insert(params)
        return params

    def _match(self, tokens):
        # A fresh request reaches the scheduler without a mamba slot; the match
        # must clip before COW allocates one (production: alloc_for_extend).
        self._rid += 1
        req = Req(
            rid=f"mamba-clip-{self._rid}",
            origin_input_text="",
            origin_input_ids=array("q"),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        self.assertFalse(req.kv.holds_mamba)
        result = self.cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", tokens)),
                cow_mamba=True,
                req=req,
            )
        )
        return req, result

    def test_split_fragment_without_any_state_serves_nothing(self):
        # 1..128 owns a state at 128. The second insert diverges at 64, so the
        # split fragment (64 tokens) is left state-less while its child keeps
        # the 128-token state further down this branch.
        self._insert(list(range(1, 129)), 128)
        self._insert(list(range(1, 65)) + list(range(9001, 9065)), 128)

        req, result = self._match(list(range(1, 65)))

        self.assertEqual(len(result.device_indices), 0)
        self.assertEqual(result.full_kv_hit_length, 64)
        self.assertEqual(result.best_match_node, self.cache.tree_core.root_node.id)
        self.assertIsNone(result.mamba_branching_seqlen)
        self.assertIsNone(req.kv.mamba_cow_src_index)
        # Nothing to copy: the request stays empty and will clear on alloc.
        self.assertFalse(req.kv.holds_mamba)

    def test_split_fragment_clips_to_deepest_state_bearing_ancestor(self):
        # 704 and 1408 both own states. The third insert diverges at 960, so the
        # 960 fragment is state-less; the reusable state is the 704 ancestor.
        insert_704 = self._insert(list(range(1, 705)), 704)
        self._insert(list(range(1, 1409)), 1408)
        self._insert(list(range(1, 961)) + list(range(9001, 9065)), 1024)

        req, result = self._match(list(range(1, 961)))

        self.assertEqual(len(result.device_indices), 704)
        self.assertEqual(result.full_kv_hit_length, 960)
        self.assertIsNone(result.mamba_branching_seqlen)
        self.assertIsNotNone(req.kv.mamba_cow_src_index)
        self.assertEqual(
            _slot(req.kv.mamba_cow_src_index), _slot(insert_704.mamba_value)
        )

    def test_state_holding_req_keeps_device_match_boundary(self):
        # A request that already owns a state continues from it; clipping to a
        # shallower tree ancestor would strand that state, so it is not clipped.
        self._insert(list(range(1, 705)), 704)
        self._insert(list(range(1, 1409)), 1408)
        self._insert(list(range(1, 961)) + list(range(9001, 9065)), 1024)

        req = self._req()
        self.assertTrue(req.kv.holds_mamba)
        result = self.cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", list(range(1, 961)))),
                cow_mamba=True,
                req=req,
            )
        )

        self.assertEqual(len(result.device_indices), 960)
        self.assertIsNone(result.mamba_branching_seqlen)


if __name__ == "__main__":
    unittest.main()
