"""Invariants from the radix-cache page-reuse poisoning investigation.

Production incident: warm probe(12) -> 32k row -> probe(32) dropped gsm8k
accuracy from 31/32 to 25/32 (floods=0, output coherent but reasoning wrong).
The poisoning enters through the full-KV page-reuse channel. This suite pins
the invariants that audit established:

T1/T2 (regression guards, expected green):
  - Evict -> reinsert must never double-account page ids: live nodes are
    pairwise disjoint, and no live node references a page the allocator has
    freed and not re-allocated.
  - match_prefix never serves pages freed by an eviction.

S1 (expected red until fixed):
  - COW dst-slot hygiene bypass. The regular scheduler path hands every
    freshly (re)assigned mamba slot an empty ReplaySSM ring
    (HybridReqToTokenPool.alloc: mamba_needs_clear=True and
    replayssm_write_pos[idx]=0, memory_pool.py). The radix COW path
    (MambaComponent.finalize_match_result_in_cache) instead allocates the dst
    slot directly via req_to_token_pool.mamba_allocator.alloc, and
    HybridReqToTokenPool.free_mamba_cache releases slots without touching
    replayssm_write_pos. A dst slot recycled from a decode request therefore
    keeps a stale ring cursor + stale ring records; the decode/verify kernels
    read ring[:write_pos] and ingest the previous owner's records.

Run on CPU (no GPU needed):
  CUDA_VISIBLE_DEVICES="" python -m pytest \
      test/registered/unit/mem_cache/test_radix_page_reuse_poison.py -v
"""

import contextlib
import unittest
from array import array
from dataclasses import dataclass
from typing import Optional
from unittest import mock

import torch
from unified_tree_core_inspection_interface import UnifiedTreeCoreInspectionInterface
from unified_tree_core_inspector import UnifiedTreeCoreInspector

from sglang.kernels.ops.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_cache.tree_core_registry import (
    _TREE_CORE_REGISTRY,
    resolve_tree_core_backend,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import (
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# A dedicated test backend selection, mirroring
# test_unified_radix_cache_unittest.py: the python inspector keeps these tests
# independent of the production backend default.
_TREE_CORE_TEST_BACKEND: Optional[str] = "python"


def _selected_tree_core_test_backend() -> str:
    return _TREE_CORE_TEST_BACKEND


@dataclass(frozen=True)
class CacheConfig:
    # Tree
    page_size: int = 1
    components: tuple[ComponentType, ...] = (ComponentType.FULL,)

    # Layer split (only matters for SWA/Mamba)
    num_layers: int = 24
    full_attention_layer_ids: tuple[int, ...] = (3, 7, 11, 15, 19, 23)

    # SWA
    sliding_window_size: Optional[int] = None

    # Mamba
    enable_mamba_extra_buffer: bool = False
    mamba_cache_size: int = 20
    mamba_intermediate_size: int = 256
    mamba_n_groups: int = 1
    mamba_num_heads: int = 2
    mamba_head_dim: int = 16
    mamba_state_size: int = 16
    mamba_conv_kernel: int = 4
    enable_int8_mamba_checkpoint: bool = False
    enable_linear_replayssm: bool = False

    # Model / pool
    kv_size: int = 256
    max_num_reqs: int = 10
    max_context_len: int = 512
    head_num: int = 2
    head_dim: int = 64
    dtype: torch.dtype = torch.bfloat16
    eviction_policy: str = "lru"
    is_eagle: bool = False

    @property
    def has_mamba(self) -> bool:
        return ComponentType.MAMBA in self.components

    @property
    def has_swa(self) -> bool:
        return ComponentType.SWA in self.components

    @property
    def non_full_layer_ids(self) -> list[int]:
        full = set(self.full_attention_layer_ids)
        return [i for i in range(self.num_layers) if i not in full]


def build_fixture(cfg: CacheConfig):
    """Create (cache, allocator, req_to_token_pool) from a CacheConfig.

    Fixture pattern copied from test_unified_radix_cache_unittest.py. The only
    addition is ``cfg.enable_linear_replayssm`` so the ReplaySSM ring cursor
    (replayssm_write_pos) exists for the COW dst-slot hygiene test.
    """
    server_args = ServerArgs(
        model_path="dummy",
        page_size=cfg.page_size,
        enable_int8_mamba_checkpoint=cfg.enable_int8_mamba_checkpoint,
    )
    # The mamba component reads mamba_cache_chunk_size, whose property otherwise
    # loads the HF config for self.model_path — impossible for the dummy model.
    # Mirror the property's default for a dummy HF config: FLA_CHUNK_SIZE.
    server_args._mamba_cache_chunk_size = max(FLA_CHUNK_SIZE, cfg.page_size)
    set_global_server_args_for_scheduler(server_args)
    # get_device() raises without any accelerator; this suite must also run on
    # a CPU-only box (e.g. CUDA_VISIBLE_DEVICES=9 on a busy GPU host).
    device = get_device() if torch.cuda.is_available() else "cpu"

    mamba2_cache_params = None
    if cfg.has_mamba:
        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            shape = Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=cfg.mamba_intermediate_size,
                n_groups=cfg.mamba_n_groups,
                num_heads=cfg.mamba_num_heads,
                head_dim=cfg.mamba_head_dim,
                state_size=cfg.mamba_state_size,
                conv_kernel=cfg.mamba_conv_kernel,
            )
            mamba2_cache_params = Mamba2CacheParams(
                shape=shape, layers=cfg.non_full_layer_ids
            )
        req_to_token_pool = HybridReqToTokenPool(
            size=cfg.max_num_reqs,
            mamba_size=cfg.mamba_cache_size,
            mamba_spec_state_size=cfg.max_num_reqs,
            max_context_len=cfg.max_context_len,
            device=device,
            enable_memory_saver=False,
            cache_params=mamba2_cache_params,
            mamba_layer_ids=cfg.non_full_layer_ids,
            enable_mamba_extra_buffer=cfg.enable_mamba_extra_buffer,
            speculative_num_draft_tokens=3,
            enable_linear_replayssm=cfg.enable_linear_replayssm,
        )
    else:
        req_to_token_pool = ReqToTokenPool(
            size=cfg.max_num_reqs,
            max_context_len=cfg.max_context_len,
            device=device,
            enable_memory_saver=False,
        )

    if cfg.has_swa:
        raise NotImplementedError("SWA components are out of scope for this suite")
    elif cfg.has_mamba:
        kv_pool = HybridLinearKVPool(
            size=cfg.kv_size,
            dtype=cfg.dtype,
            page_size=cfg.page_size,
            head_num=cfg.head_num,
            head_dim=cfg.head_dim,
            full_attention_layer_ids=cfg.full_attention_layer_ids,
            device=device,
            enable_memory_saver=False,
            mamba_pool=req_to_token_pool.mamba_pool,
        )
        allocator = TokenToKVPoolAllocator(
            size=cfg.kv_size,
            dtype=cfg.dtype,
            device=device,
            kvcache=kv_pool,
            need_sort=False,
        )
    else:
        kv_pool = MHATokenToKVPool(
            size=cfg.kv_size,
            page_size=cfg.page_size,
            dtype=cfg.dtype,
            head_num=cfg.head_num,
            head_dim=cfg.head_dim,
            layer_num=cfg.num_layers,
            device=device,
            enable_memory_saver=False,
        )
        allocator = TokenToKVPoolAllocator(
            size=cfg.kv_size,
            dtype=cfg.dtype,
            device=device,
            kvcache=kv_pool,
            need_sort=False,
        )

    cache_init_params = CacheInitParams(
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=cfg.page_size,
        disable=False,
        sliding_window_size=cfg.sliding_window_size,
        tree_components=cfg.components,
        enable_mamba_extra_buffer=cfg.enable_mamba_extra_buffer,
        eviction_policy=cfg.eviction_policy,
        is_eagle=cfg.is_eagle,
    )
    requested_backend = _selected_tree_core_test_backend()
    selected_backend = resolve_tree_core_backend(requested_backend, cache_init_params)

    def python_inspector_factory(params, components):
        return UnifiedTreeCoreInspector(params, components)

    def rust_inspector_factory(params, _components):
        from rust_unified_tree_core_inspector import RustUnifiedTreeCoreInspector

        return RustUnifiedTreeCoreInspector(params)

    with (
        envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override(requested_backend),
        mock.patch.dict(
            _TREE_CORE_REGISTRY,
            {"python": python_inspector_factory, "rust": rust_inspector_factory},
        ),
    ):
        cache = UnifiedRadixCache(params=cache_init_params)
    assert cache._tree_core_backend == selected_backend
    assert isinstance(cache.tree_core, UnifiedTreeCoreInspectionInterface), (
        "This suite requires a TreeCore backend that implements "
        "UnifiedTreeCoreInspectionInterface"
    )
    cache.cache_init_params = cache_init_params

    return cache, allocator, req_to_token_pool


def _node_children(cache, node_id):
    return cache.tree_core.get_child_node_ids(node_id)


def _device_value(cache, node_id, component_type):
    return cache.tree_core.get_component_device_value(node_id, component_type)


def _live_full_values(cache) -> dict:
    """node_id -> flattened FULL device value (token/page ids) for every live node.

    The root's FULL value is an empty list by design (UnifiedTreeCore), so both
    tensor and list values are handled and empty values are skipped.
    """
    values = {}
    stack = [cache.root_node_handle()]
    while stack:
        node_id = stack.pop()
        value = _device_value(cache, node_id, ComponentType.FULL)
        if value is not None:
            ids = (
                value.reshape(-1).tolist()
                if isinstance(value, torch.Tensor)
                else list(value)
            )
            if ids:
                values[node_id] = ids
        stack.extend(_node_children(cache, node_id))
    return values


@contextlib.contextmanager
def _tracked_free_pages(allocator):
    """Yield the running set of pages freed but not yet re-allocated.

    ``free`` adds ids, ``alloc`` removes them, so the set means "freed and
    currently unallocated". A live node referencing an id in this set is double
    accounting: the page is both owned by the tree and available for reuse.
    Legitimate page reuse (freed -> re-alloc -> live) does not false-positive.
    """
    freed = set()
    orig_alloc = allocator.alloc
    orig_free = allocator.free

    def tracking_alloc(need_size):
        result = orig_alloc(need_size)
        if result is not None:
            freed.difference_update(result.reshape(-1).tolist())
        return result

    def tracking_free(free_index):
        if free_index is not None and free_index.numel() > 0:
            freed.update(free_index.reshape(-1).tolist())
        return orig_free(free_index)

    with (
        mock.patch.object(allocator, "alloc", tracking_alloc),
        mock.patch.object(allocator, "free", tracking_free),
    ):
        yield freed


class TestPageIdDisjointnessAfterEvictReinsert(CustomTestCase):
    """T1: evict -> reinsert must never double-account full-KV page ids."""

    cfg = CacheConfig(page_size=4, components=(ComponentType.FULL,), kv_size=256)

    def _insert_full(self, cache, allocator, tokens):
        value = allocator.alloc(len(tokens))
        self.assertIsNotNone(value)
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=value[: len(tokens)],
            )
        )

    def test_live_page_ids_disjoint_after_evict_reinsert(self):
        cache, allocator, _ = build_fixture(self.cfg)
        for i in range(6):
            self._insert_full(cache, allocator, [i * 100 + j for j in range(1, 17)])

        with _tracked_free_pages(allocator) as freed:
            # Evict the two LRU leaves (16 tokens each with page_size=4).
            cache.evict(EvictParams(num_tokens=32))
            self.assertEqual(
                len(freed),
                32,
                "evicting two 16-token leaves must release exactly their pages",
            )
            # Reinsert disjoint keys; the allocator may legitimately hand back
            # freed pages, which the running freed set accounts for.
            for base in (600, 700):
                self._insert_full(cache, allocator, [base + j for j in range(1, 17)])

        live = _live_full_values(cache)
        self.assertEqual(
            sum(len(v) for v in live.values()),
            6 * 16,
            "6 live leaves x 16 tokens must survive evict + reinsert",
        )
        page_sets = [set(v) for v in live.values()]
        for i, a in enumerate(page_sets):
            for b in page_sets[i + 1 :]:
                self.assertFalse(
                    a & b, f"page ids double-accounted across nodes: {sorted(a & b)}"
                )
        for node_id, value in live.items():
            stale = set(value) & freed
            self.assertFalse(
                stale,
                f"node {node_id} references pages still free in the allocator: "
                f"{sorted(stale)}",
            )
        cache.sanity_check()


class TestMatchNeverServesEvictedPages(CustomTestCase):
    """T2: match_prefix must never serve pages freed by an eviction."""

    cfg = CacheConfig(page_size=4, components=(ComponentType.FULL,), kv_size=256)

    def _insert_full(self, cache, allocator, tokens):
        value = allocator.alloc(len(tokens))
        self.assertIsNotNone(value)
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=value[: len(tokens)],
            )
        )
        return value[: len(tokens)]

    def test_match_after_evict_serves_no_freed_pages(self):
        cache, allocator, _ = build_fixture(self.cfg)
        key_a = [1000 + i for i in range(1, 17)]
        value_a = self._insert_full(cache, allocator, key_a)
        pages_a = set(value_a.reshape(-1).tolist())

        with _tracked_free_pages(allocator) as freed:
            # Evict A's leaf; it is the only leaf, so the eviction quota lands
            # on exactly its pages.
            cache.evict(EvictParams(num_tokens=len(key_a)))
            self.assertEqual(
                freed,
                pages_a,
                "evicting the only leaf must free exactly A's pages",
            )
            # Reinsert a disjoint key. Its pages may legitimately reuse A's
            # freed pages; matching A must still never be served through them.
            key_c = [2000 + i for i in range(1, 17)]
            value_c = self._insert_full(cache, allocator, key_c)

            m_a = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", key_a))))
            self.assertLess(
                len(m_a.device_indices),
                len(key_a),
                "a fully evicted prefix must not be served in full",
            )
            served_a = set(m_a.device_indices.reshape(-1).tolist())
            self.assertFalse(
                served_a & freed,
                f"match served evicted (unallocated) pages: {sorted(served_a & freed)}",
            )

            m_c = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", key_c))))
            self.assertEqual(
                set(m_c.device_indices.reshape(-1).tolist()),
                set(value_c.reshape(-1).tolist()),
                "the surviving key must still match to its own pages",
            )
        cache.sanity_check()


class TestMambaCowDstSlotHygiene(CustomTestCase):
    """S1: the COW dst slot must be handed over hygienic (empty ReplaySSM ring).

    Regular path hygiene (HybridReqToTokenPool.alloc, memory_pool.py): a
    freshly (re)assigned mamba slot gets ``replayssm_write_pos[idx] = 0`` so
    decode/verify kernels treat the ring as empty and read only the checkpoint.
    The COW path (MambaComponent.finalize_match_result_in_cache) allocates the
    dst slot directly via ``mamba_allocator.alloc`` and only sets
    ``mamba_needs_clear = False`` — it never resets the ring cursor. Since
    ``free_mamba_cache`` does not sanitize released slots either, a dst slot
    recycled from a decode request carries a non-zero ``replayssm_write_pos``
    and stale ring records into its new owner. RED until fixed.
    """

    cfg = CacheConfig(
        page_size=1,
        components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,
        mamba_cache_size=8,
        enable_linear_replayssm=True,
    )

    def _make_req(self, rid):
        return Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=array("q"),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )

    @unittest.expectedFailure
    def test_cow_dst_slot_has_empty_replayssm_ring(self):
        # A/B verdict (2026-11-11, review image e067c7d49f98 vs 4ac1e08a5deb):
        # resetting the ring cursor at match time regressed the poison repro
        # from 27/32 floods=0 to 12/32 floods=15 — the delayed copy_from relies
        # on the dst cursor state, so the early reset is harmful. Keep this as
        # expectedFailure to document the fragile cross-layer contract until a
        # safe hygiene point is found.
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        write_pos = req_to_token_pool.mamba_pool.replayssm_write_pos
        self.assertIsNotNone(
            write_pos,
            "fixture must enable the ReplaySSM decode ring (replayssm_write_pos)",
        )
        mamba_allocator = req_to_token_pool.mamba_allocator

        tokens_a = [11, 12, 13, 14]
        value_a = allocator.alloc(len(tokens_a))
        self.assertIsNotNone(value_a)
        insert_req = self._make_req("insert-a")
        req_to_token_pool.alloc([insert_req])
        params = InsertParams(
            key=RadixKey(array("q", tokens_a)),
            value=value_a[: len(tokens_a)],
        )
        params.mamba_value = insert_req.kv.mamba_pool_idx.unsqueeze(0)
        cache.insert(params)

        # Build a stale slot through the real request lifecycle: assign a slot
        # via the scheduler path, accumulate pending ring records (a decode
        # that has not flushed), then release the slot.
        stale_req = self._make_req("stale")
        self.assertIsNotNone(req_to_token_pool.alloc([stale_req]))
        stale_slot = int(stale_req.kv.mamba_pool_idx)
        # Regular-path hygiene: a fresh slot starts with an empty ring.
        self.assertEqual(int(write_pos[stale_slot]), 0)
        write_pos[stale_slot] = 7  # simulate un-flushed ReplaySSM ring records
        req_to_token_pool.free_mamba_cache(stale_req)
        # FACT: free_mamba_cache does not sanitize the slot (it frees the id
        # and clears req-side fields; replayssm_write_pos stays non-zero).
        self.assertEqual(
            int(write_pos[stale_slot]),
            7,
            "free must leave the ring cursor untouched for this test's premise",
        )

        # Leave the stale slot at the front of the mamba free list so the COW
        # dst allocation must recycle exactly that slot.
        self.assertIn(stale_slot, mamba_allocator.free_slots.tolist())
        held = []
        while int(mamba_allocator.free_slots[0]) != stale_slot:
            slot = mamba_allocator.alloc(1)
            self.assertIsNotNone(slot, "stale slot missing from the free list")
            held.append(slot)

        # Radix mamba hit with COW, matching production: the matching request
        # is fresh (holds no mamba slot yet), so the mamba finalize allocates
        # its dst slot directly from mamba_allocator.
        req = self._make_req("match-cow")
        self.assertFalse(req.kv.holds_mamba)
        result = cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", tokens_a)),
                cow_mamba=True,
                req=req,
            )
        )
        self.assertIsNotNone(result.best_match_node)
        self.assertIsNotNone(
            req.kv.mamba_cow_src_index,
            "a radix mamba hit must arm the deferred COW on the request",
        )
        self.assertIsNotNone(req.kv.mamba_pool_idx, "COW must allocate a dst slot")
        self.assertEqual(
            int(req.kv.mamba_pool_idx),
            stale_slot,
            "the COW dst slot must be the recycled (stale) slot",
        )
        # The implementation marks the slot as needing no clear...
        self.assertFalse(req.kv.mamba_needs_clear)
        # ...but hands over a dirty ring cursor: decode/verify reads
        # ring[:write_pos] and would ingest the previous owner's records.
        self.assertEqual(
            int(write_pos[req.kv.mamba_pool_idx]),
            0,
            "COW dst slot must start with an empty ReplaySSM ring "
            "(write_pos == 0), mirroring the regular alloc path",
        )


if __name__ == "__main__":
    unittest.main()
