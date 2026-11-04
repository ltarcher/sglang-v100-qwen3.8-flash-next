"""Unit test: real pair-split mapping formulas vs the CPU protocol spec.

test_kv_host_tier_pair_split.py replays the parity protocol with its own
``_Rank`` formulas; this test pins the REAL ``KVHostTier`` mapping methods
(``pair_owns_page``, ``pair_local_page``, ``map_slots``, ``map_slot_table``)
to those same expectations, so a drift between test spec and implementation
fails here instead of only at boot.

Checked invariants (docs/v100/KV_HOST_TIER_DESIGN.md 6.1):
- exactly one rank owns every pair page (parity split);
- each rank's local pages are a bijection onto 0..pair_pages/2-1;
- map_slots: owned page -> local row slot, peer page -> padding page
  (write-side discard), padding slots -> phys_slot_base window;
- map_slot_table: owner bit 30 selects the peer, row fields agree with
  map_slots for owned pages and with the peer's map_slots for peer pages
  (both ranks must resolve the same storage row for every logical page);
- block granularity > 1 keeps the bijection and the parity grouping.
"""

import torch

from sglang.srt.mem_cache.kv_host_tier import KVHostTier

PAGE = 8


def _make_tier(pair_pages: int, block: int, rank_bit: int) -> KVHostTier:
    t = object.__new__(KVHostTier)
    t.page_size = PAGE
    t._page_shift = PAGE.bit_length() - 1
    t.pair_pages = pair_pages
    t.phys_pages = pair_pages // 2
    t.pair_block = block
    t.pair_rank_bit = rank_bit
    t.logical_tokens = pair_pages * PAGE
    t.phys_slot_base = (pair_pages // 2) * PAGE + 64
    t.pair_split = True
    return t


class TestPairMappingFormulas:
    def test_every_page_has_exactly_one_owner(self):
        for block in (1, 2, 4):
            r0 = _make_tier(16, block, 0)
            r1 = _make_tier(16, block, 1)
            for gp in range(16):
                assert r0.pair_owns_page(gp) != r1.pair_owns_page(gp)

    def test_local_pages_are_a_bijection(self):
        for block in (1, 2, 4):
            r0 = _make_tier(16, block, 0)
            r1 = _make_tier(16, block, 1)
            for r in (r0, r1):
                rows = [
                    r.pair_local_page(gp) for gp in range(16) if r.pair_owns_page(gp)
                ]
                assert sorted(rows) == list(range(8))

    def test_map_slots_routes_owned_and_padding(self):
        r0 = _make_tier(16, 1, 0)
        slots = torch.arange(16 * PAGE, dtype=torch.int64)
        slots[:PAGE] = r0.phys_slot_base + torch.arange(PAGE)
        phys = r0.map_slots(slots)
        # Logical page 0 is the reserved padding page (slots[0:PAGE] above);
        # real logical pages start at 1.
        for lp in range(1, 16):
            gp = lp % 16
            row = r0.pair_local_page(gp)
            off = torch.arange(PAGE)
            if r0.pair_owns_page(gp):
                assert torch.equal(phys[lp * PAGE : (lp + 1) * PAGE], row * PAGE + off)
            else:
                # Peer-owned writes are discarded into the padding page.
                assert torch.equal(
                    phys[lp * PAGE : (lp + 1) * PAGE], r0.phys_pages * PAGE + off
                )
        # Padding-page input slots pass through to the reserved window.
        pad = r0.phys_slot_base + torch.arange(PAGE)
        assert torch.equal(phys[:PAGE], pad)

    def test_map_slot_table_matches_protocol_expectations(self):
        # Protocol spec (post polarity fix): the owner bit is rank-RELATIVE
        # -- bit=1 marks a page the reading rank does NOT own, i.e.
        # ``(group_parity != pair_rank_bit)``. The low bits are always the
        # owner rank's row: the pair-half bijection page, scaled to a token
        # slot. An absolute-parity bit here was the TP1/TP3 bug: it inverted
        # every odd rank's table so each page read its parity neighbor.
        def spec_entry(t: KVHostTier, lp: int) -> int:
            gp = lp % t.pair_pages
            group = gp // t.pair_block
            owner_bit = ((group & 1) != t.pair_rank_bit) << 30
            row = (group >> 1) * t.pair_block + gp % t.pair_block
            off = 0  # assertions read each page's first slot (offset 0)
            return owner_bit | row * PAGE + off

        for block in (1, 2):
            for rank_bit in (0, 1):
                t = _make_tier(16, block, rank_bit)
                slots = torch.arange(16 * PAGE, dtype=torch.int32)
                slots[:PAGE] = t.phys_slot_base + torch.arange(PAGE)
                table = t.map_slot_table(slots)
                for lp in range(1, 16):
                    entry = int(table[lp * PAGE])
                    assert entry == spec_entry(t, lp), (block, rank_bit, lp)
                    # Bit=1 means "peer owns it" -- the negation of the
                    # rank's own ownership predicate.
                    assert (entry >> 30) == (0 if t.pair_owns_page(lp) else 1), (
                        block,
                        rank_bit,
                        lp,
                    )

    def test_both_ranks_resolve_the_same_storage_row(self):
        # Dual-pointer reads: the owner bit is rank-RELATIVE (bit=1 means
        # "peer owns it", from the reading rank's perspective), so the two
        # ranks' tables differ in the bit but must encode the SAME owner
        # row for every logical page -- each then picks local vs imported
        # peer buffers from the bit versus its own rank_bit. (The old
        # absolute-parity bit made rank1's table read the parity neighbor:
        # the TP1/TP3 corruption.)
        r0 = _make_tier(16, 1, 0)
        r1 = _make_tier(16, 1, 1)
        slots = torch.arange(16 * PAGE, dtype=torch.int32)
        t0 = r0.map_slot_table(slots)
        t1 = r1.map_slot_table(slots)
        for lp in range(1, 16):
            e0, e1 = int(t0[lp * PAGE]), int(t1[lp * PAGE])
            gp = lp % 16
            assert (e0 >> 30) == (0 if r0.pair_owns_page(gp) else 1), lp
            assert (e1 >> 30) == (0 if r1.pair_owns_page(gp) else 1), lp
            assert (e0 & ((1 << 30) - 1)) == (e1 & ((1 << 30) - 1)), (
                "owner row identical on both ranks",
                lp,
            )
            owner = r1 if (gp & 1) else r0
            assert owner.pair_owns_page(gp)
            row = (e0 & ((1 << 30) - 1)) // PAGE
            assert row == owner.pair_local_page(gp)

    def test_pair_peer_kv_returns_mask_plus_buffers(self):
        # Consumer contract: compact kernels take the 3-tuple
        # (owner_mask, peer_k, peer_v); the sm70 decode op takes only
        # (peer_k, peer_v) -- the wiring at qwen_sparse_attn_backend
        # unwraps it. A drift here shows up as an int-without-dtype
        # AttributeError at CUDA graph capture.
        t = _make_tier(16, 1, 0)
        k = torch.zeros(4, 2)
        v = torch.zeros_like(k)
        t._peer_kv_buffers = {0: (k, v)}
        out = t.pair_peer_kv(0)
        assert out is not None
        assert out[0] == 1 << 30
        assert out[1] is k and out[2] is v
        t.pair_split = False
        assert t.pair_peer_kv(0) is None

    def test_full_layout_map_slots_is_pure_modulo(self):
        t = object.__new__(KVHostTier)
        t.page_size = PAGE
        t._page_shift = PAGE.bit_length() - 1
        t.phys_pages = 4
        t.logical_tokens = 16 * PAGE
        t.phys_slot_base = 4 * PAGE + 64
        t.pair_split = False
        slots = torch.arange(16 * PAGE, dtype=torch.int64)
        slots[:PAGE] = t.phys_slot_base + torch.arange(PAGE)
        phys = t.map_slots(slots)
        off = torch.arange(PAGE)
        for lp in range(16):
            want = (lp % 4) * PAGE + off
            assert torch.equal(phys[lp * PAGE : (lp + 1) * PAGE], want)
