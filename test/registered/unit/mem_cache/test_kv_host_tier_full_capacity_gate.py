"""Unit test: FULL-layout pool capacity gate (review M2, 2026-11-10).

FULL layout maps logical page ``lp`` to device row ``lp % phys_pages``. That
mapping is only injective when ``logical_pages <= phys_pages``; with
oversubscription two live logical pages can be congruent (mod phys_pages) and
alias one device row, where neither page-in nor the write-restore kernels can
serve both:

- write side: both programs in one forward's ``_p3_write_restore_kernel``
  launch race non-atomically on tags/owner for the same row, and their
  ``set_kv_buffer`` stores clobber each other's bytes;
- read side: decode consumers (``sm70_longctx_decode``,
  ``qwen_sparse_kv_extraction_compact_triton``) trust page-in without tag
  checks, and one congruent pair in the selected set reads the other's bytes.

Production runs PAIR_PARITY, whose pair-half bijection admits logical pages
up to 2x per-rank physical with no aliasing (gated at boot by
``_init_pair_split``). FULL therefore refuses to boot oversubscribed.

Also pinned: for every accepted FULL config the slot table is injective over
real logical pages -- the property the gate exists to preserve.
"""

import pytest
import torch

from sglang.srt.mem_cache.kv_host_tier import KVHostTier

PAGE = 8


def _make_full_tier(logical_pages: int, phys_pages: int) -> KVHostTier:
    t = object.__new__(KVHostTier)
    t.page_size = PAGE
    t._page_shift = PAGE.bit_length() - 1
    t.phys_pages = phys_pages
    t.logical_pages = logical_pages
    t.logical_tokens = logical_pages * PAGE
    t.phys_slot_base = phys_pages * PAGE
    t.pair_split = False
    return t


class TestFullCapacityGate:
    def test_oversubscribed_full_boot_refuses(self):
        # The historical phase3 deployment shape (docs §10.1: logical
        # 540,672 tokens vs PHYS 327,680) is exactly the wrapped form the
        # gate must refuse.
        t = _make_full_tier(logical_pages=16, phys_pages=4)
        with pytest.raises(ValueError, match="lp % phys_pages"):
            t._require_full_pool_capacity()

    def test_capacity_fit_full_boot_passes(self):
        t = _make_full_tier(logical_pages=4, phys_pages=4)
        t._require_full_pool_capacity()  # modulo is the identity; injective

    def test_error_message_names_the_escalation_paths(self):
        t = _make_full_tier(logical_pages=5, phys_pages=4)
        with pytest.raises(ValueError) as exc:
            t._require_full_pool_capacity()
        msg = str(exc.value)
        assert "SGLANG_V100_KV_TIER_PHYS_TOKENS" in msg
        assert "--max-total-tokens" in msg
        assert "PAIR_PARITY" in msg


class TestFullMappingInjectivity:
    def test_slot_table_is_injective_over_real_pages_when_capacity_fits(self):
        # The property behind the gate: with logical_pages <= phys_pages the
        # pure-modulo mapping assigns every real logical page a distinct
        # device row, so page-in/write-restore tag bookkeeping can never
        # see two live pages claim one row.
        t = _make_full_tier(logical_pages=4, phys_pages=4)
        slots = torch.arange(4 * PAGE, dtype=torch.int64)
        slots[:PAGE] = t.phys_slot_base + torch.arange(PAGE)  # padding page
        table = t.map_slot_table(slots)
        first_slot_per_page = [int(table[lp * PAGE]) for lp in range(1, 4)]
        assert len(set(first_slot_per_page)) == 3
        assert all(s // PAGE < 4 for s in first_slot_per_page)

    def test_slot_table_aliases_pages_when_oversubscribed(self):
        # The hazard the gate refuses: lp=1 and lp=5 are congruent mod 4 and
        # resolve to the same device row -- neither page-in nor the write
        # path can serve both.
        t = _make_full_tier(logical_pages=16, phys_pages=4)
        slots = torch.arange(16 * PAGE, dtype=torch.int64)
        slots[:PAGE] = t.phys_slot_base + torch.arange(PAGE)
        table = t.map_slot_table(slots)
        assert int(table[1 * PAGE]) // PAGE == int(table[5 * PAGE]) // PAGE
