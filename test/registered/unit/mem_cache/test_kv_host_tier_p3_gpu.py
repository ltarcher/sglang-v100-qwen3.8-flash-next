"""GPU unit tests: Phase 3 FULL-layout write path plus Phase 1/2 kernels.

The pair-split read path has its own file (test_kv_host_tier_pair_gpu_kernels)
and the CPU protocol sims (test_kv_host_tier_p3_protocol) replay the
claim/save/restore logic with python formulas. This file runs the REAL Triton
kernels so kernel-level drift (indexing, gates, counters) fails here:

- ``_p3_compact`` dedup (sort -> segment heads -> cumsum -> scatter)
- ``_p3_write_restore_kernel`` save-evict-then-claim
- ``_p3_mirror_kernel`` tag-gated host mirror
- ``_p3_page_in_kernel`` save-then-restore decode page-in
- ``on_build_phase3`` + ``page_in_selected_phase3`` end-to-end: mirror the
  previous build's pages, claim this build's writes, then run the
  concurrent-reader restore chain across an aliasing slot
- ``_build_closed_pages`` boundary mask/clamp/sentinel
- Phase 2 ``_page_in_kernel`` / ``_degrade_kernel`` gates (protect, mirror
  validity, resident/garbled bits, qNaN garble)
- Phase 1 ``flush`` dedup, invalid-slot clamp and truncation
"""

import pytest
import torch

from sglang.srt.mem_cache.kv_host_tier import (
    _CNT_FLUSHES,
    _CNT_MIRRORED_PAGES,
    _CNT_TRUNCATED,
    KVHostTier,
    _degrade_kernel,
    _page_in_kernel,
    _p3_mirror_kernel,
    _p3_page_in_kernel,
    _p3_write_restore_kernel,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

DEV = "cuda:0"
PAGE, HEADS, DIM = 8, 1, 4
PAGE_ELEMS = PAGE * HEADS * DIM  # 32, power of two
PHYS, LOGICAL = 4, 16
ROW_TOKENS = (PHYS + 1) * PAGE  # device buffer rows incl. the padding page


def _page_val(lp: int, kv: int) -> float:
    """Canonical content of logical page ``lp``: every token row equal."""
    return float(1000 + lp + 0.5 * kv)


def _write_page(dev: torch.Tensor, lp: int, kv: int = 0) -> None:
    """Fill device page slot ``lp % PHYS`` with lp's canonical content."""
    pp = lp % PHYS
    dev[pp * PAGE : (pp + 1) * PAGE] = _page_val(lp, kv)


def _page_view(buf: torch.Tensor, lp: int) -> torch.Tensor:
    """Flat int16 view of logical page ``lp`` (kernel PAGE_ELEMS stride)."""
    return buf[lp * PAGE_ELEMS : (lp + 1) * PAGE_ELEMS]


class _FullTier:
    """A minimal FULL-layout Phase 3 tier stub (object.__new__, no init)."""

    def __init__(self):
        t = object.__new__(KVHostTier)
        t.page_size = PAGE
        t._page_shift = PAGE.bit_length() - 1
        t._page_elems = PAGE_ELEMS
        t.kv_heads, t.head_dim = HEADS, DIM
        t.kv_dtype = torch.float16
        t.phys_pages, t.logical_pages = PHYS, LOGICAL
        t.logical_tokens = LOGICAL * PAGE
        t.phys_slot_base = PHYS * PAGE
        t.device_pages = PHYS + 1
        t.pair_split = False
        t.pair_pages = 0
        t.pair_block = 1
        t.pair_rank_bit = 0
        t.layer_num = 1
        t.device = torch.device(DEV)
        self.t = t
        # int16 storage directly: the launches .view(torch.int16) everything,
        # which is a no-op here, and integer page values compare exactly.
        self.k = torch.zeros(ROW_TOKENS, HEADS, DIM, dtype=torch.int16, device=DEV)
        self.v = torch.zeros_like(self.k)
        t._kv_buffers = [(self.k, self.v)]
        self.host = torch.zeros(
            (2, 1, LOGICAL, PAGE, HEADS, DIM), dtype=torch.int16, pin_memory=True
        )
        t._host_logical = self.host
        t._p3_tags = torch.full((1, PHYS + 1), -1, dtype=torch.int32, device=DEV)
        t._p3_owner = torch.zeros((1, PHYS + 1), dtype=torch.int32, device=DEV)
        t._p3_gen_buf = torch.zeros(1, dtype=torch.int32, device=DEV)
        t._p3_hvalid = torch.zeros((1, LOGICAL), dtype=torch.uint8, device=DEV)
        t._p3_cap = LOGICAL + 1
        t._p3_uniq = [
            torch.zeros(t._p3_cap, dtype=torch.long, device=DEV) for _ in range(2)
        ]
        t._p3_count = [torch.zeros(1, dtype=torch.int32, device=DEV) for _ in range(2)]
        t._p3_cur = 0
        t._p3_pending_bound = 0
        t._p3_pi_uniq = torch.zeros(t._p3_cap, dtype=torch.long, device=DEV)
        t._p3_pi_count = torch.zeros(1, dtype=torch.int32, device=DEV)
        t._p3_dummy = torch.zeros(1, dtype=torch.int16, device=DEV)
        t._p3_force_host = False
        t._p3_check = False
        t.counters = torch.zeros(21, dtype=torch.int64, device=DEV)
        t.step_seq = 0
        t.pagein = False
        t.layer_slot = {0: 0}


def _launch_write_restore(
    t, uniq_vals, tags=None, owner=None, hvalid=None, gen=0, host=None, dev=None
):
    tags = t._p3_tags[0] if tags is None else tags
    owner = t._p3_owner[0] if owner is None else owner
    hvalid = t._p3_hvalid[0] if hvalid is None else hvalid
    host = t._host_logical[0, 0] if host is None else host
    dev = t._kv_buffers[0][0] if dev is None else dev
    uniq = torch.tensor(uniq_vals, dtype=torch.long, device=DEV)
    count = torch.tensor([len(uniq_vals)], dtype=torch.int32, device=DEV)
    gen_buf = torch.tensor([gen], dtype=torch.int32, device=DEV)
    restore = torch.zeros(1, dtype=torch.int32, device=DEV)
    save = torch.zeros(1, dtype=torch.int32, device=DEV)
    _p3_write_restore_kernel[(len(uniq_vals),)](
        host.reshape(-1),
        dev.reshape(-1),
        uniq,
        count,
        tags,
        owner,
        gen_buf,
        hvalid,
        PHYS,
        LOGICAL,
        restore,
        save,
        PAGE_ELEMS=PAGE_ELEMS,
    )
    torch.cuda.synchronize()
    return restore, save


class TestP3Compact:
    def test_dedup_keeps_sorted_uniques_and_sentinel(self):
        t = object.__new__(KVHostTier)
        t._p3_cap = LOGICAL + 1
        uniq = torch.zeros(t._p3_cap, dtype=torch.long, device=DEV)
        count = torch.zeros(1, dtype=torch.int32, device=DEV)
        pages = torch.tensor([1, 1, 3, LOGICAL, 3, 0, 1], dtype=torch.long, device=DEV)
        t._p3_compact(pages, uniq, count)
        torch.cuda.synchronize()
        assert int(count) == 4
        assert uniq[:4].tolist() == [0, 1, 3, LOGICAL]

    def test_all_duplicates_collapse_to_one(self):
        t = object.__new__(KVHostTier)
        t._p3_cap = LOGICAL + 1
        uniq = torch.zeros(t._p3_cap, dtype=torch.long, device=DEV)
        count = torch.zeros(1, dtype=torch.int32, device=DEV)
        pages = torch.full((37,), 7, dtype=torch.long, device=DEV)
        t._p3_compact(pages, uniq, count)
        torch.cuda.synchronize()
        assert int(count) == 1
        assert int(uniq[0]) == 7


class TestP3WriteRestoreKernel:
    def test_claim_fresh_slot(self):
        ft = _FullTier()
        t = ft.t
        restore, save = _launch_write_restore(t, [7], gen=5)
        torch.cuda.synchronize()
        assert int(save) == 0 and int(restore) == 0
        assert int(t._p3_tags[0][7 % PHYS]) == 7
        assert int(t._p3_owner[0][7 % PHYS]) == 5

    def test_save_evicted_page_before_overwrite(self):
        # Slot 0 holds lp 2 (fresh bytes, never mirrored, hvalid unset). A
        # write to lp 6 (6 % 4 == 2) must save lp 2's device bytes into the
        # host mirror before the claim -- a concurrent reader that later
        # loses the tag falls back to that copy.
        ft = _FullTier()
        t = ft.t
        _write_page(ft.k, 2)
        t._p3_tags[0][2] = 2
        restore, save = _launch_write_restore(t, [6], gen=5)
        torch.cuda.synchronize()
        assert int(save) == 1
        assert int(t._p3_hvalid[0][2]) == 1
        assert torch.equal(
            _page_view(ft.host[0, 0].reshape(-1), 2).cpu(),
            _page_view(ft.k.reshape(-1), 2).cpu(),
        )
        assert int(t._p3_tags[0][2]) == 6

    def test_restore_from_host_mirror_on_reclaim(self):
        # hvalid[5] set with a valid host copy: claiming lp 5 into its slot
        # restores the host bytes to the device first (restore counter).
        ft = _FullTier()
        t = ft.t
        ft.host[0, 0].reshape(-1)[5 * PAGE_ELEMS : (5 + 1) * PAGE_ELEMS] = int(
            _page_val(5, 0)
        )
        t._p3_hvalid[0][5] = 1
        restore, save = _launch_write_restore(t, [5], gen=3)
        torch.cuda.synchronize()
        assert int(restore) == 1 and int(save) == 0
        assert torch.equal(
            _page_view(ft.k.reshape(-1), 5 % PHYS),
            torch.full(
                (PAGE_ELEMS,), int(_page_val(5, 0)), dtype=torch.int16, device=DEV
            ),
        )

    def test_sentinel_and_invalid_old_tag_skipped(self):
        ft = _FullTier()
        t = ft.t
        # Old tag -1 (fresh pool): no save may fire.
        restore, save = _launch_write_restore(t, [1, LOGICAL], gen=1)
        torch.cuda.synchronize()
        assert int(save) == 0 and int(restore) == 0
        assert int(t._p3_tags[0][1]) == 1  # real page claimed
        assert int(t._p3_tags[0][0]) == -1  # sentinel entry did nothing

    def test_pair_peer_pages_skipped(self):
        # Under PAIR the kernel must not touch peer-owned pages: their rows
        # live in the pair member's buffers, and saving/claiming here would
        # alias a different page.
        ft = _FullTier()
        t = ft.t
        uniq = torch.tensor([1, 2], dtype=torch.long, device=DEV)  # 1 odd -> peer
        count = torch.tensor([2], dtype=torch.int32, device=DEV)
        gen_buf = torch.tensor([4], dtype=torch.int32, device=DEV)
        restore = torch.zeros(1, dtype=torch.int32, device=DEV)
        save = torch.zeros(1, dtype=torch.int32, device=DEV)
        _p3_write_restore_kernel[(2,)](
            ft.host[0, 0].reshape(-1),
            ft.k.reshape(-1),
            uniq,
            count,
            t._p3_tags[0],
            t._p3_owner[0],
            gen_buf,
            t._p3_hvalid[0],
            PHYS,
            LOGICAL,
            restore,
            save,
            PAGE_ELEMS=PAGE_ELEMS,
            PAIR=1,
            pair_pages=8,
            pair_block=1,
            rank_bit=0,
        )
        torch.cuda.synchronize()
        assert int(save) == 0 and int(restore) == 0
        assert int(t._p3_tags[0][1]) == 2, "self-owned even page claimed"
        assert int(t._p3_tags[0][0]) == -1, "peer-owned odd page untouched"


class TestP3MirrorKernel:
    def _launch(self, ft, uniq_vals, pair=False):
        t = ft.t
        uniq = torch.tensor(uniq_vals, dtype=torch.long, device=DEV)
        count = torch.tensor([len(uniq_vals)], dtype=torch.int32, device=DEV)
        skip = torch.zeros(1, dtype=torch.int32, device=DEV)
        counter = torch.zeros(1, dtype=torch.int32, device=DEV)
        _p3_mirror_kernel[(len(uniq_vals),)](
            ft.k.reshape(-1),
            ft.host[0, 0].reshape(-1),
            uniq,
            count,
            t._p3_tags[0],
            t._p3_hvalid[0],
            PHYS,
            LOGICAL,
            skip,
            counter,
            PAGE_ELEMS=PAGE_ELEMS,
            PAIR=1 if pair else 0,
            pair_pages=8,
            pair_block=1,
            rank_bit=0,
        )
        torch.cuda.synchronize()
        return skip, counter

    def test_tag_match_copies_to_host(self):
        ft = _FullTier()
        t = ft.t
        _write_page(ft.k, 3)
        t._p3_tags[0][3] = 3
        skip, counter = self._launch(ft, [3])
        assert int(counter) == 1 and int(skip) == 0
        assert int(t._p3_hvalid[0][3]) == 1
        assert torch.equal(
            _page_view(ft.host[0, 0].reshape(-1), 3).cpu(),
            _page_view(ft.k.reshape(-1), 3).cpu(),
        )

    def test_tag_mismatch_skips_and_counts(self):
        # The slot was re-claimed by another logical page: its host copy
        # stays whatever it was, and the skip counter makes it visible.
        ft = _FullTier()
        t = ft.t
        _write_page(ft.k, 3)
        t._p3_tags[0][3] = 9  # slot 3 now belongs to lp 9
        skip, counter = self._launch(ft, [3])
        assert int(skip) == 1 and int(counter) == 0
        assert int(t._p3_hvalid[0][3]) == 0

    def test_pair_peer_pages_never_mirrored(self):
        ft = _FullTier()
        t = ft.t
        skip, counter = self._launch(ft, [1], pair=True)
        assert int(skip) == 0 and int(counter) == 0
        assert int(t._p3_hvalid[0][1]) == 0


class TestP3PageInKernel:
    def _launch(self, ft, uniq_vals, gen=7):
        t = ft.t
        uniq = torch.tensor(uniq_vals, dtype=torch.long, device=DEV)
        count = torch.tensor([len(uniq_vals)], dtype=torch.int32, device=DEV)
        gen_buf = torch.tensor([gen], dtype=torch.int32, device=DEV)
        restore = torch.zeros(1, dtype=torch.int32, device=DEV)
        save = torch.zeros(1, dtype=torch.int32, device=DEV)
        _p3_page_in_kernel[(len(uniq_vals),)](
            ft.host[0, 0].reshape(-1),
            ft.k.reshape(-1),
            uniq,
            count,
            t._p3_tags[0],
            t._p3_owner[0],
            gen_buf,
            t._p3_hvalid[0],
            PHYS,
            LOGICAL,
            restore,
            save,
            PAGE_ELEMS=PAGE_ELEMS,
        )
        torch.cuda.synchronize()
        return restore, save

    def test_tag_hit_is_a_noop(self):
        ft = _FullTier()
        t = ft.t
        _write_page(ft.k, 2)
        t._p3_tags[0][2] = 2
        before = ft.k.clone()
        restore, save = self._launch(ft, [2])
        assert int(restore) == 0 and int(save) == 0
        assert torch.equal(ft.k, before)

    def test_tag_miss_saves_then_restores(self):
        # Slot 0 holds lp 2 (fresh); a decode page-in selects lp 6, which
        # aliases the same slot. lp 2's newest bytes must land in host[2]
        # before the restore overwrites the slot with host[6].
        ft = _FullTier()
        t = ft.t
        _write_page(ft.k, 2)
        t._p3_tags[0][2] = 2
        ft.host[0, 0].reshape(-1)[6 * PAGE_ELEMS : (6 + 1) * PAGE_ELEMS] = int(
            _page_val(6, 0)
        )
        t._p3_hvalid[0][6] = 1
        restore, save = self._launch(ft, [6], gen=9)
        torch.cuda.synchronize()
        assert int(save) == 1 and int(restore) == 1
        assert torch.equal(
            _page_view(ft.host[0, 0].reshape(-1), 2).cpu(),
            torch.full((PAGE_ELEMS,), int(_page_val(2, 0)), dtype=torch.int16).cpu(),
        )
        assert torch.equal(
            _page_view(ft.k.reshape(-1), 6 % PHYS),
            torch.full(
                (PAGE_ELEMS,), int(_page_val(6, 0)), dtype=torch.int16, device=DEV
            ),
        )
        assert int(t._p3_tags[0][2]) == 6
        assert int(t._p3_owner[0][2]) == 9

    def test_miss_without_valid_host_copy_claims_only(self):
        ft = _FullTier()
        t = ft.t
        t._p3_tags[0][1] = 3
        restore, save = self._launch(ft, [1])
        torch.cuda.synchronize()
        # old tag 3 is valid -> its bytes are saved; lp 1 has no host copy
        # (freshly allocated) -> restore count stays 0.
        assert int(save) == 1 and int(restore) == 0
        assert int(t._p3_tags[0][1]) == 1


class TestOnBuildPhase3EndToEnd:
    def test_mirror_prev_claim_cur_then_concurrent_reader_chain(self):
        # build1 writes lp1+lp2 -> build2 (no writes) mirrors them ->
        # build3 writes lp5, which aliases lp1's slot (both % 4 == 1) ->
        # a decode page-in of lp1 must save lp5's pending bytes into the
        # host mirror and serve lp1 from the mirror, then a later page-in
        # of lp5 restores it. This is the read-side invariant the protocol
        # sim pins, replayed on the real kernels through the real hooks.
        ft = _FullTier()
        t = ft.t
        slots12 = torch.arange(PAGE, 3 * PAGE, device=DEV)  # lp1, lp2
        ret = t.on_build_phase3(slots12)
        torch.cuda.synchronize()
        assert torch.equal(ret.cpu(), t.map_slots(slots12).cpu())
        assert int(t._p3_gen_buf) == 1
        for s, lp in ((8, 1), (16, 2)):
            ft.k[s : s + PAGE] = int(_page_val(lp, 0))
            ft.v[s : s + PAGE] = int(_page_val(lp, 1))

        t.on_build_phase3(None)  # mirrors build1's pending pages
        torch.cuda.synchronize()
        assert int(t._p3_gen_buf) == 2
        for lp in (1, 2):
            assert torch.equal(
                ft.host[0, 0][lp].cpu(),
                ft.k.view(-1, PAGE, HEADS, DIM)[lp % PHYS].cpu(),
            )
            assert torch.equal(
                ft.host[1, 0][lp].cpu(),
                ft.v.view(-1, PAGE, HEADS, DIM)[lp % PHYS].cpu(),
            )
            assert int(t._p3_hvalid[0][lp]) == 1

        slots5 = torch.arange(5 * PAGE, 6 * PAGE, device=DEV)  # lp5 -> slot 1
        ret5 = t.on_build_phase3(slots5)
        torch.cuda.synchronize()
        assert torch.equal(ret5.cpu(), torch.arange(PAGE, 2 * PAGE))
        # write lp5's content (model side, after the claim)
        ft.k[ret5] = int(_page_val(5, 0))
        ft.v[ret5] = int(_page_val(5, 1))

        # decode layer selects lp1: tag miss (slot holds lp5) -> save lp5's
        # pending bytes, restore lp1 from its mirrored host copy.
        table = torch.arange(LOGICAL * PAGE, dtype=torch.int32, device=DEV)[None, :]
        t.page_in_selected_phase3(table, torch.tensor([[10]], device=DEV), 0)
        torch.cuda.synchronize()
        assert torch.equal(
            ft.host[0, 0][5].cpu(),
            torch.full(
                (PAGE, HEADS, DIM), int(_page_val(5, 0)), dtype=torch.int16
            ).cpu(),
        )
        assert torch.equal(
            ft.k.view(-1, PAGE, HEADS, DIM)[1],
            torch.full(
                (PAGE, HEADS, DIM), int(_page_val(1, 0)), dtype=torch.int16, device=DEV
            ),
        )
        # a later layer selects lp5: tag miss again -> lp1's correct bytes
        # saved back, lp5 restored from its (just-saved) host copy.
        t.page_in_selected_phase3(table, torch.tensor([[44]], device=DEV), 0)
        torch.cuda.synchronize()
        assert torch.equal(
            ft.k.view(-1, PAGE, HEADS, DIM)[1],
            torch.full(
                (PAGE, HEADS, DIM), int(_page_val(5, 0)), dtype=torch.int16, device=DEV
            ),
        )
        assert int(t._p3_tags[0][1]) == 5


class TestBuildClosedPages:
    def _tier(self):
        t = object.__new__(KVHostTier)
        t.page_size = PAGE
        t._page_shift = 3
        t.device = torch.device(DEV)
        t.device_pages = PHYS + 1
        t._pi_cap = PHYS + 2
        t._pi_uniq = torch.zeros(t._pi_cap, dtype=torch.long, device=DEV)
        t._pi_count = torch.zeros(1, dtype=torch.int32, device=DEV)
        return t

    def test_mask_clamp_and_sentinel(self):
        t = self._tier()
        # Row 0: tokens 0..31 (pages 0..3); row 1: tokens 24..55 (pages
        # 3..6, clamped onto the sentinel device_pages=5 past page 4).
        req = torch.arange(32, dtype=torch.int64, device=DEV)[None, :]
        req = torch.cat(
            [req, torch.arange(24, 56, dtype=torch.int64, device=DEV)[None, :]]
        )
        closed = torch.tensor([1, 3], dtype=torch.int64, device=DEV)
        t._build_closed_pages(req, torch.tensor([0, 1], device=DEV), closed)
        torch.cuda.synchronize()
        # Row 0 closed page 0; row 1 closed pages 3,4 and the clamped
        # sentinel (page 5 == device_pages, skipped by kernels).
        uniq = sorted(set(t._pi_uniq[: int(t._pi_count)].tolist()))
        assert uniq == [0, 3, 4, 5]

    def test_out_of_range_row_and_bound_clamped(self):
        t = self._tier()
        # 40 slots per row: position 32 is legal, so closed_last clamped to
        # dp touches pages 0..4; row index 7 is out of range -> clamps to 0.
        req = torch.arange(40, dtype=torch.int64, device=DEV)[None, :]
        closed = torch.tensor([99], dtype=torch.int64, device=DEV)  # > dp
        t._build_closed_pages(req, torch.tensor([7], device=DEV), closed)
        torch.cuda.synchronize()
        uniq = sorted(set(t._pi_uniq[: int(t._pi_count)].tolist()))
        assert uniq == [0, 1, 2, 3, 4]


class TestPhase2PageInKernel:
    def _run(self, valid, protect):
        num_pages = 5
        cap = num_pages + 1
        hk = torch.zeros(num_pages * PAGE_ELEMS, dtype=torch.int16, pin_memory=True)
        # zeros_like drops pin_memory, and unpinned host pointers cannot be
        # dereferenced from Triton (UVA) -- the prod path uses pinned slices.
        hv = torch.zeros(num_pages * PAGE_ELEMS, dtype=torch.int16, pin_memory=True)
        for p in range(num_pages):
            hk[p * PAGE_ELEMS : (p + 1) * PAGE_ELEMS] = 300 + p
            hv[p * PAGE_ELEMS : (p + 1) * PAGE_ELEMS] = 400 + p
        k = torch.zeros(num_pages * PAGE_ELEMS, dtype=torch.int16, device=DEV)
        v = torch.zeros_like(k)
        uniq = torch.tensor([0, 2, 3, num_pages], dtype=torch.long, device=DEV)
        count = torch.tensor([4], dtype=torch.int32, device=DEV)
        resident = torch.zeros(num_pages, dtype=torch.uint8, device=DEV)
        garbled = torch.zeros(num_pages, dtype=torch.uint8, device=DEV)
        garbled[2] = 1
        garbled[3] = 1
        valid_buf = torch.tensor([valid], dtype=torch.int32, device=DEV)
        protect_buf = torch.tensor(protect, dtype=torch.uint8, device=DEV)
        _page_in_kernel[(cap,)](
            hk,
            hv,
            k,
            v,
            uniq,
            count,
            resident,
            garbled,
            valid_buf,
            protect_buf,
            num_pages,
            PAGE_ELEMS=PAGE_ELEMS,
            USE_PROTECT=True,
        )
        torch.cuda.synchronize()
        return k, v, resident, garbled

    def test_restore_and_gates(self):
        k, v, resident, garbled = self._run(valid=1, protect=[0, 0, 0, 1, 0, 0])
        # page 0: clean but non-resident -> restored.
        assert torch.all(k[:PAGE_ELEMS] == 300)
        assert resident[0] == 1 and garbled[0] == 0
        # page 2: garbled -> restored, garbled cleared.
        assert torch.all(k[2 * PAGE_ELEMS : 3 * PAGE_ELEMS] == 302)
        assert resident[2] == 1 and garbled[2] == 0
        # page 3: protected -> untouched (a live write must never be
        # reverted by its stale host copy).
        assert torch.all(k[3 * PAGE_ELEMS : 4 * PAGE_ELEMS] == 0)
        assert resident[3] == 0 and garbled[3] == 1
        # sentinel page 5: skipped.
        assert torch.all(v[:PAGE_ELEMS] == 400)

    def test_mirror_valid_gate(self):
        # Before the first degrade the mirror holds no snapshot: a forced
        # capture/replay launch must never copy uninitialized bytes.
        k, _, resident, garbled = self._run(valid=0, protect=[0] * 6)
        assert torch.all(k == 0)
        assert torch.all(resident == 0)


class TestPhase2DegradeKernel:
    def test_garble_sets_nan_and_bits(self):
        num_pages = 4
        cap = num_pages + 1
        k = torch.ones(num_pages * PAGE_ELEMS, dtype=torch.float16, device=DEV)
        v = torch.ones_like(k)
        uniq = torch.tensor([1, num_pages], dtype=torch.long, device=DEV)
        count = torch.tensor([2], dtype=torch.int32, device=DEV)
        resident = torch.ones(num_pages, dtype=torch.uint8, device=DEV)
        garbled = torch.zeros(num_pages, dtype=torch.uint8, device=DEV)
        _degrade_kernel[(cap, 2)](
            k.view(torch.int16),
            v.view(torch.int16),
            uniq,
            count,
            resident,
            garbled,
            num_pages,
            PAGE_ELEMS=PAGE_ELEMS,
        )
        torch.cuda.synchronize()
        kf = k.view(-1, PAGE_ELEMS)
        assert torch.isnan(kf[1].view(torch.float16)).all(), "closed page is qNaN"
        assert not torch.isnan(kf[0].view(torch.float16)).any(), "open page untouched"
        assert torch.isnan(v.view(-1, PAGE_ELEMS)[1].view(torch.float16)).all(), (
            "V half garbled too (single-program-per-page would miss this)"
        )
        assert resident[1] == 0 and garbled[1] == 1
        assert resident[0] == 1 and garbled[0] == 0


class TestPhase1Flush:
    def _tier(self):
        t = object.__new__(KVHostTier)
        t.phase3 = False
        t.pagein = False
        t.page_size = PAGE
        t.kv_heads, t.head_dim, t.kv_dtype = HEADS, DIM, torch.float16
        t.device = torch.device(DEV)
        t.device_pages = 4
        t.layer_num = 1
        k = (
            torch.arange(4, dtype=torch.float16, device=DEV)
            .repeat_interleave(PAGE)[:, None, None]
            .expand(4 * PAGE, HEADS, DIM)
            .contiguous()
        )
        t._kv_buffers = [(k, k + 1)]
        t.host_pages = 4
        t._cap = 4
        t._host = torch.zeros(
            (2, 1, 4, PAGE, HEADS, DIM), dtype=torch.float16, pin_memory=True
        )
        t._stage = torch.zeros(
            (2, 1, 4, PAGE, HEADS, DIM), dtype=torch.float16, device=DEV
        )
        t._uniq = torch.zeros(4, dtype=torch.long, device=DEV)
        t.resident = torch.zeros((1, 4), dtype=torch.uint8, device=DEV)
        t.clock_ref = torch.zeros(4, dtype=torch.uint8, device=DEV)
        t.counters = torch.zeros(21, dtype=torch.int64, device=DEV)
        t.rep_k = None
        t._copy_stream = torch.cuda.Stream(device=DEV)
        t._flush_ev = torch.cuda.Event()
        t._pending = []
        t.step_seq = 0
        t._flush_count = 0
        return t, k

    def test_dedup_invalid_clamp_and_residency(self):
        t, k = self._tier()
        # 8 tokens: pages 0,0,1,1,0(dup), invalid -5 (clamps to page 0),
        # token 40 (clamps to the last page 3), page 3 again -> uniq [0,1,3].
        tokens = torch.tensor([0, 1, 2, 3, 8, 9, -5, 40], dtype=torch.long, device=DEV)
        t.flush(tokens)
        torch.cuda.synchronize()
        assert t._uniq[:3].tolist() == [0, 1, 3]
        dev = k.view(4, PAGE, HEADS, DIM)
        host = t._host[0, 0]  # host slot i mirrors page _uniq[i]
        for i in range(4):
            assert torch.equal(
                host[i].view(PAGE, HEADS, DIM), dev[int(t._uniq[i])].cpu()
            )
        # Residency marks every uniq slot: real pages 0,1,3 plus the stale
        # slot 3 (page 0, already resident) -- harmless by design.
        assert t.resident[0].tolist() == [1, 1, 0, 1]
        assert int(t.counters[_CNT_FLUSHES]) == 1
        assert int(t.counters[_CNT_MIRRORED_PAGES]) == 4, "full-capacity D2H"
        assert int(t.counters[_CNT_TRUNCATED]) == 0

    def test_truncation_keeps_tail(self):
        t, k = self._tier()
        # 40 tokens clamp into pages 0..3; 40 > cap*page_size=32 -> keep the
        # LAST 32 tokens (pages 1,2,3,3...) and count one truncation.
        tokens = torch.arange(40, dtype=torch.long, device=DEV)
        t.flush(tokens)
        torch.cuda.synchronize()
        assert int(t.counters[_CNT_TRUNCATED]) == 1
        assert t._uniq[:3].tolist() == [1, 2, 3]
        dev = k.view(4, PAGE, HEADS, DIM)
        host = t._host[0, 0]
        for i in range(4):
            assert torch.equal(
                host[i].view(PAGE, HEADS, DIM), dev[int(t._uniq[i])].cpu()
            )
