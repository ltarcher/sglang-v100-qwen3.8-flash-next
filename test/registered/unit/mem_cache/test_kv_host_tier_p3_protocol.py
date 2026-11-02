"""Protocol-level unit test for the Phase 3 save-evict-then-claim contract.

The real kernels are Triton device code; this test replays their exact
claim/save/restore/owner logic on CPU tensors and asserts the invariant that
makes the concurrent-reader fallback correct: after any sequence of claims,
a page whose tag was stolen must find its newest device bytes in the host
mirror, and a forward's own pending pages (owner == gen) must never be
overwritten by its own page-in.

This is the regression test for the x2 needle-recall corruption: the old
per-build-erased protect bitmap let a concurrent build reclaim (and
overwrite) a predecessor's pending page between its write and its mirror,
and the reader's tag-miss fallback then served a stale host copy.
"""

import torch

PAGE, ELEMS = 8, 4
PHYS, LOGICAL = 4, 16


class _Sim:
    def __init__(self):
        self.dev = torch.zeros(PHYS, PAGE, ELEMS, dtype=torch.int64)
        self.host = torch.zeros(LOGICAL, PAGE, ELEMS, dtype=torch.int64)
        self.hvalid = torch.zeros(LOGICAL, dtype=torch.bool)
        self.tags = torch.full((PHYS,), -1, dtype=torch.int64)
        self.owner = torch.zeros(PHYS, dtype=torch.int64)
        self.saves = 0
        self.restores = 0
        self.collisions = 0

    # _p3_write_restore_kernel body (save-evict-then-claim)
    def claim(self, writer, claim_pages):
        for lp in claim_pages:
            pp = lp % PHYS
            old_lp = self.tags[pp].item()
            if old_lp != lp:
                if 0 <= old_lp < LOGICAL:
                    self.host[old_lp] = self.dev[pp]
                    self.hvalid[old_lp] = True
                    self.saves += 1
                if self.hvalid[lp]:
                    self.dev[pp] = self.host[lp]
                    self.restores += 1
            self.tags[pp] = lp
            self.owner[pp] = writer

    # _p3_mirror_kernel body
    def mirror(self, mirror_pages):
        for lp in mirror_pages:
            pp = lp % PHYS
            if self.tags[pp].item() == lp:
                self.host[lp] = self.dev[pp]
                self.hvalid[lp] = True

    # _p3_page_in_kernel body
    def page_in(self, writer, pages):
        for lp in pages:
            pp = lp % PHYS
            if self.tags[pp].item() == lp:
                continue
            if self.owner[pp].item() == writer:
                self.collisions += 1
                continue
            old_lp = self.tags[pp].item()
            if 0 <= old_lp < LOGICAL:
                self.host[old_lp] = self.dev[pp]
                self.hvalid[old_lp] = True
                self.saves += 1
            if self.hvalid[lp]:
                self.dev[pp] = self.host[lp]
                self.restores += 1
            self.tags[pp] = lp
            self.owner[pp] = writer

    def write(self, writer, pages):
        for lp in pages:
            pp = lp % PHYS
            self.dev[pp, writer % PAGE] = torch.full((ELEMS,), 1000 + lp)

    def row(self, lp, slot):
        return torch.full((ELEMS,), 1000 + lp)


class TestP3SaveEvictClaim:
    def _fresh(self):
        return _Sim()

    def test_alias_after_mirror_preserves_writer_bytes(self):
        # A writes page 3, A's next build mirrors it, then B aliases slot 3:
        # B's claim must save A's newest bytes into host[3].
        s = self._fresh()
        s.claim(1, [3])
        s.write(1, [3])
        s.mirror([3])
        s.claim(2, [7])
        assert s.saves == 1
        assert torch.equal(s.host[3, 1], s.row(3, 1))
        assert s.tags[3 % PHYS].item() == 7

    def test_alias_before_mirror_pending_window(self):
        # The corruption window: B claims slot 3 BETWEEN A's write and A's
        # mirror. The unconditional save keeps A's pending bytes readable.
        s = self._fresh()
        s.claim(1, [3])
        s.write(1, [3])
        s.claim(2, [7])  # alias before any mirror
        assert s.saves == 1
        assert torch.equal(s.host[3, 1], s.row(3, 1))
        assert s.hvalid[3]

    def test_page_in_saves_evicted_writer_bytes(self):
        # A's decode page-in aliases B's pending page: B's newest bytes must
        # land in host[lp_B] before the overwrite, so B's later tag-miss
        # fallback reads its own content.
        s = self._fresh()
        s.claim(1, [3])
        s.claim(2, [7])
        s.write(2, [7])
        s.page_in(1, [3])  # aliases slot 3 (lp 7 % 4)
        # The page-in also restores lp 3's host copy through the stale-empty
        # mirror; the invariant that matters is B's pending bytes surviving.
        assert torch.equal(s.host[7, 2], s.row(7, 2))
        assert torch.equal(s.dev[3 % PHYS, 1], s.host[3, 1])

    def test_page_in_never_reverts_own_pending_write(self):
        s = self._fresh()
        s.claim(1, [3])
        s.write(1, [3])
        before = s.dev[3 % PHYS].clone()
        s.page_in(1, [7])  # same slot, own pending page
        assert s.collisions == 1
        assert torch.equal(s.dev[3 % PHYS], before)

    def test_reallocation_full_rewrite_beats_stale_copy(self):
        # A freed page's stale mirror can never be re-read: reallocation
        # rewrites every row before any read, and the new claim restores the
        # (now stale) host copy first.
        s = self._fresh()
        s.claim(1, [3])
        s.write(1, [3])
        s.claim(2, [7])
        s.write(2, [7])
        s.claim(3, [3])  # re-claim lp 3: save of lp 7 fires
        assert torch.equal(s.host[7, 2], s.row(7, 2))
        s.write(3, [3])
        assert torch.equal(s.dev[3 % PHYS, 3], s.row(3, 3))
