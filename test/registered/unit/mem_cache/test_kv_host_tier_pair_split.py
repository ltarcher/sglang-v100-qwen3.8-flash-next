"""Protocol-level unit test for the pair-split (PAIR_PARITY) layout.

Two simulated ranks replay the parity protocol on CPU tensors: each rank
stores its parity half of the shared head's pages (owner-computes;
peer-owned writes are discarded into the padding page), and every read
resolves the page from the OWNING rank's row -- the owner-bit dual-pointer
read path, where the physical table entry for a peer-owned page carries
the owner bit plus the owner's row index and the kernels select the base
pointer (local buffer or the imported pair-member buffers) from it.

The invariant under test: a read of logical page lp always returns exactly
the bytes the canonical writer computed for lp, and the row-bijection
formulas agree on both ranks (a wrong formula makes the two ranks disagree
about which row holds which page).
"""

import torch

PAGE, ELEMS = 8, 4
PAIR_PAGES = 8  # pair-total physical pages (4 local rows per rank)
BLOCK = 1
LOGICAL_PAGES = PAIR_PAGES  # v1: no oversubscription
# Owner bit folded into the physical table entry (v2 dual-pointer design).
OWNER_BIT = 1 << 30


class _Rank:
    """One rank's tier state: local rows + tags."""

    def __init__(self, rank_bit: int):
        self.rank_bit = rank_bit
        self.local_pages = PAIR_PAGES // 2
        self.dev = torch.full(
            (self.local_pages + 1, PAGE, ELEMS), -1, dtype=torch.int64
        )
        self.tags = torch.full((self.local_pages + 1,), -1, dtype=torch.int64)
        self.peer = None

    # -- mapping (identical formulas on both ranks) -----------------------

    def owns(self, gp: int) -> bool:
        return (gp // BLOCK) & 1 == self.rank_bit

    def local_row(self, gp: int) -> int:
        # The owner's row for gp: the rank that owns gp computes the same
        # formula, so both ranks agree on which row holds the page.
        return (gp // (2 * BLOCK)) * BLOCK + gp % BLOCK

    def phys_entry(self, lp: int) -> int:
        """Read-side physical table entry for logical page lp."""
        gp = lp % PAIR_PAGES
        owner_rank_bit = (gp // BLOCK) & 1
        return (owner_rank_bit << 30) | self.local_row(gp)

    # -- read path (owner-bit dual pointer) -------------------------------

    def read(self, lp: int) -> torch.Tensor:
        entry = self.phys_entry(lp)
        owner_bit = entry >> 30
        row = entry & ((1 << 30) - 1)
        buf = self if owner_bit == self.rank_bit else self.peer
        return buf.dev[row]


class _Canonical:
    """The true content each logical page must read back as."""

    def __init__(self):
        self.content = {}

    def write(self, rank: _Rank, lp: int) -> None:
        row = torch.full((PAGE, ELEMS), 1000 + lp, dtype=torch.int64)
        gp = lp % PAIR_PAGES
        self.content[lp] = row
        if rank.owns(gp):
            rank.dev[rank.local_row(gp)] = row
            rank.tags[rank.local_row(gp)] = lp
        # The non-owner's computed copy is discarded (padding page).


class TestPairSplitParity:
    def _pair(self):
        r0, r1 = _Rank(0), _Rank(1)
        r0.peer, r1.peer = r1, r0
        return _Canonical(), r0, r1

    def test_row_bijection_agrees_across_ranks(self):
        # Both ranks compute the same owner row for every pair page.
        r0, r1 = _Rank(0), _Rank(1)
        for gp in range(PAIR_PAGES):
            assert r0.local_row(gp) == r1.local_row(gp)
        # And the rows are distinct within each rank's parity set.
        for bit, r in ((0, r0), (1, r1)):
            rows = [r.local_row(gp) for gp in range(PAIR_PAGES) if r.owns(gp)]
            assert sorted(rows) == list(range(PAIR_PAGES // 2))

    def test_owner_bit_selects_the_right_buffer(self):
        canonical, r0, r1 = self._pair()
        for lp in range(8):
            writer = r0 if lp % 2 == 0 else r1
            canonical.write(writer, lp)
        for lp in range(8):
            assert torch.equal(r0.read(lp), canonical.content[lp])
            assert torch.equal(r1.read(lp), canonical.content[lp])

    def test_block_granularity_keeps_bijection(self):
        global BLOCK
        old = BLOCK
        try:
            BLOCK = 2
            r0, r1 = _Rank(0), _Rank(1)
            for gp in range(PAIR_PAGES):
                assert r0.local_row(gp) == r1.local_row(gp)
                # Owner bit matches the block parity on both ranks.
                assert r0.owns(gp) != r1.owns(gp)
            rows0 = [r0.local_row(gp) for gp in range(PAIR_PAGES) if r0.owns(gp)]
            assert len(set(rows0)) == len(rows0) == PAIR_PAGES // 2
        finally:
            BLOCK = old

    def test_phys_entry_encodes_owner(self):
        r0, _ = _Rank(0), _Rank(1)
        # gp even -> owner bit 0; gp odd -> owner bit 1 (block=1).
        assert (r0.phys_entry(0) >> 30) == 0
        assert (r0.phys_entry(1) >> 30) == 1
        assert (r0.phys_entry(1) & ((1 << 30) - 1)) == r0.local_row(1)
