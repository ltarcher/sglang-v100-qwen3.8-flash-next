"""Unit test: Phase 3 tier remap safety under free/reuse and verify-shaped tables.

CPU-level coverage for docs/v100/KV_HOST_TIER_DESIGN.md §9 items 1 and 2.

Item 1 (verify replay index semantics): the graph-replay tier tables
(``_tier3_refresh_graph_tables``) are full-width buffers refreshed
out-of-graph from req_to_token. Verify-shaped tables carry rows whose tail
columns are stale logical slots freed by earlier requests -- the PR#6
req_to_token +4-rows/padding region. The remapping must route every
non-real slot to the reserved padding window for the whole int32 domain:
no garbage value, however stale, may resolve to a real physical row. The
FULL-layout read side (``map_slot_table`` without pair-split) had no
coverage at all before this test.

Item 2 (remap vs radix/retract release consistency): freeing a logical
page is pure allocator bookkeeping -- no tier hook clears tags/hvalid, by
design. Consistency is carried entirely by the tag protocol (save-at-steal
plus "reallocation fully rewrites the page before any read"). The tests
replay the kernel claim/save/restore/mirror logic (same approach as
test_kv_host_tier_p3_protocol.py) over release-shaped sequences: radix
eviction, retract re-prefill of a pending write set, and the PAIR_PARITY
owner guard on freed peer-owned pages.

True-machine surfaces (real graph replay with MTP verify, page-in racing
a live radix eviction, 3x131k admission) are not coverable on CPU and are
tracked in §9 as the server-level regression step.
"""

import torch

from sglang.srt.mem_cache.kv_host_tier import KVHostTier

PAGE = 8


def _make_full_tier(phys_pages: int, logical_pages: int) -> KVHostTier:
    t = object.__new__(KVHostTier)
    t.page_size = PAGE
    t._page_shift = PAGE.bit_length() - 1
    t.phys_pages = phys_pages
    t.logical_tokens = logical_pages * PAGE
    t.phys_slot_base = phys_pages * PAGE + 64
    t.pair_split = False
    return t


def _make_pair_tier(pair_pages: int, block: int, rank_bit: int) -> KVHostTier:
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


class TestRemapGarbageSafety:
    """§9 #1: the remap must be memory-safe over the whole int32 domain.

    Graph-replay tables are refreshed from req_to_token whose padding rows
    (PR#6 +4 rows) and tail columns hold stale slots from freed requests.
    The consumers read only in-range positions, but a remap that resolved
    garbage to a real row would make any future bounds slip a silent OOB
    read instead of hitting the discard padding window.
    """

    def test_full_map_slot_table_domain_safety(self):
        t = _make_full_tier(phys_pages=4, logical_pages=16)
        probe = torch.tensor(
            [
                -1,
                -2**31,
                2**31 - 1,
                0,
                PAGE - 1,  # reserved padding page 0
                PAGE,  # first real token
                t.logical_tokens - 1,  # last real token
                t.logical_tokens,  # first out-of-range
                t.phys_slot_base,  # already-physical-looking value
                12345,
            ],
            dtype=torch.int32,
        )
        out = t.map_slot_table(probe)
        for i, s in enumerate(probe.tolist()):
            entry = int(out[i])
            real = PAGE <= s < t.logical_tokens
            if real:
                row = entry // PAGE
                assert 0 <= row < t.phys_pages, (s, entry)
            else:
                # Non-real slots land exactly in the reserved padding
                # window -- the store kernel's skip index and the read
                # kernels' range checks both recognize this window.
                assert entry == t.phys_slot_base + (s % PAGE), (s, entry)

    def test_full_map_slots_write_side_domain_safety(self):
        t = _make_full_tier(phys_pages=4, logical_pages=16)
        probe = torch.tensor(
            [-1, 2**31 - 1, 0, PAGE - 1, PAGE, t.logical_tokens - 1,
             t.logical_tokens, 987654],
            dtype=torch.int64,
        )
        out = t.map_slots(probe)
        for i, s in enumerate(probe.tolist()):
            phys = int(out[i])
            real = PAGE <= s < t.logical_tokens
            if real:
                assert 0 <= phys // PAGE < t.phys_pages, (s, phys)
            else:
                assert phys == t.phys_slot_base + (s % PAGE), (s, phys)

    def test_pair_map_slot_table_garbage_routes_to_padding_window(self):
        # Stale garbage in a verify table must never be routed to a real
        # pair row on either rank: the padding-window entry carries no
        # owner bit, so neither the local nor the peer buffer is dereferenced.
        for rank_bit in (0, 1):
            t = _make_pair_tier(pair_pages=16, block=1, rank_bit=rank_bit)
            garbage = torch.tensor(
                [-1, 2**31 - 1, t.logical_tokens, 777, t.phys_slot_base + 3],
                dtype=torch.int32,
            )
            out = t.map_slot_table(garbage)
            for i, s in enumerate(garbage.tolist()):
                entry = int(out[i])
                assert entry == t.phys_slot_base + (s % PAGE), (rank_bit, s, entry)
                assert entry >> 30 == 0, (rank_bit, s, entry)

    def test_pair_map_slot_table_real_rows_stay_in_range(self):
        # The complementary property: for real slots the low 30 bits always
        # name a row inside the local half (owner_row < phys_pages), so a
        # bounded consumer can trust any owner-bit-cleared entry.
        for block in (1, 2):
            for rank_bit in (0, 1):
                t = _make_pair_tier(16, block, rank_bit)
                slots = torch.arange(16 * PAGE, dtype=torch.int32)
                slots[:PAGE] = t.phys_slot_base + torch.arange(PAGE)
                out = t.map_slot_table(slots)
                for lp in range(1, 16):
                    entry = int(out[lp * PAGE])
                    row = (entry & ((1 << 30) - 1)) // PAGE
                    assert 0 <= row < t.phys_pages, (block, rank_bit, lp, entry)

    def test_verify_shaped_table_stale_rows_never_alias_real_pages(self):
        # A verify-replay graph buffer: bs valid rows plus the PR#6 padding
        # rows, whose columns hold stale slots from freed requests. Every
        # padding-row entry must hit the padding window; every valid-row
        # real slot must remap identically wherever it appears (draft rows
        # and target rows of the same request read the same logical slot).
        t = _make_full_tier(phys_pages=4, logical_pages=16)
        bs = 3
        valid = torch.arange(PAGE, 16 * PAGE, dtype=torch.int32).reshape(15, PAGE)
        table = torch.zeros((bs + 4, valid.shape[1]), dtype=torch.int32)
        table[:bs] = valid[:bs]
        # PR#6 padding rows: stale slots from freed earlier requests.
        table[bs:] = torch.randint(
            0, 2**31 - 1, (4, valid.shape[1]), dtype=torch.int32
        )
        table[bs + 1, :] = -1  # negative garbage too
        out = t.map_slot_table(table)
        assert int(out[bs:].max()) < t.phys_slot_base + PAGE
        assert bool((out[bs:] >= t.phys_slot_base).all()), "padding rows must land in the reserved window"
        # Same logical slot -> same physical entry everywhere it appears.
        flat = out[:bs].reshape(-1)
        for k in range(0, bs * PAGE, PAGE):
            assert int(flat[k]) == int(t.map_slot_table(valid[:bs].reshape(-1))[k])


class _ReleaseSim:
    """CPU replay of the Phase 3 kernels, extended with a free() action.

    Mirrors _p3_write_restore_kernel (claim), _p3_mirror_kernel (mirror),
    and _p3_page_in_kernel (page_in) exactly, including the PAIR owner
    guards. free() is the allocator/radix/retract release path: pure
    bookkeeping, no tier state is cleared -- which is precisely the
    invariant under test.
    """

    def __init__(self, phys_pages=4, logical_pages=16, pair=False,
                 pair_pages=16, pair_block=1, rank_bit=0):
        self.phys_pages = phys_pages
        self.logical_pages = logical_pages
        self.pair = pair
        self.pair_pages = pair_pages
        self.pair_block = pair_block
        self.rank_bit = rank_bit
        self.dev = torch.zeros(phys_pages, PAGE, dtype=torch.int64)
        self.host = torch.zeros(logical_pages, PAGE, dtype=torch.int64)
        self.hvalid = torch.zeros(logical_pages, dtype=torch.bool)
        self.tags = torch.full((phys_pages,), -1, dtype=torch.int64)
        self.owner = torch.zeros(phys_pages, dtype=torch.int64)
        self.gen = 0
        self.saves = 0
        self.restores = 0
        self.live = {}  # lp -> incarnation tag

    def _pp(self, lp):
        if self.pair:
            gp = lp % self.pair_pages
            assert ((gp // self.pair_block) & 1) == self.rank_bit, (
                "peer-owned page must be filtered by the caller guard"
            )
            return (gp // (2 * self.pair_block)) * self.pair_block + gp % self.pair_block
        return lp % self.phys_pages

    def _owns(self, lp):
        if not self.pair:
            return True
        gp = lp % self.pair_pages
        return ((gp // self.pair_block) & 1) == self.rank_bit

    def free(self, lps):
        # Radix eviction / retract release: allocator bookkeeping only.
        for lp in lps:
            self.live.pop(lp, None)

    def claim(self, writer, claim_pages):
        self.gen += 1
        for lp in claim_pages:
            if not self._owns(lp):
                continue  # peer-owned pages are the pair member's claim
            pp = self._pp(lp)
            old_lp = int(self.tags[pp])
            if old_lp != lp:
                if 0 <= old_lp < self.logical_pages:
                    self.host[old_lp] = self.dev[pp]
                    self.hvalid[old_lp] = True
                    self.saves += 1
                if self.hvalid[lp]:
                    self.dev[pp] = self.host[lp]
                    self.restores += 1
            self.tags[pp] = lp
            self.owner[pp] = self.gen

    def write(self, writer, lp):
        if not self._owns(lp):
            return
        self.dev[self._pp(lp), writer % PAGE] = 1000 + lp
        self.live[lp] = writer

    def mirror(self, mirror_pages):
        for lp in mirror_pages:
            if not self._owns(lp):
                continue
            pp = self._pp(lp)
            if int(self.tags[pp]) == lp:
                self.host[lp] = self.dev[pp]
                self.hvalid[lp] = True

    def page_in(self, pages):
        for lp in pages:
            if not self._owns(lp):
                continue
            pp = self._pp(lp)
            if int(self.tags[pp]) == lp:
                continue
            old_lp = int(self.tags[pp])
            if 0 <= old_lp < self.logical_pages:
                self.host[old_lp] = self.dev[pp]
                self.hvalid[old_lp] = True
                self.saves += 1
            if self.hvalid[lp]:
                self.dev[pp] = self.host[lp]
                self.restores += 1
            self.tags[pp] = lp
            self.owner[pp] = self.gen

    def read(self, lp):
        # Reader oracle: tag hit reads the device slot; a miss falls back
        # to the host mirror (hvalid must hold -- a read with no valid
        # copy anywhere is the gather_unvalid bug class).
        pp = self._pp(lp)
        if int(self.tags[pp]) == lp:
            return int(self.dev[pp, self.live[lp] % PAGE])
        assert self.hvalid[lp], (lp, "tag-miss read without a valid host copy")
        return int(self.host[lp, self.live[lp] % PAGE])


class TestReleaseReuseProtocol:
    """§9 #2: radix-evict / retract release sequences keep every read
    serving its own incarnation's bytes, with no tier hook on free()."""

    def test_radix_evicted_slot_reuse_keeps_reader_fallback_correct(self):
        # Page A is resident and mirrored, then radix-evicted; its physical
        # slot is reused by a new request's page B. A still-live warm reader
        # of another page sharing the alias window must see displaced pages'
        # newest bytes via the mirror, and B must read its own bytes.
        s = _ReleaseSim()
        s.claim(1, [3])
        s.write(1, 3)
        s.mirror([3])
        s.free([3])  # radix evict
        s.claim(2, [7])  # realloc: steals slot 3
        s.write(2, 7)
        assert s.saves >= 1
        assert int(s.host[3, 1]) == 1003, "displaced A's newest bytes must be saved"
        # B's read through its own slot, and A's stale mirror is inert.
        assert s.read(7) == 1007
        assert int(s.dev[3, 2]) == 1007

    def test_retract_of_pending_writes_then_realloc(self):
        # Request A decodes one step (its write pages are pending writes,
        # owner == gen, not yet mirrored) and is retracted before the next
        # build's mirror runs. The freed slots are immediately reclaimed by
        # request B. The next build's mirror phase must (a) skip stolen
        # slots through the tag gate and (b) never resurrect A's freed
        # pages into B's slots; B's content stays exact throughout.
        s = _ReleaseSim()
        s.claim(1, [1, 2])
        s.write(1, 1)
        s.mirror([1])
        # A's decode step: pending write, not yet mirrored.
        s.claim(2, [2])
        s.write(2, 2)
        s.free([2])  # retract releases the decode tail
        # B reclaims the freed slot and writes it before any mirror.
        s.claim(3, [2])
        s.write(3, 2)
        # The next build mirrors what it believes are the previous build's
        # pages (the freed pending set): the tag gate must skip lp 2
        # because B now owns the slot.
        s.mirror([2])
        assert s.read(2) == 1002, "B's bytes must survive the stale mirror pass"
        assert int(s.host[2, 3]) == 1002
        # A's freed page must not have leaked into the slot.
        assert int(self._slot_content_for(s, 2)) == 1002

    @staticmethod
    def _slot_content_for(s, lp):
        return s.dev[lp % s.phys_pages, int(s.tags[lp % s.phys_pages]) % PAGE] \
            if int(s.tags[lp % s.phys_pages]) == lp else -1

    def test_interleaved_release_reuse_reads_own_incarnation(self):
        # A dense interleaving of claims, writes, mirrors, page-ins and
        # frees (the radix-evict + re-prefill + decode mix): after every
        # step, every live page must read its own incarnation's bytes
        # through the protocol fallback path.
        s = _ReleaseSim()
        seq = [
            ("claim", 1, [1, 5, 9]), ("write", 1, 1), ("write", 1, 5),
            ("mirror", None, [1, 5, 9]),
            ("free", None, [5]),  # radix evict of the mirrored page
            ("claim", 2, [13]),  # reuses slot 5 % 4 == 1
            ("write", 2, 13), ("write", 1, 9),
            ("pagein", None, [1, 13]),  # decode reads displacing slots
            ("claim", 3, [9]),  # re-prefill rewrites the aliased slot
            ("write", 3, 9),
            ("mirror", None, [9, 13]),
            ("free", None, [1]),  # retract frees a pending-ish page
            ("pagein", None, [9, 13]),
            ("claim", 4, [4]),  # reuses slot 0 (was lp 1's? 1%4=1, 4%4=0)
            ("write", 4, 4),
        ]
        for op, writer, pages in seq:
            if op == "claim":
                s.claim(writer, pages)
            elif op == "write":
                s.write(writer, pages)
            elif op == "mirror":
                s.mirror(pages)
            elif op == "pagein":
                s.page_in(pages)
            elif op == "free":
                s.free(pages)
        for lp, writer in s.live.items():
            assert s.read(lp) == 1000 + lp, (lp, writer)

    def test_pair_freed_peer_owned_page_is_untouched_by_local_rank(self):
        # PAIR_PARITY: a freed peer-owned page's storage lives in the pair
        # member's buffers. The local rank's claim/mirror/page_in guards
        # must skip it entirely -- never save it (its bytes belong to the
        # peer's tag protocol) and never claim its local row (which maps a
        # different logical page under the bijection).
        s = _ReleaseSim(
            phys_pages=8, pair=True, pair_pages=16, pair_block=1, rank_bit=0
        )
        # Rank 0 owns even group pages: peer_lp=3 (gp 3, odd) is
        # peer-owned; the local row it would alias under a wrong mapping
        # is row gp//2 = 1, which belongs to lp 2 (gp 2, even).
        peer_lp = 3
        self_local = [2, 4, 6, 8, 10, 12, 14]
        s.claim(1, self_local)
        s.write(1, 2)
        s.free([peer_lp])  # radix evict of a peer-owned page
        # The local rank's build still walks mixed pages (its own table
        # slice covers the whole logical range); guards must filter.
        s.claim(2, [peer_lp, 4])
        s.write(2, peer_lp)  # owner-computes: locally a no-op
        s.mirror([peer_lp])
        s.page_in([peer_lp])
        for lp, writer in s.live.items():
            assert s.read(lp) == 1000 + lp, (lp, writer)
        # Row 1 is still owned by its own logical page, not by the freed
        # peer page.
        assert int(s.tags[1]) == 2
