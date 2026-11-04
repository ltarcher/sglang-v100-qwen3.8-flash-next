"""GPU unit tests: pair-split owner-bit dual-pointer read path (3 kernels).

Each kernel that consumes the physical slot table gets a PAIR test (owner
bit selects between the local pool and the imported peer buffers) and a
FULL regression (modulo mapping, behavior unchanged without peers):

- ``_p3_gather_kernel`` (tier page-in gather)
- ``qwen_sparse_kv_extraction_compact_triton`` (FA2/trtllm fallback extract)
- ``sm70_cuda_qsa_decode`` (JIT decode attention, numeric vs torch ref)

The tier stubs are built with ``object.__new__`` and the same mapping
attributes the boot path sets; ``map_slot_table`` produces the real
physical table (owner bit 30 folded into peer-owned entries).
"""

import ctypes

import pytest
import torch

from sglang.srt.layers.attention.qsa.sparse_attn import (
    qwen_sparse_kv_extraction_compact_triton,
)
from sglang.srt.layers.attention.tilelang_fa_v100._decode_cuda import (
    sm70_cuda_qsa_decode,
)
from sglang.srt.mem_cache.kv_host_tier import (
    _PAIR_OWNER_MASK,
    KVHostTier,
    _p3_gather_kernel,
)
from sglang.srt.mem_cache.pair_split_ipc import ImportedDeviceBuffer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

DEV = "cuda:0"
PAGE = 8
PAIR_PAGES, LOCAL_PAGES = 16, 8
LOGICAL_TOKENS = PAIR_PAGES * PAGE
PAD_BASE = LOCAL_PAGES * PAGE + 64
OWNER_MASK = (1 << 30) - 1

# Production shape (acceptance boot): page 64, PHYS=663040 -> 10360 pair
# pages, 5180 local rows per rank. The boundary values below are the ones
# the acceptance corruption probes exercise (needle page ~195 = peer-owned
# odd half for even ranks) plus the edges of the owner-row bijection.
PROD_PAGE = 64
PROD_PAIR_PAGES = 10360
PROD_LOCAL_PAGES = 5180


def _prod_tier(pair_block: int = 1, rank_bit: int = 0) -> KVHostTier:
    t = object.__new__(KVHostTier)
    t.page_size = PROD_PAGE
    t._page_shift = PROD_PAGE.bit_length() - 1
    t.pair_pages = PROD_PAIR_PAGES
    t.phys_pages = PROD_LOCAL_PAGES
    t.pair_block = pair_block
    t.pair_rank_bit = rank_bit
    t.logical_tokens = PROD_PAIR_PAGES * PROD_PAGE
    t.phys_slot_base = PROD_LOCAL_PAGES * PROD_PAGE + 64
    t.pair_split = True
    return t


def _pair_tier(rank_bit: int = 0) -> KVHostTier:
    t = object.__new__(KVHostTier)
    t.page_size = PAGE
    t._page_shift = PAGE.bit_length() - 1
    t.pair_pages = PAIR_PAGES
    t.phys_pages = LOCAL_PAGES
    t.pair_block = 1
    t.pair_rank_bit = rank_bit
    t.logical_tokens = LOGICAL_TOKENS
    t.phys_slot_base = PAD_BASE
    t.pair_split = True
    return t


def _pair_phys_table(rank_bit: int = 0) -> torch.Tensor:
    logical = torch.arange(LOGICAL_TOKENS, dtype=torch.int64)
    logical[:PAGE] = PAD_BASE + torch.arange(PAGE)
    return _pair_tier(rank_bit).map_slot_table(logical.to(torch.int32)).to(DEV)


class TestP3GatherPair:
    def test_pair_owner_bit_reads_local_or_peer(self):
        # Local pool: even gp rows hold 1000+gp; peer pool: odd gp 2000+gp.
        dev_buf = torch.zeros(
            LOCAL_PAGES * PAGE + PAGE, 4, dtype=torch.float32, device=DEV
        )
        peer_buf = torch.zeros_like(dev_buf)
        for gp in range(PAIR_PAGES):
            row = gp // 2
            if gp % 2 == 0:
                dev_buf[row * PAGE : (row + 1) * PAGE] = 1000.0 + gp
            else:
                peer_buf[row * PAGE : (row + 1) * PAGE] = 2000.0 + gp
        host_buf = torch.full(
            (LOGICAL_TOKENS + PAGE, 4), 9999.0, dtype=torch.float32, device=DEV
        )
        phys = _pair_phys_table()

        # One token per pair page; lp 0 is the reserved padding page.
        row_tokens = torch.arange(PAIR_PAGES, device=DEV) * PAGE + 3
        slots_rows = phys[row_tokens]
        out = torch.empty(len(slots_rows), 4, dtype=torch.float32, device=DEV)
        gen = torch.zeros(1, dtype=torch.int32, device=DEV)
        counters = [torch.zeros(1, dtype=torch.int32, device=DEV) for _ in range(4)]
        tags = torch.full((LOCAL_PAGES + 1,), -1, dtype=torch.int32, device=DEV)
        hvalid = torch.zeros(LOGICAL_TOKENS // PAGE + 1, dtype=torch.int32, device=DEV)
        owner = torch.full((LOCAL_PAGES + 1,), -1, dtype=torch.int32, device=DEV)

        _p3_gather_kernel[(1,)](
            dev_buf.reshape(-1),
            host_buf.reshape(-1),
            tags,
            hvalid,
            owner,
            slots_rows,
            out,
            len(slots_rows),
            LOCAL_PAGES,
            LOGICAL_TOKENS,
            gen,
            counters[0],
            counters[1],
            counters[2],
            counters[3],
            FORCE_HOST=0,
            CHECK=0,
            LOGICAL_SHIFT=3,
            PAGE_SIZE=PAGE,
            ROW_ELEMS=4,
            PAGE_ELEMS=PAGE * 4,
            BLOCK_ROWS=16,
            PAIR=1,
            peer_ptr=peer_buf.reshape(-1),
        )
        torch.cuda.synchronize()
        assert int(counters[0]) == 0, "no row may fall back to the host mirror"
        for i, gp in enumerate(range(PAIR_PAGES)):
            want = 0.0 if gp == 0 else (1000.0 + gp if gp % 2 == 0 else 2000.0 + gp)
            assert torch.all(out[i] == want), (i, gp, out[i].tolist())

    def test_full_regression_tag_gated_reads(self):
        full_phys = 4
        t2 = object.__new__(KVHostTier)
        t2.page_size = PAGE
        t2._page_shift = 3
        t2.phys_pages = full_phys
        t2.logical_tokens = LOGICAL_TOKENS
        t2.phys_slot_base = full_phys * PAGE + 64
        t2.pair_split = False
        full_dev = torch.zeros(
            full_phys * PAGE + PAGE, 4, dtype=torch.float32, device=DEV
        )
        for p in range(full_phys):
            full_dev[p * PAGE : (p + 1) * PAGE] = 500.0 + p
        full_slots = torch.arange(LOGICAL_TOKENS, dtype=torch.int32)
        full_slots[:PAGE] = t2.phys_slot_base + torch.arange(PAGE, dtype=torch.int32)
        rows = torch.tensor([0] + [gp * PAGE + 5 for gp in range(1, 16)])
        slots = full_slots[rows].to(DEV)
        tags = torch.full((full_phys,), -1, dtype=torch.int32, device=DEV)
        for p in range(full_phys):
            tags[p] = p
        hvalid = torch.zeros(LOGICAL_TOKENS // PAGE + 1, dtype=torch.int32, device=DEV)
        owner = torch.full((full_phys,), -1, dtype=torch.int32, device=DEV)
        out = torch.empty(len(rows), 4, dtype=torch.float32, device=DEV)
        gen = torch.zeros(1, dtype=torch.int32, device=DEV)
        counters = [torch.zeros(1, dtype=torch.int32, device=DEV) for _ in range(4)]

        _p3_gather_kernel[(1,)](
            full_dev.reshape(-1),
            torch.full(
                (LOGICAL_TOKENS + PAGE, 4), 9999.0, dtype=torch.float32, device=DEV
            ).reshape(-1),
            tags,
            hvalid,
            owner,
            slots,
            out,
            len(rows),
            full_phys,
            LOGICAL_TOKENS,
            gen,
            counters[0],
            counters[1],
            counters[2],
            counters[3],
            FORCE_HOST=0,
            CHECK=0,
            LOGICAL_SHIFT=3,
            PAGE_SIZE=PAGE,
            ROW_ELEMS=4,
            PAGE_ELEMS=PAGE * 4,
            BLOCK_ROWS=16,
            PAIR=0,
            peer_ptr=full_dev.reshape(-1),
        )
        torch.cuda.synchronize()
        for i, r in enumerate(rows.tolist()):
            lp = r // PAGE
            # Padding entry and second-pass rows miss the tag -> host mirror.
            want = 9999.0 if (r == 0 or lp >= full_phys) else 500.0 + (lp % full_phys)
            assert torch.all(out[i] == want), (i, r, out[i].tolist(), want)
        assert int(counters[0]) == 1 + 12


class TestCompactKvPair:
    HEADS, DIM, BATCH, TOPK = 2, 16, 3, 6

    def _run(self, pair: bool, rank_bit: int = 0):
        heads, dim, batch, topk = self.HEADS, self.DIM, self.BATCH, self.TOPK
        if pair:
            t = _pair_tier(rank_bit)
            dev_k = torch.zeros(
                (LOCAL_PAGES + 1) * PAGE, heads, dim, dtype=torch.float16, device=DEV
            )
            dev_v = torch.zeros_like(dev_k)
            peer_k = torch.zeros_like(dev_k)
            peer_v = torch.zeros_like(dev_k)
            for gp in range(1, PAIR_PAGES):
                row = gp // 2
                val = 100.0 + gp if gp % 2 == 0 else 200.0 + gp
                local = gp % 2 == rank_bit
                (dev_k if local else peer_k)[row * PAGE : (row + 1) * PAGE] = val
                (dev_v if local else peer_v)[row * PAGE : (row + 1) * PAGE] = val + 0.5
            phys = _pair_phys_table(rank_bit)
            extra = (_PAIR_OWNER_MASK, peer_k, peer_v)
        else:
            t = object.__new__(KVHostTier)
            t.page_size = PAGE
            t._page_shift = 3
            t.phys_pages = 4
            t.logical_tokens = LOGICAL_TOKENS
            t.phys_slot_base = 4 * PAGE + 64
            t.pair_split = False
            dev_k = torch.zeros(5 * PAGE, heads, dim, dtype=torch.float16, device=DEV)
            peer_k = peer_v = None
            for p in range(4):
                dev_k[p * PAGE : (p + 1) * PAGE] = 50.0 + p
            full_slots = torch.arange(LOGICAL_TOKENS, dtype=torch.int32)
            full_slots[:PAGE] = t.phys_slot_base + torch.arange(PAGE, dtype=torch.int32)
            phys = t.map_slot_table(full_slots).to(DEV)
            extra = None
            dev_v = dev_k + 0.5
        table = phys[None, :].repeat(batch, 1).contiguous()
        req_indices = torch.arange(batch, dtype=torch.int32, device=DEV)
        indices = (
            (torch.arange(1, topk + 1, dtype=torch.int64, device=DEV) * PAGE)[None, :]
            .repeat(batch, 1)
            .contiguous()
        )
        seq_lens = torch.full(
            (batch,), (topk + 1) * PAGE, dtype=torch.int32, device=DEV
        )
        total = batch * topk
        out_k = torch.empty(total, heads, dim, dtype=torch.float16, device=DEV)
        out_v = torch.empty_like(out_k)
        cu = torch.arange(0, total + 1, topk, dtype=torch.int32, device=DEV)
        qwen_sparse_kv_extraction_compact_triton(
            dev_k,
            dev_v,
            table,
            req_indices,
            indices,
            seq_lens,
            cu,
            out_k,
            out_v,
            batch,
            topk,
            pair_peer_kv=extra,
        )
        torch.cuda.synchronize()
        return out_k, out_v

    def test_pair_owner_bit_extraction(self):
        out_k, out_v = self._run(pair=True)
        for b in range(self.BATCH):
            for j in range(self.TOPK):
                lp = 1 + j
                want = (100.0 + lp) if lp % 2 == 0 else (200.0 + lp)
                assert out_k[b * self.TOPK + j, 0, 0].item() == want, (b, j)
                assert out_v[b * self.TOPK + j, 1, self.DIM - 1].item() == want + 0.5

    def test_pair_owner_bit_extraction_rank1(self):
        # Odd-rank (TP1/TP3) perspective: the table's owner bit is relative
        # to the reading rank, so the extraction must still land on the same
        # storage rows (polarity regression, gather-side twin).
        out_k, out_v = self._run(pair=True, rank_bit=1)
        for b in range(self.BATCH):
            for j in range(self.TOPK):
                lp = 1 + j
                want = (100.0 + lp) if lp % 2 == 0 else (200.0 + lp)
                assert out_k[b * self.TOPK + j, 0, 0].item() == want, (b, j)
                assert out_v[b * self.TOPK + j, 1, self.DIM - 1].item() == want + 0.5

    def test_full_regression(self):
        out_k, out_v = self._run(pair=False)
        for b in range(self.BATCH):
            for j in range(self.TOPK):
                lp = 1 + j
                want = 50.0 + (lp % 4)
                assert out_k[b * self.TOPK + j, 0, 0].item() == want, (b, j)
                assert out_v[b * self.TOPK + j, 0, 0].item() == want + 0.5


class TestSm70DecodePair:
    HEADS, DIM = 6, 256

    def test_pair_numeric_vs_torch_reference(self):
        torch.manual_seed(7)
        dev_k = torch.zeros(
            (LOCAL_PAGES + 1) * PAGE, 1, self.DIM, dtype=torch.float16, device=DEV
        )
        dev_v = torch.zeros_like(dev_k)
        peer_k = torch.zeros_like(dev_k)
        peer_v = torch.zeros_like(dev_k)
        for gp in range(1, PAIR_PAGES):
            row = gp // 2
            gen = torch.Generator(device="cpu").manual_seed(100 + gp)
            kt = torch.randn(PAGE, 1, self.DIM, generator=gen).half()
            vt = torch.randn(PAGE, 1, self.DIM, generator=gen).half()
            kdst, vdst = (dev_k, dev_v) if gp % 2 == 0 else (peer_k, peer_v)
            kdst[row * PAGE : (row + 1) * PAGE] = kt.to(DEV)
            vdst[row * PAGE : (row + 1) * PAGE] = vt.to(DEV)
        phys = _pair_phys_table()
        table = phys[None, :]
        positions = [10, 20, 27, 40, 55, 66, 71, 90, 103, 120]
        batch = 1
        indices = torch.tensor([positions], dtype=torch.int32, device=DEV)
        seq_lens = torch.tensor([LOGICAL_TOKENS], dtype=torch.int32, device=DEV)
        req_indices = torch.zeros(batch, dtype=torch.int32, device=DEV)
        q = torch.randn(batch, self.HEADS, self.DIM, dtype=torch.float16, device=DEV)
        scale = 0.125

        out = sm70_cuda_qsa_decode(
            q,
            dev_k,
            dev_v,
            table,
            req_indices,
            indices,
            seq_lens,
            scale,
            # Real boot passes the IPC-imported pointer wrappers, not
            # tensors -- _qsa_cache_view must pass them through.
            peer_kv=(
                ImportedDeviceBuffer(
                    ctypes.c_void_p(peer_k.data_ptr()), peer_k.numel(), peer_k.dtype
                ),
                ImportedDeviceBuffer(
                    ctypes.c_void_p(peer_v.data_ptr()), peer_v.numel(), peer_v.dtype
                ),
            ),
        ).reshape(batch, self.HEADS, self.DIM)

        rows = []
        for p in positions:
            entry = int(phys[p])
            src_k, src_v = (peer_k, peer_v) if (entry >> 30) else (dev_k, dev_v)
            rows.append((src_k[entry & OWNER_MASK], src_v[entry & OWNER_MASK]))
        kk = torch.stack([r[0].reshape(-1) for r in rows])
        vv = torch.stack([r[1].reshape(-1) for r in rows])
        qf = q[0].float()
        probs = torch.softmax(qf @ kk.float().T * scale, dim=-1)
        ref = (probs @ vv.float()).half()
        diff = (out[0].float() - ref.float()).abs().max().item()
        assert diff < 0.05, diff

    def test_full_regression_no_peer_buffers(self):
        torch.manual_seed(11)
        t2 = object.__new__(KVHostTier)
        t2.page_size = PAGE
        t2._page_shift = 3
        t2.phys_pages = 4
        t2.logical_tokens = LOGICAL_TOKENS
        t2.phys_slot_base = 4 * PAGE + 64
        t2.pair_split = False
        full_k = torch.zeros(5 * PAGE, 1, self.DIM, dtype=torch.float16, device=DEV)
        full_v = torch.zeros_like(full_k)
        for p in range(4):
            gen = torch.Generator(device="cpu").manual_seed(200 + p)
            full_k[p * PAGE : (p + 1) * PAGE] = (
                torch.randn(PAGE, 1, self.DIM, generator=gen).half().to(DEV)
            )
            full_v[p * PAGE : (p + 1) * PAGE] = (
                torch.randn(PAGE, 1, self.DIM, generator=gen).half().to(DEV)
            )
        full_table = t2.map_slot_table(
            torch.arange(LOGICAL_TOKENS, dtype=torch.int32)
        ).to(DEV)[None, :]
        batch = 1
        pos = [10, 15, 25, 30]
        idx = torch.tensor([pos], dtype=torch.int32, device=DEV)
        seq_lens = torch.tensor([LOGICAL_TOKENS], dtype=torch.int32, device=DEV)
        req_indices = torch.zeros(batch, dtype=torch.int32, device=DEV)
        q = torch.randn(batch, self.HEADS, self.DIM, dtype=torch.float16, device=DEV)

        out = sm70_cuda_qsa_decode(
            q, full_k, full_v, full_table, req_indices, idx, seq_lens, 0.125
        ).reshape(batch, self.HEADS, self.DIM)
        rows = [full_table[0, p].item() for p in pos]
        kk = torch.stack([full_k[r].reshape(-1) for r in rows])
        vv = torch.stack([full_v[r].reshape(-1) for r in rows])
        probs = torch.softmax(q[0].float() @ kk.float().T * 0.125, dim=-1)
        ref = (probs @ vv.float()).half()
        diff = (out[0].float() - ref.float()).abs().max().item()
        assert diff < 0.05, diff


class TestPairMappingProductionBoundaries:
    """Owner routing at production shape (page 64, 10360 pair pages).

    The acceptance corruption (needle at page ~195) suspects the owner-bit
    routing table at page/parity boundaries; these checks pin the mapping
    itself, independently of any kernel. Expected values are derived from
    the docs/v100/KV_HOST_TIER_DESIGN.md §6.1 bijection by hand:
    owner_bit = (gp // block) & 1, owner_row = (gp // (2*block)) * block
    + gp % block, entry = owner_bit << 30 | owner_row * 64 + off.
    """

    def _entry(self, rank_bit, token, pair_block=1):
        t = _prod_tier(pair_block=pair_block, rank_bit=rank_bit)
        table = torch.tensor([token], dtype=torch.int32)
        return int(t.map_slot_table(table)[0])

    def test_owner_routing_parity_pages(self):
        BIT = 1 << 30
        # Even ranks own even groups: page 195 (odd group, the probe page)
        # belongs to the peer, owner row 97; gp 1 -> peer row 0.
        assert self._entry(0, 195 * 64) == BIT + 97 * 64
        assert self._entry(0, 195 * 64 + 63) == BIT + 97 * 64 + 63
        assert self._entry(0, 194 * 64 + 31) == 97 * 64 + 31
        assert self._entry(0, 1 * 64 + 5) == BIT + 5
        # Odd ranks mirror it (token < 64 is the reserved page: padding
        # branch, parity-independent).
        assert self._entry(1, 195 * 64 + 31) == 97 * 64 + 31
        assert self._entry(1, 194 * 64) == BIT + 97 * 64
        assert self._entry(1, 1 * 64 + 5) == 5
        assert self._entry(0, 5) == _prod_tier().phys_slot_base + 5

    def test_owner_routing_local_pool_edges(self):
        BIT = 1 << 30
        # gp 5179 (odd, peer for even ranks) -> owner row 2589;
        # gp 5180 (even) -> row 2590: the bijection crosses the local pool
        # midpoint without shifting.
        assert self._entry(0, 5179 * 64) == BIT + 2589 * 64
        assert self._entry(0, 5180 * 64 + 63) == 2590 * 64 + 63
        # Last pair page gp 10359 (odd): owned by odd ranks (bit 0, row
        # 5179 -> slot 331519, exactly one below the kernel validity bound
        # phys_pages * PAGE_SIZE = 331520); peer-owned for even ranks.
        assert self._entry(1, 10359 * 64 + 63) == 5179 * 64 + 63
        assert self._entry(0, 10359 * 64 + 63) == BIT + 5179 * 64 + 63
        assert self._entry(1, 10359 * 64 + 63) & ((1 << 30) - 1) == 331519

    def test_owner_routing_block_boundary(self):
        BIT = 1 << 30
        # block=256: gp 255 ends even group 0 (local rows 0..255), gp 256
        # starts odd group 1 (peer rows 0..255) -- the group boundary must
        # switch parity without moving the row base.
        assert self._entry(0, 255 * 64, 256) == 255 * 64
        assert self._entry(0, 256 * 64 + 63, 256) == BIT + 0 * 64 + 63
        assert self._entry(0, 511 * 64, 256) == BIT + 255 * 64
        assert self._entry(0, 512 * 64 + 1, 256) == 256 * 64 + 1

    def test_write_side_padding_routing(self):
        # map_slots (write side): peer-owned pages must land on the padding
        # row (phys_pages) so the store kernel's skip index discards them;
        # non-real slots keep the reserved padding base.
        t = _prod_tier()
        slots = torch.tensor(
            [0 * 64 + 3, 195 * 64 + 7, 5, 5180 * 64 + 63], dtype=torch.int32
        )
        out = t.map_slots(slots).tolist()
        assert out[0] == t.phys_slot_base + 3, "page-0 tokens are non-real: padding"
        assert out[1] == 5180 * 64 + 7, "peer page 195 -> padding row"
        assert out[2] == t.phys_slot_base + 5, "reserved slot stays padding"
        assert out[3] == 2590 * 64 + 63, "even gp 5180 -> local row 2590"


class TestPairGatherPageBoundary:
    """Gather kernel (PAIR=1) across page boundaries at production page
    size 64, including the probe page 195 and the last-page validity edge.
    Scaled to 1024 pair pages (512 local rows) to keep buffers small.
    """

    GPAGES, LPAGES = 1024, 512

    def _run(self, tokens, rank_bit: int = 0):
        gpages, lpages = self.GPAGES, self.LPAGES
        t = object.__new__(KVHostTier)
        t.page_size = 64
        t._page_shift = 6
        t.pair_pages = gpages
        t.phys_pages = lpages
        t.pair_block = 1
        t.pair_rank_bit = rank_bit
        t.logical_tokens = gpages * 64
        t.phys_slot_base = lpages * 64 + 64
        t.pair_split = True

        dev_buf = torch.zeros(lpages * 64 + 64, 4, dtype=torch.float32, device=DEV)
        peer_buf = torch.zeros_like(dev_buf)
        for gp in range(1, gpages):
            row = gp // 2
            val = 1000.0 + gp if gp % 2 == 0 else 2000.0 + gp
            (dev_buf if gp % 2 == rank_bit else peer_buf)[row * 64 : (row + 1) * 64] = (
                val
            )
        phys = t.map_slot_table(
            torch.arange(gpages * 64, dtype=torch.int64).to(torch.int32)
        ).to(DEV)
        slots_rows = phys[torch.tensor(tokens, dtype=torch.int64, device=DEV)]
        out = torch.empty(len(tokens), 4, dtype=torch.float32, device=DEV)
        gen = torch.zeros(1, dtype=torch.int32, device=DEV)
        counters = [torch.zeros(1, dtype=torch.int32, device=DEV) for _ in range(4)]
        tags = torch.full((lpages + 1,), -1, dtype=torch.int32, device=DEV)
        hvalid = torch.zeros(gpages + 1, dtype=torch.int32, device=DEV)
        owner = torch.full((lpages + 1,), -1, dtype=torch.int32, device=DEV)
        _p3_gather_kernel[((len(tokens) + 15) // 16,)](
            dev_buf.reshape(-1),
            torch.zeros(gpages * 64 + 64, 4, dtype=torch.float32, device=DEV).reshape(
                -1
            ),
            tags,
            hvalid,
            owner,
            slots_rows,
            out,
            len(tokens),
            lpages,
            gpages * 64,
            gen,
            counters[0],
            counters[1],
            counters[2],
            counters[3],
            FORCE_HOST=0,
            CHECK=0,
            LOGICAL_SHIFT=6,
            PAGE_SIZE=64,
            ROW_ELEMS=4,
            PAGE_ELEMS=64 * 4,
            BLOCK_ROWS=16,
            PAIR=1,
            peer_ptr=peer_buf.reshape(-1),
        )
        torch.cuda.synchronize()
        assert int(counters[0]) == 0, "no row may fall back to the host mirror"
        return out

    def test_boundary_pages_read_owner_half(self):
        tokens = [
            193 * 64 + 63,  # tail of odd gp -> peer row 96
            194 * 64,  # head of even gp -> local
            194 * 64 + 63,  # page edge inside one owner
            195 * 64,  # head of odd gp -> peer row 97 (probe page)
            195 * 64 + 63,  # tail of the probe page -> peer
            196 * 64 + 5,  # back to local
            511 * 64 + 63,  # odd gp near local-pool edge -> peer row 255
            1022 * 64 + 31,  # second-to-last even gp -> local row 511
            1023 * 64 + 63,  # last gp, slot 32767 = validity bound - 1
        ]
        out = self._run(tokens)
        for i, tok in enumerate(tokens):
            gp = tok // 64
            want = 0.0 if gp == 0 else (1000.0 + gp if gp % 2 == 0 else 2000.0 + gp)
            assert torch.all(out[i] == want), (i, tok, gp, out[i].tolist())

    def test_rank1_table_reads_the_same_owner_rows(self):
        # The TP1/TP3 polarity regression (map_slot_table owner bit must be
        # rank-relative): rank1's table has local/peer swapped versus rank0,
        # yet every row must still resolve to the same storage bytes. The old
        # absolute-parity bit inverted the whole table on odd ranks and each
        # page read its parity neighbor -- "mostly correct, occasionally
        # corrupt" (acceptance 4/6 with one wrong answer).
        tokens = [
            193 * 64 + 63,
            194 * 64,
            195 * 64 + 63,
            511 * 64 + 63,
            1022 * 64 + 31,
            1023 * 64 + 63,
        ]
        out = self._run(tokens, rank_bit=1)
        for i, tok in enumerate(tokens):
            gp = tok // 64
            want = 1000.0 + gp if gp % 2 == 0 else 2000.0 + gp
            assert torch.all(out[i] == want), (i, tok, gp, out[i].tolist())

    def test_gather_multiple_programs_and_padding_rows(self):
        # 19 rows -> grid ceil(19/16) = 2: the BLOCK_ROWS tiling must not
        # drop or garble the tail program, and a padding-page row (token 5,
        # non-real) yields zeros through the invalid mask.
        tokens = [gp * 64 + 17 for gp in range(1, 17)] + [1023 * 64 + 63, 5]
        out = self._run(tokens)
        assert out.shape[0] == 18
        for i, tok in enumerate(tokens):
            gp = tok // 64
            want = 0.0 if gp == 0 else (1000.0 + gp if gp % 2 == 0 else 2000.0 + gp)
            assert torch.all(out[i] == want), (i, tok, gp, out[i].tolist())
