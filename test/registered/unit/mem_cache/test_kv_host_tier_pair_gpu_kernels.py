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

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU"
)

DEV = "cuda:0"
PAGE = 8
PAIR_PAGES, LOCAL_PAGES = 16, 8
LOGICAL_TOKENS = PAIR_PAGES * PAGE
PAD_BASE = LOCAL_PAGES * PAGE + 64
OWNER_MASK = (1 << 30) - 1


def _pair_tier() -> KVHostTier:
    t = object.__new__(KVHostTier)
    t.page_size = PAGE
    t._page_shift = PAGE.bit_length() - 1
    t.pair_pages = PAIR_PAGES
    t.phys_pages = LOCAL_PAGES
    t.pair_block = 1
    t.pair_rank_bit = 0
    t.logical_tokens = LOGICAL_TOKENS
    t.phys_slot_base = PAD_BASE
    t.pair_split = True
    return t


def _pair_phys_table() -> torch.Tensor:
    logical = torch.arange(LOGICAL_TOKENS, dtype=torch.int64)
    logical[:PAGE] = PAD_BASE + torch.arange(PAGE)
    return _pair_tier().map_slot_table(logical.to(torch.int32)).to(DEV)


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

    def _run(self, pair: bool):
        heads, dim, batch, topk = self.HEADS, self.DIM, self.BATCH, self.TOPK
        if pair:
            t = _pair_tier()
            dev_k = torch.zeros(
                (LOCAL_PAGES + 1) * PAGE, heads, dim, dtype=torch.float16, device=DEV
            )
            dev_v = torch.zeros_like(dev_k)
            peer_k = torch.zeros_like(dev_k)
            peer_v = torch.zeros_like(dev_k)
            for gp in range(1, PAIR_PAGES):
                row = gp // 2
                val = 100.0 + gp if gp % 2 == 0 else 200.0 + gp
                (dev_k if gp % 2 == 0 else peer_k)[
                    row * PAGE : (row + 1) * PAGE
                ] = val
                (dev_v if gp % 2 == 0 else peer_v)[
                    row * PAGE : (row + 1) * PAGE
                ] = val + 0.5
            phys = _pair_phys_table()
            extra = (_PAIR_OWNER_MASK, peer_k, peer_v)
        else:
            t = object.__new__(KVHostTier)
            t.page_size = PAGE
            t._page_shift = 3
            t.phys_pages = 4
            t.logical_tokens = LOGICAL_TOKENS
            t.phys_slot_base = 4 * PAGE + 64
            t.pair_split = False
            dev_k = torch.zeros(
                5 * PAGE, heads, dim, dtype=torch.float16, device=DEV
            )
            peer_k = peer_v = None
            for p in range(4):
                dev_k[p * PAGE : (p + 1) * PAGE] = 50.0 + p
            full_slots = torch.arange(LOGICAL_TOKENS, dtype=torch.int32)
            full_slots[:PAGE] = t.phys_slot_base + torch.arange(
                PAGE, dtype=torch.int32
            )
            phys = t.map_slot_table(full_slots).to(DEV)
            extra = None
            dev_v = dev_k + 0.5
        table = phys[None, :].repeat(batch, 1).contiguous()
        req_indices = torch.arange(batch, dtype=torch.int32, device=DEV)
        indices = (
            (torch.arange(1, topk + 1, dtype=torch.int64, device=DEV) * PAGE)[
                None, :
            ]
            .repeat(batch, 1)
            .contiguous()
        )
        seq_lens = torch.full((batch,), (topk + 1) * PAGE, dtype=torch.int32, device=DEV)
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
            full_k[p * PAGE : (p + 1) * PAGE] = torch.randn(
                PAGE, 1, self.DIM, generator=gen
            ).half().to(DEV)
            full_v[p * PAGE : (p + 1) * PAGE] = torch.randn(
                PAGE, 1, self.DIM, generator=gen
            ).half().to(DEV)
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
