"""Active-request KV host tier (streaming KV) -- Phase 1 mirror pipeline.

Implements components 1-3 of docs/v100/KV_HOST_TIER_DESIGN.md §3.2 for the
QSA full-KV pool:

1. Pinned fp16 host pool sized per rank (SGLANG_V100_KV_HOST_TIER_GB),
   page-granular at the KV pool page size.
2. Residency manager: device-side page bitmaps (resident / Clock
   second-chance reference bits) plus advisory counters. Phase 1 never
   evicts and never degrades reads, so the Clock sweep consumer is Phase 2;
   only the bits are maintained here.
3. Pooled per-page K representatives (64-token group means), the buffers
   the Phase 2 selector will read once KV pages actually move to host.
   Lazily allocated via ensure_rep_k() -- Phase 1 only feeds the pipeline,
   and eager allocation showed up as the last straw on the boot-time
   device-memory edge.

Phase 1 is mirror-only: the attention read path never consumes the host
copy. The mirror dedups the flush's token slots to unique pages entirely
on-device (fixed-capacity scatter, no host sync), gathers them into a
small staging buffer on the compute stream, and enqueues the D2H on a
dedicated copy stream ordered behind the gather -- outputs are
bit-identical to the tier being absent; the steady-state cost is the
mirror DMA itself (~2% of prefill at 4k chunks) plus a few small kernels
per forward. Two perf traps are documented in flush(): per-token gather
moves ~64x the real bytes, and torch.unique / boolean indexing both
host-sync and stall the deep prefill pipeline.

Known Phase 1 limitations (resolved by Phase 2 page-in):
- The host pool is a rolling window: a device page's host slot is its
  position in the current flush, not a stable address.
- Writes issued inside captured CUDA graphs (graph-replay decode/verify)
  update the pool but do not enqueue a flush; the residency bits for those
  pages stay unset until the next out-of-graph forward flushes.
- Resident/clock bits may briefly lag the copy stream (reads of the bits
  are advisory statistics in Phase 1).

Phase 2 (SGLANG_V100_KV_TIER_PAGEIN=1) adds the device-driven read path.
A second pinned buffer, ``_host_final``, holds a stable full-pool mirror
whose host slot for a device page is the page number itself. The manual
degrade policy (SGLANG_V100_KV_TIER_DEGRADE_FLAG) mirrors the whole pool
into ``_host_final`` on a one-shot trigger and garbage-fills the closed
device pages with fp16 qNaN, so every later read of them must go through
the Triton page-in kernel, which restores pages in-place straight from
the pinned host addresses (UVA) -- zero host sync and CUDA-graph safe:
the compaction writes fixed device buffers and the kernel launch is
captured with the graph, so replays restore whatever the captured
indexer selects, using only device-visible state. Residency is per
full-attention layer (``resident[layer, page]``): a degrade garbages all
layers at once but page-in restores one layer at a time. Protect bits
mark pages written after the last mirror -- their host copies are stale
by those writes, and page-in must never revert a live write (set_kv_buffer
runs before the per-layer page-in inside the same forward).
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Counter slots inside the device-side counters vector.
_CNT_FLUSHES = 0
_CNT_MIRRORED_PAGES = 1
_CNT_MIRRORED_BYTES = 2
_CNT_TRUNCATED = 3
_CNT_READ_STEPS = 4
_CNT_READ_PAGES = 5
_CNT_READ_HITS = 6
_CNT_READ_MISSES = 7
_CNT_PAGEIN_STEPS = 8
_CNT_DEGRADES = 9
_CNT_SLOTS = 10

_LOG_EVERY_N_FLUSHES = 1000

# Fixed per-flush mirror capacity, in pages. A full 4k prefill chunk touches
# 64 pages; the slack absorbs decode tokens appended to the same flush and
# keeps the dedup buffer shape independent of the token count (no sync).
_MIRROR_CAP = 128


def attach_kv_host_tier(
    pool, full_attention_layer_ids: List[int]
) -> Optional["KVHostTier"]:
    """Attach the tier to a QSA KV pool, or return None when disabled.

    Called at the end of QSATokenToKVPool.__init__; the returned value is
    stored as ``pool.kv_host_tier`` (None when SGLANG_V100_KV_HOST_TIER is
    unset, which keeps every consumer on a plain ``is not None`` check).
    """
    if not envs.SGLANG_V100_KV_HOST_TIER.get():
        return None
    return KVHostTier(
        pool, full_attention_layer_ids, pagein=envs.SGLANG_V100_KV_TIER_PAGEIN.get()
    )


@triton.jit
def _page_in_kernel(
    hk_ptr,
    hv_ptr,
    k_ptr,
    v_ptr,
    uniq_ptr,
    count_ptr,
    resident_ptr,
    garbled_ptr,
    valid_ptr,
    protect_ptr,
    num_pages,
    PAGE_ELEMS: tl.constexpr,
    USE_PROTECT: tl.constexpr,
):
    """Restore missing pages in-place from the pinned host mirror (UVA).

    One program per uniq slot, restoring both the K and V halves of the
    page. ``uniq`` holds deduped device page numbers with a ``num_pages``
    sentinel for invalid entries; ``count`` is a device scalar, so the
    launch is host-sync-free and CUDA-graph safe. Both halves must live in
    one program: with a separate program per half, the two programs race
    on the shared residency bit -- whichever half finishes first sets
    ``resident`` and the other early-outs, leaving half a page as qNaN.
    Programs are idempotent per page: duplicates across replays would write
    identical bytes. Residency is tracked per layer (resident[li, page]).

    Two extra device-side gates keep replays safe without host state: the
    restore early-outs while ``valid`` is zero (the mirror holds no real
    snapshot before the first degrade, so a forced capture/replay launch
    can never copy uninitialized bytes over live pages), and ``garbled``
    marks pages whose device content is known-garbage (degraded and not yet
    restored); a successful restore clears the page's garbled flag alongside
    setting residency, which keeps flush-time residency marking from ever
    re-blocking a restore.
    """
    i = tl.program_id(0)
    if i >= tl.load(count_ptr):
        return
    if tl.load(valid_ptr) == 0:
        return
    page = tl.load(uniq_ptr + i)
    if page >= num_pages:
        return
    if USE_PROTECT:
        if tl.load(protect_ptr + page) != 0:
            return
    if tl.load(resident_ptr + page) != 0:
        return
    offs = tl.arange(0, PAGE_ELEMS)
    base = page * PAGE_ELEMS + offs
    tl.store(k_ptr + base, tl.load(hk_ptr + base))
    tl.store(v_ptr + base, tl.load(hv_ptr + base))
    tl.store(resident_ptr + page, 1)
    tl.store(garbled_ptr + page, 0)


@triton.jit
def _degrade_kernel(
    k_ptr,
    v_ptr,
    uniq_ptr,
    count_ptr,
    resident_ptr,
    garbled_ptr,
    num_pages,
    PAGE_ELEMS: tl.constexpr,
):
    """Garbage-fill degraded device pages so silent non-page-in reads fail.

    Fills each closed page with the fp16 qNaN pattern (0x7E00), clears the
    per-layer residency bit, and sets the per-layer garbled flag. Any read
    that bypasses page-in therefore produces NaNs instead of silently
    correct-looking output, and flush-time residency marking skips the page
    until page-in has actually restored it (per layer).
    """
    i = tl.program_id(0)
    kv = tl.program_id(1)
    if i >= tl.load(count_ptr):
        return
    page = tl.load(uniq_ptr + i)
    if page >= num_pages:
        return
    offs = tl.arange(0, PAGE_ELEMS)
    pat = tl.full((PAGE_ELEMS,), 0x7E00, tl.int16)
    if kv == 0:
        tl.store(k_ptr + page * PAGE_ELEMS + offs, pat)
    else:
        tl.store(v_ptr + page * PAGE_ELEMS + offs, pat)
    tl.store(resident_ptr + page, 0)
    tl.store(garbled_ptr + page, 1)


class KVHostTier:
    """Pinned host mirror + residency bookkeeping for the QSA full-KV pool."""

    def __init__(
        self, pool, full_attention_layer_ids: List[int], pagein: bool = False
    ):
        page_tokens_env = envs.SGLANG_V100_KV_HOST_PAGE_TOKENS.get()
        if page_tokens_env != pool.page_size:
            raise ValueError(
                "SGLANG_V100_KV_HOST_PAGE_TOKENS must equal the KV pool page "
                f"size in Phase 1: got {page_tokens_env}, pool page_size="
                f"{pool.page_size}"
            )
        pool_gb = envs.SGLANG_V100_KV_HOST_TIER_GB.get()
        if pool_gb <= 0:
            raise ValueError(
                f"SGLANG_V100_KV_HOST_TIER_GB must be positive, got {pool_gb}"
            )

        self.pool = pool
        self.page_size = int(pool.page_size)
        # Pools expose device as either str or torch.device; normalize once so
        # Phase 2 checks like self.device.type work on both.
        self.device = (
            torch.device(pool.device) if isinstance(pool.device, str) else pool.device
        )
        self.layer_ids = list(full_attention_layer_ids)
        self.layer_num = len(self.layer_ids)
        if self.layer_num == 0:
            raise ValueError("KV host tier requires full-attention layers")

        # Resolve per-layer device buffers once. Shapes must agree across
        # layers: [rows, kv_heads, head_dim], rows page-aligned.
        self._kv_buffers: List[tuple] = []
        ref_shape = None
        for gid in self.layer_ids:
            k_buf = pool.get_key_buffer(gid)
            v_buf = pool.get_value_buffer(gid)
            if k_buf.shape != v_buf.shape or k_buf.dtype != v_buf.dtype:
                raise ValueError(
                    f"QSA K/V buffers disagree at layer {gid}: "
                    f"{tuple(k_buf.shape)}/{k_buf.dtype} vs "
                    f"{tuple(v_buf.shape)}/{v_buf.dtype}"
                )
            if k_buf.shape[0] % self.page_size:
                raise ValueError(
                    f"KV buffer rows {k_buf.shape[0]} not page-aligned "
                    f"(page_size={self.page_size})"
                )
            if ref_shape is None:
                ref_shape = tuple(k_buf.shape)
                self.kv_dtype = k_buf.dtype
                self.kv_heads = int(k_buf.shape[1])
                self.head_dim = int(k_buf.shape[2])
            elif tuple(k_buf.shape) != ref_shape:
                raise ValueError(
                    f"KV buffer shape differs across layers: "
                    f"{tuple(k_buf.shape)} vs {ref_shape}"
                )
            self._kv_buffers.append((k_buf, v_buf))
        self.device_pages = ref_shape[0] // self.page_size

        page_bytes = self.page_size * self.kv_heads * self.head_dim * 2
        layer_page_bytes = 2 * self.layer_num * page_bytes  # K + V across layers
        budget_bytes = pool_gb << 30
        self.host_pages = min(self.device_pages, budget_bytes // layer_page_bytes)
        if self.host_pages <= 0:
            raise ValueError(
                f"SGLANG_V100_KV_HOST_TIER_GB={pool_gb} is too small for "
                f"{layer_page_bytes} bytes per page across "
                f"{self.layer_num} layers"
            )

        from sglang.srt.mem_cache.pool_host.base import host_memory_budget_bytes

        if budget_bytes > host_memory_budget_bytes():
            logger.warning(
                "kv_host_tier: requested %.1f GiB exceeds the host memory "
                "budget %.1f GiB; allocation may fail.",
                budget_bytes / (1 << 30),
                host_memory_budget_bytes() / (1 << 30),
            )

        # Component 1: one pinned fp16 host block laid out as the device page
        # view [kv, layer, page, page_tokens, kv_heads, head_dim]. A single
        # contiguous block lets flush issue ONE async D2H per step; per-layer
        # copies of gather outputs would hit PyTorch's synchronizing copy
        # path (copy_kernel_cuda -> cudaStreamSynchronize) and stall the
        # prefill pipeline for the whole chunk every flush.
        page_shape = (2, self.layer_num, self.host_pages, self.page_size,
                      self.kv_heads, self.head_dim)
        self._host = torch.empty(page_shape, dtype=self.kv_dtype, pin_memory=True)

        # Per-flush mirror capacity actually used: capped both by the module
        # constant and by the host pool size (a tiny SGLANG_V100_KV_HOST_TIER_GB
        # yields fewer host pages than a full chunk touches).
        self._cap = min(_MIRROR_CAP, self.host_pages)
        # Device staging for the same layout, fixed capacity: at most
        # self._cap unique pages are mirrored per flush (see flush for the
        # dedup scheme). The extra slack over the 64 pages a full 4k prefill
        # chunk touches absorbs the decode tokens the overlap scheduler
        # appends to the same flush.
        self._stage = torch.empty(
            (2, self.layer_num, self._cap) + page_shape[3:],
            dtype=self.kv_dtype,
            device=self.device,
        )
        # Dedup scratch: unique page indices of the current flush, fixed
        # capacity. Slots beyond the real page count keep stale page numbers
        # from previous flushes (or 0 initially); re-mirroring them is
        # harmless, so the D2H always copies the full capacity and no
        # device->host sync is ever needed.
        self._uniq = torch.zeros(
            self._cap, dtype=torch.long, device=self.device
        )

        # Component 2: device-side residency state and counters. Residency
        # is per full-attention layer: a degraded page is garbaged across all
        # layers at once, but page-in restores one layer at a time, so a
        # single flat bitmap would let layer 0's restore hide layer 1's.
        self.resident = torch.zeros(
            (self.layer_num, self.device_pages), dtype=torch.uint8, device=self.device
        )
        self.clock_ref = torch.zeros(
            self.device_pages, dtype=torch.uint8, device=self.device
        )
        self.counters = torch.zeros(_CNT_SLOTS, dtype=torch.int64, device=self.device)

        # Component 3: per-page pooled K representatives (64-token group
        # means), fp16, one slot per device page. Phase 1 only feeds the
        # pipeline (writes are dropped until Phase 2), so the buffer is
        # allocated lazily via ensure_rep_k(): on this box the post-capture
        # runtime headroom is ~3.5 GiB against ~3.4 GiB of first-use Triton
        # kernel/workspace loads, and the eager 41 MiB here turned a
        # borderline boot into a deterministic warmup OOM.
        self.rep_k: Optional[torch.Tensor] = None

        # ------------------------------------------------------------------
        # Phase 2 page-in state. The pinned _host block above stays the
        # Phase 1 rolling flush window; page-in reads the separate full-pool
        # mirror _host_final, whose host slot for a device page is the page
        # number itself (stable address, UVA-dereferenced by the Triton
        # page-in kernel).
        # ------------------------------------------------------------------
        self.pagein = bool(pagein)
        self.layer_slot = {gid: i for i, gid in enumerate(self.layer_ids)}
        self.pending_degrade = False
        self.degrade_ever = False
        self._degrade_seq = 0
        if self.pagein:
            if self.device.type != "cuda":
                raise ValueError("KV host tier page-in requires a CUDA device")
            if self.kv_dtype != torch.float16:
                raise ValueError(
                    f"KV host tier page-in assumes fp16 KV, got {self.kv_dtype}"
                )
            # tl.arange in the page-in/degrade kernels needs a power-of-two
            # page extent.
            self._page_elems = self.page_size * self.kv_heads * self.head_dim
            if self._page_elems & (self._page_elems - 1):
                raise ValueError(
                    f"Page extent {self._page_elems} (page_size={self.page_size}"
                    f" x kv_heads={self.kv_heads} x head_dim={self.head_dim})"
                    " is not a power of two"
                )
            self._page_shift = self.page_size.bit_length() - 1
            final_shape = (2, self.layer_num, self.device_pages) + (
                self.page_size,
                self.kv_heads,
                self.head_dim,
            )
            self._host_final = torch.empty(
                final_shape, dtype=self.kv_dtype, pin_memory=True
            )
            # +1 sentinel slot: compaction clamps garbage page numbers onto
            # device_pages and the kernels skip page >= num_pages.
            self._protect = torch.zeros(
                self.device_pages + 1, dtype=torch.uint8, device=self.device
            )
            # Per-layer garbled bitmap: 1 = the page's device content is
            # known-garbage (degraded, not yet restored on this layer).
            # flush() consults it so residency marking can never re-block a
            # pending restore, and page-in clears it on a successful restore.
            self._garbled = torch.zeros(
                (self.layer_num, self.device_pages),
                dtype=torch.uint8,
                device=self.device,
            )
            # Device-visible "the mirror holds a real snapshot" flag. Page-in
            # early-outs while it is zero, so forced capture/replay launches
            # before the first degrade can never copy the uninitialized
            # _host_final over live pages.
            self._mirror_valid = torch.zeros(
                1, dtype=torch.int32, device=self.device
            )
            # One-shot degrade-flag semantics: a degrade consumes the flag;
            # rm + touch of the file re-arms it (see on_build_extend).
            self._flag_consumed = False
            self._pi_cap = self.device_pages + 1
            self._pi_uniq = torch.zeros(
                self._pi_cap, dtype=torch.long, device=self.device
            )
            self._pi_count = torch.zeros(1, dtype=torch.int32, device=self.device)
            self._ev_started = torch.cuda.Event()
            self._ev_done = torch.cuda.Event()
            self._degrade_flag_path = (
                envs.SGLANG_V100_KV_TIER_DEGRADE_FLAG.get() or None
            )
            logger.info(
                "kv_host_tier page-in enabled: full-pool mirror %.2f GiB/rank"
                " pinned, degrade flag %s",
                self._full_pool_gib(),
                self._degrade_flag_path or "<unset>",
            )

        self._copy_stream = torch.cuda.Stream(device=self.device)
        # Recorded at the end of every flush's copy-stream work; consumers
        # that must observe the final residency bits (extend-build restore,
        # degrade garble) wait on it before touching those bits.
        self._flush_ev = torch.cuda.Event()
        self._pending: List[torch.Tensor] = []
        self.step_seq = 0
        self._flush_count = 0

        logger.info(
            "kv_host_tier attached: %d layers, page=%d tokens, device_pages=%d, "
            "host_pages=%d (%.1f GiB pinned)",
            self.layer_num,
            self.page_size,
            self.device_pages,
            self.host_pages,
            self.host_pages * layer_page_bytes / (1 << 30),
        )

    # ------------------------------------------------------------------
    # Write path (mirror on write)
    # ------------------------------------------------------------------

    def flush(self, out_cache_loc: Optional[torch.Tensor]) -> None:
        """Mirror the KV pages written this forward and bump the step.

        Called once per forward from the attention backend's metadata
        build (out-of-graph paths). ``out_cache_loc`` is None for captured
        graph replays, where only the step counter advances.
        """
        self.step_seq += 1
        if torch.cuda.is_current_stream_capturing():
            return
        if out_cache_loc is not None and out_cache_loc.numel() > 0:
            self._pending.append(out_cache_loc)
        if not self._pending:
            return
        self._flush_count += 1

        pages = torch.cat(self._pending).to(torch.long)
        self._pending.clear()
        # Do NOT boolean-mask invalid slots: CUDA boolean indexing lowers to
        # nonzero, whose output count must be copied back to the host, forcing
        # a sync every flush and stalling the deep prefill pipeline. Clamp
        # invalid slots onto page 0 instead (the reserved padding slot);
        # mirroring it is harmless under the rolling-window host layout.
        pages = pages.clamp(min=0, max=self.device_pages * self.page_size - 1) >> (
            self.page_size.bit_length() - 1
        )
        if pages.numel() > self._cap * self.page_size:
            self.counters[_CNT_TRUNCATED] += 1
            pages = pages[-self._cap * self.page_size :]

        # Dedup to unique page indices, without any device->host sync. This
        # matters at real scale: a 4k chunk's out_cache_loc names each of its
        # 64 pages ~64 times, so gathering per token would move ~3 GiB per
        # flush (measured 2296 -> 1581 tok/s on the target box; the naive
        # alternative, torch.unique, host-syncs and stalls the pipeline the
        # same way). Instead: sort, mark segment boundaries, place each new
        # segment head at a dense index via cumsum, and scatter the page
        # numbers into the fixed-capacity _uniq buffer. Everything after the
        # real page count is stale slots from earlier flushes -- re-mirroring
        # them costs bandwidth only.
        sorted_pages = torch.sort(pages).values
        heads = torch.empty_like(sorted_pages, dtype=torch.int32)
        # fill_, not heads[0] = 1: integer-index assignment lowers to a
        # pageable H2D copy_, which cudaStreamIsCapturing rejects.
        heads[0].fill_(1)
        heads[1:] = sorted_pages[1:] != sorted_pages[:-1]
        dest = torch.cumsum(heads, dim=0, dtype=torch.long) - 1
        self._uniq.scatter_(0, dest.clamp(max=self._cap - 1), sorted_pages)

        stage = self._stage
        for li, (k_buf, v_buf) in enumerate(self._kv_buffers):
            torch.index_select(
                k_buf.view(-1, self.page_size, self.kv_heads, self.head_dim),
                0,
                self._uniq,
                out=stage[0, li],
            )
            torch.index_select(
                v_buf.view(-1, self.page_size, self.kv_heads, self.head_dim),
                0,
                self._uniq,
                out=stage[1, li],
            )
            if self.rep_k is not None:
                # Duplicate indices in _uniq write the same page mean from
                # this same gather, so the index_copy_ race is harmless.
                self.rep_k[li].index_copy_(
                    0, self._uniq, stage[0, li].mean(dim=1)
                )

        writes_done = torch.cuda.Event()
        writes_done.record()
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_event(writes_done)
            # Async D2H, one contiguous block per (kv, layer): stage[kv, li]
            # and host[kv, li, :cap] are both contiguous _MIRROR_CAP-page
            # blocks, so each copy_ takes the fast async path. Host and stage
            # capacities match by construction here, but a single flat-prefix
            # copy would still misalign the [kv, layer] stride -- copy per
            # block.
            for li in range(self.layer_num):
                for kv in range(2):
                    self._host[kv, li, : self._cap].copy_(
                        stage[kv, li], non_blocking=True
                    )
            if self.pagein:
                # Mark residency only for pages that are not sitting garbled:
                # _uniq keeps stale slots from earlier flushes, and marking a
                # degraded-but-unrestored page resident would block its
                # restore forever (the residency gate skips resident pages).
                self.resident[:, self._uniq] = (
                    self._garbled[:, self._uniq] == 0
                ).to(torch.uint8)
            else:
                self.resident[:, self._uniq] = 1
            self.clock_ref[self._uniq] = 1
            self.counters[_CNT_FLUSHES] += 1
            self.counters[_CNT_MIRRORED_PAGES] += self._cap
            self.counters[_CNT_MIRRORED_BYTES] += self._cap * 2 * self.layer_num * (
                self.page_size * self.kv_heads * self.head_dim * 2
            )
            self._flush_ev.record(self._copy_stream)

        if self._flush_count % _LOG_EVERY_N_FLUSHES == 0:
            self.log_stats()

    def ensure_rep_k(self) -> torch.Tensor:
        """Allocate the Phase 2 representative buffer on first use."""
        if self.rep_k is None:
            self.rep_k = torch.zeros(
                (self.layer_num, self.device_pages, self.kv_heads, self.head_dim),
                dtype=torch.float16,
                device=self.device,
            )
        return self.rep_k

    # ------------------------------------------------------------------
    # Phase 2: manual degrade + page-in (SGLANG_V100_KV_TIER_PAGEIN=1)
    # ------------------------------------------------------------------

    def _full_pool_gib(self) -> float:
        return (
            2
            * self.layer_num
            * self.device_pages
            * self.page_size
            * self.kv_heads
            * self.head_dim
            * 2
        ) / (1 << 30)

    def _compact_pages(self, pages: torch.Tensor) -> None:
        """Dedup page numbers into (_pi_uniq, _pi_count) with no host sync.

        Sort -> segment heads -> cumsum -> scatter, the same scheme flush()
        uses but into the page-in buffers. Distinct values are bounded by
        device_pages + 1 (real pages plus the sentinel), exactly the buffer
        capacity, so the capacity clamp never folds two distinct pages.
        """
        flat = pages.reshape(-1).to(torch.long)
        sorted_pages = torch.sort(flat).values
        heads = torch.empty_like(sorted_pages, dtype=torch.int32)
        # fill_, not heads[0] = 1: integer-index assignment lowers to a
        # pageable H2D copy_, which stream capture rejects. Everything here
        # must stay capture-legal -- the launch is captured by decode graphs.
        heads[0].fill_(1)
        heads[1:] = sorted_pages[1:] != sorted_pages[:-1]
        dest = torch.cumsum(heads, dim=0, dtype=torch.long) - 1
        self._pi_uniq.scatter_(0, dest.clamp(max=self._pi_cap - 1), sorted_pages)
        self._pi_count.copy_(dest[-1] + 1)

    def _build_closed_pages(
        self,
        req_to_token: torch.Tensor,
        rows: torch.Tensor,
        closed_last_page: torch.Tensor,
    ) -> None:
        """Dedup the device pages holding tokens [0, closed_last*page_size) per row.

        Only page-start slots strictly below the per-row bound are
        dereferenced, and both degrade and extend callers bound those by a
        written position, so stale req_to_token entries are never read.
        Out-of-range pages clamp onto the sentinel (device_pages) and are
        skipped by the kernels.
        """
        dp = self.device_pages
        pos = torch.arange(dp, device=self.device, dtype=torch.long) * self.page_size
        bound = closed_last_page.to(torch.long).clamp(max=dp) * self.page_size
        mask = pos.unsqueeze(0) < bound.unsqueeze(1)
        rows_l = rows.to(torch.long).clamp(0, req_to_token.shape[0] - 1)
        slots = req_to_token[
            rows_l.unsqueeze(1), pos.clamp(max=req_to_token.shape[1] - 1)
        ]
        pages = (slots >> self._page_shift).clamp(0, dp)
        pages = torch.where(mask, pages, torch.full_like(pages, dp))
        self._compact_pages(pages)

    def _launch_page_in_layer(self, li: int, use_protect: bool) -> None:
        """One page-in launch for layer slot ``li`` (UVA host dereference)."""
        hk = self._host_final[0, li].view(torch.int16)
        hv = self._host_final[1, li].view(torch.int16)
        k_buf, v_buf = self._kv_buffers[li]
        _page_in_kernel[(self._pi_cap,)](
            hk,
            hv,
            k_buf.view(torch.int16),
            v_buf.view(torch.int16),
            self._pi_uniq,
            self._pi_count,
            self.resident[li],
            self._garbled[li],
            self._mirror_valid,
            self._protect,
            self.device_pages,
            PAGE_ELEMS=self._page_elems,
            USE_PROTECT=use_protect,
            num_warps=8,
        )

    def _degrade(
        self, req_to_token: torch.Tensor, rows: torch.Tensor, closed_last_page: torch.Tensor
    ) -> None:
        """One-shot degrade: mirror the full pool, then garbage the closed pages.

        The D2H is a whole-pool contiguous mirror (constant host length, no
        sync) instead of a uniq-scatter gather, which torch cannot express
        device-side; page-in later restores per page through the UVA kernel,
        so the scatter path is never needed. Compute waits for the mirror
        before garbage-filling, and stale protect bits clear after the
        mirror -- post-mirror content is in _host_final for every page, so
        everything is page-in-eligible again.
        """
        self._build_closed_pages(req_to_token, rows, closed_last_page)
        started, done = self._ev_started, self._ev_done
        started.record()
        page_view = (self.page_size, self.kv_heads, self.head_dim)
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_event(started)
            for li, (k_buf, v_buf) in enumerate(self._kv_buffers):
                self._host_final[0, li].copy_(
                    k_buf.view(-1, *page_view), non_blocking=True
                )
                self._host_final[1, li].copy_(
                    v_buf.view(-1, *page_view), non_blocking=True
                )
            done.record()
        torch.cuda.current_stream().wait_event(done)
        # Order the garble's residency/garbled clears behind any in-flight
        # flush marking on the copy stream -- a one-shot path, so waiting for
        # the previous flush's DMA here is free next to the full-pool mirror.
        torch.cuda.current_stream().wait_event(self._flush_ev)
        self._mirror_valid.fill_(1)
        self._protect.zero_()
        for li in range(self.layer_num):
            k_buf, v_buf = self._kv_buffers[li]
            _degrade_kernel[(self._pi_cap, 2)](
                k_buf.view(torch.int16),
                v_buf.view(torch.int16),
                self._pi_uniq,
                self._pi_count,
                self.resident[li],
                self._garbled[li],
                self.device_pages,
                PAGE_ELEMS=self._page_elems,
                num_warps=8,
            )
        # Eager full restore (one-shot degrade policy): put every closed page
        # back right after the garble, so all later decode/verify reads hit
        # resident pages and the replayed in-graph page-in becomes a no-op.
        # The replay path of that selected-page page-in showed unexplained
        # divergence (eager verify correct, replayed verify NaN), and a
        # one-shot degrade is a correctness harness rather than a capacity
        # policy -- ~63 pages x layers ~= 190 MiB per rank is negligible next
        # to the full-pool mirror this path just took. The in-graph page-in
        # stays for Phase 3's automatic partial degradation, where pages
        # genuinely live on host between forwards.
        for li in range(self.layer_num):
            self._launch_page_in_layer(li, use_protect=False)
        self.pending_degrade = False
        self.degrade_ever = True
        self._flag_consumed = True
        self._degrade_seq += 1
        self.counters[_CNT_DEGRADES] += 1
        # Log python-side facts only; counters.item() would host-sync.
        logger.info(
            "kv_host_tier degrade #%d: mirrored full pool (%.2f GiB/rank), "
            "garbage-filled closed pages",
            self._degrade_seq,
            self._full_pool_gib(),
        )

    def on_build_extend(
        self, req_to_token: torch.Tensor, rows: torch.Tensor, seq_lens: torch.Tensor
    ) -> None:
        """Extend-build hook: arm on the degrade flag, restore closed pages.

        Restore covers every page holding tokens [0, S-1] per row. The
        restore must respect the protect bits: a page whose tail rows were
        written after the mirror (decode/verify tokens) has a stale host
        copy for those rows, and this forward rewrites only its own new
        rows, so an ungated restore would revert earlier live writes.
        Residency and protection together dedup against earlier restores:
        pages written out of graph are resident, pages written inside
        captured graphs are protected, and both are skipped.
        """
        if rows.numel() == 0:
            return
        if self._degrade_flag_path:
            if os.path.exists(self._degrade_flag_path):
                # One-shot arm: a degrade consumes the flag; the reset happens
                # when the file disappears, so rm + touch re-arms it.
                if not self._flag_consumed:
                    self.pending_degrade = True
            else:
                self._flag_consumed = False
        if not self.degrade_ever:
            return
        closed_last = (
            (seq_lens.to(torch.long) - 1).clamp_min(0) >> self._page_shift
        ) + 1
        self._build_closed_pages(req_to_token, rows, closed_last)
        # The restore must observe the last flush's final residency bits --
        # its copy-stream marking can otherwise race this eager launch.
        torch.cuda.current_stream().wait_event(self._flush_ev)
        for li in range(self.layer_num):
            self._launch_page_in_layer(li, use_protect=True)
        self.counters[_CNT_PAGEIN_STEPS] += 1

    def on_build_decode_like(
        self,
        req_to_token: torch.Tensor,
        rows: torch.Tensor,
        seq_lens: torch.Tensor,
        append_lo: torch.Tensor,
        write_ahead: int,
    ) -> None:
        """Decode-like build hook: degrade if armed, then protect write pages.

        ``append_lo`` is the first logical position this forward writes
        (seq_len - 1 for decode; seq_len for verify rows, which write
        ``write_ahead`` tokens from there). Degrade runs first so the
        mirror snapshots the pool before this forward's writes; the protect
        scatter lands after and survives. The pages holding those positions
        are never in the closed set, but they still need protection:
        set_kv_buffer runs before the per-layer page-in inside the same
        forward, and their host mirror copies are stale by those writes.
        Freshly allocated pages can be long-degraded ones, so without the
        protect bits a page-in would revert the token just written.
        """
        if rows.numel() == 0:
            return
        if self.pending_degrade:
            self._degrade(req_to_token, rows, append_lo >> self._page_shift)
        pos = append_lo.to(torch.long).unsqueeze(1) + torch.arange(
            write_ahead, device=self.device, dtype=torch.long
        ).unsqueeze(0)
        pos = pos.clamp(0, req_to_token.shape[1] - 1)
        rows_l = rows.to(torch.long).clamp(0, req_to_token.shape[0] - 1)
        slots = req_to_token[rows_l.unsqueeze(1), pos]
        pages = (slots >> self._page_shift).clamp(0, self.device_pages)
        self._protect[pages.reshape(-1)] = 1

    def page_in_selected(
        self,
        req_to_token: torch.Tensor,
        rows: torch.Tensor,
        positions: torch.Tensor,
        tier_layer: int,
        force: bool,
    ) -> None:
        """Restore one layer's selected pages before its attention read.

        Called from _forward_paged_attention for every full-attention layer
        on every decode-like forward, including at graph capture: the
        compaction writes the fixed (_pi_uniq, _pi_count) buffers and the
        launch is captured, so at replay the freshly selected pages (the
        captured indexer recomputes them) are restored from pinned RAM with
        zero host involvement. ``force`` bypasses the pre-degrade early-out
        because graph capture must include the path unconditionally; eager
        calls early-out until the first degrade. Extend reads are covered by
        the build-time restore of all closed pages instead.
        """
        if rows.numel() == 0 or positions.numel() == 0:
            return
        if not force and not self.degrade_ever:
            return
        rows_l = rows.to(torch.long).clamp(0, req_to_token.shape[0] - 1)
        pos = positions.to(torch.long)
        valid = pos >= 0
        pos = pos.clamp(0, req_to_token.shape[1] - 1)
        slots = req_to_token[rows_l.unsqueeze(1), pos]
        pages = torch.where(
            valid,
            slots >> self._page_shift,
            torch.full_like(slots, self.device_pages),
        ).clamp(0, self.device_pages)
        self._compact_pages(pages)
        self._launch_page_in_layer(tier_layer, use_protect=True)
        self.counters[_CNT_PAGEIN_STEPS] += 1

    # ------------------------------------------------------------------
    # Read path (advisory residency query; no read-path behavior change)
    # ------------------------------------------------------------------

    def note_read(
        self,
        req_to_token: torch.Tensor,
        req_rows: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Count resident/miss pages for one forward's selected tokens.

        ``positions`` are the expanded per-row selected logical token
        positions; slots resolve through ``req_to_token`` exactly like the
        SM70 sparse decode kernel does. Statistics only in Phase 1.
        """
        rows = req_rows.to(torch.long)
        pos = positions.to(torch.long)
        pos = pos.clamp(min=0)
        rows = rows.clamp(min=0, max=req_to_token.shape[0] - 1)
        slots = req_to_token[rows.unsqueeze(1), pos]
        pages = slots >> (self.page_size.bit_length() - 1)
        valid = (slots >= 0) & (pages < self.device_pages)
        # Clamp both ends instead of boolean-masking (which would host-sync
        # through nonzero every decode step); invalid rows land on page 0,
        # whose bit flip is harmless for the advisory Phase 1 statistics.
        pages = pages.clamp(min=0, max=self.device_pages - 1)
        # Layer 0's bits are representative for the advisory stats; reading
        # all layers would multiply the gather per decode step for nothing.
        bits = self.resident[0][pages]
        hit = (bits != 0) & valid
        miss = (bits == 0) & valid
        self.counters[_CNT_READ_STEPS] += 1
        self.counters[_CNT_READ_PAGES] += valid.sum()
        self.counters[_CNT_READ_HITS] += hit.sum()
        self.counters[_CNT_READ_MISSES] += miss.sum()
        self.clock_ref[pages] = 1

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def log_stats(self) -> None:
        self._copy_stream.synchronize()
        c = self.counters.tolist()
        reads = c[_CNT_READ_HITS] + c[_CNT_READ_MISSES]
        hit_rate = c[_CNT_READ_HITS] / reads * 100 if reads else 0.0
        logger.info(
            "kv_host_tier stats: flushes=%d mirrored_pages=%d mirrored_MiB=%d "
            "truncated=%d read_steps=%d read_pages=%d hit_rate=%.1f%% "
            "pagein_steps=%d degrades=%d",
            c[_CNT_FLUSHES],
            c[_CNT_MIRRORED_PAGES],
            c[_CNT_MIRRORED_BYTES] >> 20,
            c[_CNT_TRUNCATED],
            c[_CNT_READ_STEPS],
            c[_CNT_READ_PAGES],
            hit_rate,
            c[_CNT_PAGEIN_STEPS],
            c[_CNT_DEGRADES],
        )
