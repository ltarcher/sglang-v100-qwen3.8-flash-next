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

from sglang.srt.environ import KVHeadLayout, envs
from sglang.srt.mem_cache.pair_split_ipc import imported_as_tensor

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
# Phase 3 (shrunk device pool, direct-modulo mapping).
_CNT_P3_MIRROR_PAGES = 10
_CNT_P3_WRITE_RESTORES = 11
_CNT_P3_PAGEIN_RESTORES = 12
# Historical slot: the page-in collision gate this counter used to observe
# was removed -- every page-in tag-miss now saves and restores instead of
# skipping. Kept as a zeroed slot so the stats line stays column-stable.
_CNT_P3_COLLISIONS = 13
_CNT_P3_HOST_GATHER_ROWS = 14
# Diagnostics: a mirror whose tag gate rejects the page (host copy stays
# invalid), and an extend gather that reads a host page with no valid copy.
_CNT_P3_MIRROR_SKIPS = 15
_CNT_P3_GATHER_UNVALID = 16
# Debug invariants (QSA_P3_CHECK=1): rows whose page is tag-matched, mirrored
# (hvalid) and not written by this forward must read identical bytes from the
# device slot and the host mirror; a mismatch means the tag claims a page the
# slot does not hold, or the mirror stored another page's bytes.
_CNT_P3_CHECKED_ROWS = 17
_CNT_P3_HOST_DEV_MISMATCH = 18
# Claim-time eviction: a slot being re-claimed held a page whose current bytes
# only existed on device (its mirror was stale), so they were saved back before
# the overwrite. Without this a concurrent reader of that page falls back to
# the stale host copy and reads the wrong context.
_CNT_P3_SAVES = 19
# Pair-split: a page-in fetched a peer-owned page whose row the pair member
# no longer holds (its tag moved). v1 refuses pair-pool oversubscription at
# boot, so any nonzero count means the no-oversubscription invariant broke --
# the affected reads served wrong bytes.
_CNT_P3_PEER_MISS = 20
_CNT_SLOTS = 21

# Extend-read gather: rows per program. ROW_ELEMS is one token's K or V row
# (kv_heads * head_dim); a tile of rows keeps the launches bounded at 512K.
_P3_GATHER_ROWS = 16

# Pair-split physical-table encoding: bit 30 marks a page the PAIR member
# owns (row index is then into the member's buffers, not ours). int32
# entries leave bits 0..29 for the row offset -- 536M tokens, far above any
# pool this fork serves. Slot value -1 (invalid) stays unambiguous because
# valid entries always carry bit 30 == 0 or 1 with a nonnegative remainder.
_PAIR_OWNER_SHIFT = 30
_PAIR_OWNER_MASK = 1 << _PAIR_OWNER_SHIFT
# Kernel-side twin: Triton resolves module globals in @jit bodies only as
# tl.constexpr instances, while the Python-side mapping code needs a plain int.
_PAIR_OWNER_MASK_C = tl.constexpr(_PAIR_OWNER_MASK)

_LOG_EVERY_N_FLUSHES = 1000

# Phase 3 logs per build step: a 516K single request is only ~250 steps, so the
# 1000-flush cadence above would never fire for it.
_LOG_EVERY_N_STEPS = 64

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


@triton.jit
def _p3_mirror_kernel(
    src_ptr,
    dst_ptr,
    uniq_ptr,
    count_ptr,
    tags_ptr,
    hvalid_ptr,
    phys_pages,
    logical_pages,
    skip_ptr,
    counter_ptr,
    PAGE_ELEMS: tl.constexpr,
    PAIR: tl.constexpr = 0,
    pair_pages=0,
    pair_block=0,
    rank_bit=0,
):
    """Copy one logical page from its device slot into the pinned host mirror.

    Phase 3's mapping is pure modulo, so the device slot for logical page
    ``lp`` is ``lp % phys_pages``. A page is only mirrored while its tag still
    names it: with at most ``phys_pages`` distinct writes per forward (checked
    on the host side) nothing overwrote it in between, but the tag gate keeps
    a bug from copying another page's bytes into the mirror.

    The store targets pinned host memory directly (UVA); Phase 2's page-in
    proved device accesses to that memory from a Triton kernel are supported.
    """
    i = tl.program_id(0)
    if i >= tl.load(count_ptr):
        return
    lp = tl.load(uniq_ptr + i)
    if lp >= logical_pages:
        return
    if PAIR:
        # Peer-owned pages are claimed (and mirrored) by the pair member's
        # build in ITS rows; our rows never hold them.
        gp = lp % pair_pages
        if ((gp // pair_block) & 1) != rank_bit:
            return
        pp = (gp // (2 * pair_block)) * pair_block + gp % pair_block
    else:
        pp = lp % phys_pages
    if tl.load(tags_ptr + pp) != lp:
        # The slot was re-claimed by another logical page: its host copy stays
        # whatever it was, and this page's rows must have been rewritten.
        tl.atomic_add(skip_ptr, 1)
        return
    offs = tl.arange(0, PAGE_ELEMS)
    src = src_ptr + pp * PAGE_ELEMS + offs
    dst = dst_ptr + lp * PAGE_ELEMS + offs
    tl.store(dst, tl.load(src))
    tl.store(hvalid_ptr + lp, 1)
    tl.atomic_add(counter_ptr, 1)


@triton.jit
def _p3_write_restore_kernel(
    host_ptr,
    dev_ptr,
    uniq_ptr,
    count_ptr,
    tags_ptr,
    owner_ptr,
    gen_ptr,
    hvalid_ptr,
    phys_pages,
    logical_pages,
    counter_ptr,
    save_ptr,
    PAGE_ELEMS: tl.constexpr,
    PAIR: tl.constexpr = 0,
    pair_pages=0,
    pair_block=0,
    rank_bit=0,
):
    """Save-evict-then-write for this forward's write pages.

    A write may touch only part of a page (decode appends one token to the
    previous page), while modulo reuse means the device slot may hold another
    logical page. When the slot holds a different page (old tag), save its
    current device bytes back to the host mirror first, unconditionally: they
    are the newest bytes of that page (its writer's rows were not mirrored
    yet), and a concurrent reader that loses the tag must fall back to the
    host copy. The save is safe even when the old page was never mirrored
    (hvalid == 0): the tag names the last writer, so the slot holds that
    page's bytes, and a free page's stale mirror can never be re-read because
    reallocation fully rewrites the page before any read. Then restore the
    host copy of ``lp`` itself when one exists (``hvalid``; a freshly
    allocated page has none and is fully rewritten), and claim the slot with
    the tag + the current generation (loaded from ``gen_ptr``): the
    generation marks the page as this forward's pending write set, so its
    host copy is stale for the rows this forward is about to write.
    """
    i = tl.program_id(0)
    if i >= tl.load(count_ptr):
        return
    lp = tl.load(uniq_ptr + i)
    if lp >= logical_pages:
        return
    if PAIR:
        # Peer-owned pages are claimed by the pair member's build in ITS
        # rows; saving or claiming them here would alias a different page.
        gp = lp % pair_pages
        if ((gp // pair_block) & 1) != rank_bit:
            return
        # Local rows follow the pair-half bijection, matching map_slots and
        # page-in, so a self-owned write and a fetched peer page can never
        # collide.
        pp = (gp // (2 * pair_block)) * pair_block + gp % pair_block
    else:
        pp = lp % phys_pages
    gen = tl.load(gen_ptr)
    old_lp = tl.load(tags_ptr + pp)
    if old_lp != lp:
        if (old_lp >= 0) & (old_lp < logical_pages):
            offs = tl.arange(0, PAGE_ELEMS)
            tl.store(
                host_ptr + old_lp * PAGE_ELEMS + offs,
                tl.load(dev_ptr + pp * PAGE_ELEMS + offs),
            )
            tl.store(hvalid_ptr + old_lp, 1)
            tl.atomic_add(save_ptr, 1)
        if tl.load(hvalid_ptr + lp) != 0:
            offs = tl.arange(0, PAGE_ELEMS)
            tl.store(
                dev_ptr + pp * PAGE_ELEMS + offs,
                tl.load(host_ptr + lp * PAGE_ELEMS + offs),
            )
            tl.atomic_add(counter_ptr, 1)
    tl.store(tags_ptr + pp, lp.to(tl.int32))
    tl.store(owner_ptr + pp, gen)


@triton.jit
def _p3_page_in_kernel(
    host_ptr,
    dev_ptr,
    uniq_ptr,
    count_ptr,
    tags_ptr,
    owner_ptr,
    gen_ptr,
    hvalid_ptr,
    phys_pages,
    logical_pages,
    restore_ptr,
    save_ptr,
    PAGE_ELEMS: tl.constexpr,
    PAIR: tl.constexpr = 0,
    pair_pages=0,
    pair_block=0,
    rank_bit=0,
):
    """Restore one layer's selected logical pages before its decode read.

    The generation is loaded from ``gen_ptr``, not baked into the launch:
    these kernels are captured into the decode CUDA graph, and a frozen
    python-int generation would make slots claimed by the capture-time build
    match "owner == gen" for the life of the process.

    The tag is the truth: a matching tag means the device slot already holds
    this logical page (it may have been written this forward), so nothing is
    copied. Any tag-miss is save-then-restore, unconditionally -- including
    slots this forward's build claimed or a previous layer's page-in
    restored: the save pushes the displaced page's newest device bytes
    (this forward's pending write rows included) into its host mirror, and
    the restore brings the selected page in for this layer's read. The next
    layer's page-in repeats the dance; every read is preceded by its own
    layer's restore, so no reader ever sees another page's bytes.
    """
    i = tl.program_id(0)
    if i >= tl.load(count_ptr):
        return
    lp = tl.load(uniq_ptr + i)
    if lp >= logical_pages:
        return
    gen = tl.load(gen_ptr)
    if PAIR:
        # Pair-split v2 design (owner-bit dual-pointer): the decode kernels
        # read peer-owned pages straight from the pair member's buffers via
        # the owner bit in the physical table, so page-in does nothing for
        # them here. Counting a miss would need the dual-pointer table that
        # slice does not build yet; leave the local row untouched.
        gp = lp % pair_pages
        if ((gp // pair_block) & 1) != rank_bit:
            return
        pp = (gp // (2 * pair_block)) * pair_block + gp % pair_block
    else:
        pp = lp % phys_pages
    if tl.load(tags_ptr + pp) == lp:
        return
    old_lp = tl.load(tags_ptr + pp)
    if (old_lp >= 0) & (old_lp < logical_pages):
        offs = tl.arange(0, PAGE_ELEMS)
        tl.store(
            host_ptr + old_lp * PAGE_ELEMS + offs,
            tl.load(dev_ptr + pp * PAGE_ELEMS + offs),
        )
        tl.store(hvalid_ptr + old_lp, 1)
        tl.atomic_add(save_ptr, 1)
    if tl.load(hvalid_ptr + lp) != 0:
        offs = tl.arange(0, PAGE_ELEMS)
        tl.store(
            dev_ptr + pp * PAGE_ELEMS + offs,
            tl.load(host_ptr + lp * PAGE_ELEMS + offs),
        )
        tl.atomic_add(restore_ptr, 1)
    tl.store(tags_ptr + pp, lp.to(tl.int32))
    tl.store(owner_ptr + pp, gen)


@triton.jit
def _p3_gather_kernel(
    dev_ptr,
    host_ptr,
    tags_ptr,
    hvalid_ptr,
    owner_ptr,
    slots_ptr,
    out_ptr,
    n_rows,
    phys_pages,
    logical_tokens,
    gen_ptr,
    host_rows_ptr,
    unvalid_ptr,
    checked_ptr,
    mismatch_ptr,
    FORCE_HOST: tl.constexpr,
    CHECK: tl.constexpr,
    LOGICAL_SHIFT: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ROW_ELEMS: tl.constexpr,
    PAGE_ELEMS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    PAIR: tl.constexpr = 0,
    peer_ptr=None,
):
    """Sparse extend read across the split device/host slot space.

    Each row picks its source by tag: the device slot when it holds the
    logical page, the pinned host mirror otherwise. No page-in, so a page the
    same forward just wrote to an aliasing slot cannot clobber a read. The
    generation is loaded from ``gen_ptr`` so a captured launch reads the
    current build's value at replay.

    Debug modes (both off in production):
    ``FORCE_HOST`` reads every mirrored page (``hvalid`` set) from the host,
    so a run that still matches the plain path proves the mirror content
    correct and isolates the tag-gated device read. ``CHECK`` compares the
    device and host bytes of every tag-matched, mirrored page that is not
    part of this forward's pending write set (owner != gen) and counts rows
    that disagree. Pending pages are excluded because their host copies are
    stale by design until the next build mirrors them; every other page must
    agree, which is what makes the concurrent-reader fallback to the host
    copy safe.
    """
    gen = tl.load(gen_ptr)
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    mask_r = rows < n_rows
    slots = tl.load(slots_ptr + rows, mask=mask_r, other=0).to(tl.int64)
    if PAIR:
        # Owner-bit dual pointer: under PAIR the gather consumes the PHYS
        # slot table (map_slot_table output), where bit 30 marks a page the
        # PAIR member owns and the low bits index the member's buffers.
        peer = (slots & _PAIR_OWNER_MASK_C) != 0
        slots = slots & ~_PAIR_OWNER_MASK_C
        # The low bits are the owner row's LOCAL token slot, so the FULL
        # range check (entry >= PAGE_SIZE, i.e. not the reserved page 0)
        # does not apply: row 0's tokens are legal content. Validity is the
        # row bound instead -- padding entries (phys_slot_base + off) land
        # at page phys_pages + 8 and fall outside.
        valid = mask_r & (slots < phys_pages * PAGE_SIZE)
    else:
        peer = tl.zeros((1,), tl.int1)
        valid = mask_r & (slots >= PAGE_SIZE) & (slots < logical_tokens)
    lp = slots >> LOGICAL_SHIFT
    off = slots - (lp << LOGICAL_SHIFT)
    if PAIR:
        # No PAIR-pool oversubscription (boot gate): a written page always
        # sits in its owner's rows, so the tag/host-fallback dance is
        # FULL-only. Every valid row reads device-side -- local rows from
        # our pool, peer rows straight from the pair member's imported
        # buffers. FORCE_HOST/CHECK compare host mirrors by logical page,
        # which the PAIR entry encoding (owner row, not logical page)
        # cannot address; they remain FULL-only debug gates.
        pp = lp
        tagmatch = valid
        on_device = valid
        # hvalid/host-mirror state is FULL-only (see above); a zero fill
        # keeps the downstream unvalid counter compilable and inert.
        hvalid_v = tl.zeros((BLOCK_ROWS,), tl.int32)
    else:
        pp = lp % phys_pages
        tags = tl.load(tags_ptr + pp, mask=valid, other=-1)
        hvalid_v = tl.load(hvalid_ptr + lp, mask=valid, other=0)
        tagmatch = valid & (tags == lp)
        if FORCE_HOST:
            on_device = tagmatch & (hvalid_v == 0)
        else:
            on_device = tagmatch
    offs = tl.arange(0, ROW_ELEMS)
    dev_idx = pp[:, None] * PAGE_ELEMS + off[:, None] * ROW_ELEMS + offs[None, :]
    host_idx = lp[:, None] * PAGE_ELEMS + off[:, None] * ROW_ELEMS + offs[None, :]
    value = tl.load(
        dev_ptr + dev_idx, mask=(on_device & ~peer)[:, None], other=0.0
    ) + tl.load(
        peer_ptr + dev_idx, mask=(on_device & peer)[:, None], other=0.0
    ) + tl.load(
        host_ptr + host_idx, mask=(valid & ~on_device)[:, None], other=0.0
    )
    tl.store(
        out_ptr + rows[:, None] * ROW_ELEMS + offs[None, :],
        value,
        mask=mask_r[:, None],
    )
    tl.atomic_add(host_rows_ptr, tl.sum((valid & ~on_device).to(tl.int32)))
    if CHECK:
        if PAIR:
            # The PAIR entry carries the owner row, not the logical page, so
            # the host mirror cannot be addressed; peer rows have no local
            # copy at all. A peer-source mismatch counter is deferred until
            # a read-back harness exists.
            checked = valid & tl.zeros((1,), tl.int1)
        else:
            # The host copy of a page this forward claimed is stale by this
            # forward's own writes until the next build mirrors it
            # (owner == gen); everything else was saved/mirrored by the
            # claiming build and must be byte-identical.
            checked = (
                valid
                & tagmatch
                & (hvalid_v != 0)
                & (tl.load(owner_ptr + pp, mask=valid, other=-1) != gen)
            )
        both_dev = tl.load(dev_ptr + dev_idx, mask=checked[:, None], other=0.0)
        both_host = tl.load(host_ptr + host_idx, mask=checked[:, None], other=0.0)
        bad = (both_dev != both_host).to(tl.int32)
        tl.atomic_add(checked_ptr, tl.sum(checked.to(tl.int32)))
        tl.atomic_add(mismatch_ptr, tl.sum((tl.sum(bad, 1) > 0).to(tl.int32)))
    # A host read of a page with no valid copy yields uninitialized bytes; the
    # counter makes that visible instead of looking like a model-quality drop.
    unvalid = valid & ~on_device
    if tl.sum(unvalid.to(tl.int32)) > 0:
        tl.atomic_add(
            unvalid_ptr,
            tl.sum(((hvalid_v == 0) & unvalid).to(tl.int32)),
        )


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

        # ------------------------------------------------------------------
        # Phase 3 (docs/v100/KV_HOST_TIER_DESIGN.md §10): the device full-KV
        # pool is smaller than the logical token budget. Below this level every
        # logical id (allocator, req_to_token, radix, compressed indexer) is
        # untouched; only the full-attention buffers shrink. Phase 1/2
        # machinery is not allocated at all -- its pinned blocks are 32 GiB
        # per rank by default.
        # ------------------------------------------------------------------
        self.phys_tokens = int(
            getattr(pool, "phys_size", ref_shape[0] - self.page_size)
        )
        self.logical_tokens = int(getattr(pool, "logical_size", self.phys_tokens))
        self.phase3 = self.phys_tokens != self.logical_tokens
        # Pair-split state defaults (FULL layout): the kernel launches read
        # these unconditionally; _init_pair_split overwrites them.
        self.pair_split = False
        self.pair_pages = 0
        self.pair_block = 1
        self.pair_rank_bit = 0
        self.peer_rank = -1
        if self.phase3:
            self._init_phase3(pagein)
            return

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
    # Phase 3: shrunk device pool, direct modulo mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _host_mem_fields() -> str:
        """One-line /proc/meminfo view around the boot-time pinned allocations.

        The Phase 3 mirror is the last multi-GiB pinned block a rank allocates,
        and on a 126 GiB box it lands while the checkpoint page cache is still
        warm; logging both sides of the allocation is what separates "the
        allocator itself costs more than the shape" from "someone else is
        holding memory at this instant".
        """
        keys = ("MemFree", "MemAvailable", "Cached", "Shmem", "AnonPages", "Mlocked")
        try:
            with open("/proc/meminfo", "r") as f:
                text = f.read()
        except OSError:
            return "meminfo unavailable"
        seen = {}
        for line in text.splitlines():
            name, _, rest = line.partition(":")
            if name in keys:
                seen[name] = int(rest.split()[0]) // (1 << 20)
        return " ".join(f"{k}={seen.get(k, -1)}GiB" for k in keys)

    def _init_phase3(self, pagein: bool) -> None:
        if pagein:
            raise ValueError(
                "SGLANG_V100_KV_TIER_PAGEIN is the Phase 2 degrade harness; "
                "leave it unset when SGLANG_V100_KV_TIER_PHYS_TOKENS shrinks "
                "the device pool (Phase 3 has its own read path)"
            )
        if self.device.type != "cuda":
            raise ValueError("Phase 3 KV host tier requires a CUDA device")
        if self.kv_dtype != torch.float16:
            raise ValueError(
                f"Phase 3 KV host tier assumes fp16 KV, got {self.kv_dtype}"
            )
        if self.page_size & (self.page_size - 1):
            raise ValueError(
                f"Phase 3 needs a power-of-two KV page size, got {self.page_size}"
            )
        self._page_shift = self.page_size.bit_length() - 1
        self._page_elems = self.page_size * self.kv_heads * self.head_dim
        if self._page_elems & (self._page_elems - 1):
            raise ValueError(
                f"Page extent {self._page_elems} (page_size={self.page_size}"
                f" x kv_heads={self.kv_heads} x head_dim={self.head_dim})"
                " is not a power of two"
            )
        row_elems = self.kv_heads * self.head_dim
        if row_elems & (row_elems - 1):
            raise ValueError(
                f"One token's K/V row ({row_elems} = kv_heads {self.kv_heads}"
                f" x head_dim {self.head_dim}) is not a power of two; the"
                " Phase 3 gather kernel needs tl.arange over it"
            )
        self.phys_pages = self.phys_tokens // self.page_size
        self.logical_pages = self.logical_tokens // self.page_size
        if self.device_pages != self.phys_pages + 1:
            raise ValueError(
                f"Device KV buffer has {self.device_pages} pages; Phase 3 needs "
                f"phys_pages={self.phys_pages} plus the reserved padding page"
            )
        # Logical page lp lives in device slot lp % phys_pages. The reserved
        # padding page (index phys_pages) absorbs writes to the padding slot
        # and page so they can never alias a real logical page.
        self.phys_slot_base = self.phys_pages * self.page_size
        if envs.SGLANG_V100_KV_HEAD_LAYOUT.get() == KVHeadLayout.PAIR_PARITY:
            self._init_pair_split()
        else:
            self._require_full_pool_capacity()

        mirror_elems = (
            2
            * self.layer_num
            * self.logical_pages
            * self.page_size
            * self.kv_heads
            * self.head_dim
        )
        mirror_gib = mirror_elems * self.kv_dtype.itemsize / (1 << 30)
        # Measure the allowance before the allocation itself shrinks MemAvailable;
        # checking afterwards always reports a fictionially tight budget.
        from sglang.srt.mem_cache.pool_host.base import host_memory_budget_bytes

        budget_gib = host_memory_budget_bytes() / (1 << 30)
        if mirror_gib > budget_gib:
            logger.warning(
                "kv_host_tier phase3: full logical mirror %.2f GiB/rank may "
                "exceed the host memory budget %.2f GiB.",
                mirror_gib,
                budget_gib,
            )
        # Weight loading stages tensors through pinned memory, and the caching
        # host allocator keeps those blocks resident after they are freed. They
        # are the same unevictable pages the mirror is about to ask for.
        host_empty = getattr(torch._C, "_host_emptyCache", None)
        if host_empty is not None:
            host_empty()
            logger.info(
                "kv_host_tier phase3: released cached pinned blocks (%s)",
                self._host_mem_fields(),
            )
        logger.info(
            "kv_host_tier phase3: allocating %.2f GiB/rank pinned mirror (%s)",
            mirror_gib,
            self._host_mem_fields(),
        )
        self._host_logical = torch.empty(
            (2, self.layer_num, self.logical_pages)
            + (self.page_size, self.kv_heads, self.head_dim),
            dtype=self.kv_dtype,
            pin_memory=True,
        )
        logger.info(
            "kv_host_tier phase3: pinned mirror allocated (%s)",
            self._host_mem_fields(),
        )

        # Per-layer tags/owner: every full-attention layer has its own device
        # buffer, so "which logical page is in slot pp" is per layer; a shared
        # tag would let one layer's restore claim a slot for its neighbours.
        # owner is the build generation of the last claim: readers exclude
        # this forward's pending pages (their host copies are stale by its
        # own writes until the next build mirrors them), and it replaces the
        # per-build-erased protect bitmap that a concurrent build used to
        # wipe while its predecessor was still reading the pages.
        self._p3_tags = torch.full(
            (self.layer_num, self.phys_pages + 1),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self._p3_owner = torch.zeros(
            (self.layer_num, self.phys_pages + 1), dtype=torch.int32, device=self.device
        )
        # The generation lives in a device scalar, not a python int: the
        # decode CUDA graph captures the page-in kernels, so a frozen python
        # int would pin the capture-time generation into every replay --
        # slots claimed by that long-gone build would keep matching
        # "owner == gen" forever and their selected-page restores would be
        # skipped for the life of the process. The build bumps the scalar
        # on-device; captured kernels read the current value at replay.
        self._p3_gen_buf = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._p3_hvalid = torch.zeros(
            (self.layer_num, self.logical_pages), dtype=torch.uint8, device=self.device
        )
        # Write-page compaction ping-pong: [cur] is the pending list the next
        # build mirrors, [1-cur] is being filled. Capacity covers every
        # logical page plus the invalid sentinel, so compaction never folds
        # two distinct pages.
        self._p3_cap = self.logical_pages + 1
        self._p3_uniq = [
            torch.zeros(self._p3_cap, dtype=torch.long, device=self.device)
            for _ in range(2)
        ]
        self._p3_count = [
            torch.zeros(1, dtype=torch.int32, device=self.device) for _ in range(2)
        ]
        self._p3_cur = 0
        self._p3_pending_bound = 0
        # Decode page-in compaction scratch (recomputed per layer; the
        # selected set differs per layer and must not touch the pending list).
        self._p3_pi_uniq = torch.zeros(
            self._p3_cap, dtype=torch.long, device=self.device
        )
        self._p3_pi_count = torch.zeros(1, dtype=torch.int32, device=self.device)
        # Pointer stand-in for constexpr-dead branches: Triton binds every
        # pointer arg at launch even when the branch is compiled away.
        self._p3_dummy = torch.zeros(1, dtype=self.kv_dtype, device=self.device)
        # Debug-only read-path switches (production leaves both unset):
        # FORCE_HOST reads every mirrored page from the host mirror, CHECK
        # counts device/host disagreements on tag-matched pages.
        self._p3_force_host = os.environ.get("QSA_P3_FORCE_HOST") == "1"
        self._p3_check = os.environ.get("QSA_P3_CHECK") == "1"

        self.pagein = False
        self.layer_slot = {gid: i for i, gid in enumerate(self.layer_ids)}
        self.rep_k = None
        self._pending: List[torch.Tensor] = []
        self.counters = torch.zeros(_CNT_SLOTS, dtype=torch.int64, device=self.device)
        self._copy_stream = torch.cuda.Stream(device=self.device)
        self._flush_ev = torch.cuda.Event()
        self.step_seq = 0
        self._flush_count = 0
        logger.info(
            "kv_host_tier phase3: layers=%d, page=%d tokens, logical_pages=%d, "
            "phys_pages=%d, host logical mirror %.2f GiB/rank, debug read path "
            "force_host=%s check=%s",
            self.layer_num,
            self.page_size,
            self.logical_pages,
            self.phys_pages,
            mirror_gib,
            self._p3_force_host,
            self._p3_check,
        )

    def _require_full_pool_capacity(self) -> None:
        """FULL layout refuses oversubscription (review M2, 2026-11-10).

        The wraparound mapping ``lp % phys_pages`` is only injective when
        ``logical_pages <= phys_pages``. Oversubscribed, two live logical
        pages can be congruent mod phys_pages and alias one device row:
        their write-restore programs race non-atomically on tags/owner and
        their stores clobber each other, and decode consumers (which trust
        page-in without tag checks) read whichever congruent page won the
        row last -- silent corruption the old experiments logged as
        monotonically degrading recall (KV_HOST_TIER_DESIGN.md §10 wrap
        note). Wrapped FULL needs an associative remap redesign before it
        can be re-enabled; until then the pool must fit.
        """
        if self.logical_pages > self.phys_pages:
            raise ValueError(
                "FULL layout wraps logical pages with lp % phys_pages, which "
                "aliases two live pages onto one device row once the pool is "
                f"oversubscribed: logical_pages={self.logical_pages} > "
                f"phys_pages={self.phys_pages}. Raise SGLANG_V100_KV_TIER_"
                "PHYS_TOKENS, lower --max-total-tokens, or use "
                "SGLANG_V100_KV_HEAD_LAYOUT=PAIR_PARITY (its pair-half "
                "bijection admits logical pages up to 2x per-rank physical)."
            )

    def _init_pair_split(self) -> None:
        """Pair-split layout (docs/v100/KV_HOST_TIER_DESIGN.md §6.1).

        ``self.phys_pages`` here is this rank's LOCAL page count (the pool
        was built with PHYS/2 rows); the PAIR-total physical space is
        ``pair_pages = 2 * phys_pages``. Logical page ``gp = lp %
        pair_pages`` belongs to the parity group ``gp // block``: even
        groups are local, odd groups belong to the pair member (which rank
        is "even" is fixed by the TP rank, so both sides agree without
        communication). Self-owned pages map to local device row
        ``local_page_index(gp)``; peer-owned pages are resolved through the
        owner bit in the physical slot table (v2 dual-pointer read path),
        so the attention kernels read the pair member's imported buffers
        directly instead of paging peer pages into local rows.

        v1 constraint: no oversubscription of the PAIR pool. The peer-owne
        d pages must always sit in the peer's device slots -- a peer that
        paged them out would need a cross-process host-mirror read path
        that does not exist yet.
        """
        from sglang.srt.distributed.parallel_state import get_parallel

        self.pair_split = True
        self.pair_block = envs.SGLANG_V100_KV_HEAD_LAYOUT_BLOCK.get()
        self.pair_pages = 2 * self.phys_pages
        rank = get_parallel().attn_tp_rank
        # Group parity inside the pair: even TP ranks own even groups. The
        # pair members are (2i, 2i+1); QKVParallelLinear's shard_id =
        # kv_tp_rank // replicas places both members on the same head, so
        # both compute the same head and each stores its parity half.
        self.pair_rank_bit = rank & 1
        self.peer_rank = rank ^ 1
        # v1 scaffolding landed the ownership map, write-side parity guards
        # and the IPC rendezvous; the read path (owner-bit dual-pointer
        # decode/gather tables, docs/v100/KV_HOST_TIER_DESIGN.md §6.1 v2)
        # is wired by the _p3_gather_kernel PAIR slice below. The former
        # SGLANG_V100_KV_PAIR_SPLIT_TESTBENCH admission gate was removed on
        # 2026-11-10: pair-accept validated (2026-11-04) and serving in
        # production since 2026-11-05.
        if self.logical_pages > self.pair_pages:
            raise ValueError(
                "Pair-split v1 requires the logical pool to fit the PAIR-total "
                f"physical space: logical_pages={self.logical_pages} > "
                f"pair_pages={self.pair_pages}. Raise SGLANG_V100_KV_TIER_PHYS_"
                "TOKENS (the PAIR total) or lower --max-total-tokens."
            )
        # Tag mirror of the pair member: refreshed at every build so page-in
        # knows which of the peer's slots currently hold the pages it reads.
        # phys layout mirrors ours (both ranks build symmetric pools).
        self._peer_tags = torch.full(
            (self.layer_num, self.phys_pages + 1),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        # Imported peer K/V buffers, set by attach_peer_buffers() after the
        # IPC rendezvous. Kernels never dereference them until then; boot
        # refuses to serve without them.
        self._peer_kv_buffers: Optional[List[tuple]] = None
        logger.info(
            "kv_host_tier pair-split: rank=%d peer=%d block=%d pair_pages=%d "
            "local_pages=%d logical_pages=%d",
            rank,
            self.peer_rank,
            self.pair_block,
            self.pair_pages,
            self.phys_pages,
            self.logical_pages,
        )

    def pair_owns_page(self, group_page: int) -> bool:
        """Whether the local rank stores global pair page ``group_page``."""
        return (group_page // self.pair_block) & 1 == self.pair_rank_bit

    def pair_local_page(self, group_page: int) -> int:
        """Local device page index for a self-owned pair page."""
        # Drop one parity bit per block: groups alternate, so the local
        # index is the group sequence number times the block size plus the
        # offset inside the block.
        group = group_page // self.pair_block
        return (group >> 1) * self.pair_block + group_page % self.pair_block

    def map_slots(self, slots: torch.Tensor) -> torch.Tensor:
        """Logical token slots -> physical device slots.

        FULL layout: pure modulo. PAIR_PARITY: self-owned pages map to
        their local row; PEER-OWNED pages map to the reserved padding page
        so the store kernel's skip index discards them -- the peer computes
        the same head and stores its own parity half (owner-computes; there
        are no cross-rank writes).
        """
        slots = slots.to(torch.long)
        page = slots >> self._page_shift
        off = slots & (self.page_size - 1)
        real = (slots >= self.page_size) & (slots < self.logical_tokens)
        if getattr(self, "pair_split", False):
            gp = page % self.pair_pages
            owned = ((gp // self.pair_block) & 1) == self.pair_rank_bit
            local = (gp // (2 * self.pair_block)) * self.pair_block + gp % self.pair_block
            phys = torch.where(owned, local, self.phys_pages) * self.page_size + off
        else:
            phys = (page % self.phys_pages) * self.page_size + off
        return torch.where(real, phys, self.phys_slot_base + off)

    def map_slot_table(self, table: torch.Tensor) -> torch.Tensor:
        """int32 logical slot table -> int32 physical slot table (read side).

        v2 owner-bit dual pointer (docs/v100/KV_HOST_TIER_DESIGN.md 6.1):
        entry = owner_bit << _PAIR_OWNER_SHIFT | owner_row * page_size + off.
        Owner row is the pair-half bijection; read-side consumers
        (``_p3_gather_kernel``, ``qwen_sparse_kv_extraction_compact_triton``,
        ``sm70_qsa_decode``) select the local or the pair member's imported
        base pointer from the bit. Peer-owned pages are read straight from
        the peer's buffers -- page-in skips them. (map_slots is the
        write-side counterpart: it routes peer-owned writes to the padding
        page, whose bytes the pair member produces independently.)
        """
        page = table >> self._page_shift
        off = table & (self.page_size - 1)
        real = (table >= self.page_size) & (table < self.logical_tokens)
        if getattr(self, "pair_split", False):
            # v2 owner-bit dual pointer (docs/v100/KV_HOST_TIER_DESIGN.md
            # 6.1): entry = owner_bit << _PAIR_OWNER_SHIFT | owner_row * page
            # + off. Owner row = the pair-half bijection; the consumer
            # selects the local or the imported peer base pointer from the
            # bit.
            gp = page % self.pair_pages
            # The bit must read "this entry's bytes live in the pair member's
            # buffers" FROM THIS RANK's perspective: parity != pair_rank_bit
            # means the peer owns it. An absolute-parity bit is only correct
            # on even ranks -- on TP1/TP3 it inverts the whole table and every
            # page reads its parity neighbor (the corruption the acceptance
            # probes caught as 4/6 with one wrong answer).
            owner_bit = (((gp // self.pair_block) & 1) != self.pair_rank_bit).to(
                torch.int32
            ) << _PAIR_OWNER_SHIFT
            owner_row = (
                gp // (2 * self.pair_block)
            ) * self.pair_block + gp % self.pair_block
            phys = owner_bit + owner_row * self.page_size + off
        else:
            phys = (page % self.phys_pages) * self.page_size + off
        return torch.where(real, phys, self.phys_slot_base + off).to(torch.int32)

    def _p3_compact(
        self, pages: torch.Tensor, uniq: torch.Tensor, count: torch.Tensor
    ) -> None:
        """Dedup logical page numbers into (uniq, count) with no host sync."""
        sorted_pages = torch.sort(pages.reshape(-1)).values
        heads = torch.empty_like(sorted_pages, dtype=torch.int32)
        # fill_, not heads[0] = 1: integer index assignment lowers to a
        # pageable H2D copy_, which stream capture rejects.
        heads[0].fill_(1)
        heads[1:] = sorted_pages[1:] != sorted_pages[:-1]
        dest = torch.cumsum(heads, dim=0, dtype=torch.long) - 1
        uniq.scatter_(0, dest.clamp(max=self._p3_cap - 1), sorted_pages)
        count.copy_(dest[-1] + 1)

    def attach_peer_buffers(self, peer_k_buffers, peer_v_buffers, peer_tags) -> None:
        """Install the pair member's imported full-KV buffers and tag tables.

        Called once at boot after the IPC rendezvous; both ranks build
        symmetric pools, so the tensors line up layer-for-layer with ours.
        Serving refuses to start without them under PAIR_PARITY (checked in
        ensure_peer_ready).
        """
        if not self.pair_split:
            return
        if len(peer_k_buffers) != self.layer_num:
            raise ValueError(
                f"Peer buffer count {len(peer_k_buffers)} != layers "
                f"{self.layer_num}"
            )
        self._peer_kv_buffers = list(zip(peer_k_buffers, peer_v_buffers))
        self._peer_tags_source = list(peer_tags)

    def ensure_peer_ready(self) -> None:
        """Boot-time hard gate: pair-split needs the peer buffers attached."""
        if self.pair_split and self._peer_kv_buffers is None:
            raise RuntimeError(
                "SGLANG_V100_KV_HEAD_LAYOUT=PAIR_PARITY requires the pair "
                "member's KV buffers (IPC import failed or was skipped)"
            )

    def _refresh_peer_tags(self) -> None:
        """Copy the pair member's tag tables into the local mirror.

        P2P device-to-device, one small copy per layer on the current
        stream: the build already ordered everything else, so the refresh
        lands before this forward's page-in and gather.
        """
        for li in range(self.layer_num):
            # The source is an IPC-imported pointer wrapper; torch copy_
            # needs a real tensor, so wrap the allocation zero-copy.
            self._peer_tags[li].copy_(
                imported_as_tensor(
                    self._peer_tags_source[li], self._peer_tags[li].shape
                ),
                non_blocking=True,
            )

    def _p3_logical_pages(self, slots: torch.Tensor) -> torch.Tensor:
        s = slots.reshape(-1).to(torch.long)
        page = s >> self._page_shift
        real = (s >= self.page_size) & (s < self.logical_tokens)
        return torch.where(real, page, torch.full_like(page, self.logical_pages))

    def _p3_mirror_layer(self, li: int, uniq, count, bound, counter) -> None:
        k_buf, v_buf = self._kv_buffers[li]
        skip = self.counters[
            _CNT_P3_MIRROR_SKIPS : _CNT_P3_MIRROR_SKIPS + 1
        ]
        for kv, buf in ((0, k_buf), (1, v_buf)):
            _p3_mirror_kernel[(bound,)](
                buf.view(torch.int16),
                self._host_logical[kv, li].view(torch.int16),
                uniq,
                count,
                self._p3_tags[li],
                self._p3_hvalid[li],
                self.phys_pages,
                self.logical_pages,
                skip,
                counter,
                PAGE_ELEMS=self._page_elems,
                PAIR=self.pair_split,
                pair_pages=self.pair_pages,
                pair_block=self.pair_block,
                rank_bit=self.pair_rank_bit,
                num_warps=8,
            )

    def on_build_phase3(self, out_cache_loc: Optional[torch.Tensor]):
        """Book this forward's writes and return the physical write slots.

        Order matters and is enforced by the stream: mirror the previous
        build's write pages (their host copies become current), then
        save-evict-then-claim this forward's write pages (the evicted page's
        newest bytes go back to the host mirror before the overwrite, which
        is what keeps a concurrent reader's host fallback correct). Each
        build bumps the generation, so pending pages are marked by owner ==
        gen and stay marked until another build actually reclaims the slot.
        All of it runs before the model forward, so the extend gather and the
        decode page-in of this forward observe consistent host copies.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Phase 3 KV host tier does not support CUDA-graph capture; "
                "serve with --disable-cuda-graph"
            )
        if self.pair_split:
            # Refresh the pair member's tag mirror before this forward's
            # page-in/gather consult it: after the refresh, "peer tag ==
            # lp" means the peer's row holds the page's current bytes.
            self._refresh_peer_tags()
        self.step_seq += 1
        # Bump the generation on-device: the captured page-in kernels load
        # this scalar at replay, so the bump must be a device op ordered
        # before them (it runs on the same stream, out-of-graph).
        self._p3_gen_buf += 1
        if self.step_seq % _LOG_EVERY_N_STEPS == 0:
            # flush() returns early in Phase 3, so the counter dump has to hang
            # off the build hook instead.
            self.log_stats()
        if self._p3_pending_bound > 0:
            counter = self.counters[
                _CNT_P3_MIRROR_PAGES : _CNT_P3_MIRROR_PAGES + 1
            ]
            for li in range(self.layer_num):
                self._p3_mirror_layer(
                    li,
                    self._p3_uniq[self._p3_cur],
                    self._p3_count[self._p3_cur],
                    self._p3_pending_bound,
                    counter,
                )
        self._p3_pending_bound = 0

        cur = 1 - self._p3_cur
        bound = 0
        if out_cache_loc is not None and out_cache_loc.numel():
            bound = min(int(out_cache_loc.numel()), self._p3_cap)
            pages = self._p3_logical_pages(out_cache_loc)
            self._p3_compact(pages, self._p3_uniq[cur], self._p3_count[cur])
            uniq, count = self._p3_uniq[cur], self._p3_count[cur]
            counter = self.counters[
                _CNT_P3_WRITE_RESTORES : _CNT_P3_WRITE_RESTORES + 1
            ]
            save = self.counters[_CNT_P3_SAVES : _CNT_P3_SAVES + 1]
            for li in range(self.layer_num):
                k_buf, v_buf = self._kv_buffers[li]
                for kv, buf in ((0, k_buf), (1, v_buf)):
                    _p3_write_restore_kernel[(bound,)](
                        self._host_logical[kv, li].view(torch.int16),
                        buf.view(torch.int16),
                        uniq,
                        count,
                        self._p3_tags[li],
                        self._p3_owner[li],
                        self._p3_gen_buf,
                        self._p3_hvalid[li],
                        self.phys_pages,
                        self.logical_pages,
                        counter,
                        save,
                        PAGE_ELEMS=self._page_elems,
                        PAIR=self.pair_split,
                        pair_pages=self.pair_pages,
                        pair_block=self.pair_block,
                        rank_bit=self.pair_rank_bit,
                        num_warps=8,
                    )
        else:
            self._p3_count[cur].zero_()
        self._p3_cur = cur
        self._p3_pending_bound = bound
        if out_cache_loc is None:
            return None
        return self.map_slots(out_cache_loc)

    def page_in_selected_phase3(
        self,
        slot_table: torch.Tensor,
        positions: torch.Tensor,
        tier_layer: int,
    ) -> None:
        """Restore one layer's selected logical pages before its decode read.

        ``positions`` are logical per-row positions, resolved through the
        dense logical ``slot_table`` (batch-local rows) exactly like the SM70
        decode kernel's table. Deduped on-device, so a top-k set touching a
        page dozens of times copies it once.
        """
        if positions.numel() == 0 or slot_table.numel() == 0:
            return
        n_rows = int(positions.shape[0])
        max_len = int(slot_table.shape[1])
        rows = torch.arange(n_rows, device=self.device, dtype=torch.long).clamp(
            max=slot_table.shape[0] - 1
        )
        pos = positions.to(torch.long)
        valid = (pos >= 0) & (pos < max_len)
        pos = pos.clamp(0, max_len - 1)
        s = slot_table[rows.unsqueeze(1), pos].to(torch.long)
        real = valid & (s >= self.page_size) & (s < self.logical_tokens)
        pages = torch.where(
            real, s >> self._page_shift, torch.full_like(s, self.logical_pages)
        )
        bound = min(int(pages.numel()), self._p3_cap)
        self._p3_compact(pages, self._p3_pi_uniq, self._p3_pi_count)
        restore = self.counters[
            _CNT_P3_PAGEIN_RESTORES : _CNT_P3_PAGEIN_RESTORES + 1
        ]
        save = self.counters[_CNT_P3_SAVES : _CNT_P3_SAVES + 1]
        k_buf, v_buf = self._kv_buffers[tier_layer]
        for kv, buf in ((0, k_buf), (1, v_buf)):
            _p3_page_in_kernel[(bound,)](
                self._host_logical[kv, tier_layer].view(torch.int16),
                buf.view(torch.int16),
                self._p3_pi_uniq,
                self._p3_pi_count,
                self._p3_tags[tier_layer],
                self._p3_owner[tier_layer],
                self._p3_gen_buf,
                self._p3_hvalid[tier_layer],
                self.phys_pages,
                self.logical_pages,
                restore,
                save,
                PAGE_ELEMS=self._page_elems,
                PAIR=self.pair_split,
                pair_pages=self.pair_pages,
                pair_block=self.pair_block,
                rank_bit=self.pair_rank_bit,
                num_warps=8,
            )

    def gather_kv_rows(
        self,
        tier_layer: int,
        k_buf: torch.Tensor,
        v_buf: torch.Tensor,
        slots: torch.Tensor,
        out_k: torch.Tensor,
        out_v: torch.Tensor,
    ) -> None:
        """Gather logical rows from device *or* the pinned host mirror.

        Extend reads materialize the whole prefix, which modulo mapping
        cannot hold on device. Reading the host copy in place avoids
        restoring pages the same forward may be aliasing with its own writes.
        """
        n = int(slots.numel())
        if n == 0:
            return
        grid = (triton.cdiv(n, _P3_GATHER_ROWS),)
        counter = self.counters[
            _CNT_P3_HOST_GATHER_ROWS : _CNT_P3_HOST_GATHER_ROWS + 1
        ]
        unvalid = self.counters[
            _CNT_P3_GATHER_UNVALID : _CNT_P3_GATHER_UNVALID + 1
        ]
        checked = self.counters[
            _CNT_P3_CHECKED_ROWS : _CNT_P3_CHECKED_ROWS + 1
        ]
        mismatch = self.counters[
            _CNT_P3_HOST_DEV_MISMATCH : _CNT_P3_HOST_DEV_MISMATCH + 1
        ]
        tags = self._p3_tags[tier_layer]
        owner = self._p3_owner[tier_layer]
        # Under pair-split, ``slots`` arrives as the PHYS slot table
        # (owner-bit encoded, map_slot_table output) and ``peer_buf`` is the
        # pair member's imported buffer for this layer + K/V index.
        for kv, buf, out, peer_buf in self._gather_buffers(tier_layer, k_buf, v_buf, out_k, out_v):
            _p3_gather_kernel[grid](
                buf,
                self._host_logical[kv, tier_layer],
                tags,
                self._p3_hvalid[tier_layer],
                owner,
                slots,
                out,
                n,
                self.phys_pages,
                self.logical_tokens,
                self._p3_gen_buf,
                counter,
                unvalid,
                checked,
                mismatch,
                FORCE_HOST=self._p3_force_host,
                CHECK=self._p3_check,
                LOGICAL_SHIFT=self._page_shift,
                PAGE_SIZE=self.page_size,
                ROW_ELEMS=self.kv_heads * self.head_dim,
                PAGE_ELEMS=self._page_elems,
                BLOCK_ROWS=_P3_GATHER_ROWS,
                PAIR=self.pair_split,
                peer_ptr=peer_buf,
                num_warps=4,
            )

    def _gather_buffers(self, tier_layer, k_buf, v_buf, out_k, out_v):
        """Per-K/V launch tuples, with the peer buffer under pair-split."""
        if not self.pair_split:
            # Triton binds pointer args even in constexpr-dead branches;
            # hand it a real 1-element dummy instead of None.
            dummy = self._p3_dummy
            return ((0, k_buf, out_k, dummy), (1, v_buf, out_v, dummy))
        peer_k, peer_v = self._peer_kv_buffers[tier_layer]
        return ((0, k_buf, out_k, peer_k), (1, v_buf, out_v, peer_v))

    def pair_peer_kv(self, tier_layer: int):
        """(owner_mask, peer_k, peer_v) for this layer, or None off pair-split.

        Read-side consumers of the physical slot table (FA2 extraction
        gather, sm70 decode kernel) select the pair member's imported
        buffers per row from the owner bit encoded by ``map_slot_table``.
        """
        if not self.pair_split:
            return None
        peer_k, peer_v = self._peer_kv_buffers[tier_layer]
        return (_PAIR_OWNER_MASK, peer_k, peer_v)

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
        if self.phase3:
            # Phase 3 writes are booked/mirrored by on_build_phase3.
            return
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
        if self.phase3:
            logger.info(
                "kv_host_tier phase3 stats: steps=%d mirror_pages=%d "
                "write_restores=%d pagein_restores=%d collisions=%d "
                "saves=%d host_gather_rows=%d mirror_skips=%d "
                "gather_unvalid=%d checked_rows=%d host_dev_mismatch=%d "
                "peer_miss=%d",
                self.step_seq,
                c[_CNT_P3_MIRROR_PAGES],
                c[_CNT_P3_WRITE_RESTORES],
                c[_CNT_P3_PAGEIN_RESTORES],
                c[_CNT_P3_COLLISIONS],
                c[_CNT_P3_SAVES],
                c[_CNT_P3_HOST_GATHER_ROWS],
                c[_CNT_P3_MIRROR_SKIPS],
                c[_CNT_P3_GATHER_UNVALID],
                c[_CNT_P3_CHECKED_ROWS],
                c[_CNT_P3_HOST_DEV_MISMATCH],
                c[_CNT_P3_PEER_MISS],
            )
            return
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
