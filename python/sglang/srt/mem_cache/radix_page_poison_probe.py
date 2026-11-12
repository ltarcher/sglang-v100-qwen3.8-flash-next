"""Poison-hunt probe: tree-owned FULL page set vs allocator pops.

Debug-only (SGLANG_DEBUG_MAMBA_RADIX). Maintains a host-side set of device
pages currently owned by radix tree nodes (BASE/FULL component values) and
checks every allocator pop against it. A hit means the allocator handed out
a page the tree still references — the pointer-level double-booking that
corrupts served prefix KV (silent numeric damage, no floods).

Scope: BASE/FULL pages only (SWA/mamba live in separate slot spaces).
"""

import hashlib
import logging
from typing import Iterable, Optional, Union

import torch

logger = logging.getLogger(__name__)

_TREE_OWNED_PAGES: set[int] = set()
# All pages currently handed out (request-owned or tree-owned). A pop that
# intersects this set means two live owners share a page — req-vs-req (or
# draft-vs-target) double-booking, invisible to the tree-owned check.
_LIVE_PAGES: set[int] = set()


def _page_ids(value: Union[torch.Tensor, Iterable[int]], page_size: int) -> set[int]:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return set()
        return set((value.long() // page_size).tolist())
    return {int(v) // page_size for v in value}


def tree_own_pages(value: torch.Tensor, page_size: int) -> None:
    _TREE_OWNED_PAGES.update(_page_ids(value, page_size))


def tree_release_pages(value: torch.Tensor, page_size: int) -> None:
    pages = _page_ids(value, page_size)
    _TREE_OWNED_PAGES.difference_update(pages)
    _LIVE_PAGES.difference_update(pages)


def probe_reset() -> None:
    _TREE_OWNED_PAGES.clear()
    _LIVE_PAGES.clear()


def alloc_take_pages(pages: torch.Tensor, page_size: int, where: str) -> None:
    """Register an allocator handout; log when a pop is already live."""
    ids = _page_ids(pages, page_size)
    hit = ids & _LIVE_PAGES
    if hit:
        logger.error(
            "[radix-dbg] DOUBLE-BOOKING(live) at %s: allocator popped %d "
            "already-live pages, sample=%s (live set size=%d)",
            where,
            len(hit),
            sorted(hit)[:16],
            len(_LIVE_PAGES),
        )
    _LIVE_PAGES.update(ids)


def alloc_release_pages(indices: torch.Tensor, page_size: int) -> None:
    _LIVE_PAGES.difference_update(_page_ids(indices, page_size))


def check_alloc_pages(pages: torch.Tensor, page_size: int, where: str) -> None:
    """Log (do not raise) when an allocator pop returns a tree-owned page."""
    hit = _page_ids(pages, page_size) & _TREE_OWNED_PAGES
    if hit:
        logger.error(
            "[radix-dbg] DOUBLE-BOOKING at %s: allocator popped %d tree-owned "
            "pages, sample=%s (tree set size=%d)",
            where,
            len(hit),
            sorted(hit)[:16],
            len(_TREE_OWNED_PAGES),
        )


def check_write_pages(
    pages: Union[torch.Tensor, Iterable[int]], page_size: int, where: str
) -> None:
    """Log when a KV WRITE target lands on a tree-owned page.

    Covers the MTP/verify write path, whose slot targets come from the
    req_to_token table (or move_kv_cache sources/destinations) rather than
    from an allocator pop — the allocator cannot see these, so alloc-level
    probes are blind to them. A hit means the speculative path is about to
    overwrite a page a radix node still serves.
    """
    hit = _page_ids(pages, page_size) & _TREE_OWNED_PAGES
    if hit:
        logger.error(
            "[radix-dbg] WRITE-INTO-TREE at %s: write target covers %d "
            "tree-owned pages, sample=%s (tree set size=%d)",
            where,
            len(hit),
            sorted(hit)[:16],
            len(_TREE_OWNED_PAGES),
        )


def _allocatable_pages(alloc) -> Optional[torch.Tensor]:
    """Pages the allocator may hand out right now.

    Deliberately excludes deferred-release lists: those are not yet
    allocatable, so a row pointing at one cannot be silently reused yet and
    would produce false positives. ``class_free_pages`` is the classed
    sharding allocator's (page_interleave) per-class free stacks.
    """
    lists = getattr(alloc, "class_free_pages", None)
    if lists is not None:
        live = [t for t in lists if t.numel() > 0]
        return torch.cat(live) if live else None
    return None


def check_write_against_free(
    pages: Union[torch.Tensor, Iterable[int]],
    page_size: int,
    alloc,
    where: str,
) -> None:
    """Log when a KV WRITE target is a page the allocator can already reissue.

    This is the direct signature of "the request row still points at a freed
    page": the write itself looks legal (the row is the requester's own), but
    the page may be handed to another request at any moment, so the write
    either lands in unowned memory or overwrites a later owner's KV.
    """
    ids = _page_ids(pages, page_size)
    if not ids:
        return
    free = _allocatable_pages(alloc)
    if free is None:
        return
    hit = ids & set(torch.unique(free).tolist())
    if hit:
        total = int(free.numel())
        logger.error(
            "[radix-dbg] WRITE-INTO-FREE at %s: %d write-target pages are "
            "allocatable, sample=%s (free pages=%d)",
            where,
            len(hit),
            sorted(hit)[:16],
            total,
        )


def mamba_state_digest(
    req_to_token_pool,
    slots: Union[torch.Tensor, Iterable[int]],
    where: str,
    *,
    physical: bool = False,
) -> Optional[str]:
    """Hash the linear-attention state rows of ``slots`` for A/B comparison.

    Covers conv + temporal plus any slot-sibling state registered on the pool
    (short-conv / ngram context), i.e. everything that rides on a mamba slot.
    Equal digests across a fresh run's donate and a later hit's COW source mean
    the stored state was not mutated in place; a mismatch pins the tree-stored
    state (or the slot it names) as the divergence source.

    Debug-only: syncs the device. Returns the 8-byte hex digest, or None when
    the slots are empty / the layout is unsupported.
    """
    try:
        idx = (
            slots
            if isinstance(slots, torch.Tensor)
            else torch.as_tensor(list(slots), dtype=torch.long)
        ).reshape(-1)
        if idx.numel() == 0:
            return None
        pool = req_to_token_pool
        mamba_pool = pool.mamba_pool
        temporal = mamba_pool.mamba_cache.temporal
        idx = idx.to(device=temporal.device, dtype=torch.long)
        if not physical:
            idx = pool.translate_mamba_indices(idx)
        digest = hashlib.blake2b(digest_size=8)
        for t in [*mamba_pool.mamba_cache.conv, temporal]:
            if t is None or t.numel() == 0:
                continue
            # Slot axis is 1 for both conv ([layer, slot, ...]) and temporal.
            sel = t.index_select(1, idx)
            digest.update(sel.detach().float().contiguous().cpu().numpy().tobytes())
        slot_count = getattr(mamba_pool, "size", None)
        for sibling in getattr(mamba_pool, "_slot_siblings", ()):
            for name, t in vars(sibling).items():
                if not isinstance(t, torch.Tensor) or t.numel() == 0:
                    continue
                if t.ndim > 1 and t.shape[1] == slot_count:
                    axis = 1
                elif t.shape[0] == slot_count:
                    axis = 0
                else:
                    continue
                digest.update(name.encode())
                sel = t.index_select(axis, idx)
                digest.update(sel.detach().float().contiguous().cpu().numpy().tobytes())
        hexdigest = digest.hexdigest()
        logger.info(
            "[radix-dbg] mamba-state %s slots=%s digest=%s",
            where,
            idx.tolist()[:8],
            hexdigest,
        )
        return hexdigest
    except Exception as exc:  # a debug probe must never break serving
        logger.warning("[radix-dbg] mamba-state digest failed at %s: %r", where, exc)
        return None
