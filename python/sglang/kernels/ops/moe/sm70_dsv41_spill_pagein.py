"""WO-13 D4-G: SM70 UVA page-in of spilled expert rows into a landing pool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _module() -> Module:
    cap = torch.cuda.get_device_capability()
    if cap[0] != 7:
        raise RuntimeError(
            f"sm70_dsv41 spill_page_in requires SM70 (Volta); got SM{cap[0]}{cap[1]}"
        )
    return load_jit(
        "sm70_dsv41_spill_pagein",
        cuda_files=["sm70_dsv41_spill_pagein.cuh"],
        cuda_wrappers=[
            ("spill_page_in", "sm70_dsv41::spill_page_in"),
            ("spill_page_in_cached", "sm70_dsv41::spill_page_in_cached"),
        ],
        extra_include_paths=[str(KERNEL_PATH / "csrc")],
    )


def spill_page_in(
    topk_ids: torch.Tensor,
    land_ids: torch.Tensor,
    slot_host_row: torch.Tensor,
    map_table: torch.Tensor,
    host_map: torch.Tensor,
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    row_bytes: torch.Tensor,
) -> None:
    """Remap ``topk_ids`` and copy UVA host rows into landing slots in place."""
    _module().spill_page_in(
        topk_ids,
        land_ids,
        slot_host_row,
        map_table,
        host_map,
        src_ptrs,
        dst_ptrs,
        row_bytes,
    )


def spill_page_in_cached(
    topk_ids: torch.Tensor,
    land_ids: torch.Tensor,
    slot_host_row: torch.Tensor,
    map_table: torch.Tensor,
    host_map: torch.Tensor,
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    row_bytes: torch.Tensor,
    cache_lut: torch.Tensor,
    cache_slot_key: torch.Tensor,
    cache_epoch: torch.Tensor,
    cache_clock: torch.Tensor,
    lut_offset: int,
) -> None:
    """Same remap+copy, against a persistent cross-call expert cache.

    ``cache_lut`` is the rank-global flat (layer, expert) -> slot table and
    ``lut_offset`` is this layer's ``layer_ordinal * n_logical`` base:
    cache_slot_key stores flat LUT indices, so eviction back-invalidates the
    owning layer's entry. Hits remap without any copy; misses claim the
    argmin-epoch slot (MoE4All-style global LRU with batch-epoch
    protection). All cache state is device-side, so CUDA-graph replays stay
    deterministic.
    """
    _module().spill_page_in_cached(
        topk_ids,
        land_ids,
        slot_host_row,
        map_table,
        host_map,
        src_ptrs,
        dst_ptrs,
        row_bytes,
        cache_lut,
        cache_slot_key,
        cache_epoch,
        cache_clock,
        lut_offset,
    )
