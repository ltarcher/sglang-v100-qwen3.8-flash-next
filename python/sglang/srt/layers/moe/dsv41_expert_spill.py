"""DSV4.1 routed-expert spill: LRU of MXFP4 rows onto pinned host.

Attention / routers / shared experts stay GPU. Generic ``--cpu-offload-gb``
wraps whole ``DeepseekV4DecoderLayer`` modules (CSA2 + indexer + MoE) and
``to(device)``-s them on every forward. This module spills only routed-expert
rows.

``maybe_spill_model_routed_experts`` attaches the plan. Shrink + host copies
happen only when ``SGLANG_DSV41_EXPERT_SPILL_APPLY`` is on. Prefill still
``ensure()`` + ``map_ids`` (LRU; not CUDA-graph safe). Decode uses a host
MXFP4 GEMV via a mapped mailbox when ``SGLANG_DSV41_HOST_GEMV`` is on,
else UVA page-in into a shared landing pool. Both remap with a
capturable kernel so Marlin/GEMV can stay in the decode graph.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.mem_cache.dsv41_host_placement import (
    EngramNumaError,
    engram_numa_node,
    gpu_numa_node,
    mmap_numa_thp,
    node_available_bytes,
    read_numa_nodes,
)
from sglang.srt.mem_cache.dsv41_v100_budget import (
    DEFAULT_SPILL_GIB,
    N_LAYERS,
    N_ROUTED_EXPERTS,
    expert_scale_bytes,
    mxfp4_expert_bytes,
)

logger = logging.getLogger(__name__)

GIB = 1024**3

# FusedMoE expert-dim parameters after MXFP4/NVFP4 Marlin pack. Shrink them
# together so w13/w2/scales stay row-aligned with the LRU slot map. The two
# scale-inv names are the MXFP4 checkpoint layout, the two scale_2 names the
# NVFP4 checkpoint layout; after the SM70 Marlin pack both families carry the
# per-expert global scales instead (w13_scale2 / w2_scale2), which the LRU
# must swap just like any other expert-dim row because the fold-down makes
# them expert-specific. The per-expert input_scale rows exist only on NVFP4
# checkpoints and must mirror too: the loader routes every expert-indexed
# load, and a spilled expert with no host row would hit the no-GPU-slot
# guard. (SM70 Marlin never consumes them -- process_weights_after_loading
# reduces them to a scalar -- but the loads must land somewhere.)
_EXPERT_PARAM_ATTRS = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_weight_scale_inv",
    "w2_weight_scale_inv",
    "w13_weight_scale_2",
    "w2_weight_scale_2",
    "w13_scale2",
    "w2_scale2",
    "w13_input_scale",
    "w2_input_scale",
    "w13_weight_bias",
    "w2_weight_bias",
)


@dataclass(frozen=True)
class RoutedExpertSpillPlan:
    local_routed: int
    n_shared: int
    n_kept_routed: int
    n_spilled: int
    bytes_per_expert: int
    spill_bytes: int
    kept_bytes: int

    @property
    def spill_gib(self) -> float:
        return self.spill_bytes / GIB

    @property
    def kept_gib(self) -> float:
        return self.kept_bytes / GIB


def plan_routed_expert_spill(
    *,
    spill_gib: float,
    local_routed: int,
    bytes_per_expert: int,
    n_shared: int = 1,
    n_layers: int = N_LAYERS,
) -> RoutedExpertSpillPlan:
    """How many local routed experts leave HBM at this spill budget.

    Shared-expert slots are never spilled. ``spill_gib`` is per rank, across
    all layers (the dry-run number).
    """
    if local_routed < 0 or bytes_per_expert < 0:
        raise ValueError("local_routed and bytes_per_expert must be >= 0")
    spill_bytes = int(round(max(spill_gib, 0.0) * GIB))
    layer_budget = spill_bytes // max(n_layers, 1)
    n_spilled = 0 if bytes_per_expert == 0 else min(
        local_routed, layer_budget // bytes_per_expert
    )
    # If integer division under-spills the rank budget, spill extra experts
    # from the last layers conceptually — we still report rank-level bytes.
    n_kept_routed = local_routed - n_spilled
    kept_bytes = n_kept_routed * bytes_per_expert * n_layers
    actual_spill = n_spilled * bytes_per_expert * n_layers
    return RoutedExpertSpillPlan(
        local_routed=local_routed,
        n_shared=n_shared,
        n_kept_routed=n_kept_routed,
        n_spilled=n_spilled,
        bytes_per_expert=bytes_per_expert,
        spill_bytes=actual_spill,
        kept_bytes=kept_bytes,
    )


def v1_spill_plan(spill_gib: float = DEFAULT_SPILL_GIB, ep_size: int = 8) -> RoutedExpertSpillPlan:
    local_routed = N_ROUTED_EXPERTS // ep_size
    per_expert = mxfp4_expert_bytes() + expert_scale_bytes()
    return plan_routed_expert_spill(
        spill_gib=spill_gib,
        local_routed=local_routed,
        bytes_per_expert=per_expert,
        n_shared=1,
    )


def marlin_mxfp4_packed_expert_bytes(
    hidden_size: int, intermediate_size_per_partition: int
) -> int:
    """Bytes for one expert after MXFP4 Marlin pack (int8 + e8m0 scales)."""
    fp4_block_k = 32
    inter = (intermediate_size_per_partition + 127) // 128 * 128
    hidden = (hidden_size + 255) // 256 * 256
    w13 = 2 * inter * (hidden // 2)
    w2 = hidden * (inter // 2)
    s13 = 2 * inter * (hidden // fp4_block_k)
    s2 = hidden * (inter // fp4_block_k)
    return w13 + w2 + s13 + s2


def marlin_nvfp4_packed_expert_bytes(
    hidden_size: int, intermediate_size_per_partition: int
) -> int:
    """Bytes for one expert after NVFP4 SM70 Marlin pack.

    Codes: [K/16, 2N] int32 for w13 (i.e. K*N fp4 elements at 0.5 B), same
    density for w2. Scales: one e4m3 per 16 inputs per output. Plus the two
    per-expert fp32 global scales (fold-down makes the w13 one
    expert-specific, so the LRU swaps it with the rows).
    """
    hidden = hidden_size
    inter = intermediate_size_per_partition
    w13 = inter * hidden
    w2 = inter * hidden // 2
    s13 = 2 * inter * hidden // 16
    s2 = hidden * inter // 16
    return w13 + w2 + s13 + s2 + 16


# Construction-time exemption for FusedMoE layers that must keep every routed
# expert GPU-resident even when the process-wide spill env is set. The NextN
# draft MoE is the canonical user: a single full-size MoE layer run on every
# decode step whose forward dispatch has no spill LRU/remap wiring (that wiring
# attaches only to the target model via finalize_after_quant_processing), so a
# shrunk slot count would leave its unremapped topk_ids indexing past the
# weight tensor.
_spill_exempt_depth = 0


@contextmanager
def exempt_fused_moe_from_spill():
    global _spill_exempt_depth
    _spill_exempt_depth += 1
    try:
        yield
    finally:
        _spill_exempt_depth -= 1


def plan_gpu_expert_slots(
    *,
    num_local_experts: int,
    n_shared: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    n_layers: Optional[int] = None,
    is_nvfp4: bool = False,
) -> tuple[int, Optional[RoutedExpertSpillPlan]]:
    """GPU expert-dim length for create_weights. APPLY shrinks at alloc time."""
    spill_gib = float(envs.SGLANG_DSV41_EXPERT_SPILL_GB.get() or 0.0)
    apply = bool(envs.SGLANG_DSV41_EXPERT_SPILL_APPLY.get())
    if not apply or spill_gib <= 0 or num_local_experts <= 0:
        return num_local_experts, None
    if _spill_exempt_depth > 0:
        logger.info(
            "DSV4.1 expert spill exempt at construction: keeping all %d routed "
            "experts GPU-resident (NextN draft / spill-unwired MoE).",
            num_local_experts,
        )
        return num_local_experts, None
    if n_layers is None:
        # DSV4.1's 40 was the planning default; other stacks override (GLM
        # has 42 MoE layers), or via SGLANG_DSV41_EXPERT_SPILL_N_LAYERS.
        n_layers = int(envs.SGLANG_DSV41_EXPERT_SPILL_N_LAYERS.get() or N_LAYERS)
    local_routed = max(num_local_experts - n_shared, 0)
    # Spill GiB is sized for the 40-layer 384-expert target (48 local routed
    # at EP8). DSpark draft is 128/EP8=16 and must stay fully GPU-resident.
    # Overridable for other stacks / forced-spill smoke tests.
    target_local = int(
        envs.SGLANG_DSV41_EXPERT_SPILL_MIN_LOCAL.get() or (N_ROUTED_EXPERTS // 8)
    )
    if 0 < local_routed < target_local:
        logger.info(
            "DSV4.1 expert spill skipped: local routed %d < target shard %d. "
            "Keeping %d GPU slots (DSpark draft / small MoE).",
            local_routed,
            target_local,
            num_local_experts,
        )
        return num_local_experts, None
    if is_nvfp4:
        bytes_per = marlin_nvfp4_packed_expert_bytes(
            hidden_size, intermediate_size_per_partition
        )
    else:
        bytes_per = marlin_mxfp4_packed_expert_bytes(
            hidden_size, intermediate_size_per_partition
        )
    plan = plan_routed_expert_spill(
        spill_gib=spill_gib,
        local_routed=local_routed,
        bytes_per_expert=bytes_per,
        n_shared=n_shared,
        n_layers=n_layers,
    )
    if plan.n_spilled <= 0:
        return num_local_experts, plan
    gpu_n = plan.n_kept_routed + n_shared
    # The spill GiB knob is sized for the 40-layer 384-expert target. A 3-stage
    # DSpark draft is 128 experts (~0.8 GiB) and would compute gpu_n=0 if it
    # inherited that LRU. Keep the small MoE entirely on GPU.
    if gpu_n <= 0 or plan.n_spilled >= local_routed:
        logger.info(
            "DSV4.1 expert spill skipped: would host-spill all %d local routed "
            "experts (gpu slots %d). Keeping %d GPU slots. Draft/small MoE; "
            "the 384-expert target LRU is unchanged.",
            local_routed,
            gpu_n,
            num_local_experts,
        )
        return num_local_experts, None
    logger.info(
        "DSV4.1 expert create_weights: GPU slots %d (kept routed %d + shared %d), "
        "host-spill %d of %d local routed (%.1f GiB/rank packed)",
        gpu_n,
        plan.n_kept_routed,
        n_shared,
        plan.n_spilled,
        local_routed,
        plan.spill_gib,
    )
    return gpu_n, plan


def spill_host_is_marlin_packed(
    hosts: Dict[str, torch.Tensor], gpu_w13: torch.Tensor, gpu_scale: Optional[torch.Tensor]
) -> bool:
    """True only if host w13 *and* packed scales already match the GPU Marlin layout.

    Checkpoint host is ``w13_weight_scale_inv`` in ``[E, N, K/32]``. SM70 pack
    deletes that and writes ``w13_weight_scale`` in ``[E, K/32, N]``. Matching
    w13 trailing shape alone used to skip pack while leaving scales on the
    checkpoint name, so LRU swapped w13/w2 but Marlin kept the GPU slot's
    scales.
    """
    host_w13 = hosts.get("w13_weight")
    if host_w13 is None:
        return False
    if tuple(host_w13.shape[1:]) != tuple(gpu_w13.shape[1:]):
        return False
    if host_w13.dtype != gpu_w13.dtype:
        raise RuntimeError(
            "DSV4.1 spill host w13 trailing shape matches GPU "
            f"{tuple(gpu_w13.shape[1:])} but dtype {host_w13.dtype} != {gpu_w13.dtype}"
        )
    host_scale = hosts.get("w13_weight_scale")
    if host_scale is None or gpu_scale is None:
        return False
    if tuple(host_scale.shape[1:]) != tuple(gpu_scale.shape[1:]) or (
        host_scale.dtype != gpu_scale.dtype
    ):
        return False
    # NVFP4 checkpoints also name their raw E4M3 scales "w13_weight_scale"
    # with the same trailing shape and dtype as the packed Marlin tensor, so
    # shape alone cannot separate checkpoint from packed. The pack additionally
    # emits the per-expert global scales, which only exist post-pack.
    if host_scale.dtype == torch.float8_e4m3fn and "w13_scale2" not in hosts:
        return False
    return True


def alloc_spill_host_buffers(moe: nn.Module, plan: RoutedExpertSpillPlan) -> None:
    """Pinned host rows for spilled experts, matching each GPU expert-dim param."""
    if plan.n_spilled <= 0:
        return
    # Checkpoint-layout rows, filled by the weight loader; repacked and pinned
    # after Marlin pack (pin_spill_host_numa). Not registered here: pinning
    # 80 GiB before the load would only add to the loader's peak.
    #
    # Place them on the *same* NUMA node the pinned mirror of this
    # layer will use. With the default (preferred node 1) policy and node 1
    # full of Engram hugetlb, these 10 GiB/rank overflowed onto node 0 and
    # coexisted with the growing pinned node-0 half -> node-0 OOM
    # (CONSTRAINT_MEMORY_POLICY) during repack. Same node keeps the per-node
    # footprint flat through the transition.
    node: Optional[int] = None
    layer_id = int(getattr(moe, "layer_id", 0))
    use_numa = bool(envs.SGLANG_ENABLE_DSV41_EXPERT_SPILL_NUMA.get()) and torch.cuda.is_available()
    if use_numa:
        nodes = _spill_numa_nodes()
        node = nodes[layer_id % len(nodes)]
        if node < 0:
            node = None
    specs = _spill_row_specs(moe)
    if node is None:
        hosts, mms = _alloc_spill_rows(specs, n_rows=plan.n_spilled, node=None)
    else:
        hosts, mms, node = _alloc_spill_rows_with_failover(
            specs, n_rows=plan.n_spilled, preferred=node, layer_id=layer_id
        )
    moe._dsv41_spill_host = hosts  # type: ignore[attr-defined]
    moe._dsv41_spill_host_ctor_mms = mms  # type: ignore[attr-defined]
    moe._dsv41_spill_numa_node = node  # type: ignore[attr-defined]
    moe._dsv41_expert_spill_plan = plan  # type: ignore[attr-defined]


# (param attr, dtype, row shape) of one expert-dim param's host rows.
_SpillRowSpec = tuple[str, torch.dtype, tuple[int, ...]]


def _spill_row_specs(moe: nn.Module) -> List[_SpillRowSpec]:
    specs = []
    for attr in _EXPERT_PARAM_ATTRS:
        p = getattr(moe, attr, None)
        if p is None or not isinstance(p, torch.nn.Parameter) or p.ndim < 1:
            continue
        if int(p.shape[0]) == 0:
            continue
        specs.append((attr, p.dtype, tuple(p.shape[1:])))
    return specs


def _alloc_spill_rows(
    specs: List[_SpillRowSpec],
    *,
    n_rows: int,
    node: Optional[int],
) -> tuple[Dict[str, torch.Tensor], List]:
    hosts: Dict[str, torch.Tensor] = {}
    mms: List = []
    try:
        for attr, dtype, row_shape in specs:
            numel = n_rows * math.prod(row_shape)
            if node is not None and numel > 0:
                mms.append(mmap_numa_thp(numel * dtype.itemsize, node=node))
                hosts[attr] = torch.frombuffer(mms[-1], dtype=dtype, count=numel).view(
                    n_rows, *row_shape
                )
            else:
                # Pageable on purpose: fresh pinned allocations cost 2x their
                # size in RSS on the 4xV100 host; the OOM-killer wins at create.
                hosts[attr] = torch.empty(n_rows, *row_shape, dtype=dtype, device="cpu")
    except BaseException:
        # The traceback keeps this frame alive; unmap the rows placed so far now.
        hosts.clear()
        mms.clear()
        raise
    return hosts, mms


def _alloc_spill_rows_with_failover(
    specs: List[_SpillRowSpec],
    *,
    n_rows: int,
    preferred: int,
    layer_id: int,
) -> tuple[Dict[str, torch.Tensor], List, int]:
    """Rows of one layer, all on one node: the stripe node if it has room."""
    nbytes = sum(
        n_rows * math.prod(shape) * dtype.itemsize for _, dtype, shape in specs
    )
    last_err: Optional[EngramNumaError] = None
    for node in _spill_numa_failover_order(preferred, nbytes=nbytes):
        try:
            hosts, mms = _alloc_spill_rows(specs, n_rows=n_rows, node=node)
        except EngramNumaError as e:
            last_err = e
            logger.warning(
                "DSV4.1 spill rows L%s: placing on node %d failed (%s); trying the next node",
                layer_id,
                node,
                e,
            )
            continue
        if node != preferred:
            logger.warning(
                "DSV4.1 spill rows L%s: on node %d instead of stripe node %d",
                layer_id,
                node,
                preferred,
            )
        return hosts, mms, node
    raise EngramNumaError(
        f"spill rows L{layer_id}: no NUMA node took {nbytes} bytes"
    ) from last_err


@contextmanager
def _gc_scans_only_new_objects():
    """Limit gc.collect() to objects created inside the block.

    Repack and pin call gc.collect() several times per layer; with the loaded
    model on the heap each full pass cost ~0.4 s (about half of the per-layer
    time, py-spy). Tensors are still freed by refcount.
    """
    gc.freeze()
    try:
        yield
    finally:
        gc.unfreeze()


def repack_spill_host_for_sm70_marlin(moe: nn.Module) -> None:
    """Pack create-time host rows the same way GPU weights were packed.

    Host buffers are checkpoint layout; SM70 Marlin rewrites GPU tensors in
    ``process_weights_after_loading``. LRU swaps require matching layouts.
    """
    hosts: Optional[Dict[str, torch.Tensor]] = getattr(moe, "_dsv41_spill_host", None)
    if not hosts or "w13_weight" not in hosts:
        return
    if getattr(moe, "_dsv41_spill_host_packed", False):
        return
    _rss_probe(moe, "repack-enter")
    gpu = moe.w13_weight
    host_w13 = hosts["w13_weight"]
    host_trail = tuple(host_w13.shape[1:])
    gpu_trail = tuple(gpu.shape[1:])
    gpu_scale = getattr(moe, "w13_weight_scale", None)
    if spill_host_is_marlin_packed(hosts, gpu, gpu_scale):
        logger.info(
            "DSV4.1 spill host already Marlin-packed w13 %s %s scale %s %s",
            host_w13.dtype,
            host_trail,
            hosts["w13_weight_scale"].dtype,
            tuple(hosts["w13_weight_scale"].shape[1:]),
        )
        moe._dsv41_spill_host_packed = True  # type: ignore[attr-defined]
        return
    logger.info(
        "DSV4.1 spill host packing w13 host=%s %s gpu=%s %s host_scale_keys=%s",
        host_w13.dtype,
        host_trail,
        gpu.dtype,
        gpu_trail,
        sorted(k for k in hosts if "scale" in k),
    )
    dummy = nn.Module()
    dummy.orig_dtype = torch.float16
    # Host rows stay on CPU. Do not park GPU w13: restoring it OOMed
    # (384 MiB) on a full card. SM70 pack streams one expert at a time and
    # writes the packed E-stack on CPU when the dummy is not CUDA.
    dummy.w13_weight = nn.Parameter(hosts["w13_weight"].contiguous(), requires_grad=False)
    dummy.w2_weight = nn.Parameter(hosts["w2_weight"].contiguous(), requires_grad=False)
    raw_s13 = hosts.get("w13_weight_scale")
    raw_s2 = hosts.get("w2_weight_scale")
    if raw_s13 is None or raw_s2 is None:
        raise RuntimeError(
            "DSV4.1 spill host pack needs w13/w2 scales "
            f"(keys={sorted(hosts)})"
        )
    # NVFP4 checkpoints ship raw E4M3 block scales plus per-expert fp32
    # weight_scale_2; MXFP4 ships ue8m0 under the *_scale_inv name. Dispatch
    # on the block-scale dtype so both families pack through their own
    # SM70 helper.
    is_nvfp4 = raw_s13.dtype == torch.float8_e4m3fn
    if is_nvfp4:
        dummy.w13_weight_scale = nn.Parameter(
            raw_s13.contiguous(), requires_grad=False
        )
        dummy.w2_weight_scale = nn.Parameter(raw_s2.contiguous(), requires_grad=False)
        ckpt_s13_2 = hosts.get("w13_weight_scale_2")
        ckpt_s2_2 = hosts.get("w2_weight_scale_2")
        if ckpt_s13_2 is None or ckpt_s2_2 is None:
            raise RuntimeError(
                "DSV4.1 NVFP4 spill host pack needs weight_scale_2 rows "
                f"(keys={sorted(hosts)})"
            )
        dummy.w13_weight_scale_2 = nn.Parameter(
            ckpt_s13_2.contiguous(), requires_grad=False
        )
        dummy.w2_weight_scale_2 = nn.Parameter(
            ckpt_s2_2.contiguous(), requires_grad=False
        )
        from sglang.srt.layers.quantization.modelopt_quant import (
            prepare_moe_nvfp4_layer_for_sm70_marlin,
        )

        pack_fn = prepare_moe_nvfp4_layer_for_sm70_marlin
    else:
        s13 = hosts.get("w13_weight_scale_inv", raw_s13)
        s2 = hosts.get("w2_weight_scale_inv", raw_s2)
        dummy.w13_weight_scale_inv = nn.Parameter(
            s13.contiguous(), requires_grad=False
        )
        dummy.w2_weight_scale_inv = nn.Parameter(s2.contiguous(), requires_grad=False)
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            _prepare_moe_mxfp4_layer_for_sm70_marlin,
        )

        pack_fn = _prepare_moe_mxfp4_layer_for_sm70_marlin

    if torch.cuda.is_available():
        free_b, _total_b = torch.cuda.mem_get_info()
        logger.info(
            "DSV4.1 spill host pack GPU free=%.1f MiB before dummy pack (%s)",
            free_b / (1024**2),
            "nvfp4" if is_nvfp4 else "mxfp4",
        )
    try:
        pack_fn(dummy)
        _rss_probe(moe, "pack-fn-done")
        packed: Dict[str, torch.Tensor] = {}
        for attr in (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_scale2",
            "w2_scale2",
            "w13_weight_bias",
            "w2_weight_bias",
        ):
            t = getattr(dummy, attr, None)
            if t is None:
                continue
            packed[attr] = t.detach().to("cpu").contiguous()
            del t
        moe._dsv41_spill_host = packed  # type: ignore[attr-defined]
        moe._dsv41_spill_host_packed = True  # type: ignore[attr-defined]
        _rss_probe(moe, "repack-done")
        logger.info(
            "DSV4.1 spill host Marlin-packed w13 %s %s -> %s %s",
            host_w13.dtype,
            host_trail,
            packed["w13_weight"].dtype,
            tuple(packed["w13_weight"].shape[1:]),
        )
    finally:
        if getattr(dummy, "workspace", None) is not None:
            del dummy.workspace
        del dummy
        gc.collect()
        torch.cuda.empty_cache()


# cudaHostRegisterMapped: pins for DMA *and* maps into the CUDA address space
# so a later in-graph UVA page-in can read the mirror directly.
_CUDA_HOST_REGISTER_MAPPED = 0x02


def _spill_numa_nodes() -> List[int]:
    nodes: List[int] = []
    for s in envs.SGLANG_DSV41_EXPERT_SPILL_NUMA_NODES.get():
        try:
            nodes.append(int(s))
        except ValueError:
            logger.warning("SGLANG_DSV41_EXPERT_SPILL_NUMA_NODES: ignoring %r", s)
    if nodes:
        return nodes
    # Unset: the GPU-local node first, then the other memory nodes.
    gpu = gpu_numa_node()
    return [gpu, *sorted(n for n in read_numa_nodes() if n != gpu)]


# Arbitrary margin: covers the reclaim watermarks and the other ranks
# placing the same layer at the same time.
_SPILL_NODE_HEADROOM_BYTES = 2 * GIB


def _spill_numa_failover_order(preferred: int, *, nbytes: int) -> List[int]:
    """The stripe node first, then the other configured nodes.

    Nodes without ``nbytes`` of free or page-cache memory go last: a bound
    allocation there reclaims and swaps on that node, and can end in an OOM
    kill, instead of failing.
    """
    ordered: List[int] = []
    for n in [preferred, *_spill_numa_nodes()]:
        if n not in ordered:
            ordered.append(n)

    def lacks_room(n: int) -> bool:
        node = n if n >= 0 else gpu_numa_node()
        return node_available_bytes(node) < nbytes + _SPILL_NODE_HEADROOM_BYTES

    return sorted(ordered, key=lacks_room)


def _pin_spill_hosts_on_node(
    hosts: Dict[str, torch.Tensor],
    node: int,
) -> tuple[Dict[str, torch.Tensor], List, int, int]:
    """THP-bind + cudaHostRegister every spill row onto ``node``.

    On ``EngramNumaError`` the already-registered pointers are unregistered
    and the mappings dropped so the caller can retry another node.
    """
    mms: List = []
    pinned: Dict[str, torch.Tensor] = {}
    registered_ptrs: List[int] = []
    total = 0
    registered = 0
    cudart = torch.cuda.cudart()
    bind_node = node if node >= 0 else gpu_numa_node()
    try:
        for attr, t in hosts.items():
            t = t.contiguous()
            nbytes = t.numel() * t.element_size()
            if nbytes == 0:
                pinned[attr] = t
                continue
            mm = mmap_numa_thp(nbytes, node=bind_node)
            mms.append(mm)
            p = torch.frombuffer(mm, dtype=t.dtype, count=t.numel()).view(t.shape)
            p.copy_(t)
            err = cudart.cudaHostRegister(
                p.data_ptr(), nbytes, _CUDA_HOST_REGISTER_MAPPED
            )
            if int(err) != 0:
                logger.warning(
                    "DSV4.1 spill mirror cudaHostRegister(%s, %d bytes) failed err=%s; "
                    "row stays NUMA-bound but pageable",
                    attr,
                    nbytes,
                    int(err),
                )
            else:
                registered += nbytes
                registered_ptrs.append(int(p.data_ptr()))
            total += nbytes
            pinned[attr] = p
    except Exception:
        for ptr in registered_ptrs:
            try:
                cudart.cudaHostUnregister(ptr)
            except Exception:
                pass
        del pinned
        del mms
        raise
    return pinned, mms, total, registered


def _rss_probe(moe: nn.Module, tag: str) -> None:
    if not envs.SGLANG_DSV41_EXPERT_SPILL_RSS_LOG.get():
        return
    n = getattr(moe, "_dsv41_spill_probe_n", 0)
    if n > 5:
        return
    moe._dsv41_spill_probe_n = n + 1  # type: ignore[attr-defined]
    with open("/proc/self/status") as _f:
        _status = {
            ln.split(":")[0]: ln.split(":")[1].strip()
            for ln in _f
            if ln.startswith(("VmRSS", "RssAnon", "RssFile"))
        }
    # VmRSS counts mmap'd checkpoint page cache; RssAnon is the process's own
    # resident memory (spill mirror, pinned host, activations).
    logger.info(
        "DSV4.1 spill RSS probe %s L%s: VmRSS=%s RssAnon=%s RssFile=%s",
        tag, n,
        _status.get("VmRSS"), _status.get("RssAnon"), _status.get("RssFile"),
    )


def _ref_census(moe: nn.Module, layer_ordinal: int) -> None:
    """TEMP (RAM bring-up): reconcile host RSS against tensor buckets.

    Big pageable storages reconcile exactly with the design residency, yet RSS
    climbs ~440 MiB/layer -- attribute the excess: dedup CPU storages by
    data_ptr into pinned / big-pageable / mid-pageable buckets and log totals
    plus the top mid-size shapes.
    """
    if not envs.SGLANG_DSV41_SPILL_REF_CENSUS.get():
        return
    if layer_ordinal > 8:
        return
    seen_ptrs = set()
    pin_total = page_big = page_mid = 0
    mid_shapes: Dict[tuple, int] = {}
    pin_shapes: Dict[tuple, int] = {}
    n_page_big = 0
    with open("/proc/self/status") as _f:
        _rss = next(
            ln.split(":")[1].strip() for ln in _f if ln.startswith("VmRSS")
        )
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, torch.Tensor) or obj.device.type != "cpu":
                continue
            nb = obj.numel() * obj.element_size()
            if nb < (1 << 20):
                continue
            ptr = int(obj.untyped_storage().data_ptr())
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            if obj.is_pinned():
                pin_total += nb
                k = (tuple(obj.shape), str(obj.dtype))
                pin_shapes[k] = pin_shapes.get(k, 0) + nb
            elif nb >= (64 << 20):
                page_big += nb
                n_page_big += 1
            else:
                page_mid += nb
                k = (tuple(obj.shape), str(obj.dtype))
                mid_shapes[k] = mid_shapes.get(k, 0) + nb
        except Exception:
            continue
    gib = 1024**3
    logger.info(
        "DSV4.1 mem census L%s: VmRSS=%s pinned=%.2fG page_big=%.2fG(%d) "
        "page_mid=%.2fG",
        layer_ordinal,
        _rss,
        pin_total / gib,
        page_big / gib,
        n_page_big,
        page_mid / gib,
    )
    top_pin = sorted(pin_shapes.items(), key=lambda kv: -kv[1])[:4]
    for (shape, dtype), nb in top_pin:
        logger.info(
            "DSV4.1 mem census L%s: pinned %d MiB %s %s",
            layer_ordinal,
            nb // (1024**2),
            shape,
            dtype,
        )
    top_mid = sorted(mid_shapes.items(), key=lambda kv: -kv[1])[:8]
    for (shape, dtype), nb in top_mid:
        logger.info(
            "DSV4.1 mem census L%s: mid %d MiB %s %s",
            layer_ordinal,
            nb // (1024**2),
            shape,
            dtype,
        )


def _register_spill_hosts_inplace(moe: nn.Module, layer_ordinal: int) -> bool:
    """Pin the packed host mirror via cudaHostRegister -- in place, no copy.

    The copy-based fallback below holds the full pageable mirror while the
    pinned copy is built. On the 4xV100 RAM bring-up that peak
    (pageable-full + pinned-growing, 4 TP ranks, no swap) crossed the OS OOM
    killer at SGLANG_DSV41_EXPERT_SPILL_GB=18 and 20. Registering the
    existing mapping adds no memory at all: the pages are already resident,
    they only become unevictable and UVA device-accessible -- exactly what
    the D4-G landing page-in kernel dereferences. The ctor rows are plain
    allocator blocks (mmap-backed, page-aligned); if any register call still
    fails, every registration is rolled back and the copy path runs as
    before.
    """
    hosts: Optional[Dict[str, torch.Tensor]] = getattr(moe, "_dsv41_spill_host", None)
    if not hosts or getattr(moe, "_dsv41_spill_host_pinned", False):
        return False
    _rss_probe(moe, "pin-register-enter")
    cudart = torch.cuda.cudart()
    page = os.sysconf("SC_PAGESIZE") if hasattr(os, "sysconf") else 4096
    mask = page - 1
    # The attrs share allocator pages (the tiny *_scale2 rows live inside the
    # w13/w2 row blocks), so per-tensor registration overlaps itself: the
    # second register of a shared page fails with cudaErrorHostMemoryAlreadyRegistered
    # and any later rollback unmaps a page a surviving span still "holds".
    # Union the page-aligned spans first and register each disjoint run once.
    # glibc places large blocks as back-to-back mmaps, so a naive page
    # rounding of a tensor's range reaches into the neighboring mapping
    # (the next layer's ctor rows, pack temporaries). Registering those
    # pages poisons them for their owner and the driver surfaces
    # cudaErrorHostMemoryAlreadyRegistered later, far from the cause. Clip
    # every span to the /proc/self/maps VMAs it actually lives in first.
    vmas: List[tuple] = []
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split()
                start_s, end_s = parts[0].split("-")
                vmas.append((int(start_s, 16), int(end_s, 16)))
        vmas.sort()
    except OSError:
        vmas = []
    if not vmas:
        logger.warning(
            "DSV4.1 spill mirror L%s: /proc/self/maps unreadable; "
            "falling back to copy pin",
            layer_ordinal,
        )
        return _plain_pin_spill_hosts(moe, layer_ordinal)
    # A tensor is only safe to register if its padded page span exactly owns
    # its backing VMA (glibc serves large blocks as dedicated mmaps, header
    # included). Small scale rows live in the shared heap arena: registering
    # arena pages pins foreign allocations, and once glibc trims/frees them
    # the driver's registration goes stale -- the next cudaHostAlloc dies
    # with cudaErrorHostMemoryAlreadyRegistered far from the cause. Migrate
    # those attrs into private anonymous mmaps first (same pattern the ctor
    # path uses for NUMA placement); the page-in kernel UVA-reads every
    # attr, so each must end up registered.
    mms: List = []
    raw_spans: List[tuple] = []
    for attr, t in hosts.items():
        nbytes = t.numel() * t.element_size()
        if nbytes == 0:
            continue
        if not t.is_contiguous():
            logger.warning(
                "DSV4.1 spill mirror %s not contiguous; falling back to copy pin",
                attr,
            )
            return _plain_pin_spill_hosts(moe, layer_ordinal)
        lo = int(t.data_ptr()) & ~mask
        hi = (int(t.data_ptr()) + nbytes + mask) & ~mask
        owner = next(
            ((s, e) for s, e in vmas if s <= lo and hi <= e), None
        )
        if owner is None:
            logger.warning(
                "DSV4.1 spill mirror %s spans no single VMA; falling back "
                "to copy pin",
                attr,
            )
            # Already-migrated attrs view private mmaps: keep the mappings
            # alive for the copy path to swap out (closing here would leave
            # the hosts dict reading unmapped memory).
            moe._dsv41_spill_host_mms = mms  # type: ignore[attr-defined]
            return _plain_pin_spill_hosts(moe, layer_ordinal)
        if owner != (lo, hi):
            import mmap as _mmap_mod

            mm_obj = _mmap_mod.mmap(-1, nbytes)
            moved = torch.frombuffer(mm_obj, dtype=t.dtype, count=t.numel()).view(
                t.shape
            )
            moved.copy_(t)
            hosts[attr] = moved
            mms.append(mm_obj)
            lo = int(moved.data_ptr()) & ~mask
            hi = (int(moved.data_ptr()) + nbytes + mask) & ~mask
        raw_spans.append((lo, hi))
    if not raw_spans:
        return _plain_pin_spill_hosts(moe, layer_ordinal)
    merged: List[List[int]] = []
    for base, end in sorted(raw_spans):
        if merged and base <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([base, end])
    registered_spans: List[tuple] = []
    for base, end in merged:
        err = int(
            cudart.cudaHostRegister(base, end - base, _CUDA_HOST_REGISTER_MAPPED)
        )
        if err != 0:
            logger.warning(
                "DSV4.1 spill mirror in-place cudaHostRegister(%d bytes) "
                "failed err=%d; rolling back",
                end - base,
                err,
            )
            for rb, re_ in reversed(registered_spans):
                try:
                    cudart.cudaHostUnregister(rb)
                except Exception:
                    pass
            moe._dsv41_spill_host_mms = mms  # type: ignore[attr-defined]
            return _plain_pin_spill_hosts(moe, layer_ordinal)
        registered_spans.append((base, end))
    registered = sum(end - base for base, end in registered_spans)
    moe._dsv41_spill_host_mms = mms  # type: ignore[attr-defined]
    moe._dsv41_spill_host_pinned = True  # type: ignore[attr-defined]
    moe._dsv41_spill_host_node = None  # type: ignore[attr-defined]
    # The mirror rows were registered in place, so nothing was replaced: any
    # construct-time mappings they view must stay alive (same rule as the
    # pageable path).
    gc.collect()
    _rss_probe(moe, "pin-register-done")
    logger.info(
        "DSV4.1 spill mirror L%s: %.0f MiB pinned in place via "
        "cudaHostRegister (no copy)",
        layer_ordinal,
        registered / (1024**2),
    )
    return True


def _plain_pin_spill_hosts(moe: nn.Module, layer_ordinal: int) -> bool:
    """Pin the packed host mirror with the CUDA allocator (no NUMA placement).

    Pinned memory is UVA device-accessible, which the D4-G landing page-in
    kernel requires: it dereferences the host rows' VAs directly. A pageable
    mirror faults there, so this is the mandatory fallback when
    set_mempolicy/mbind is unavailable (containers without CAP_SYS_NICE).
    """
    hosts: Optional[Dict[str, torch.Tensor]] = getattr(moe, "_dsv41_spill_host", None)
    if not hosts or getattr(moe, "_dsv41_spill_host_pinned", False):
        return False
    _rss_probe(moe, "pin-enter")
    # The mirror was packed through ~42 layers of construction transients; the
    # default allocator retains a multi-GiB heap high-water mark that a fresh
    # pinned allocation of the same size cannot reuse. Hand it back before
    # doubling the mirror's footprint -- measured +5-7 GiB/rank resident at
    # this point on the 4xV100 RAM bring-up (no swap: it is all unevictable).
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    total = 0
    for attr, host in list(hosts.items()):
        # Swap per tensor and drop the pageable source immediately: on a
        # RAM-tight host (4 TP ranks pinning ~GB-scale mirrors) the original
        # all-at-once copy peaked at 2x the mirror set and the kernel
        # OOM-killed the container.
        pinned = host.pin_memory()
        total += pinned.numel() * pinned.element_size()
        hosts[attr] = pinned
        del host
    _rss_probe(moe, "pin-swapped")
    moe._dsv41_spill_host = hosts  # type: ignore[attr-defined]
    moe._dsv41_spill_host_mms = []  # type: ignore[attr-defined]
    moe._dsv41_spill_host_pinned = True  # type: ignore[attr-defined]
    moe._dsv41_spill_host_node = None  # type: ignore[attr-defined]
    # Construct-time checkpoint-layout mappings: the pinned copies replaced
    # their tensors, so dropping the mmap handles lets them unmap now.
    moe._dsv41_spill_host_ctor_mms = None  # type: ignore[attr-defined]
    gc.collect()
    _ref_census(moe, layer_ordinal)
    logger.info(
        "DSV4.1 spill mirror L%s: %.0f MiB pinned (plain CUDA allocator, "
        "no NUMA placement)",
        layer_ordinal,
        total / (1024**2),
    )
    return True


def pin_spill_host_numa(moe: nn.Module, layer_ordinal: int) -> Optional[int]:
    """Move the packed host mirror into node-local THP mappings and
    ``cudaHostRegister`` them (mapped).

    ``repack_spill_host_for_sm70_marlin`` leaves the rows as pageable tensors
    from the default allocator; under memory pressure those were paged out
    and every LRU miss became a swap-in through the RAID. Layers are striped over the configured nodes so the mirror
    spreads across sockets. Spill=12 can exhaust the GPU-local node's THP
    remainder (1G hugepages already hold Engram); retry the other stripe
    node instead of aborting (explicit overflow, not silent UPI). Returns
    the node used, or None if skipped.
    """
    hosts: Optional[Dict[str, torch.Tensor]] = getattr(moe, "_dsv41_spill_host", None)
    if not hosts or not torch.cuda.is_available():
        return None
    if getattr(moe, "_dsv41_spill_host_pinned", False):
        return getattr(moe, "_dsv41_spill_host_node", None)
    if spill_landing_slots() <= 0:
        # Without the landing pool nothing reads the mirror through UVA: the
        # D4-G page-in kernel is the only device-side consumer and D4-H's host
        # GEMV reads the rows on the CPU. A pageable mirror is enough, and on
        # hosts where a fresh pinned allocation bills 2x its size in RSS
        # (measured on the 4xV100 RAM bring-up) pinning a full 19 GiB/rank
        # mirror would double the footprint and lose to the OOM killer. The
        # LRU then swaps through pageable staging (slower swaps, same results).
        moe._dsv41_spill_host_pinned = False  # type: ignore[attr-defined]
        moe._dsv41_spill_host_node = None  # type: ignore[attr-defined]
        # NOTE: ctor mms are left alone -- the mirror tensors still view that
        # memory (no replacement happened here), so the handles must outlive
        # them.
        gc.collect()
        _rss_probe(moe, "pin-skipped-pageable")
        logger.info(
            "DSV4.1 spill mirror L%s: kept pageable (%.0f MiB, landing pool "
            "disabled -> Python LRU swaps)",
            layer_ordinal,
            sum(h.numel() * h.element_size() for h in hosts.values()) / (1024**2),
        )
        return None
    if not bool(envs.SGLANG_ENABLE_DSV41_EXPERT_SPILL_NUMA.get()):
        if _register_spill_hosts_inplace(moe, layer_ordinal):
            return None
        _plain_pin_spill_hosts(moe, layer_ordinal)
        return None
    preferred = getattr(moe, "_dsv41_spill_numa_node", None)
    if preferred is None:
        nodes = _spill_numa_nodes()
        preferred = nodes[layer_ordinal % len(nodes)]
    last_err: Optional[BaseException] = None
    pinned: Optional[Dict[str, torch.Tensor]] = None
    mms: List = []
    total = 0
    registered = 0
    node = preferred
    nbytes = sum(t.numel() * t.element_size() for t in hosts.values())
    for node in _spill_numa_failover_order(int(preferred), nbytes=nbytes):
        try:
            pinned, mms, total, registered = _pin_spill_hosts_on_node(hosts, node)
            if node != preferred:
                logger.warning(
                    "DSV4.1 spill mirror L%s: pinned on node %d instead of node %d",
                    layer_ordinal,
                    node,
                    preferred,
                )
            last_err = None
            break
        except EngramNumaError as e:
            last_err = e
            logger.warning(
                "DSV4.1 spill mirror L%s: placing on node %d failed (%s); trying the next node",
                layer_ordinal,
                node,
                e,
            )
    if last_err is not None or pinned is None:
        # Containers without CAP_SYS_NICE cannot set_mempolicy(BIND). The
        # mirror still must be pinned: the D4-G landing page-in kernel reads
        # host rows through UVA, which pageable memory cannot serve.
        _plain_pin_spill_hosts(moe, layer_ordinal)
        return None
    moe._dsv41_spill_host = pinned  # type: ignore[attr-defined]
    moe._dsv41_spill_host_mms = mms  # type: ignore[attr-defined]
    moe._dsv41_spill_host_pinned = registered == total  # type: ignore[attr-defined]
    moe._dsv41_spill_host_node = node  # type: ignore[attr-defined]
    del hosts
    # Construct-time checkpoint-layout mappings: the repack replaced their
    # tensors, so dropping the mmap handles lets them unmap now (same node
    # as the pinned rows -> footprint flat, not doubled).
    moe._dsv41_spill_host_ctor_mms = None  # type: ignore[attr-defined]
    gc.collect()
    logger.info(
        "DSV4.1 spill mirror L%s: %.0f MiB on NUMA node %d, pinned+mapped %.0f MiB",
        layer_ordinal,
        total / (1024**2),
        node,
        registered / (1024**2),
    )
    return node


_COLD_SET_CACHE: Dict[str, Optional[torch.Tensor]] = {}


def _load_cold_set_table(path: str) -> Optional[torch.Tensor]:
    """``cold_ids`` int64 [layers, ep, S], coldest first, from JSON or a torch file. Cached per path."""
    if path in _COLD_SET_CACHE:
        return _COLD_SET_CACHE[path]
    table: Optional[torch.Tensor] = None
    try:
        if path.endswith(".json"):
            # The shipped table is JSON: plain data, nothing to unpickle.
            with open(path) as f:
                obj = json.load(f)
        else:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        t = obj["cold_ids"] if isinstance(obj, dict) else obj
        table = torch.as_tensor(t, dtype=torch.int64)
        if table.ndim != 3:
            raise ValueError(f"cold_ids must be [layers, ep, S], got {tuple(table.shape)}")
        logger.info(
            "DSV4.1 spill cold set %s: layers=%d ep=%d S=%d source=%s",
            path,
            *table.shape,
            (obj.get("source") if isinstance(obj, dict) else None),
        )
    except Exception as e:  # noqa: BLE001 - fall back to tail placement, loudly
        logger.error("DSV4.1 spill cold set %s unusable (%s); using tail placement", path, e)
        table = None
    _COLD_SET_CACHE[path] = table
    return table


def spill_placement(moe: nn.Module) -> tuple[List[int], List[int]]:
    """(kept_ids, cold_ids) local routed ids for this (layer, ep rank).

    Host row of a cold expert = its index in ``cold_ids``; GPU slot of a kept
    expert = its index in ``kept_ids``; shared experts follow the kept rows.
    Without a cold-set table this is today's placement: cold = tail
    ``[n_kept_routed, n_routed)``, kept = ``[0, n_kept_routed)``.
    """
    cached = getattr(moe, "_dsv41_spill_placement", None)
    # Loader threads hit this concurrently (deepseek_v4 load_weights).
    # Placement is the publish flag: only return it once both slot maps exist.
    if (
        cached is not None
        and getattr(moe, "_dsv41_spill_kept_slot", None) is not None
        and getattr(moe, "_dsv41_spill_host_slot", None) is not None
    ):
        return cached
    plan: RoutedExpertSpillPlan = moe._dsv41_expert_spill_plan  # type: ignore[attr-defined]
    n_routed = int(getattr(moe, "_num_local_routed", 0)) or plan.local_routed
    cold: Optional[List[int]] = None
    path = envs.SGLANG_DSV41_EXPERT_SPILL_COLD_SET.get()
    if path and plan.n_spilled > 0:
        table = _load_cold_set_table(path)
        # Rows are indexed by model layer_id (same space as the upstream
        # expert-distribution recorder); layers before first_k_dense_replace
        # are dense and never consulted.
        layer = int(getattr(moe, "layer_id", -1))
        rank = int(getattr(moe, "moe_ep_rank", 0))
        if table is not None and 0 <= layer < table.shape[0] and rank < table.shape[1]:
            ids: List[int] = []
            for x in table[layer, rank].tolist():
                if 0 <= x < n_routed and x not in ids:
                    ids.append(int(x))
                if len(ids) == plan.n_spilled:
                    break
            if len(ids) == plan.n_spilled:
                cold = sorted(ids)
            else:
                logger.warning(
                    "DSV4.1 spill cold set L%d r%d has %d usable ids < n_spilled %d; tail placement",
                    layer, rank, len(ids), plan.n_spilled,
                )
        elif table is not None:
            logger.warning(
                "DSV4.1 spill cold set has no entry for layer %d rank %d; tail placement",
                layer, rank,
            )
    moe._dsv41_spill_cold_source = "table" if cold is not None else "tail"  # type: ignore[attr-defined]
    if cold is None:
        cold = list(range(plan.n_kept_routed, n_routed))
    cold_set = set(cold)
    kept = [i for i in range(n_routed) if i not in cold_set]
    placement = (kept, cold)
    moe._dsv41_spill_kept_slot = {e: s for s, e in enumerate(kept)}  # type: ignore[attr-defined]
    moe._dsv41_spill_host_slot = {e: s for s, e in enumerate(cold)}  # type: ignore[attr-defined]
    moe._dsv41_spill_placement = placement  # type: ignore[attr-defined]
    return placement


def spilled_expert_host_row(
    moe: nn.Module, param: torch.Tensor, expert_id: int
) -> Optional[tuple[torch.Tensor, int]]:
    """If this local expert is spilled, return (host_tensor, host_row)."""
    plan: Optional[RoutedExpertSpillPlan] = getattr(moe, "_dsv41_expert_spill_plan", None)
    hosts: Optional[Dict[str, torch.Tensor]] = getattr(moe, "_dsv41_spill_host", None)
    if plan is None or not hosts or plan.n_spilled <= 0:
        return None
    n_routed = int(getattr(moe, "_num_local_routed", 0))
    if expert_id < 0 or expert_id >= n_routed:
        return None
    spill_placement(moe)
    host_row = moe._dsv41_spill_host_slot.get(expert_id)  # type: ignore[attr-defined]
    if host_row is None:
        return None
    src_ptr = param.data_ptr()
    for attr, host in hosts.items():
        gp = getattr(moe, attr, None)
        if gp is not None and gp.data_ptr() == src_ptr:
            return host, host_row
    return None


def remap_shared_expert_gpu_index(moe: nn.Module, expert_id: int) -> int:
    """GPU slot of a non-spilled local expert when GPU tensors are pre-shrunk.

    Kept routed experts sit at their index in ``kept_ids`` (identity for the
    tail placement); shared slots follow the kept rows.
    """
    plan: Optional[RoutedExpertSpillPlan] = getattr(moe, "_dsv41_expert_spill_plan", None)
    n_routed = int(getattr(moe, "_num_local_routed", 0))
    if plan is None or plan.n_spilled <= 0:
        return expert_id
    if expert_id >= n_routed:
        return plan.n_kept_routed + (expert_id - n_routed)
    spill_placement(moe)
    slot = moe._dsv41_spill_kept_slot.get(expert_id)  # type: ignore[attr-defined]
    if slot is None:
        raise KeyError(f"local expert {expert_id} is spilled; no GPU slot")
    return slot


class RoutedExpertLru:
    """Row-wise LRU over expert-dim tensors, spilling cold routed rows.

    Shared expert rows are the tail ``n_shared`` and never leave the device.
    ``ensure(ids)`` copies spilled rows into GPU slots, evicting the LRU hot
    routed expert onto the pinned host copy. If ``unique(ids)`` exceeds
    ``n_kept_routed``, last-resort eviction thrashes within the batch (the
    last ensured id is resident; earlier ones may have been evicted). Optional
    ``siblings`` (w2, scales) share the same slot map so Marlin w13/w2 stay
    aligned.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        n_shared: int = 1,
        n_spilled: int,
        pin_memory: bool = False,
        siblings: Optional[Sequence[torch.Tensor]] = None,
        hosts: Optional[Sequence[torch.Tensor]] = None,
        logical_n_experts: Optional[int] = None,
        already_shrunk: bool = False,
        kept_ids: Optional[Sequence[int]] = None,
        cold_ids: Optional[Sequence[int]] = None,
    ):
        if weight.ndim < 1:
            raise ValueError("expert weight must have an expert dimension")
        n_experts = int(logical_n_experts or weight.shape[0])
        if n_shared < 0 or n_spilled < 0:
            raise ValueError("n_shared and n_spilled must be >= 0")
        if n_shared + n_spilled > n_experts:
            raise ValueError(
                f"n_shared={n_shared} + n_spilled={n_spilled} > n_experts={n_experts}"
            )
        extra = list(siblings or ())
        gpu_rows = n_experts - n_spilled if already_shrunk else n_experts
        for t in extra:
            if int(t.shape[0]) != int(weight.shape[0]):
                raise ValueError(
                    f"sibling expert dim {t.shape[0]} != primary {weight.shape[0]}"
                )
        if already_shrunk and int(weight.shape[0]) != gpu_rows:
            raise ValueError(
                f"already-shrunk GPU dim {weight.shape[0]} != kept+shared {gpu_rows}"
            )
        self.n_experts = n_experts
        self.n_shared = n_shared
        self.n_routed = n_experts - n_shared
        self.n_spilled = n_spilled
        self.n_kept_routed = self.n_routed - n_spilled
        self._pin_memory = pin_memory
        # Which local routed ids start on host. Default = tail.
        if cold_ids is None:
            cold_list = list(range(self.n_kept_routed, self.n_routed))
        else:
            cold_list = sorted(int(i) for i in cold_ids)
        if len(cold_list) != n_spilled or any(
            i < 0 or i >= self.n_routed for i in cold_list
        ) or len(set(cold_list)) != len(cold_list):
            raise ValueError(f"cold_ids must be {n_spilled} distinct routed ids: {cold_list}")
        cold_set = set(cold_list)
        if kept_ids is None:
            kept_list = [i for i in range(self.n_routed) if i not in cold_set]
        else:
            kept_list = [int(i) for i in kept_ids]
            if len(kept_list) != self.n_kept_routed or cold_set.intersection(kept_list):
                raise ValueError("kept_ids must be the routed ids not in cold_ids")
        self._kept_ids = kept_list
        self._cold_ids = cold_list
        self._cold_set = cold_set  # membership: "its home is the host mirror"
        self._gpus: List[torch.Tensor] = [weight, *extra]
        if hosts is not None:
            host_list = list(hosts)
            if len(host_list) != len(self._gpus):
                raise ValueError(
                    f"hosts len {len(host_list)} != gpu tensors {len(self._gpus)}"
                )
            self._hosts = host_list
        else:
            self._hosts = [
                self._alloc_host(t, pin_memory=pin_memory) for t in self._gpus
            ]
        self._host_mm = getattr(self, "_host_mm", None)
        # gpu_slot -> expert id for routed slots [0, n_kept_routed)
        self._slot_to_id = list(kept_list)
        self._id_to_slot = {e: s for s, e in enumerate(kept_list)}
        self._host_slot_to_id = list(cold_list)
        self._id_to_host_slot = {e: s for s, e in enumerate(cold_list)}
        self._lru: OrderedDict[int, None] = OrderedDict(
            (i, None) for i in range(self.n_kept_routed)
        )
        self.applied = already_shrunk
        self._staging: Optional[List[torch.Tensor]] = None
        self._map_table: Optional[torch.Tensor] = None
        self._host_map_table: Optional[torch.Tensor] = None
        self.n_swaps = 0
        # Last-resort evictions when unique(batch) > n_kept_routed (intra-batch thrash).
        self.n_thrash = 0
        self._swap_row_bytes: Optional[int] = None

    def _alloc_host(self, weight: torch.Tensor, *, pin_memory: bool) -> torch.Tensor:
        """Host holds only spilled routed rows (logical n_kept .. n_routed-1)."""
        device = weight.device
        n_host = self.n_spilled
        row_shape = weight.shape[1:]
        row_numel = int(weight.reshape(weight.shape[0], -1).shape[1]) if n_host else 0
        host_bytes = n_host * row_numel * weight.element_size()
        use_numa = (
            pin_memory
            and device.type == "cuda"
            and bool(envs.SGLANG_ENABLE_DSV41_EXPERT_SPILL_NUMA.get())
            and host_bytes > 0
        )
        if use_numa:
            node = engram_numa_node()
            if node < 0:
                node = gpu_numa_node()
            try:
                host_mm = mmap_numa_thp(host_bytes, node=node)
            except EngramNumaError:
                # No CAP_SYS_NICE (containers): plain allocator, no pin.
                logger.warning(
                    "DSV4.1 LRU host NUMA bind failed; using pageable rows"
                )
                host_mm = None
            if host_mm is not None:
                if getattr(self, "_host_mms", None) is None:
                    self._host_mms = []
                self._host_mms.append(host_mm)
                host = torch.frombuffer(
                    host_mm, dtype=weight.dtype, count=n_host * row_numel
                ).view(n_host, *row_shape)
                err = torch.cuda.cudart().cudaHostRegister(
                    host.data_ptr(), host_bytes, _CUDA_HOST_REGISTER_MAPPED
                )
                if int(err) != 0:
                    logger.warning(
                        "cudaHostRegister expert-spill %s bytes failed err=%s; "
                        "NUMA bind still holds",
                        host_bytes,
                        int(err),
                    )
            else:
                host = torch.empty(
                    n_host,
                    *row_shape,
                    dtype=weight.dtype,
                    device="cpu",
                    pin_memory=pin_memory and device.type == "cuda",
                )
        else:
            host = torch.empty(
                n_host,
                *row_shape,
                dtype=weight.dtype,
                device="cpu",
                pin_memory=pin_memory and device.type == "cuda",
            )
        if n_host and weight.shape[0] >= self.n_routed:
            host.copy_(weight[self._cold_ids].detach().to("cpu"))
        return host

    @property
    def _gpu(self) -> torch.Tensor:
        return self._gpus[0]

    @property
    def _host(self) -> torch.Tensor:
        return self._hosts[0]

    @property
    def gpu_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._gpus)

    def swap_row_bytes(self) -> int:
        """Per-swap bytes moved in each direction (one host row, all attrs)."""
        if self._swap_row_bytes is None:
            self._swap_row_bytes = sum(
                int(h[0].numel()) * int(h.element_size()) for h in self._hosts
            )
        return self._swap_row_bytes

    def _shrink_one(self, tensor: torch.Tensor) -> torch.Tensor:
        kept = tensor[self._kept_ids].clone()
        if not self.n_shared:
            return kept
        shared = tensor[self.n_routed :].clone()
        return torch.cat([kept, shared], dim=0)

    def apply_shrink(self) -> torch.Tensor:
        """Drop spilled routed rows from every GPU tensor; shared tail is kept."""
        if self.applied:
            return self._gpus[0]
        self._gpus = [self._shrink_one(t) for t in self._gpus]
        self.applied = True
        return self._gpus[0]

    def shrunk_tensors(self) -> List[torch.Tensor]:
        if not self.applied:
            raise RuntimeError("apply_shrink() first")
        return list(self._gpus)

    def physical_ids(self, logical_ids: Iterable[int]) -> list[int]:
        """Map logical expert ids onto the shrunken GPU table (after apply_shrink)."""
        out = []
        for i in logical_ids:
            i = int(i)
            if i < 0 or i >= self.n_experts:
                raise IndexError(i)
            if i >= self.n_routed:
                # shared tail sits after kept routed rows
                out.append(self.n_kept_routed + (i - self.n_routed))
                continue
            slot = self._id_to_slot.get(i)
            if slot is None:
                raise KeyError(
                    f"expert {i} is spilled; call ensure() before the MoE runner"
                )
            out.append(slot)
        return out

    def _ensure_device_tables(self, like: torch.Tensor) -> None:
        """GPU map_table (logical -> slot) and host_map (logical -> host row)."""
        device = like.device
        dtype = torch.int32
        need = (
            self._map_table is None
            or self._host_map_table is None
            or self._map_table.device != device
            or self._map_table.dtype != dtype
            or int(self._map_table.numel()) != self.n_experts
        )
        if not need:
            return
        cpu_map = torch.full((self.n_experts,), -1, dtype=dtype)
        cpu_host = torch.full((self.n_experts,), -1, dtype=dtype)
        for logical, slot in self._id_to_slot.items():
            cpu_map[logical] = int(slot)
        for logical, hs in self._id_to_host_slot.items():
            cpu_host[logical] = int(hs)
        if self.n_shared:
            for i in range(self.n_shared):
                cpu_map[self.n_routed + i] = self.n_kept_routed + i
        self._map_table = cpu_map.to(device=device)
        self._host_map_table = cpu_host.to(device=device)

    def map_ids(self, logical_ids: torch.Tensor) -> torch.Tensor:
        """Vectorized ``physical_ids``; negative ids (EP remote) pass through."""
        self._ensure_device_tables(logical_ids)
        table = self._map_table
        assert table is not None
        n = int(table.shape[0])
        idx = logical_ids.clamp(min=0, max=max(n - 1, 0)).to(torch.int64)
        mapped = table[idx].to(dtype=logical_ids.dtype)
        return torch.where(logical_ids < 0, logical_ids, mapped)

    def _ensure_staging(self) -> List[torch.Tensor]:
        if self._staging is not None:
            return self._staging
        pin = bool(self._pin_memory) and any(t.is_cuda for t in self._gpus)
        self._staging = [
            torch.empty(
                host.shape[1:],
                dtype=host.dtype,
                device="cpu",
                pin_memory=pin,
            )
            for host in self._hosts
        ]
        return self._staging

    def _swap_row(
        self,
        gpu: torch.Tensor,
        host: torch.Tensor,
        staging: torch.Tensor,
        *,
        victim_slot: int,
        victim_id: int,
        logical: int,
    ) -> None:
        hs = self._id_to_host_slot[logical]
        incoming = host[hs]
        victim = gpu[victim_slot]
        if tuple(incoming.shape) != tuple(victim.shape) or incoming.dtype != victim.dtype:
            raise RuntimeError(
                "DSV4.1 spill LRU swap layout mismatch: "
                f"host{tuple(incoming.shape)} {incoming.dtype} vs "
                f"gpu{tuple(victim.shape)} {victim.dtype}. "
                "Host pack must run after GPU process_weights_after_loading."
            )
        # Pinned/host-registered copy_. Do not clone().to() — that is pageable
        # HtoD and stalls the CPU while NCCL waits (decode profile: 9k copies).
        staging.copy_(victim)
        victim.copy_(incoming)
        incoming.copy_(staging)
        if not self.applied and gpu.shape[0] > logical:
            gpu[logical].copy_(gpu[victim_slot])

    # Residents touched within this many most-recent slots are never chosen as
    # a cold-first victim (a cold expert that is hot *right now* should not
    # thrash against the next cold miss); plain LRU decides then.
    _MRU_PROTECT = 8

    def _pick_victim(self, needed: set, current: Optional[int] = None) -> int:
        """Slot to evict. Among residents outside the MRU window,
        prefer one whose home is the host mirror (a cold expert brought in
        earlier) over a kept expert, so the frequency-hot set stays on the
        GPU; otherwise the oldest resident. Never a slot needed by a
        currently-resident batch id while a non-needed occupant exists.

        Last resort (unique(batch) > n_slots): evict the oldest LRU occupant
        that is not ``current``. The evicted id maps to -1 afterwards, so the
        MoE path splits such batches into slot-sized passes first
        (``run_moe_with_expert_spill``).
        """
        n = len(self._lru)
        protect = min(self._MRU_PROTECT, n // 2)
        fallback = None
        for idx, slot in enumerate(self._lru):  # oldest first
            occupant = self._slot_to_id[slot]
            if occupant in needed:
                continue
            if idx < n - protect and occupant in self._cold_set:
                return slot
            if fallback is None:
                fallback = slot
        if fallback is not None:
            return fallback
        for slot in self._lru:
            occupant = self._slot_to_id[slot]
            if occupant == current:
                continue
            return slot
        raise RuntimeError("every GPU expert slot is needed by this batch")

    def ensure(
        self,
        logical_ids: Iterable[int],
    ) -> None:
        """Make each routed id resident in a GPU slot, spilling LRU victims."""
        ids = [int(i) for i in logical_ids]
        batch = {i for i in ids if 0 <= i < self.n_routed}
        if not self.applied and self._kept_ids != list(range(self.n_kept_routed)):
            raise RuntimeError("non-tail spill placement requires apply_shrink() first")
        logged_thrash = False
        for logical in ids:
            if logical < 0:
                continue
            if logical >= self.n_experts:
                raise IndexError(logical)
            if logical >= self.n_routed:
                continue  # shared, always resident
            slot = self._id_to_slot.get(logical)
            if slot is not None:
                self._lru.move_to_end(slot)
                continue
            if self.n_kept_routed == 0:
                raise RuntimeError("no GPU slots to hold a routed expert")
            # Occupied slots, not future batch ids: an expert not yet loaded
            # does not hold a GPU row, so it must not pin a victim.
            needed = set(self._id_to_slot).intersection(batch)
            victim_slot = self._pick_victim(needed, current=logical)
            victim_id = self._slot_to_id[victim_slot]
            if victim_id in needed:
                self.n_thrash += 1
                if not logged_thrash:
                    logged_thrash = True
                    logger.info(
                        "DSV4.1 spill LRU intra-batch thrash: unique=%d slots=%d "
                        "(working set exceeds GPU expert slots)",
                        len(batch),
                        self.n_kept_routed,
                    )
            del self._lru[victim_slot]
            hs = self._id_to_host_slot.pop(logical)
            staging = self._ensure_staging()
            for gpu, host, st in zip(self._gpus, self._hosts, staging):
                self._swap_row(
                    gpu,
                    host,
                    st,
                    victim_slot=victim_slot,
                    victim_id=victim_id,
                    logical=logical,
                )
            self._id_to_host_slot[victim_id] = hs
            self._host_slot_to_id[hs] = victim_id
            del self._id_to_slot[victim_id]
            self._id_to_slot[logical] = victim_slot
            self._slot_to_id[victim_slot] = logical
            self._lru[victim_slot] = None
            self.n_swaps += 1
            table = self._map_table
            if table is not None:
                table[victim_id] = -1
                table[logical] = victim_slot
            host_table = self._host_map_table
            if host_table is not None:
                host_table[logical] = -1
                host_table[victim_id] = hs
    def gather_rows(self, logical_ids: Iterable[int]) -> torch.Tensor:
        self.ensure(logical_ids)
        ids = list(logical_ids)
        slots = self.physical_ids(ids) if self.applied else [
            self._id_to_slot[i] if i < self.n_routed else i for i in ids
        ]
        return self._gpus[0][slots]


def cpu_offload_gb_tax() -> str:
    return (
        "--cpu-offload-gb uses OffloaderV1, which walks decoder layers in order "
        "and moves whole module parameters (attention + MoE) to pinned host. "
        "Each forward does state_dict().to(device), so CSA2, the indexer, and "
        "shared experts pay a PCIe round trip even though they must stay GPU. "
        "It also races Engram host tables if combined "
        "(see handle_offload_compatibility for PLE). Prefer "
        "SGLANG_DSV41_EXPERT_SPILL_GB + this LRU."
    )


def _expert_dim_params(moe: nn.Module) -> Dict[str, torch.nn.Parameter]:
    n = int(getattr(moe, "num_local_experts", 0))
    gpu_n = int(getattr(moe, "_dsv41_gpu_expert_slots", n) or n)
    out: Dict[str, torch.nn.Parameter] = {}
    for attr in _EXPERT_PARAM_ATTRS:
        p = getattr(moe, attr, None)
        if p is None or not isinstance(p, torch.nn.Parameter) or p.ndim < 1:
            continue
        if int(p.shape[0]) in (n, gpu_n):
            out[attr] = p
    return out


# Per-rank swap-rate telemetry:
# (layer-calls, swaps, decode layer-calls, decode swaps, intra-batch thrash
# swaps, swap bytes per direction; PCIe traffic is twice that: HtoD page-in
# plus the DtoH write-back of the victim).
_SWAP_STATS = [0, 0, 0, 0, 0, 0]
_SWAP_LOG_EVERY = 40 * 200  # ~200 forward passes of 40 MoE layers


def _note_swaps(
    swaps: int, n_tokens: int, thrash: int = 0, row_bytes: int = 0
) -> None:
    s = _SWAP_STATS
    s[0] += 1
    s[1] += swaps
    s[4] += thrash
    s[5] += swaps * row_bytes
    if n_tokens <= 2:  # decode-shaped (bs 1-2 at np=1)
        s[2] += 1
        s[3] += swaps
    if s[0] >= _SWAP_LOG_EVERY:
        n_layers = 40
        logger.info(
            "DSV4.1 spill swaps: %d over %d layer-calls (%.2f/layer-call); decode-shaped "
            "%d over %d (%.1f swaps per %d-layer token); intra-batch thrash %d; "
            "swapped %.2f GiB per direction",
            s[1], s[0], s[1] / max(s[0], 1),
            s[3], s[2], n_layers * s[3] / max(s[2], 1), n_layers,
            s[4],
            s[5] / (1024**3),
        )
        s[:] = [0] * len(s)


def spill_landing_slots() -> int:
    """D4-G landing slots (0 disables in-graph page-in)."""
    try:
        return max(int(envs.SGLANG_DSV41_SPILL_LANDING.get() or 0), 0)
    except Exception:
        return 0


def spill_decode_landing_active() -> bool:
    """Whether decode-time landing page-in is in play at all.

    True only when experts were actually spilled to the host
    (SGLANG_DSV41_EXPERT_SPILL_APPLY) and the landing pool is enabled. The
    graph-capture width filter and landing warmup must not engage otherwise:
    a model that never spilled (e.g. Qwen3.8 with SGLANG_DSV41_SPILL_LANDING
    left at its default) has no page-in and no drop hazard, and filtering
    would strip every spec capture shape and refuse to boot.
    """
    return (
        bool(envs.SGLANG_DSV41_EXPERT_SPILL_APPLY.get())
        and spill_landing_slots() > 0
    )


def _decode_shaped_max_tokens() -> int:
    """Max T that still uses landing page-in instead of the prefill LRU.

    Correctness bound: the assign kernel silently drops every expert past
    ``min(n_landing, kMaxLanding)`` unique misses, so a T is landing-safe only
    when ``landing >= T * topk`` (GLM-5.3 routed top-k is 8; P0 measured
    active drops at landing=12 with T=2 already). ``slots // 8`` encodes that,
    capped at 6 by the decode graph width. The 2-token floor keeps bs=2 greedy
    decode on the captured path only while 2*8 still fits the pool; below 16
    slots those shapes must go eager, and below 8 slots nothing fits -- 0
    routes every batch to the eager path (captured shapes then fail loudly at
    the width guard instead of dropping experts silently).
    """
    slots = spill_landing_slots()
    if slots < 8:
        return 0
    if slots < 16:
        return 1
    return max(2, min(slots // 8, 6))


def _decode_shaped_topk(topk_ids: torch.Tensor) -> bool:
    return bool(topk_ids.numel()) and int(topk_ids.shape[0]) <= _decode_shaped_max_tokens()


def _cuda_graph_capturing() -> bool:
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _prefill_landing_ready(moe: nn.Module) -> bool:
    """Wide (prefill/extend) batches may route through the landing pool.

    Only when the pool holds every logical expert of a layer: a wide batch
    can request all of them in one call, so anything narrower would silently
    drop experts -- exactly what the decode gate exists to prevent. When the
    gate holds, one device-side assign+copy per layer call replaces the host
    LRU (its per-slice syncs, swap churn, and 1:1 write-back).
    """
    if not envs.SGLANG_DSV41_SPILL_PREFILL_LANDING.get():
        return False
    pool = getattr(moe, "_dsv41_landing_pool", None)
    lru = getattr(moe, "_dsv41_expert_lru", None)
    if pool is None or lru is None:
        return False
    return spill_landing_slots() >= int(lru.n_experts)


@dataclass
class SpillLandingPool:
    """Rank-global landing weight table reused every decode layer."""

    n_landing: int
    tensors: Dict[str, torch.Tensor]
    attrs: List[str]
    dst_ptrs: torch.Tensor
    row_bytes: torch.Tensor
    slot_host_row: torch.Tensor
    land_ids: Optional[torch.Tensor]
    quant_info: object
    # Persistent global LRU cache state (P1): the landing pool stops being a
    # per-call scratch and becomes a rank-global expert cache keyed
    # (layer, expert) -> slot. All device-side; safe under CUDA graphs.
    cache_lut: Optional[torch.Tensor] = None
    cache_slot_key: Optional[torch.Tensor] = None
    cache_epoch: Optional[torch.Tensor] = None
    cache_clock: Optional[torch.Tensor] = None


_LANDING_POOLS: Dict[int, SpillLandingPool] = {}


# ===== P2: whole-layer prefill bank =====
#
# Wide batches have union coverage (~every spilled expert of a layer is
# routed), so their page-in set per layer is static -- exactly the rows the
# cold-set placement put in the host mirror. The bank copies those rows into
# dedicated landing slots with one contiguous pinned HtoD per tensor and
# remaps topk ids through static tables, replacing the per-claim assign
# kernel. Two banks alternate across layers, so after the first forward both
# already hold the right rows: steady-state prefill issues no page-in at all
# until a decode landing claim evicts a bank slot (host-tracked dirty flag).


class _PrefillBank:
    """Rank-global state of the two alternating whole-layer banks."""

    def __init__(
        self,
        *,
        n_cold: int,
        mods: List[nn.Module],
        tables: List[torch.Tensor],
    ):
        self.n_cold = n_cold
        self.n_mods = len(mods)
        self.mods = mods
        # Per module: int32 [n_logical], cold expert -> bank slot, else -1.
        self.tables = tables
        # Which module each bank currently holds (-1 = nothing valid yet).
        self.bank_module = [-1, -1]
        # Fill-completion event per bank (guard for the Marlin stream).
        self.events: List[Optional[torch.cuda.Event]] = [None, None]
        self.stream: Optional[torch.cuda.Stream] = None
        self.dirty = True
        # Host-side fill accounting: bank fills bypass spill_assign_cached,
        # so the cache-clock ledger cannot see them.
        self.fills = 0
        self.fill_rows = 0


_PREFILL_BANK: Optional[_PrefillBank] = None

# Modules with an attached landing pool, in enumeration order; the bank
# sorts them by layer_id on first use.
_SPILL_BANK_REGISTRY: List[nn.Module] = []


def _prefill_bank_enabled() -> bool:
    try:
        return bool(envs.SGLANG_DSV41_SPILL_PREFILL_BANK.get())
    except Exception:
        return False


def _mark_spill_bank_dirty() -> None:
    """A dynamic landing claim may have overwritten a bank slot.

    Host-tracked on purpose: the assign kernel runs data-dependent on the
    device, so the host cannot know which slots were claimed. Decode cannot
    interleave with a prefill forward, so the flag only needs to be
    conservative across batch boundaries.
    """
    bank = _PREFILL_BANK
    if bank is not None:
        bank.dirty = True


def _prefill_bank_slot_table(moe: nn.Module, parity: int) -> torch.Tensor:
    """Static remap table: cold expert -> landing slot in its bank."""
    lru: RoutedExpertLru = moe._dsv41_expert_lru  # type: ignore[attr-defined]
    spill_placement(moe)
    host_slot = moe._dsv41_spill_host_slot  # type: ignore[attr-defined]
    off = parity * int(lru.n_spilled)
    tab = torch.full((int(lru.n_experts),), -1, dtype=torch.int32)
    for e, row in host_slot.items():
        tab[e] = int(row) + off
    return tab.to(moe.w13_weight.device)


def _bank_fill(moe: nn.Module, bank_idx: int, state: _PrefillBank) -> None:
    """Copy one module's spilled half into its bank on the copy stream."""
    if state.stream is None:
        state.stream = torch.cuda.Stream()
    pool: SpillLandingPool = moe._dsv41_landing_pool  # type: ignore[attr-defined]
    hosts: Dict[str, torch.Tensor] = moe._dsv41_spill_host  # type: ignore[attr-defined]
    n = state.n_cold
    lo = bank_idx * n
    with torch.cuda.stream(state.stream):
        for attr in pool.attrs:
            pool.tensors[attr][lo : lo + n].copy_(
                hosts[attr][:n], non_blocking=True
            )
        ev = torch.cuda.Event()
        ev.record(state.stream)
        state.events[bank_idx] = ev
    state.fills += 1
    state.fill_rows += n


def _prefill_bank_dispatch(moe: nn.Module, topk_ids: torch.Tensor) -> bool:
    """Static-bank page-in for one wide MoE call. False = caller falls back.

    Consumes topk_ids in place (kept/shared -> LRU slot, spilled -> -1) and
    publishes ``_dsv41_land_ids`` (spilled -> bank slot, else -1), the same
    contract the landing assign kernel fulfills for the two-run Marlin join.
    """
    global _PREFILL_BANK
    state = _PREFILL_BANK
    if state is None:
        return False
    mod = int(getattr(moe, "_dsv41_bank_mod", -1))
    if not 0 <= mod < state.n_mods:
        return False
    lru: RoutedExpertLru = moe._dsv41_expert_lru  # type: ignore[attr-defined]
    pool: SpillLandingPool = moe._dsv41_landing_pool  # type: ignore[attr-defined]
    n_tok, k = int(topk_ids.shape[0]), int(topk_ids.shape[1])
    buf = pool.land_ids
    if (
        buf is None
        or buf.dtype != topk_ids.dtype
        or buf.device != topk_ids.device
        or int(buf.shape[1]) < k
        or int(buf.shape[0]) < n_tok
    ):
        if _cuda_graph_capturing():
            return False
        pool.land_ids = torch.empty(
            (max(_decode_shaped_max_tokens(), n_tok), k),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        buf = pool.land_ids
    land_ids = buf[:n_tok, :k]
    topk_ids = topk_ids.contiguous()
    flat = topk_ids.view(-1).long()
    # Negative ids (EP remote / padding) pass through untouched; torch would
    # wrap them as python-style indices into the tables.
    neg = (flat < 0).view(n_tok, k)
    safe = flat.clamp(min=0)
    # Main run: kept/shared -> LRU slot, spilled -> -1 (Marlin skips).
    topk_ids.copy_(
        torch.where(
            neg,
            topk_ids,
            lru._map_table[safe].view(n_tok, k).to(topk_ids.dtype),
        )
    )
    # Landing run: spilled -> bank slot, else -1.
    land_ids.copy_(
        torch.where(
            neg,
            torch.full_like(topk_ids, -1),
            state.tables[mod][safe].view(n_tok, k).to(topk_ids.dtype),
        )
    )
    moe._dsv41_land_ids = land_ids  # type: ignore[attr-defined]

    cur = torch.cuda.current_stream()
    # Overwrite guard: banks are filled on the side stream, so a fill into
    # the bank last used by module mod-1 must not start until that module's
    # landing run (already enqueued on the current stream) has finished.
    guard = torch.cuda.Event()
    guard.record(cur)

    need_cur = state.dirty or state.bank_module[mod % 2] != mod
    # Prefetch the next module's bank behind this module's compute. The last
    # module fills module 0 for the next chunk; a single-module stack just
    # refills at its next dispatch instead (the guard above cannot protect
    # its only bank from its own in-flight Marlin).
    nxt = (mod + 1) % state.n_mods
    need_nxt = nxt != mod and (state.dirty or state.bank_module[nxt % 2] != nxt)
    if need_cur or need_nxt:
        if state.stream is None:
            state.stream = torch.cuda.Stream()
        state.stream.wait_event(guard)
    if need_cur:
        _bank_fill(moe, mod % 2, state)
        state.bank_module[mod % 2] = mod
    ev = state.events[mod % 2]
    if ev is not None:
        cur.wait_event(ev)
    if need_nxt:
        _bank_fill(state.mods[nxt], nxt % 2, state)
        state.bank_module[nxt % 2] = nxt
    state.dirty = False
    return True


def _maybe_route_prefill_bank(moe: nn.Module, topk_ids: torch.Tensor) -> bool:
    """Gate + lazy init for the bank path; True means it handled the call."""
    global _PREFILL_BANK
    if not _prefill_bank_enabled() or _cuda_graph_capturing():
        return False
    # Decode-shaped calls stay on the dynamic assign: refilling a whole
    # bank per decode step would re-fetch the full spilled half for a
    # handful of experts.
    if _decode_shaped_topk(topk_ids):
        return False
    lru: Optional[RoutedExpertLru] = getattr(moe, "_dsv41_expert_lru", None)
    pool: Optional[SpillLandingPool] = getattr(moe, "_dsv41_landing_pool", None)
    if lru is None or pool is None or lru.n_spilled <= 0:
        return False
    if not _prefill_landing_ready(moe):
        return False
    if spill_landing_slots() < 2 * int(lru.n_spilled):
        return False
    if _PREFILL_BANK is None:
        mods = _SPILL_BANK_REGISTRY
        if len(mods) < 2:
            return False
        ordered = sorted(mods, key=lambda m: int(getattr(m, "layer_id", 0)))
        n_cold = int(ordered[0]._dsv41_expert_lru.n_spilled)  # type: ignore[attr-defined]
        if any(
            int(m._dsv41_expert_lru.n_spilled) != n_cold  # type: ignore[attr-defined]
            for m in ordered
        ):
            logger.warning(
                "DSV4.1 P2 prefill bank disabled: uneven spilled counts"
            )
            return False
        for i, m in enumerate(ordered):
            m._dsv41_bank_mod = i  # type: ignore[attr-defined]
        _PREFILL_BANK = _PrefillBank(
            n_cold=n_cold,
            mods=ordered,
            tables=[
                _prefill_bank_slot_table(m, i % 2)
                for i, m in enumerate(ordered)
            ],
        )
        logger.info(
            "DSV4.1 P2 prefill bank: %d modules x 2 banks x %d rows "
            "(static slot mapping, zero write-back)",
            len(ordered),
            n_cold,
        )
    return _prefill_bank_dispatch(moe, topk_ids)


def _start_cache_stats_monitor(pool: SpillLandingPool) -> None:
    """Log cache_counter deltas every 30 s (env-gated; 4-int D2H sync)."""

    import threading
    import time as _time

    # Every claimed slot copies the full row of every pooled tensor exactly
    # once (spill_copy_kernel), so claims x row_bytes is the exact HtoD bytes;
    # the landing path issues no DtoH at all. Pool tensors are slot-major
    # ([n_slots, ...]); sum(numel) alone would count the whole pool per claim
    # (measured as a 12x overcount at n_landing=12 before this division).
    row_bytes = sum(
        int(t.numel()) // int(t.shape[0]) * int(t.element_size())
        for t in pool.tensors.values()
    )
    prev = {"calls": 0, "hits": 0, "claims": 0, "drops": 0, "fills": 0, "frows": 0}

    def _loop() -> None:
        while True:
            _time.sleep(30.0)
            try:
                c = pool.cache_clock.cpu()
                calls, hits, claims, drops = int(c[1]), int(c[2]), int(c[3]), int(c[4])
                d_claims = claims - prev["claims"]
                gib = d_claims * row_bytes / (1024**3)
                msg = (
                    "DSV4.1 cache stats: calls=%d (+%d) lut_hits=%d (+%d) "
                    "slot_claims=%d (+%d) drops=%d (+%d) pagein=%.2f GiB "
                    "(%.1f GB/s avg)"
                )
                args = [
                    calls, calls - prev["calls"],
                    hits, hits - prev["hits"],
                    claims, d_claims,
                    drops, drops - prev["drops"],
                    gib,
                    gib * (1024**3) / 30e9,
                ]
                bank = _PREFILL_BANK
                if bank is not None:
                    # Bank fills are whole cold halves, not per-row claims;
                    # report them separately so the ledger stays honest.
                    d_fills = bank.fills - prev["fills"]
                    msg += " bank_fills=%d (+%d) bank_rows=%d"
                    args += [bank.fills, d_fills, bank.fill_rows]
                    prev["fills"] = bank.fills
                    prev["frows"] = bank.fill_rows
                logger.info(msg, *args)
                prev.update(calls=calls, hits=hits, claims=claims, drops=drops)
            except Exception as _e:  # noqa: BLE001
                logger.warning("DSV4.1 cache stats reader stopped: %s", _e)
                return

    threading.Thread(target=_loop, daemon=True, name="dsv41-cache-stats").start()


def _attach_landing_pool(moe: nn.Module, n_landing: int, layer_ordinal: int) -> None:
    if n_landing <= 0:
        return
    hosts: Optional[Dict[str, torch.Tensor]] = getattr(moe, "_dsv41_spill_host", None)
    params = _expert_dim_params(moe)
    if not hosts or not params:
        return
    attrs = [a for a in _EXPERT_PARAM_ATTRS if a in params and a in hosts]
    if not attrs:
        return
    device = params[attrs[0]].device
    if device.type != "cuda":
        return
    if "w13_weight" not in attrs or "w2_weight" not in attrs:
        logger.warning("DSV4.1 D4-G landing skipped: missing w13/w2")
        return
    if "w13_weight_scale" not in attrs or "w2_weight_scale" not in attrs:
        logger.warning("DSV4.1 D4-G landing skipped: missing Marlin scales")
        return
    key = int(device.index or 0)
    pool = _LANDING_POOLS.get(key)
    if pool is None:
        tensors: Dict[str, torch.Tensor] = {}
        for a in attrs:
            p = params[a]
            tensors[a] = torch.empty(
                (n_landing, *p.shape[1:]),
                dtype=p.dtype,
                device=device,
            )
        row_bytes = torch.tensor(
            [
                tensors[a].reshape(n_landing, -1).shape[1] * tensors[a].element_size()
                for a in attrs
            ],
            dtype=torch.int64,
            device=device,
        )
        dst_ptrs = torch.tensor(
            [tensors[a].data_ptr() for a in attrs],
            dtype=torch.int64,
            device=device,
        )
        slot_host_row = torch.empty(n_landing, dtype=torch.int32, device=device)
        from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo

        quant_info = MarlinMoeQuantInfo(
            w13_qweight=tensors["w13_weight"],
            w2_qweight=tensors["w2_weight"],
            w13_scales=tensors["w13_weight_scale"],
            w2_scales=tensors["w2_weight_scale"],
            w13_g_idx_sort_indices=None,
            w2_g_idx_sort_indices=None,
            weight_bits=4,
            is_k_full=True,
            w13_bias=tensors.get("w13_weight_bias"),
            w2_bias=tensors.get("w2_weight_bias"),
            # NVFP4: the landing rows carry the fold-down global scales too;
            # without them the Marlin kernel underflows (same as passing raw
            # checkpoint scales) and every paged-in expert computes ~zero.
            w13_global_scale=tensors.get("w13_scale2"),
            w2_global_scale=tensors.get("w2_scale2"),
            expert_map=None,
            global_num_experts=-1,
            # Landing ids are -1 wherever the main run covers the expert, so
            # the Marlin kernel must skip those blocks (see MarlinMoeQuantInfo
            # .is_expert_parallel).
            is_expert_parallel=True,
        )
        lru = getattr(moe, "_dsv41_expert_lru", None)
        map_tab = getattr(lru, "_map_table", None)
        if map_tab is None:
            return  # no device tables yet: landing unusable anyway
        n_logical = int(map_tab.shape[0])
        n_layers = int(
            envs.SGLANG_DSV41_EXPERT_SPILL_N_LAYERS.get() or N_LAYERS
        )
        dev = tensors[attrs[0]].device
        pool = SpillLandingPool(
            n_landing=n_landing,
            tensors=tensors,
            attrs=attrs,
            dst_ptrs=dst_ptrs,
            row_bytes=row_bytes,
            slot_host_row=slot_host_row,
            land_ids=None,
            quant_info=quant_info,
            # 48 KiB per rank at 42x288: a direct (layer, expert) index, no
            # hash. epoch starts at 0 with clock at 0, so every slot is
            # older than the first call's epoch (1) and the pool fills LRU.
            cache_lut=torch.full(
                (n_layers * n_logical,), -1, dtype=torch.int32, device=dev
            ),
            cache_slot_key=torch.full(
                (n_landing,), -1, dtype=torch.int32, device=dev
            ),
            cache_epoch=torch.zeros(n_landing, dtype=torch.int32, device=dev),
            # [epoch, calls, lut-hits, slots-claimed, drops, 0, 0, 0]
            cache_clock=torch.zeros(8, dtype=torch.int32, device=dev),
        )
        _LANDING_POOLS[key] = pool
        if os.environ.get("SGLANG_DSV41_SPILL_CACHE_STATS") == "1":
            _start_cache_stats_monitor(pool)
        logger.info(
            "DSV4.1 D4-G landing pool: %d slots attrs=%s %.1f MiB "
            "(persistent cache: lut %dx%d)",
            n_landing,
            attrs,
            sum(t.numel() * t.element_size() for t in tensors.values())
            / (1024 * 1024),
            n_layers,
            n_logical,
        )
    moe._dsv41_landing_pool = pool  # type: ignore[attr-defined]
    if _prefill_bank_enabled() and moe not in _SPILL_BANK_REGISTRY:
        _SPILL_BANK_REGISTRY.append(moe)
    # Per-layer LUT slice; the kernel needs no layer_id argument.
    lru = getattr(moe, "_dsv41_expert_lru", None)
    map_tab = getattr(lru, "_map_table", None)
    if pool.cache_lut is not None and map_tab is not None:
        # Flat (layer, expert) key space: cache_slot_key stores flat LUT
        # indices so eviction back-invalidates the OWNING layer's entry. (A
        # per-layer slice view would corrupt other layers' LUTs on evict.)
        n_logical = int(map_tab.shape[0])
        moe._dsv41_cache_lut = pool.cache_lut  # type: ignore[attr-defined]
        moe._dsv41_cache_lut_offset = int(layer_ordinal * n_logical)  # type: ignore[attr-defined]
    src = [int(hosts[a].data_ptr()) for a in pool.attrs]
    moe._dsv41_uva_src_ptrs = torch.tensor(src, dtype=torch.int64, device=device)  # type: ignore[attr-defined]


def page_in_spill_experts(moe: nn.Module, topk_ids: torch.Tensor) -> torch.Tensor:
    """D4-G: UVA copy of spilled hits into the landing pool; remap topk in place."""
    # ROUTEPROBE: raw per-call topk tables for the cold-set/route-stability
    # probes (scripts/glm53_cold_set_from_route_probe.py).
    _pp = envs.SGLANG_SPILL_ROUTE_PROBE.get()
    if _pp and not _cuda_graph_capturing():
        os.makedirs(_pp, exist_ok=True)
        _n = getattr(page_in_spill_experts, "_probe_n", 0)
        if _n < 6000:
            _r = int(topk_ids.device.index or 0)
            torch.save(
                topk_ids.detach().to("cpu", torch.int32),
                f"{_pp}/call_r{_r}_{_n:05d}.pt",
            )
            page_in_spill_experts._probe_n = _n + 1
    lru: RoutedExpertLru = moe._dsv41_expert_lru  # type: ignore[attr-defined]
    pool: SpillLandingPool = moe._dsv41_landing_pool  # type: ignore[attr-defined]
    if topk_ids.dtype != torch.int32 or topk_ids.dim() != 2:
        raise RuntimeError(
            f"D4-G page-in requires int32 [T, K] topk_ids, got {topk_ids.dtype} "
            f"{tuple(topk_ids.shape)}"
        )
    if topk_ids.device.type != "cuda":
        raise RuntimeError("D4-G page-in requires CUDA topk_ids")
    topk_ids = topk_ids.contiguous()
    lru._ensure_device_tables(topk_ids)
    n_tok, k = int(topk_ids.shape[0]), int(topk_ids.shape[1])
    buf = pool.land_ids
    if (
        buf is None
        or buf.dtype != topk_ids.dtype
        or buf.device != topk_ids.device
        or buf.dim() != 2
        or int(buf.shape[1]) < k
        or int(buf.shape[0]) < n_tok
    ):
        if _cuda_graph_capturing():
            raise RuntimeError(
                "D4-G landing id buffer missing during CUDA graph capture; "
                "warmup must run page_in_spill_experts first"
            )
        # Pad T to decode-shaped max (2 greedy, γ+1 when landing is 6*(γ+1))
        # so T=1 capture and a later T=verify eager/capture share storage.
        # k only ever grows the buffer: the warmup allocator may have guessed
        # the routing width from hf config defaults.
        pool.land_ids = torch.empty(
            (
                max(_decode_shaped_max_tokens(), n_tok),
                max(k, int(buf.shape[1]) if buf is not None and buf.dim() == 2 else 0),
            ),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        buf = pool.land_ids
    land_ids = buf[:n_tok, :k]
    from sglang.kernels.ops.moe.sm70_dsv41_spill_pagein import (
        spill_page_in,
        spill_page_in_cached,
    )

    lut = getattr(moe, "_dsv41_cache_lut", None)
    if os.environ.get("SGLANG_DSV41_SPILL_CACHE", "1") != "0" and lut is not None:
        spill_page_in_cached(
            topk_ids,
            land_ids,
            pool.slot_host_row,
            lru._map_table,
            lru._host_map_table,
            moe._dsv41_uva_src_ptrs,
            pool.dst_ptrs,
            pool.row_bytes,
            lut,
            pool.cache_slot_key,
            pool.cache_epoch,
            pool.cache_clock,
            int(getattr(moe, "_dsv41_cache_lut_offset", 0)),
        )
    else:
        spill_page_in(
            topk_ids,
            land_ids,
            pool.slot_host_row,
            lru._map_table,
            lru._host_map_table,
            moe._dsv41_uva_src_ptrs,
            pool.dst_ptrs,
            pool.row_bytes,
        )
    moe._dsv41_land_ids = land_ids  # type: ignore[attr-defined]
    return topk_ids


def warmup_spill_landing_for_capture(
    model: nn.Module,
    n_tok: int,
    k: int,
    device: torch.device,
) -> None:
    """Allocate ``land_ids`` at verify width before CUDA-graph capture.

    Capture throws if the buffer is still the T=1 ``max(2, n_tok)`` allocation.
    """
    if spill_landing_slots() <= 0:
        return
    pad_t = max(int(n_tok), _decode_shaped_max_tokens(), 2)
    # The routing width lives on the layer (hf config keys vary per model
    # family -- mini-GLM has no num_experts_per_tok at all), so read it from
    # the FusedMoE itself. The width must land exactly on k: page_in slices
    # buf[:n_tok, :k], and a wider buffer would make that slice
    # non-contiguous (the page-in kernel's TensorMatcher rejects strides).
    pad_k = max(int(k), 1)
    for moe in model.modules():
        if getattr(moe, "_dsv41_landing_pool", None) is not None:
            pad_k = max(pad_k, int(getattr(moe, "top_k", 0) or 0))
    for moe in model.modules():
        pool = getattr(moe, "_dsv41_landing_pool", None)
        if pool is None:
            continue
        buf = pool.land_ids
        need = (
            buf is None
            or buf.device != device
            or buf.dtype != torch.int32
            or buf.dim() != 2
            or int(buf.shape[0]) < pad_t
            or int(buf.shape[1]) < pad_k
        )
        if need:
            pool.land_ids = torch.empty(
                (pad_t, pad_k), dtype=torch.int32, device=device
            )


def ensure_spill_experts(moe: nn.Module, topk_ids: torch.Tensor) -> torch.Tensor:
    """Host LRU swap + in-place ``map_ids``. Not CUDA-graph safe.

    Wrapped with ``eager_on_graph`` so breakable decode graphs split here.
    ``map_ids`` is ``aten::index`` (IndexKernel); keep it off the captured
    Marlin segment. Writes physical ids into ``topk_ids`` in place so the
    next segment keeps the same buffer address.
    """
    lru = getattr(moe, "_dsv41_expert_lru", None)
    if lru is None or not lru.applied:
        return topk_ids
    valid = topk_ids[topk_ids >= 0]
    if valid.numel():
        before = lru.n_swaps
        before_thrash = lru.n_thrash
        lru.ensure(torch.unique(valid).detach().cpu().tolist())
        _note_swaps(
            lru.n_swaps - before,
            int(topk_ids.shape[0]),
            lru.n_thrash - before_thrash,
            lru.swap_row_bytes(),
        )
    if envs.SGLANG_DSV41_PREFILL_SYNC.get() and topk_ids.numel():
        table = lru._map_table
        logger.info(
            "DSV41 map_ids in min=%s max=%s shape=%s table_n=%s",
            int(topk_ids.min().item()),
            int(topk_ids.max().item()),
            tuple(topk_ids.shape),
            None if table is None else int(table.shape[0]),
        )
    physical = lru.map_ids(topk_ids)
    topk_ids.copy_(physical)
    return topk_ids


try:
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        eager_on_graph,
    )

    ensure_spill_experts = eager_on_graph(True)(ensure_spill_experts)
except ImportError:
    pass


_DSV41_HIDDEN = 5120


def _host_gemv_hosts_ready(moe: nn.Module) -> bool:
    hosts: Optional[Dict[str, torch.Tensor]] = getattr(moe, "_dsv41_spill_host", None)
    if not hosts:
        return False
    if not all(
        k in hosts
        for k in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    ):
        return False
    # D4-H decodes raw MXFP4 codes (e8m0 scale bytes) on the CPU. The NVFP4
    # mirror carries Marlin-repacked int32 codes with e4m3-encoded block
    # scales, which the host GEMV cannot decode -- decode falls back to the
    # D4-G landing pool / prefill LRU instead.
    if hosts["w13_weight_scale"].dtype == torch.float8_e4m3fn:
        return False
    return True


def _attach_host_gemv_bases(moe: nn.Module) -> bool:
    """Device int64[5] of host row bases + n_host. Stable for CUDA graph."""
    if not _host_gemv_hosts_ready(moe):
        return False
    device = getattr(moe, "w13_weight", None)
    if device is None or not isinstance(device, torch.Tensor) or device.device.type != "cuda":
        return False
    if getattr(moe, "_dsv41_host_gemv_bases", None) is not None:
        return True
    hosts: Dict[str, torch.Tensor] = moe._dsv41_spill_host  # type: ignore[attr-defined]
    n_host = int(hosts["w13_weight"].shape[0])
    moe._dsv41_host_gemv_bases = torch.tensor(  # type: ignore[attr-defined]
        [
            int(hosts["w13_weight"].data_ptr()),
            int(hosts["w13_weight_scale"].data_ptr()),
            int(hosts["w2_weight"].data_ptr()),
            int(hosts["w2_weight_scale"].data_ptr()),
            n_host,
        ],
        dtype=torch.int64,
        device=device.device,
    )
    moe._dsv41_host_gemv_pending = False  # type: ignore[attr-defined]
    return True


def _host_gemv_decode_ready(moe: nn.Module) -> bool:
    from sglang.kernels.ops.moe.sm70_dsv41_spill_host_gemv import (
        host_gemv_start,
        sm70_dsv41_host_gemv_available,
    )

    if not sm70_dsv41_host_gemv_available():
        return False
    if getattr(moe, "_dsv41_host_gemv_bases", None) is None:
        if _cuda_graph_capturing() or not _attach_host_gemv_bases(moe):
            return False
    return host_gemv_start()


def spill_request_host_gemv(
    moe: nn.Module,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    hidden: torch.Tensor,
) -> None:
    """D4-H: remap kept ids and post spilled hits; CPU overlaps the GPU GEMV."""
    from sglang.kernels.ops.moe.sm70_dsv41_spill_host_gemv import spill_request

    lru: RoutedExpertLru = moe._dsv41_expert_lru  # type: ignore[attr-defined]
    if topk_ids.dtype != torch.int32 or topk_ids.dim() != 2:
        raise RuntimeError(
            f"D4-H spill_request requires int32 [T, K] topk_ids, got {topk_ids.dtype} "
            f"{tuple(topk_ids.shape)}"
        )
    lru._ensure_device_tables(topk_ids)
    bases = moe._dsv41_host_gemv_bases  # type: ignore[attr-defined]
    if hidden.dim() != 2:
        hidden = hidden.view(-1, hidden.shape[-1])
    if int(hidden.shape[-1]) != _DSV41_HIDDEN:
        hidden = hidden[:, :_DSV41_HIDDEN]
    if not hidden.is_contiguous():
        hidden = hidden.contiguous()
    if topk_weights.dtype != torch.float32:
        topk_weights = topk_weights.float()
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()
    ids = topk_ids if topk_ids.is_contiguous() else topk_ids.contiguous()
    spill_request(
        ids,
        topk_weights,
        hidden,
        lru._map_table,
        lru._host_map_table,
        bases,
    )
    if ids is not topk_ids:
        topk_ids.copy_(ids)
    moe._dsv41_host_gemv_pending = True  # type: ignore[attr-defined]
    moe._dsv41_land_ids = None  # type: ignore[attr-defined]


def spill_join_host_gemv(moe: nn.Module, output: torch.Tensor) -> None:
    """D4-H: wait for the CPU GEMV and add y into the GPU MoE output."""
    from sglang.kernels.ops.moe.sm70_dsv41_spill_host_gemv import spill_join

    if not output.is_contiguous():
        raise RuntimeError("D4-H spill_join requires a contiguous output")
    spill_join(output)
    moe._dsv41_host_gemv_pending = False  # type: ignore[attr-defined]


def remap_dispatch_for_expert_spill(moe: nn.Module, dispatch_output):
    """Prefill: LRU ensure + map_ids. Decode: D4-H host GEMV or D4-G landing."""
    lru = getattr(moe, "_dsv41_expert_lru", None)
    if lru is None or not lru.applied:
        return dispatch_output
    topk_output = getattr(dispatch_output, "topk_output", None)
    if topk_output is None or not hasattr(topk_output, "topk_ids"):
        return dispatch_output
    ids = topk_output.topk_ids
    moe._dsv41_host_gemv_pending = False  # type: ignore[attr-defined]
    if _decode_shaped_topk(ids) and _host_gemv_decode_ready(moe):
        hidden = getattr(dispatch_output, "hidden_states", None)
        weights = getattr(topk_output, "topk_weights", None)
        if hidden is not None and weights is not None:
            spill_request_host_gemv(moe, ids, weights, hidden)
            return dispatch_output
    pool = getattr(moe, "_dsv41_landing_pool", None)
    # P2: wide batches take the static whole-layer bank (zero per-claim
    # assign, contiguous fills, steady-state page-in free). Decode-shaped
    # calls keep the dynamic assign and mark the banks dirty -- a claim may
    # have evicted a bank slot, and the next prefill forward refills.
    if pool is not None and _maybe_route_prefill_bank(moe, ids):
        return dispatch_output
    # Landing owns decode-shaped batches: T is bounded by the pool width
    # (_decode_shaped_topk), which encodes landing >= T*topk. This also gates
    # capture -- the old "capture always takes landing" baked over-width
    # verify graphs (MTP bs>=2, T up to bs*(draft+1)) into a pool that cannot
    # hold their unique experts, and spill_assign_cached then silently drops
    # them (P0: 29 -> 5065 drops on the MTP arm). Over-width shapes are
    # filtered out of capture_bs by the graph runner; this guard only fires
    # if one still gets here, where baking drops would be silent and the
    # eager host LRU would read back the device mid-capture (illegal).
    if (
        pool is not None
        and spill_landing_slots() > 0
        and (_decode_shaped_topk(ids) or _prefill_landing_ready(moe))
    ):
        page_in_spill_experts(moe, ids)
        # Every dynamic claim (decode, or a wide call the bank gate refused)
        # may land on a bank slot; the next prefill forward refills.
        if _prefill_bank_enabled():
            _mark_spill_bank_dirty()
    elif pool is not None and spill_landing_slots() > 0 and _cuda_graph_capturing():
        raise RuntimeError(
            f"D4-G landing width {_decode_shaped_max_tokens()} tokens cannot "
            f"hold this captured batch (T={int(ids.shape[0])}, topk="
            f"{int(ids.shape[1])}): it would silently drop experts. Filter "
            "the shape out of capture_bs or raise SGLANG_DSV41_SPILL_LANDING."
        )
    else:
        moe._dsv41_land_ids = None  # type: ignore[attr-defined]
        ensure_spill_experts(moe, ids)
    return dispatch_output
def _prefill_pass_groups(
    lru: RoutedExpertLru, topk_ids: torch.Tensor
) -> Optional[List[List[int]]]:
    """Routed-id groups that each fit the GPU slots, or None for one pass.

    A single ``ensure`` over more unique ids than slots evicts ids of the same
    batch, and ``map_ids`` then sends their tokens to -1 (expert skipped).
    Residents go in the first group so it needs the fewest swaps.
    """
    if _decode_shaped_topk(topk_ids):
        return None
    valid = topk_ids[topk_ids >= 0]
    if not valid.numel():
        return None
    routed = [i for i in torch.unique(valid).tolist() if i < lru.n_routed]
    slots = lru.n_kept_routed
    if len(routed) <= slots:
        return None
    resident = [i for i in routed if i in lru._id_to_slot]
    missing = [i for i in routed if i not in lru._id_to_slot]
    room = slots - len(resident)
    groups = [resident + missing[:room]]
    rest = missing[room:]
    groups += [rest[i : i + slots] for i in range(0, len(rest), slots)]
    return groups


def _map_pass(
    lru: RoutedExpertLru,
    logical: torch.Tensor,
    group: List[int],
    *,
    include_shared: bool,
) -> torch.Tensor:
    """Physical ids for ``group`` (plus shared once); every other choice -1."""
    before, before_thrash = lru.n_swaps, lru.n_thrash
    lru.ensure(group)
    _note_swaps(
        lru.n_swaps - before, int(logical.shape[0]), lru.n_thrash - before_thrash
    )
    physical = lru.map_ids(logical)
    member = torch.zeros(lru.n_experts, dtype=torch.bool, device=logical.device)
    member[torch.tensor(group, dtype=torch.int64, device=logical.device)] = True
    if include_shared:
        member[lru.n_routed :] = True
    keep = (logical >= 0) & member[logical.clamp(min=0).to(torch.int64)]
    return torch.where(keep, physical, torch.full_like(physical, -1))


def _prefill_reads_landing(moe: nn.Module, topk_ids: torch.Tensor) -> bool:
    return (
        bool(envs.SGLANG_DSV41_PREFILL_LANDING.get())
        and getattr(moe, "_dsv41_landing_pool", None) is not None
        and spill_landing_slots() > 0
        and bool(topk_ids.numel())
        and not _decode_shaped_topk(topk_ids)
    )


# Per-rank prefill landing telemetry: (layer-calls, host rows copied, tokens).
_LANDING_STATS = [0, 0, 0]
_LANDING_LOG_EVERY = 40 * 50  # ~50 prefill forwards of 40 MoE layers


def _note_landing_rows(rows: int, n_tokens: int) -> None:
    s = _LANDING_STATS
    s[0] += 1
    s[1] += rows
    s[2] += n_tokens
    if s[0] >= _LANDING_LOG_EVERY:
        logger.info(
            "DSV4.1 prefill landing: %.1f host rows copied per layer-call "
            "(%.0f tokens per call, %d calls)",
            s[1] / s[0], s[2] / s[0], s[0],
        )
        s[0] = s[1] = s[2] = 0


def _host_row_runs(rows: List[int]) -> List[tuple[int, int, int]]:
    """(first host row, first landing slot, length) for each run of
    consecutive host rows, so each run is one copy per tensor."""
    runs: List[tuple[int, int, int]] = []
    for slot, row in enumerate(rows):
        if runs and runs[-1][0] + runs[-1][2] == row:
            first, first_slot, length = runs[-1]
            runs[-1] = (first, first_slot, length + 1)
        else:
            runs.append((row, slot, 1))
    return runs


def _run_prefill_through_landing(
    moe: nn.Module, lru: RoutedExpertLru, dispatch_output, apply
):
    """Prefill without GPU-slot swaps: copy the host rows this batch routes to
    into the landing pool and run them as landing ids.

    No write-back and no LRU change, so decode keeps the placed GPU set.
    One host sync per layer picks the rows; kept ids run in pass 0 only.
    """
    pool: SpillLandingPool = moe._dsv41_landing_pool  # type: ignore[attr-defined]
    hosts: Dict[str, torch.Tensor] = moe._dsv41_spill_host  # type: ignore[attr-defined]
    topk_output = dispatch_output.topk_output
    logical = topk_output.topk_ids
    lru._ensure_device_tables(logical)
    index = logical.clamp(min=0).to(torch.int64)
    valid = logical >= 0
    kept = torch.where(valid, lru._map_table[index], -1).to(logical.dtype)
    host_row = torch.where(valid, lru._host_map_table[index], -1).to(logical.dtype)
    rows = torch.unique(host_row[host_row >= 0]).tolist()
    _note_landing_rows(len(rows), int(logical.shape[0]))
    n_host = int(hosts[pool.attrs[0]].shape[0])
    moe._dsv41_host_gemv_pending = False  # type: ignore[attr-defined]
    combined = None
    for start in range(0, max(len(rows), 1), pool.n_landing):
        chunk = rows[start : start + pool.n_landing]
        land_ids = None
        if chunk:
            for first, slot, length in _host_row_runs(chunk):
                for attr in pool.attrs:
                    pool.tensors[attr][slot : slot + length].copy_(
                        hosts[attr][first : first + length], non_blocking=True
                    )
            slot_of_row = torch.full(
                (n_host,), -1, dtype=logical.dtype, device=logical.device
            )
            slot_of_row[torch.tensor(chunk, device=logical.device)] = torch.arange(
                len(chunk), dtype=logical.dtype, device=logical.device
            )
            land_ids = torch.where(
                host_row >= 0, slot_of_row[host_row.clamp(min=0).to(torch.int64)], -1
            )
        moe._dsv41_land_ids = land_ids  # type: ignore[attr-defined]
        ids = kept if start == 0 else torch.full_like(kept, -1)
        out = apply(
            layer=moe,
            dispatch_output=dispatch_output._replace(
                topk_output=topk_output._replace(topk_ids=ids)
            ),
        )
        if combined is None:
            combined = out
        else:
            combined = combined._replace(
                hidden_states=combined.hidden_states + out.hidden_states
            )
    moe._dsv41_land_ids = None  # type: ignore[attr-defined]
    return combined


def run_moe_with_expert_spill(moe: nn.Module, dispatch_output, apply):
    """``apply`` once, or once per slot-sized expert group with summed outputs.

    The passes are exact: each (token, expert) choice runs in exactly one
    pass, and the MoE output is a sum over choices.
    """
    lru = getattr(moe, "_dsv41_expert_lru", None)
    topk_output = getattr(dispatch_output, "topk_output", None)
    groups = None
    if lru is not None and lru.applied and topk_output is not None:
        if _prefill_reads_landing(moe, topk_output.topk_ids):
            return _run_prefill_through_landing(moe, lru, dispatch_output, apply)
        groups = _prefill_pass_groups(lru, topk_output.topk_ids)
    if groups is None:
        dispatch_output = remap_dispatch_for_expert_spill(moe, dispatch_output)
        return apply(layer=moe, dispatch_output=dispatch_output)
    moe._dsv41_host_gemv_pending = False  # type: ignore[attr-defined]
    moe._dsv41_land_ids = None  # type: ignore[attr-defined]
    logical = topk_output.topk_ids.clone()
    combined = None
    for index, group in enumerate(groups):
        ids = _map_pass(lru, logical, group, include_shared=index == 0)
        out = apply(
            layer=moe,
            dispatch_output=dispatch_output._replace(
                topk_output=topk_output._replace(topk_ids=ids)
            ),
        )
        if combined is None:
            combined = out
        else:
            combined = combined._replace(
                hidden_states=combined.hidden_states + out.hidden_states
            )
    return combined


def maybe_spill_model_routed_experts(model: nn.Module) -> Optional[RoutedExpertSpillPlan]:
    """Attach a spill plan after Marlin pack, then pack host rows and build the LRU.

    Must run *after* ``process_weights_after_loading``. ``post_load_weights``
    is too early: host would be packed while GPU ``w13`` is still checkpoint
    layout. Plan-only mode does not pin host copies. APPLY requires
    ``remap_dispatch_for_expert_spill`` before Marlin indexes ``topk_ids``.
    """
    spill_gib = float(envs.SGLANG_DSV41_EXPERT_SPILL_GB.get() or 0.0)
    if spill_gib <= 0:
        return None
    if envs.SGLANG_SM70_U2_EXPERT_POOL.get():
        # The u2b2 pool keeps every expert resident in VRAM; the layer params
        # are uint2b2, not NVFP4, so a spill plan would have nothing to pack.
        raise ValueError(
            "SGLANG_DSV41_EXPERT_SPILL_GB is incompatible with "
            "SGLANG_SM70_U2_EXPERT_POOL=1: the u2b2 resident pool has no "
            "spill plan. Unset SGLANG_DSV41_EXPERT_SPILL_GB."
        )
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    moes = [m for m in model.modules() if isinstance(m, FusedMoE)]
    if not moes:
        logger.warning("SGLANG_DSV41_EXPERT_SPILL_GB=%s but no FusedMoE found", spill_gib)
        return None
    moe0 = moes[0]
    existing = getattr(moe0, "_dsv41_expert_spill_plan", None)
    local_routed = int(getattr(moe0, "_num_local_routed", moe0.num_local_experts))
    n_shared = int(getattr(moe0, "num_fused_shared_experts", 0))
    if existing is not None:
        plan = existing
    else:
        bytes_per = 0
        gpu_n = int(getattr(moe0, "_dsv41_gpu_expert_slots", moe0.num_local_experts))
        for _name, p in moe0.named_parameters():
            if p.ndim >= 1 and int(p.shape[0]) in (moe0.num_local_experts, gpu_n):
                bytes_per += p.numel() // int(p.shape[0]) * p.element_size()
        if bytes_per == 0:
            bytes_per = mxfp4_expert_bytes() + expert_scale_bytes()
        plan = plan_routed_expert_spill(
            spill_gib=spill_gib,
            local_routed=local_routed,
            bytes_per_expert=bytes_per,
            n_shared=n_shared,
            n_layers=max(len(moes), 1),
        )
    apply = bool(envs.SGLANG_DSV41_EXPERT_SPILL_APPLY.get())
    logger.info(
        "DSV4.1 routed-expert spill: %.1f GiB/rank plan (%d spilled of %d "
        "local routed, shared=%d stay GPU). apply=%s. %s",
        plan.spill_gib,
        plan.n_spilled,
        plan.local_routed,
        plan.n_shared,
        apply,
        "Marlin remaps topk_ids through RoutedExpertLru after ensure()."
        if apply
        else "Plan only (SGLANG_DSV41_EXPERT_SPILL_APPLY=0); GPU tensors unchanged.",
    )
    for layer_ordinal, moe in enumerate(moes):
        moe._dsv41_expert_spill_plan = plan  # type: ignore[attr-defined]
        if not (apply and plan.n_spilled):
            continue
        if envs.SGLANG_DSV41_EXPERT_SPILL_RSS_LOG.get():
            with open("/proc/self/status") as _f:
                _st = {
                    ln.split(":")[0]: ln.split(":")[1].strip()
                    for ln in _f
                    if ln.startswith(("VmRSS", "VmHWM", "VmSwap"))
                }
            logger.info(
                "DSV4.1 spill RSS L%d pre : VmRSS=%s VmHWM=%s VmSwap=%s",
                layer_ordinal,
                _st.get("VmRSS"),
                _st.get("VmHWM"),
                _st.get("VmSwap"),
            )
        if getattr(moe, "_dsv41_spill_host", None) is not None:
            with _gc_scans_only_new_objects():
                repack_spill_host_for_sm70_marlin(moe)
                # Always pin: the D4-G landing page-in kernel reads the mirror
                # through UVA, which requires device-accessible (pinned) memory.
                # pin_spill_host_numa degrades to plain pinning without the NUMA
                # env and to plain pinning again if mbind/set_mempolicy fails.
                pin_spill_host_numa(moe, layer_ordinal)
        else:
            logger.warning(
                "DSV4.1 spill APPLY: no _dsv41_spill_host on %s; "
                "LRU will snapshot GPU rows after Marlin pack",
                type(moe).__name__,
            )
        params = _expert_dim_params(moe)
        w13 = params.get("w13_weight")
        if w13 is None:
            continue
        kept_ids, cold_ids = spill_placement(moe)
        pre_hosts = getattr(moe, "_dsv41_spill_host", None)
        if pre_hosts:
            sib_attrs = [
                attr
                for attr in params
                if attr != "w13_weight" and attr in pre_hosts
            ]
            siblings = [params[attr].data for attr in sib_attrs]
            host_w13 = pre_hosts.get("w13_weight")
            host_sibs = [pre_hosts[attr] for attr in sib_attrs]
            lru = RoutedExpertLru(
                w13.data,
                n_shared=int(getattr(moe, "num_fused_shared_experts", 0)),
                n_spilled=plan.n_spilled,
                # Pinned staging for the write-back leg once the mirror is
                # pinned; pageable staging would re-introduce a sync copy.
                pin_memory=bool(getattr(moe, "_dsv41_spill_host_pinned", False)),
                siblings=siblings,
                hosts=[host_w13, *host_sibs] if host_w13 is not None else None,
                logical_n_experts=int(moe.num_local_experts),
                already_shrunk=True,
                kept_ids=kept_ids,
                cold_ids=cold_ids,
            )
            moe._dsv41_expert_lru = lru  # type: ignore[attr-defined]
            if w13.is_cuda:
                lru._ensure_device_tables(
                    torch.empty(1, dtype=torch.int32, device=w13.device)
                )
            _attach_landing_pool(moe, spill_landing_slots(), layer_ordinal)
            if envs.SGLANG_DSV41_EXPERT_SPILL_RSS_LOG.get():
                with open("/proc/self/status") as _f:
                    _st = {
                        ln.split(":")[0]: ln.split(":")[1].strip()
                        for ln in _f
                        if ln.startswith(("VmRSS", "VmHWM", "VmSwap"))
                    }
                logger.info(
                    "DSV4.1 spill RSS L%d post: VmRSS=%s VmHWM=%s VmSwap=%s",
                    layer_ordinal,
                    _st.get("VmRSS"),
                    _st.get("VmHWM"),
                    _st.get("VmSwap"),
                )
            continue
        siblings = [p.data for attr, p in params.items() if attr != "w13_weight"]
        lru = RoutedExpertLru(
            w13.data,
            n_shared=int(getattr(moe, "num_fused_shared_experts", 0)),
            n_spilled=plan.n_spilled,
            pin_memory=True,
            siblings=siblings,
            kept_ids=kept_ids,
            cold_ids=cold_ids,
        )
        lru.apply_shrink()
        shrunk = lru.shrunk_tensors()
        w13.data = shrunk[0]
        i = 1
        for attr, p in params.items():
            if attr == "w13_weight":
                continue
            p.data = shrunk[i]
            i += 1
        moe._dsv41_expert_lru = lru  # type: ignore[attr-defined]
        if w13.is_cuda:
            lru._ensure_device_tables(
                torch.empty(1, dtype=torch.int32, device=w13.device)
            )
        _attach_landing_pool(moe, spill_landing_slots(), layer_ordinal)
        if envs.SGLANG_DSV41_EXPERT_SPILL_RSS_LOG.get():
            with open("/proc/self/status") as _f:
                _st = {
                    ln.split(":")[0]: ln.split(":")[1].strip()
                    for ln in _f
                    if ln.startswith(("VmRSS", "VmHWM", "VmSwap"))
                }
            logger.info(
                "DSV4.1 spill RSS L%d post: VmRSS=%s VmHWM=%s VmSwap=%s",
                layer_ordinal,
                _st.get("VmRSS"),
                _st.get("VmHWM"),
                _st.get("VmSwap"),
            )
    if apply and plan.n_spilled:
        n_table = sum(
            1 for m in moes if getattr(m, "_dsv41_spill_cold_source", None) == "table"
        )
        logger.info(
            "DSV4.1 spill placement: cold set from table for %d/%d layers, tail for %d "
            "(SGLANG_DSV41_EXPERT_SPILL_COLD_SET=%s)",
            n_table,
            len(moes),
            len(moes) - n_table,
            envs.SGLANG_DSV41_EXPERT_SPILL_COLD_SET.get(),
        )
        n_h = 0
        for moe in moes:
            if _attach_host_gemv_bases(moe):
                n_h += 1
        if n_h:
            from sglang.kernels.ops.moe.sm70_dsv41_spill_host_gemv import (
                host_gemv_start,
                sm70_dsv41_host_gemv_available,
                sm70_dsv41_host_gemv_enabled,
            )

            if not sm70_dsv41_host_gemv_enabled():
                logger.info(
                    "DSV4.1 D4-H host GEMV off (SGLANG_DSV41_HOST_GEMV=0); "
                    "decode uses D4-G landing"
                )
            elif sm70_dsv41_host_gemv_available() and host_gemv_start():
                logger.info(
                    "DSV4.1 D4-H host GEMV: %d/%d layers, %d CPU threads "
                    "(SGLANG_DSV41_HOST_GEMV=0 restores D4-G landing)",
                    n_h,
                    len(moes),
                    int(envs.SGLANG_DSV41_HOST_GEMV_THREADS.get() or 4),
                )
            else:
                logger.warning(
                    "DSV4.1 D4-H host GEMV unavailable; decode stays on D4-G landing"
                )
    return plan
