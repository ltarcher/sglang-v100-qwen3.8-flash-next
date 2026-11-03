"""Pair-split IPC rendezvous (docs/v100/KV_HOST_TIER_DESIGN.md §6.1).

PAIR_PARITY needs the pair member's full-KV buffers and tag tables mapped
into this process so the tier's page-in and gather kernels can read them
as ordinary device pointers. The tensors are allocated once at boot and
never freed, so the classic one-shot cudaIpcGetMemHandle / OpenMemHandle
exchange over the TP group is enough -- no lifetime management, no
allocation from imported memory, and the imported pointers are fixed
before the decode CUDA graphs capture (graph-safe by construction).
"""

from __future__ import annotations

import ctypes
import logging
from typing import List

import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.cuda_wrapper import (
    CudaRTLibrary,
    cudaIpcMemHandle_t,
)

logger = logging.getLogger(__name__)

# cudaIpcMemHandle_t payload size (the wrapper's struct is 128 bytes).
_HANDLE_BYTES = 128


def _tensor_ipc_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize one CUDA tensor's IPC handle."""
    lib = CudaRTLibrary()
    handle = lib.cudaIpcGetMemHandle(tensor.data_ptr())
    return bytes(handle.internal)


class ImportedDeviceBuffer:
    """Raw imported CUDA allocation viewed as a torch-shaped operand.

    Triton kernels only need ``.data_ptr()``; the host never dereferences
    the memory through torch. The allocation is owned by the exporting
    process and lives for the whole job (KV pools are never freed), so no
    release path exists -- cudaIpcCloseMemHandle is deliberately not
    called.
    """

    def __init__(self, ptr, numel: int, dtype: torch.dtype) -> None:
        self._ptr = ptr
        self.numel = numel
        self.dtype = dtype

    def data_ptr(self) -> int:
        # Triton launches call .data_ptr() on operand objects.
        return self._ptr.value

    def view(self, dtype: torch.dtype) -> "ImportedDeviceBuffer":
        # Kernels view int16 over fp16 storage; element sizes must match so
        # the pointer and count stay valid.
        if dtype.itemsize != self.dtype.itemsize:
            raise ValueError(
                f"peer view dtype {dtype} does not match storage {self.dtype}"
            )
        return self

    def slice_rows(self, start: int) -> "ImportedDeviceBuffer":
        """Row-window view (per-layer tables inside a whole-pool export).

        IPC handles name the allocation base; consumers add the offset
        arithmetically rather than exporting one handle per row view.
        """
        shifted = ctypes.c_void_p(self._ptr.value + start * self.dtype.itemsize)
        return ImportedDeviceBuffer(shifted, self.numel - start, self.dtype)


def exchange_pair_tensors(
    local_tensors: List[torch.Tensor], peer_rank: int
) -> List[ImportedDeviceBuffer]:
    """Send our IPC handles to ``peer_rank`` and open theirs.

    One point-to-point send/recv each way on the world group carries the
    raw handle blobs; shapes and dtypes are symmetric by construction
    (both ranks build identical pools), so nothing else travels.
    """
    lib = CudaRTLibrary()
    blobs = [_tensor_ipc_bytes(t) for t in local_tensors]
    send = torch.frombuffer(b"".join(blobs), dtype=torch.uint8).clone()
    # The world group is NCCL-only (no gloo backend in the serve image), so
    # the handle blobs must ride on the device; and this is a pairwise
    # exchange, not a collective -- broadcast(src=peer) would issue two
    # different collectives on the world group (each rank joins the other's
    # src, so neither collective ever has all ranks) and deadlock. A
    # point-to-point send/recv pair matches by ordering instead.
    send_gpu = torch.empty(
        send.numel(), dtype=torch.uint8, device=local_tensors[0].device
    )
    send_gpu.copy_(send)
    recv_gpu = torch.empty_like(send_gpu)
    if dist.get_rank() < peer_rank:
        dist.send(send_gpu, dst=peer_rank)
        dist.recv(recv_gpu, src=peer_rank)
    else:
        dist.recv(recv_gpu, src=peer_rank)
        dist.send(send_gpu, dst=peer_rank)
    recv = recv_gpu.cpu()
    opened = []
    for i, t in enumerate(local_tensors):
        blob = recv[i * _HANDLE_BYTES : (i + 1) * _HANDLE_BYTES].tolist()
        handle = cudaIpcMemHandle_t()
        # memmove, not attribute assignment: the wrapper's internal field is
        # c_byte_Array_128 and a c_char_Array_128 assignment raises TypeError.
        ctypes.memmove(handle.internal, bytes(blob), _HANDLE_BYTES)
        ptr = lib.cudaIpcOpenMemHandle(handle)
        opened.append(ImportedDeviceBuffer(ptr, t.numel(), t.dtype))
    return opened
