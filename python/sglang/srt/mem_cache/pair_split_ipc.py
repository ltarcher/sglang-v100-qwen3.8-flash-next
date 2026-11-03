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

    One point-to-point broadcast each way on the world group carries the
    raw handle blobs; shapes and dtypes are symmetric by construction
    (both ranks build identical pools), so nothing else travels.
    """
    lib = CudaRTLibrary()
    blobs = [_tensor_ipc_bytes(t) for t in local_tensors]
    send = torch.frombuffer(b"".join(blobs), dtype=torch.uint8).clone()
    recv = torch.empty_like(send)
    dist.broadcast(recv, src=peer_rank)
    opened = []
    for i, t in enumerate(local_tensors):
        blob = bytes(recv[i * _HANDLE_BYTES : (i + 1) * _HANDLE_BYTES].tolist())
        handle = cudaIpcMemHandle_t()
        handle.internal = (ctypes.c_char * _HANDLE_BYTES).from_buffer_copy(blob)
        ptr = lib.cudaIpcOpenMemHandle(handle)
        opened.append(ImportedDeviceBuffer(ptr, t.numel(), t.dtype))
    return opened
