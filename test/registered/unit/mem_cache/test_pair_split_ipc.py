"""GPU unit test: pair-split IPC handle exchange (pair_split_ipc).

The rendezvous runs on the TP world group, which is NCCL-only in the serve
image -- a CPU-tensor broadcast there dies with "No backend type associated
with device type cpu". This test replays the exchange on a real 2-rank NCCL
group so that regression (broadcast moved back to host memory, or blobs
carried on a device the group rejects) fails loudly. It also checks the
imported-view semantics: peer bytes readable through the imported pointer,
and slice_rows row-window arithmetic.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from sglang.srt.mem_cache.pair_split_ipc import (
    ImportedDeviceBuffer,
    _HANDLE_BYTES,
    exchange_pair_tensors,
)

_WORLD = 2
_ELEMS = 64


def _read_imported(buf: ImportedDeviceBuffer, out: torch.Tensor) -> None:
    """Copy the raw imported allocation into a local CUDA tensor."""
    lib = CudaRTLibrary()
    lib.cudaMemcpy(
        ctypes_ptr(out.data_ptr()),
        ctypes_ptr(buf.data_ptr()),
        out.numel() * out.element_size(),
    )


def ctypes_ptr(addr: int):
    import ctypes

    return ctypes.c_void_p(addr)


def _worker(rank: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=_WORLD)
    dev = f"cuda:{rank}"

    local = torch.full((_ELEMS,), float(100 + rank), dtype=torch.float16, device=dev)
    buffers = exchange_pair_tensors([local], peer_rank=1 - rank)
    assert len(buffers) == 1
    imported = buffers[0]
    assert isinstance(imported, ImportedDeviceBuffer)
    assert imported.numel == _ELEMS

    seen = torch.empty(_ELEMS, dtype=torch.float16, device=dev)
    _read_imported(imported, seen)
    expected = 100 + (1 - rank)
    assert torch.all(seen == expected), f"rank {rank} read {seen[:4].tolist()}"

    # slice_rows must expose the tail rows of the same allocation.
    tail = imported.slice_rows(16)
    assert tail.numel == imported.numel - 16
    tail_vals = torch.empty(_ELEMS - 16, dtype=torch.float16, device=dev)
    _read_imported(tail, tail_vals)
    assert torch.all(tail_vals == expected)

    dist.barrier()
    dist.destroy_process_group()


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
class TestPairSplitIpcExchange:
    def test_two_rank_nccl_exchange_roundtrip(self):
        port = _free_port()
        mp.spawn(_worker, args=(port,), nprocs=_WORLD, join=True)

    def test_handle_bytes_matches_cuda_struct(self):
        # The wire format packs one fixed-size blob per tensor; it must
        # match the cudaIpcMemHandle_t struct the wrapper exports.
        from sglang.srt.distributed.device_communicators.cuda_wrapper import (
            cudaIpcMemHandle_t,
        )

        import ctypes

        assert _HANDLE_BYTES == 128
        assert ctypes.sizeof(cudaIpcMemHandle_t) == _HANDLE_BYTES
