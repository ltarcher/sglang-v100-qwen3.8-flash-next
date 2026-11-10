"""Staged (NVMe + pinned-staging) storage for the offloaded Qwen4-Exp PLE
n-gram table.

Phase 1 of docs/v100/PLE_NVME_TIER_DESIGN.md. Discrete GPUs (V100) cannot use
the stock ``file`` backend: the gather kernel would have to dereference a
pageable mmap pointer, which only works on unified-memory parts (GB10 / DGX
Spark, ``cudaDevAttrPageableMemoryAccessUsesHostPageTables``). This module
replaces the "device walks host page tables" trick with explicit host staging:

  L0  pinned staging buffer - one slot per flat gather id position
  L2  Linux page cache      - reclaimable; this is what frees the 47.7 GiB
                              hard-pinned table (buffered pread only, never
                              O_DIRECT, so the kernel can deduplicate IO)
  L3  per-rank shard file   - one dense file per TP shard, written once by
                              ``materialize`` after weight loading

Every gather runs as eager host code: flat ids are synced to the CPU, valid
shard-local rows are pread into staging slots by a small thread pool, invalid
slots are zeroed, and the *unchanged* gather kernel reads the staging pointer
with a widened vocabulary range and slot ids 0..N-1. Rows outside this rank's
shard land in zero slots, preserving the kernel's ``in_range`` zeroing
semantics bit-exactly.

Limitations (Phase 1): CUDA graph capture is refused (a captured gather would
replay stale staging contents). Decode-phase pre-replay staging from
CPU-computed row ids is Phase 2 of the design doc.

Like ``qwen4_exp_ple_table.py``, this module has no Triton or CUDA-kernel
imports so its IO machinery can be unit-tested on CPU.
"""

import json
import os
import struct
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from sglang.srt.environ import envs

PLE_STAGED_MAGIC = b"SGPLSTG1"
PLE_STAGED_VERSION = 1
_HEADER_FMT = "8sIIQII"  # magic, version, row_bytes, shard_rows, dtype_code, meta_len
_HEADER_STRUCT = struct.Struct(_HEADER_FMT)

_DTYPE_CODES = {
    torch.float8_e4m3fn: 1,
    torch.float8_e5m2: 2,
    torch.bfloat16: 3,
    torch.float16: 4,
}


def staged_table_file_name(shape, dtype, tag: str) -> str:
    """Deterministic per-shard file name (distinct from the file-backend mmap
    table names on purpose: the staged layout carries a header)."""
    row_bytes = shape[1] * torch.tensor([], dtype=dtype).element_size()
    return f"ple_staged_{shape[0]}x{row_bytes}_{_DTYPE_CODES[dtype]}_{tag}.bin"


def _dtype_code(dtype: torch.dtype) -> int:
    if dtype not in _DTYPE_CODES:
        raise ValueError(f"unsupported PLE staged dtype: {dtype}")
    return _DTYPE_CODES[dtype]


class PleStagedTable:
    """Dense per-shard PLE row file with a small self-describing header."""

    def __init__(self, shape, dtype: torch.dtype, table_dir: str, tag: str):
        if len(shape) != 2:
            raise ValueError(f"PLE staged table must be 2D, got shape {shape}")
        self.shape = tuple(shape)
        self.dtype = dtype
        self.row_bytes = shape[1] * torch.tensor([], dtype=dtype).element_size()
        self.shard_rows = int(shape[0])
        self.header_size = _HEADER_STRUCT.size
        os.makedirs(table_dir, exist_ok=True)
        self.path = os.path.join(table_dir, staged_table_file_name(shape, dtype, tag))
        self.data_offset = self.header_size + len(json.dumps(self._meta()).encode())
        self._fd: int | None = None

    # -- lifecycle ---------------------------------------------------------

    def _open_rw(self) -> int:
        if self._fd is None or self._fd < 0:
            self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        return self._fd

    def close(self) -> None:
        if self._fd is not None and self._fd >= 0:
            os.close(self._fd)
            self._fd = None

    # -- materialization ---------------------------------------------------

    def _read_header(self) -> tuple | None:
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return None
        if size < self.header_size:
            return None
        with open(self.path, "rb") as f:
            head = f.read(self.header_size)
            magic, version, row_bytes, shard_rows, dtype_code, meta_len = (
                _HEADER_STRUCT.unpack(head)
            )
            if magic != PLE_STAGED_MAGIC or version != PLE_STAGED_VERSION:
                return None
            meta = json.loads(f.read(meta_len))
        expected = self._meta()
        return (row_bytes, shard_rows, dtype_code, meta), size, expected

    def _meta(self) -> dict:
        return {
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "version": PLE_STAGED_VERSION,
        }

    def materialize(self, host_table: torch.Tensor, reuse: bool) -> bool:
        """Write ``host_table`` into the shard file. Returns True when bytes
        were written, False when an existing file was reused untouched."""
        expected = (
            self.row_bytes,
            self.shard_rows,
            _dtype_code(self.dtype),
            self._meta(),
        )
        if reuse:
            found = self._read_header()
            if found is not None:
                header, size, _ = found
                if (
                    header == expected
                    and size == self.data_offset + self.shard_rows * self.row_bytes
                ):
                    return False

        fd = self._open_rw()
        meta_bytes = json.dumps(expected[3]).encode()
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(
            fd,
            _HEADER_STRUCT.pack(
                PLE_STAGED_MAGIC,
                PLE_STAGED_VERSION,
                self.row_bytes,
                self.shard_rows,
                _dtype_code(self.dtype),
                len(meta_bytes),
            )
            + meta_bytes,
        )

        flat = host_table.contiguous().view(torch.uint8).reshape(-1)
        chunk_rows = max(1, (8 << 20) // self.row_bytes)  # ~8 MiB per write
        for start in range(0, self.shard_rows, chunk_rows):
            end = min(start + chunk_rows, self.shard_rows)
            chunk = flat[start * self.row_bytes : end * self.row_bytes]
            buf = self._as_write_buffer(chunk)
            n = os.write(fd, buf)
            if n != chunk.numel():
                raise OSError(f"short write into {self.path}: {n} < {chunk.numel()}")
        os.fsync(fd)
        return True

    @staticmethod
    def _as_write_buffer(chunk: torch.Tensor) -> memoryview:
        try:
            return memoryview(chunk.numpy())
        except (TypeError, RuntimeError):
            # Hugepage-backed tensors may not expose __array__; fall back to
            # one copy per chunk.
            import ctypes

            addr = chunk.data_ptr()
            nbytes = chunk.numel()
            return memoryview((ctypes.c_char * nbytes).from_address(addr))

    # -- reads -------------------------------------------------------------

    def _ensure_open(self) -> int:
        if self._fd is None:
            if not os.path.exists(self.path):
                raise FileNotFoundError(
                    f"PLE staged shard file missing: {self.path} "
                    "(materialize() must run after weight loading)"
                )
            self._fd = os.open(self.path, os.O_RDONLY)
        return self._fd

    def read_rows(self, dest: torch.Tensor, rows: list[int]) -> None:
        """Blocking debug/read path: copy file rows into ``dest`` rows."""
        fd = self._ensure_open()
        row_bytes = self.row_bytes
        dest_flat = dest.contiguous().view(torch.uint8).reshape(-1)
        import ctypes

        for slot, row in enumerate(rows):
            if not 0 <= row < self.shard_rows:
                raise IndexError(f"row {row} out of range [0, {self.shard_rows})")
            addr = dest_flat.data_ptr() + slot * row_bytes
            buf = (ctypes.c_char * row_bytes).from_address(addr)
            n = os.preadv(fd, [buf], self.data_offset + row * row_bytes)
            if n != row_bytes:
                raise OSError(f"short read from {self.path}: {n} < {row_bytes}")


class PleRowStaging:
    """Pinned staging buffer, one slot per flat gather id position.

    ``stage(ids, vocab_start, vocab_end)`` fills every slot for this gather:
    valid shard-local rows are pread from the shard file, invalid slots are
    zeroed. The fill is a pure function of the id set (the vocab range is
    fixed per instance), so the last staged ids are cached and an identical
    repeat call returns without touching the buffer; any different id set
    refills ALL slots (zeroing included), which keeps slot reuse across
    gathers safe.

    ``stage_async`` is the Phase 2 prefill double-buffering entry point
    (docs/v100/PLE_NVME_TIER_DESIGN.md §4.2/§11.9.4): the caller stages every
    PLE layer's rows at extend-forward start and the pool overlaps the preads
    with early-layer compute; the later gather calls ``fill_pending`` (which
    trusts the pending width, no second D2H) or ``stage`` (which blocks until
    the pending fill is done before refilling)."""

    # Rows per background read task: enough to amortize dispatch, small
    # enough that io_threads tasks run concurrently for a 4096-row chunk.
    _ASYNC_CHUNK_ROWS = 512

    def __init__(
        self, table: PleStagedTable, max_rows: int, io_threads: int | None = None
    ):
        if io_threads is None:
            io_threads = envs.SGLANG_QWEN4_PLE_STAGED_IO_THREADS.get()
        self.table = table
        self.max_rows = max_rows
        self.row_bytes = table.row_bytes
        self._io_threads = io_threads
        if io_threads > 1:
            self._pool = ThreadPoolExecutor(
                max_workers=io_threads, thread_name_prefix="ple-staged-io"
            )
        else:
            self._pool = None

        try:
            self.buffer = torch.empty(
                (max_rows, self.row_bytes), dtype=torch.uint8, pin_memory=True
            )
        except (RuntimeError, AssertionError):
            # CPU-only test environments (and engines without a visible CUDA
            # driver) cannot pin; the gather kernel never touches this path
            # there anyway.
            self.buffer = torch.empty((max_rows, self.row_bytes), dtype=torch.uint8)
        self._zero_row = torch.zeros(self.row_bytes, dtype=torch.uint8)
        self._arange: torch.Tensor | None = None
        # Id set currently materialized in ``buffer`` (numpy view of the last
        # staged CPU ids, at most max_rows * 8 bytes), plus the outstanding
        # ``stage_async`` fill: (ids_np, futures). ``buffer`` must not be read
        # by a gather kernel while a fill is pending.
        self._staged_ids = None
        self._pending = None

    def slot_arange(self, device: torch.device) -> torch.Tensor:
        """Device-side slot ids 0..max_rows-1 (cached; sliced per gather)."""
        if self._arange is None or self._arange.device != device:
            self._arange = torch.arange(self.max_rows, device=device, dtype=torch.long)
        return self._arange

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def _read_into_slot(self, fd: int, slot: int, row: int) -> None:
        import ctypes

        addr = self.buffer.data_ptr() + slot * self.row_bytes
        buf = (ctypes.c_char * self.row_bytes).from_address(addr)
        n = os.preadv(fd, [buf], self.table.data_offset + row * self.row_bytes)
        if n != self.row_bytes:
            raise OSError(f"short read from {self.table.path}: {n} < {self.row_bytes}")

    def _read_slice(self, fd: int, slots: list[int], rows: list[int]) -> None:
        # os.preadv releases the GIL, so sliced tasks genuinely run in
        # parallel across the pool.
        for slot, row in zip(slots, rows):
            self._read_into_slot(fd, slot, row)

    def _to_ids_np(self, ids: torch.Tensor) -> np.ndarray:
        if ids.dtype == torch.long and ids.is_cpu:
            return ids.numpy()
        return ids.detach().to(torch.long).cpu().numpy()

    def _plan_fill(
        self, ids_np: np.ndarray, vocab_start: int, vocab_end: int
    ) -> tuple[list[int], list[int]]:
        """Zero out-of-range slots now and return the (slots, rows) read
        plan. Runs on the caller thread, before any background fill starts."""
        local = ids_np - vocab_start
        valid = (local >= 0) & (local < self.table.shard_rows) & (ids_np < vocab_end)
        invalid_t = torch.from_numpy((~valid).nonzero()[0])
        if invalid_t.numel():
            self.buffer.index_fill_(0, invalid_t, 0)
        return valid.nonzero()[0].tolist(), local[valid].tolist()

    def _wait_pending(self) -> None:
        """Block until the outstanding ``stage_async`` fill (if any) completes
        and record its id set as the staged one."""
        pending = self._pending
        if pending is None:
            return
        ids_np, futures = pending
        self._pending = None
        exc = None
        for fut in futures:
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001 - re-raise first failure
                exc = exc or e
        if exc is not None:
            raise exc
        self._staged_ids = ids_np

    def stage_async(self, ids: torch.Tensor, vocab_start: int, vocab_end: int) -> None:
        """Start filling staging for ``ids`` in the background.

        Compute-ahead prefill (§11.9.4): the caller stages every PLE layer's
        rows right after batch prep, overlapping the preads with early-layer
        compute. The gather that consumes this staging must first call
        ``fill_pending`` or ``stage`` — the buffer is only readable once the
        pending fill completed."""
        n = int(ids.numel())
        if n > self.max_rows:
            raise ValueError(
                f"gather of {n} rows exceeds staging capacity {self.max_rows} "
                "(raise SGLANG_QWEN4_PLE_STAGED_MAX_ROWS)"
            )
        ids_np = self._to_ids_np(ids)
        self._staged_vocab_start = int(vocab_start)
        self._wait_pending()
        slots, rows = self._plan_fill(ids_np, vocab_start, vocab_end)
        if self._pool is None or len(rows) <= 32:
            # Small fills (eager decode, tiny verifies) stay synchronous:
            # page-cache-warm rows are a few us each, while a round of thread
            # dispatch shows up directly on the decode step.
            fd = self.table._ensure_open()
            self._read_slice(fd, slots, rows)
            self._staged_ids = ids_np
            return
        fd = self.table._ensure_open()
        chunks = min(
            self._io_threads,
            (len(rows) + self._ASYNC_CHUNK_ROWS - 1) // self._ASYNC_CHUNK_ROWS,
        )
        step = (len(rows) + chunks - 1) // chunks
        futures = [
            self._pool.submit(
                self._read_slice, fd, slots[i : i + step], rows[i : i + step]
            )
            for i in range(0, len(rows), step)
        ]
        self._pending = (ids_np, futures)

    def fill_pending(self, expect_n: int) -> bool:
        """Wait for the outstanding async fill, if any. Returns True only when
        that fill staged exactly ``expect_n`` rows, so the caller may gather
        directly without its own id D2H: the pending fill was computed from
        this forward's own batch, and one staging is only ever pending for its
        own layer's ids. A previously cached (non-pending) id set never
        counts — a same-width gather from an earlier step would otherwise be
        served stale rows."""
        pending = self._pending
        if pending is None:
            return False
        ids_np, _ = pending
        self._wait_pending()
        return ids_np.shape[0] == expect_n

    def debug_verify_ids(self, ids: torch.Tensor) -> bool:
        """PREP_DEBUG helper: compare the staged id set against a device id
        tensor (one D2H; debug builds only)."""
        staged = self._staged_ids
        if staged is None:
            return False
        cpu = ids.detach().to(torch.long).cpu()
        return cpu.numel() == staged.shape[0] and np.array_equal(cpu.numpy(), staged)

    def wait_and_match(self, ids_np: np.ndarray) -> bool:
        """Chunk-ahead consume check (docs/v100/PLE_NVME_TIER_DESIGN.md
        §11.9.5): block until this buffer's speculative fill completed, then
        compare its id set against ``ids_np`` — the ids the consuming forward
        computed itself from the real batch. The staged fill was derived
        speculatively from the previous forward's merged context, so a
        bit-exact match is the only trust basis; any mismatch sends the
        caller back to the synchronous path."""
        self._wait_pending()
        staged = self._staged_ids
        return (
            staged is not None
            and staged.shape[0] == ids_np.shape[0]
            and np.array_equal(staged, ids_np)
        )

    def stage(self, ids: torch.Tensor, vocab_start: int, vocab_end: int) -> int:
        """Fill staging for ``ids`` (1D global ids). Returns the slot count.

        Blocks until any outstanding async fill is done before touching the
        buffer, then skips the refill entirely when ``ids`` match the staged
        set."""
        n = int(ids.numel())
        if n > self.max_rows:
            raise ValueError(
                f"gather of {n} rows exceeds staging capacity {self.max_rows} "
                "(raise SGLANG_QWEN4_PLE_STAGED_MAX_ROWS)"
            )
        ids_np = self._to_ids_np(ids)
        self._wait_pending()
        staged = self._staged_ids
        if (
            staged is not None
            and staged.shape[0] == n
            and np.array_equal(ids_np, staged)
        ):
            return n

        slots, rows = self._plan_fill(ids_np, vocab_start, vocab_end)
        if rows:
            fd = self.table._ensure_open()
            if self._pool is None or len(rows) <= 32:
                self._read_slice(fd, slots, rows)
            else:
                futures = [
                    self._pool.submit(
                        self._read_slice,
                        fd,
                        slots[i : i + self._ASYNC_CHUNK_ROWS],
                        rows[i : i + self._ASYNC_CHUNK_ROWS],
                    )
                    for i in range(0, len(rows), self._ASYNC_CHUNK_ROWS)
                ]
                exc = None
                for fut in futures:
                    try:
                        fut.result()
                    except Exception as e:  # noqa: BLE001 - re-raise first failure
                        exc = exc or e
                if exc is not None:
                    raise exc
        self._staged_ids = ids_np
        return n


class PleAheadStaging:
    """Chunk-ahead prefetch pool for extend prefill (docs/v100/
    PLE_NVME_TIER_DESIGN.md §11.9.5).

    While chunk ``i`` computes on the GPU, the rows for chunks ``i+1..i+depth``
    are already being pread into extra staging buffers, so the gather's
    ``fill_pending`` tail wait (~198 ms per 64k chunk) disappears into the
    previous chunks' compute windows. ``depth`` queues ``depth`` speculative
    fills and holds ``depth + 1`` buffers — the extra one is the buffer just
    handed out as the current forward's serve, which must not be refilled
    until the next consume (the caller's id D2H there proves this forward's
    gathers finished).

    Entries are FIFO, tagged with the req pool slot. ``consume`` trusts a
    staged fill only after a bit-exact id comparison against the ids the
    consuming forward computed itself; any mismatch (mixed batches, verify,
    re-slotted requests, stale stash) drops the entry and the caller falls
    back to the synchronous path — zero correctness risk, IO timing only.
    """

    def __init__(
        self,
        table: PleStagedTable,
        depth: int,
        max_rows: int,
        io_threads: int | None = None,
    ):
        if depth < 1:
            raise ValueError("PleAheadStaging needs depth >= 1")
        self.depth = depth
        self.buffers = [
            PleRowStaging(table, max_rows, io_threads) for _ in range(depth + 1)
        ]
        # FIFO of (pool_idx, ids_np, PleRowStaging); ids kept for the dedup
        # below (each forward re-derives the whole window from its stash).
        self._queue: deque = deque()
        self._serve: PleRowStaging | None = None

    def close(self) -> None:
        for buf in self.buffers:
            buf.close()

    def submit(
        self, pool_idx: int, ids_np: np.ndarray, vocab_start: int, vocab_end: int
    ) -> bool:
        """Queue a speculative fill for a future chunk. Non-blocking; skips
        ids already queued (the sliding window re-derives every chunk each
        forward) and drops when the window is saturated. Returns True when
        the fill was queued."""
        if ids_np.shape[0] > self.buffers[0].max_rows:
            return False  # speculative: never raise, just skip the fill
        for _, staged, _buf in self._queue:
            if staged.shape == ids_np.shape and np.array_equal(staged, ids_np):
                return False
        buf = self._pick_free()
        if buf is None or len(self._queue) >= self.depth:
            return False
        buf.stage_async(torch.from_numpy(ids_np), vocab_start, vocab_end)
        self._queue.append((pool_idx, ids_np, buf))
        return True

    def consume(self, pool_idx: int, ids_np: np.ndarray) -> PleRowStaging | None:
        """Hand the head entry's buffer to the caller's gather when it was
        staged for exactly this forward's ids. The previous forward's serve
        is released here: the caller already synced its ids D2H, which proves
        that forward's gathers finished, so the buffer is safe to refill."""
        self._serve = None
        if not self._queue:
            return None
        head_pool, _head_ids, buf = self._queue.popleft()
        if head_pool != pool_idx:
            return None  # stale slot tag; the sync path refills instead
        if buf.wait_and_match(ids_np):
            self._serve = buf
            return buf
        return None

    def _pick_free(self) -> PleRowStaging | None:
        queued = {id(buf) for _, _, buf in self._queue}
        for buf in self.buffers:
            if buf is not self._serve and id(buf) not in queued:
                return buf
        return None
