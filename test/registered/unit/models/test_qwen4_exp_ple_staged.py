"""Unit tests for the staged (NVMe + pinned staging) PLE table backend.

Phase 1 of docs/v100/PLE_NVME_TIER_DESIGN.md. Everything here runs on CPU:
the module under test has no Triton or CUDA-kernel imports, matching the
contract of ``qwen4_exp_ple_table.py``. GPU-side bit-exactness (staged vs
pinned gather through the real kernel) is covered separately on hardware.

Boundary scenarios covered (high-risk areas per the streaming-KV lessons):
- rows whose file offset straddles a 4 KiB page boundary,
- slot reuse across gathers (a slot valid in one gather must be zeroed when
  it becomes invalid in the next),
- thread-count invariance (io_threads=1 vs 8 must produce identical bytes),
- shard-range edges (row 0, last row, one past the end),
- the stage() id cache (identical repeat ids must skip the refill; any
  different id set — including a prefix of the cached one — must refill).
"""

import os
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.models.qwen4_exp_ple_staged import (
    PLE_STAGED_MAGIC,
    PleRowStaging,
    PleStagedTable,
    ensure_max_rows_for_prefill,
    staged_table_file_name,
)

ROW_BYTES = 160  # Qwen3.8-Flash-Next: 16 heads x 160 B fp8 rows per token
PAGE = 4096


def _make_table(rows: int, row_bytes: int = ROW_BYTES, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, 255, (rows, row_bytes), dtype=torch.uint8, generator=gen
    ).view(torch.float8_e4m3fn)


def _make_staged(rows: int, row_bytes: int, tmpdir: str, tag: str) -> PleStagedTable:
    return PleStagedTable(
        shape=(rows, row_bytes),
        dtype=torch.float8_e4m3fn,
        table_dir=tmpdir,
        tag=tag,
    )


class TestStagedTableFileName:
    def test_deterministic_and_tagged(self):
        a = staged_table_file_name((100, ROW_BYTES), torch.float8_e4m3fn, "r0-5")
        b = staged_table_file_name((100, ROW_BYTES), torch.float8_e4m3fn, "r0-5")
        c = staged_table_file_name((100, ROW_BYTES), torch.float8_e4m3fn, "r5-9")
        assert a == b
        assert a != c
        # Must not collide with the file-backend mmap table names.
        assert "ple_staged_" in a


class TestMaterializeRoundtrip:
    def test_write_then_read_back_identical(self, tmp_path):
        rows = 1000
        table = _make_table(rows)
        staged = _make_staged(rows, ROW_BYTES, str(tmp_path), "t1")
        assert staged.materialize(table, reuse=False) is True

        staged2 = _make_staged(rows, ROW_BYTES, str(tmp_path), "t1")
        got = torch.empty(rows, ROW_BYTES, dtype=torch.uint8)
        staged2.read_rows(got, list(range(rows)))
        # Compare bytes, not fp8 values: random bytes include NaN patterns
        # (0x7F/0xFF) and NaN != NaN would make torch.equal a false negative.
        assert torch.equal(got, table.view(torch.uint8))

    def test_row_subset_read(self, tmp_path):
        rows = 500
        table = _make_table(rows, seed=3)
        staged = _make_staged(rows, ROW_BYTES, str(tmp_path), "t2")
        staged.materialize(table, reuse=False)

        got = torch.empty(7, ROW_BYTES, dtype=torch.uint8)
        staged.read_rows(got, [0, 1, 25, 26, 499, 123, 250])
        for slot, row in enumerate([0, 1, 25, 26, 499, 123, 250]):
            assert torch.equal(got[slot], table[row].view(torch.uint8)), (
                f"row {row} mismatch"
            )

    def test_page_boundary_rows(self, tmp_path):
        # row_bytes=160: row 25 starts at offset 4000 and straddles the 4096
        # page boundary; rows 24/25/26 together span three pages.
        rows = 64
        table = _make_table(rows, seed=5)
        staged = _make_staged(rows, ROW_BYTES, str(tmp_path), "pb")
        staged.materialize(table, reuse=False)
        assert 25 * ROW_BYTES // PAGE != (25 * ROW_BYTES + ROW_BYTES - 1) // PAGE

        got = torch.empty(3, ROW_BYTES, dtype=torch.uint8)
        staged.read_rows(got, [24, 25, 26])
        for slot, row in enumerate([24, 25, 26]):
            assert torch.equal(got[slot], table[row].view(torch.uint8))

    def test_header_layout(self, tmp_path):
        rows = 10
        staged = _make_staged(rows, ROW_BYTES, str(tmp_path), "hdr")
        table = _make_table(rows, seed=7)
        staged.materialize(table, reuse=False)
        path = staged.path
        size = os.path.getsize(path)
        # Data starts after the packed header AND the JSON meta block; the
        # first data byte must be table content, never header bytes.
        assert size == staged.data_offset + rows * ROW_BYTES
        with open(path, "rb") as f:
            raw = f.read()
        assert raw[:8] == PLE_STAGED_MAGIC
        assert raw[staged.data_offset] == table.view(torch.uint8).reshape(-1)[0].item()
        assert (
            raw[staged.data_offset + ROW_BYTES]
            == table.view(torch.uint8).reshape(-1)[ROW_BYTES].item()
        )


class TestMaterializeReuse:
    def test_reuse_skips_rewrite_when_header_matches(self, tmp_path):
        rows = 100
        staged = _make_staged(rows, ROW_BYTES, str(tmp_path), "reuse")
        staged.materialize(_make_table(rows, seed=1), reuse=False)
        path = staged.path
        first_mtime = os.stat(path).st_mtime_ns
        first_size = os.path.getsize(path)

        # Different content, same shape: reuse must NOT pick it up silently —
        # materialize(reuse=True) keeps the existing file untouched.
        staged2 = _make_staged(rows, ROW_BYTES, str(tmp_path), "reuse")
        wrote = staged2.materialize(_make_table(rows, seed=2), reuse=True)
        assert wrote is False
        assert os.stat(path).st_mtime_ns == first_mtime
        assert os.path.getsize(path) == first_size

    def test_mismatched_header_rewrites(self, tmp_path):
        staged = _make_staged(100, ROW_BYTES, str(tmp_path), "mm")
        staged.materialize(_make_table(100), reuse=False)

        staged2 = _make_staged(200, ROW_BYTES, str(tmp_path), "mm")
        assert staged2.materialize(_make_table(200), reuse=True) is True
        assert os.path.getsize(staged2.path) == 200 * ROW_BYTES + staged2.data_offset

    def test_reuse_flag_default_off(self, tmp_path):
        # Production-safe default: a checkpoint change with identical shape
        # must not silently serve a stale file.
        assert envs.SGLANG_QWEN4_PLE_STAGED_REUSE.get() is False


class TestRowStaging:
    def _fill_and_check(self, tmp_path, ids, vocab_start, vocab_end, io_threads):
        rows = 300
        table = _make_table(rows, seed=9)
        staged = _make_staged(rows, ROW_BYTES, str(tmp_path), f"st{io_threads}")
        staged.materialize(table, reuse=False)

        staging = PleRowStaging(
            table=staged, max_rows=len(ids) + 4, io_threads=io_threads
        )
        n = staging.stage(torch.tensor(ids, dtype=torch.long), vocab_start, vocab_end)
        assert n == len(ids)
        buf = staging.buffer[: len(ids)]
        for slot, gid in enumerate(ids):
            local = gid - vocab_start
            if 0 <= local < rows:
                assert torch.equal(buf[slot], table[local].view(torch.uint8)), (
                    f"slot {slot} (id {gid}) mismatch"
                )
            else:
                assert torch.all(buf[slot] == 0), f"slot {slot} (id {gid}) not zeroed"
        return staging

    def test_fill_valid_and_invalid(self, tmp_path):
        # vocab range [100, 400): ids 100 and 399 are edge rows, 0/99/400/999
        # are out of range and must read as zeros.
        self._fill_and_check(
            tmp_path, [100, 150, 399, 0, 99, 400, 999], 100, 400, io_threads=4
        )

    def test_slot_reuse_zeroes_stale_rows(self, tmp_path):
        rows = 50
        table = _make_table(rows, seed=11)
        staged = _make_staged(rows, ROW_BYTES, str(tmp_path), "reuse_slot")
        staged.materialize(table, reuse=False)
        staging = PleRowStaging(table=staged, max_rows=8, io_threads=2)

        # First gather against vocab range [0, 50): slots 0-7 all valid.
        staging.stage(torch.arange(0, 8), 0, 50)
        for slot in range(8):
            assert torch.equal(staging.buffer[slot], table[slot].view(torch.uint8))

        # Second gather against vocab range [8, 54): ids 4..7 now fall below
        # the range (local < 0). Slots 0-3 were valid before and must be
        # re-zeroed, not left stale; slots 4-7 map to rows 0-3.
        staging.stage(torch.arange(4, 12), 8, 54)
        for slot in range(4):
            assert torch.all(staging.buffer[slot] == 0), (
                f"slot {slot} kept stale data from the previous gather"
            )
        for slot in range(4, 8):
            local = slot - 4  # gid = 4 + slot, local = gid - 8
            assert torch.equal(staging.buffer[slot], table[local].view(torch.uint8))

    def test_thread_count_invariance(self, tmp_path):
        ids = list(range(0, 300, 3))
        one = self._fill_and_check(tmp_path, ids, 0, 300, io_threads=1)
        eight = self._fill_and_check(tmp_path, ids, 0, 300, io_threads=8)
        assert torch.equal(one.buffer[: len(ids)], eight.buffer[: len(ids)])

    def test_shard_range_edges(self, tmp_path):
        rows = 16
        staged = _make_staged(rows, ROW_BYTES, str(tmp_path), "edge")
        staged.materialize(_make_table(rows, seed=13), reuse=False)
        staging = PleRowStaging(table=staged, max_rows=6, io_threads=2)
        # first row, last row, one before, one after
        staging.stage(torch.tensor([0, 15, 16, 4294967295]), 0, 16)
        assert torch.equal(
            staging.buffer[0], _make_table(rows, seed=13)[0].view(torch.uint8)
        )
        assert torch.equal(
            staging.buffer[1], _make_table(rows, seed=13)[15].view(torch.uint8)
        )
        assert torch.all(staging.buffer[2] == 0)
        assert torch.all(staging.buffer[3] == 0)


class TestStageIdCache:
    """stage() caches the last staged id set: the fill is a pure function of
    the ids (the vocab range is fixed per staging instance), so an identical
    repeat call must skip the refill while any different id set — including a
    prefix of the cached one — must refill. The decode-prep gate shares this
    same stage() API, and fill_pending never trusts the cache for its
    D2H-skip decision (only genuine pending async fills count)."""

    def _make(self, tmp_path, rows=300, max_rows=64):
        table = _make_table(rows, seed=17)
        staged = _make_staged(rows, ROW_BYTES, str(tmp_path), "cache")
        staged.materialize(table, reuse=False)
        staging = PleRowStaging(table=staged, max_rows=max_rows, io_threads=4)
        return staging, table

    def _count_reads(self, staging):
        calls = []
        orig = staging._read_into_slot

        def counting(fd, slot, row):
            calls.append((slot, row))
            return orig(fd, slot, row)

        staging._read_into_slot = counting
        return calls

    def test_repeated_ids_skip_refill(self, tmp_path):
        staging, table = self._make(tmp_path)
        ids = torch.tensor([5, 250, 299, 7], dtype=torch.long)
        staging.stage(ids, 0, 300)
        calls = self._count_reads(staging)

        n = staging.stage(ids, 0, 300)
        assert n == ids.numel()
        assert calls == [], "identical repeat ids must not re-read the shard file"
        for slot, gid in enumerate(ids.tolist()):
            assert torch.equal(staging.buffer[slot], table[gid].view(torch.uint8))

    def test_changed_ids_refill(self, tmp_path):
        staging, _ = self._make(tmp_path)
        staging.stage(torch.tensor([5, 250, 299, 7]), 0, 300)
        calls = self._count_reads(staging)

        other = torch.tensor([5, 250, 299, 8], dtype=torch.long)
        staging.stage(other, 0, 300)
        assert len(calls) == 4, "a different id set must refill every slot"
        # Stale slot content must not survive: slot 3 changed 7 -> 8.
        rows = 300
        gen = torch.Generator().manual_seed(17)
        table = torch.randint(
            0, 255, (rows, ROW_BYTES), dtype=torch.uint8, generator=gen
        )
        assert torch.equal(staging.buffer[3], table[8].view(torch.uint8))

    def test_prefix_ids_refill(self, tmp_path):
        # A strict prefix of the cached ids is a different gather. Skipping
        # would actually be safe here (the leading slots already hold the
        # right rows), but the cache is deliberately conservative: any byte
        # difference in the id arrays refills. Lock that behavior in.
        staging, _ = self._make(tmp_path)
        ids = torch.tensor([5, 250, 299, 7], dtype=torch.long)
        staging.stage(ids, 0, 300)
        calls = self._count_reads(staging)

        n = staging.stage(ids[:3], 0, 300)
        assert n == 3
        assert len(calls) == 3, "a shorter gather with same leading ids must refill"

    def test_a_b_a_refills_last(self, tmp_path):
        # Decode gate fills ids B between two eager prefill gathers of ids A:
        # the second A must refill (B overwrote the staging).
        staging, table = self._make(tmp_path)
        a = torch.tensor([5, 250, 299, 7], dtype=torch.long)
        b = torch.tensor([11, 12, 13], dtype=torch.long)
        staging.stage(a, 0, 300)
        staging.stage(b, 0, 300)
        calls = self._count_reads(staging)

        staging.stage(a, 0, 300)
        assert len(calls) == 4
        for slot, gid in enumerate(a.tolist()):
            assert torch.equal(staging.buffer[slot], table[gid].view(torch.uint8))

    def test_capacity_check_before_cache(self, tmp_path):
        staging, _ = self._make(tmp_path, max_rows=4)
        ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        staging.stage(ids[:4], 0, 300)
        try:
            staging.stage(ids, 0, 300)
            raise AssertionError("expected capacity ValueError")
        except ValueError:
            pass


class TestStageAsyncPrefillPrefetch:
    """Phase 2 prefill compute-ahead staging (§11.9.4): stage_async starts a
    background fill; the gather consumes it via fill_pending (no second D2H)
    or stage() (which waits, then refills on mismatch)."""

    def _make(self, tmp_path, rows=300, max_rows=64, io_threads=4):
        table = _make_table(rows, seed=19)
        staged = _make_staged(rows, ROW_BYTES, str(tmp_path), "async")
        staged.materialize(table, reuse=False)
        staging = PleRowStaging(table=staged, max_rows=max_rows, io_threads=io_threads)
        return staging, table

    def test_fill_pending_consumes_async_fill(self, tmp_path):
        staging, table = self._make(tmp_path)
        ids = torch.arange(0, 64, dtype=torch.long)
        staging.stage_async(ids, 0, 300)

        assert staging.fill_pending(ids.numel()) is True
        for slot, gid in enumerate(ids.tolist()):
            assert torch.equal(staging.buffer[slot], table[gid].view(torch.uint8))
        # Consumed: a second fill_pending finds nothing pending.
        assert staging.fill_pending(3) is False

    def test_fill_pending_width_mismatch(self, tmp_path):
        staging, table = self._make(tmp_path)
        ids = torch.arange(0, 64, dtype=torch.long)
        staging.stage_async(ids, 0, 300)
        # Width mismatch: the fill still lands (buffer is correct), but the
        # caller is told not to trust it and must stage its own ids.
        assert staging.fill_pending(3) is False
        for slot, gid in enumerate(ids.tolist()):
            assert torch.equal(staging.buffer[slot], table[gid].view(torch.uint8))

    def test_stage_waits_pending_then_refills_on_mismatch(self, tmp_path):
        staging, table = self._make(tmp_path)
        a = torch.arange(0, 64, dtype=torch.long)
        b = torch.arange(100, 164, dtype=torch.long)
        staging.stage_async(a, 0, 300)
        # stage() with different ids must not race the pending fill: it waits
        # for it, then refills every slot.
        staging.stage(b, 0, 300)
        for slot, gid in enumerate(b.tolist()):
            assert torch.equal(staging.buffer[slot], table[gid].view(torch.uint8))
        # Slot 63 held row 63 from the async fill; ids 100..163 refill it, so
        # prove the refill landed by a shorter gather: the kernel only reads
        # slots[:n], and stage() zeroes invalid slots within 0..n-1 (slots at
        # or beyond n may keep stale bytes — they are never gathered).
        staging.stage(torch.arange(0, 32, dtype=torch.long), 0, 300)
        for slot in range(32):
            assert torch.equal(staging.buffer[slot], table[slot].view(torch.uint8))

    def test_stage_matches_pending_ids_without_refill(self, tmp_path):
        staging, table = self._make(tmp_path)
        ids = torch.arange(0, 64, dtype=torch.long)
        staging.stage_async(ids, 0, 300)
        assert staging.fill_pending(ids.numel()) is True

        calls = []
        orig = staging._read_into_slot

        def counting(fd, slot, row):
            calls.append((slot, row))
            return orig(fd, slot, row)

        staging._read_into_slot = counting
        n = staging.stage(ids, 0, 300)
        assert n == ids.numel()
        assert calls == [], "stage() on the consumed pending ids must not re-read"
        for slot, gid in enumerate(ids.tolist()):
            assert torch.equal(staging.buffer[slot], table[gid].view(torch.uint8))

    def test_overlapping_stage_async_serializes(self, tmp_path):
        staging, table = self._make(tmp_path)
        a = torch.arange(0, 64, dtype=torch.long)
        b = torch.arange(100, 164, dtype=torch.long)
        staging.stage_async(a, 0, 300)
        staging.stage_async(b, 0, 300)  # must wait for the first fill
        assert staging.fill_pending(b.numel()) is True
        for slot, gid in enumerate(b.tolist()):
            assert torch.equal(staging.buffer[slot], table[gid].view(torch.uint8))

    def test_small_fill_stays_synchronous(self, tmp_path):
        staging, table = self._make(tmp_path)
        ids = torch.tensor([5, 250, 299], dtype=torch.long)
        staging.stage_async(ids, 0, 300)
        # Small fills complete inline: no pending fill is left outstanding.
        assert staging._pending is None
        assert staging.fill_pending(3) is False  # nothing pending...
        for slot, gid in enumerate(ids.tolist()):
            assert torch.equal(staging.buffer[slot], table[gid].view(torch.uint8))

    def test_single_thread_fills_synchronously(self, tmp_path):
        # io_threads=1 has no pool: stage_async must still fill correctly
        # (inline), with nothing left pending for fill_pending to consume.
        staging, table = self._make(tmp_path, io_threads=1)
        ids = torch.arange(0, 40, dtype=torch.long)
        staging.stage_async(ids, 0, 300)
        assert staging.fill_pending(40) is False
        for slot, gid in enumerate(ids.tolist()):
            assert torch.equal(staging.buffer[slot], table[gid].view(torch.uint8))


class TestEnsureMaxRowsForPrefill:
    """m12 boot guard (docs/v100/CODE_REVIEW_TODO_20261110.md): stage_async
    raises on a gather wider than the staging capacity, which would kill the
    scheduler mid-serve on the first long prefill under a legal
    chunked_prefill_size > default max_rows / rows-per-token config. The boot
    helper sizes the staging for the prefill width bound instead."""

    def _patch_schedule(self, monkeypatch, chunked_prefill_size):
        import sglang.srt.runtime_context as rc

        schedule = SimpleNamespace(chunked_prefill_size=chunked_prefill_size)
        monkeypatch.setattr(rc, "get_schedule", lambda: schedule)

    def test_default_capacity_covers_default_chunk(self, monkeypatch):
        # 16384 tokens x 16 rows/token == the 262144 default: no-op, env left
        # untouched (still unset).
        monkeypatch.delenv("SGLANG_QWEN4_PLE_STAGED_MAX_ROWS", raising=False)
        self._patch_schedule(monkeypatch, 16384)
        ensure_max_rows_for_prefill(16)
        assert envs.SGLANG_QWEN4_PLE_STAGED_MAX_ROWS.get() == 262144
        assert not envs.SGLANG_QWEN4_PLE_STAGED_MAX_ROWS.is_set()

    def test_oversize_chunk_auto_raises(self, monkeypatch):
        monkeypatch.delenv("SGLANG_QWEN4_PLE_STAGED_MAX_ROWS", raising=False)
        self._patch_schedule(monkeypatch, 32768)
        ensure_max_rows_for_prefill(16)
        assert envs.SGLANG_QWEN4_PLE_STAGED_MAX_ROWS.get() == 32768 * 16

    def test_explicit_max_rows_too_small_fails_loud(self, monkeypatch):
        # An explicit SGLANG_QWEN4_PLE_STAGED_MAX_ROWS is operator intent:
        # silently overriding it would be a surprise memory bump. Refuse
        # instead of raising behind their back.
        monkeypatch.setenv("SGLANG_QWEN4_PLE_STAGED_MAX_ROWS", "4096")
        self._patch_schedule(monkeypatch, 4096)  # required = 65536 > 4096
        with pytest.raises(ValueError, match="SGLANG_QWEN4_PLE_STAGED_MAX_ROWS"):
            ensure_max_rows_for_prefill(16)

    def test_explicit_max_rows_compatible_passes(self, monkeypatch):
        monkeypatch.setenv("SGLANG_QWEN4_PLE_STAGED_MAX_ROWS", "65536")
        self._patch_schedule(monkeypatch, 4096)  # required = 65536
        ensure_max_rows_for_prefill(16)
        assert envs.SGLANG_QWEN4_PLE_STAGED_MAX_ROWS.get() == 65536

    def test_disabled_chunked_prefill_refused(self, monkeypatch):
        # -1 (chunking disabled) makes the prefill width unbounded: no finite
        # staging can cover it, so refuse at boot with guidance instead of
        # letting the first long prefill kill the scheduler.
        monkeypatch.delenv("SGLANG_QWEN4_PLE_STAGED_MAX_ROWS", raising=False)
        for bad in (-1, None, 0):
            self._patch_schedule(monkeypatch, bad)
            with pytest.raises(ValueError, match="chunked_prefill_size"):
                ensure_max_rows_for_prefill(16)


class TestEnvDefaults:
    def test_staged_env_defaults(self):
        assert envs.SGLANG_QWEN4_PLE_STAGED_DIR.get() != ""
        assert envs.SGLANG_QWEN4_PLE_STAGED_IO_THREADS.get() >= 1
        assert envs.SGLANG_QWEN4_PLE_STAGED_MAX_ROWS.get() >= 4096 * 16
