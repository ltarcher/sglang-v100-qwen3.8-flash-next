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
- shard-range edges (row 0, last row, one past the end).
"""

import os

import torch

from sglang.srt.environ import envs
from sglang.srt.models.qwen4_exp_ple_staged import (
    PLE_STAGED_MAGIC,
    PleRowStaging,
    PleStagedTable,
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


class TestEnvDefaults:
    def test_staged_env_defaults(self):
        assert envs.SGLANG_QWEN4_PLE_STAGED_DIR.get() != ""
        assert envs.SGLANG_QWEN4_PLE_STAGED_IO_THREADS.get() >= 1
        assert envs.SGLANG_QWEN4_PLE_STAGED_MAX_ROWS.get() >= 4096 * 16
