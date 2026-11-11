"""Unit tests for the staged PLE decode prep gate (Phase 2).

Phase 2 of docs/v100/PLE_NVME_TIER_DESIGN.md, revised per section 11.6.1:
sampling runs EAGERLY outside the decode CUDA graph, so the prep gate is
"compute-ahead", not "predictive" — right before a decode graph replay it
re-runs the exact same id computation the captured graph will perform
(same ``_prepare_ple_batch``/``compute_ngram_ids`` functions, same stream-
ordered pool state) and fills each PLE layer's staging buffer. There is no
prediction, hence no mismatch bitmap, no invalid markers, and no pinned prep
request buffer; the superseded section 11.3-11.5 machinery is gone.

Hash multipliers are per PLE layer (``ple_layer_index`` feeds the seed), so
every layer gathers different rows and needs its own staging fill. The gate
therefore wires one entry per layer and builds the ``_PLEBatch`` exactly
once per step.

Everything here runs on CPU: the module under test has no Triton or CUDA
imports and receives the batch builder via dependency injection (production
wires ``_prepare_ple_batch``; tests wire fakes). The per-layer hash semantics
of the real embeddings are covered on hardware by the Phase 2 bit-exact
oracle.
"""

import numpy as np
import pytest
import torch
from array import array
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.models.qwen4_exp_ple_prep import (
    PleDecodePrep,
    PlePrepEntry,
    _PLE_CHUNK_STASH,
    prefetch_staged_ple_prefill,
    record_ple_chunk_ahead,
    release_ple_chunk_stash,
)
from sglang.srt.models.qwen4_exp_ple_staged import (
    PleAheadStaging,
    PleRowStaging,
    PleStagedTable,
)

ROW_BYTES = 160
SHARD_ROWS = 64


def _make_table(rows: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, 255, (rows, ROW_BYTES), dtype=torch.uint8, generator=gen
    ).view(torch.float8_e4m3fn)


def _make_entry(
    tmpdir: str, tag: str, vocab_start: int, vocab_end: int, max_rows: int = 64
) -> tuple:
    """Returns (table_bytes_u8, staging, entry) for one fake PLE layer."""
    table = _make_table(SHARD_ROWS, seed=len(tag))
    staged = PleStagedTable(
        shape=(SHARD_ROWS, ROW_BYTES),
        dtype=torch.float8_e4m3fn,
        table_dir=tmpdir,
        tag=tag,
    )
    staged.materialize(table, reuse=False)
    staging = PleRowStaging(staged, max_rows=max_rows, io_threads=1)
    return (
        table.view(torch.uint8),
        staging,
        PlePrepEntry(
            embedding=None,  # replaced per test; kept out of the fixture
            staging=staging,
            vocab_start=vocab_start,
            vocab_end=vocab_end,
        ),
    )


class FakeEmbedding:
    """Returns the next id set from ``id_sets`` and records the batch seen.

    Id sets are shaped like the real ``compute_ngram_ids`` output (2D
    [tokens, heads_per_ngram]) so the flattening contract is exercised.
    """

    def __init__(self, id_sets: list[torch.Tensor]):
        self.id_sets = id_sets
        self.seen_batches = []

    def compute_ngram_ids(self, batch) -> torch.Tensor:
        self.seen_batches.append(batch)
        return self.id_sets[min(len(self.seen_batches), len(self.id_sets)) - 1]


class RecordingBuilder:
    """Stands in for ``_prepare_ple_batch``: records inputs, returns a canned
    batch object (identity is what downstream code relies on)."""

    def __init__(self, batch_obj):
        self.batch_obj = batch_obj
        self.calls = []

    def __call__(self, input_ids, forward_batch):
        self.calls.append((input_ids, forward_batch))
        return self.batch_obj


class TestPrepAndFill:
    def test_fills_each_layer_staging_independently(self, tmp_path):
        # Two PLE layers with DIFFERENT id sets (per-layer hash multipliers):
        # each staging must be filled from its own layer's ids.
        table_a_u8, staging_a, entry_a = _make_entry(str(tmp_path), "la", 0, 1 << 62)
        table_b_u8, staging_b, entry_b = _make_entry(str(tmp_path), "lb", 0, 1 << 62)
        ids_a = torch.arange(6).unsqueeze(1).expand(6, 4)  # [T, heads]
        ids_b = torch.arange(6, 12).unsqueeze(1).expand(6, 4)
        emb_a, emb_b = FakeEmbedding([ids_a]), FakeEmbedding([ids_b])
        entry_a.embedding, entry_b.embedding = emb_a, emb_b
        batch_obj = object()
        builder = RecordingBuilder(batch_obj)
        prep = PleDecodePrep(builder, [entry_a, entry_b])

        n = prep.prep_and_fill(torch.zeros(6, dtype=torch.long), forward_batch=None)

        assert n == 24  # 6 tokens x 4 heads, flattened
        assert len(builder.calls) == 1  # batch built once per step, not per layer
        assert emb_a.seen_batches[0] is batch_obj
        assert emb_b.seen_batches[0] is batch_obj

        got_a = staging_a.buffer[:24].view(torch.uint8)
        want_a = table_a_u8[ids_a.reshape(-1).tolist()]
        assert torch.equal(got_a, want_a)
        got_b = staging_b.buffer[:24].view(torch.uint8)
        want_b = table_b_u8[ids_b.reshape(-1).tolist()]
        assert torch.equal(got_b, want_b)

    def test_flatten_order_is_row_major(self, tmp_path):
        # Slot i must hold the row for flat_ids[i] with flat = ids.reshape(-1):
        # token-major then head-major, matching the eager gather's flattening.
        table_u8, staging, entry = _make_entry(str(tmp_path), "fm", 0, 1 << 62)
        tokens, heads = 3, 4
        ids = torch.randint(
            0, SHARD_ROWS, (tokens, heads), generator=torch.Generator().manual_seed(7)
        )
        entry.embedding = FakeEmbedding([ids])
        prep = PleDecodePrep(RecordingBuilder(object()), [entry])

        prep.prep_and_fill(torch.zeros(tokens, dtype=torch.long), forward_batch=None)

        flat = ids.reshape(-1).tolist()
        got = staging.buffer[: tokens * heads].view(torch.uint8)
        want = table_u8[flat]
        assert torch.equal(got, want)

    def test_out_of_vocab_slots_zeroed(self, tmp_path):
        # Ids outside [vocab_start, vocab_end) or beyond shard_rows must land
        # in zeroed slots (the kernel's in_range zeroing semantics).
        table_u8, staging, entry = _make_entry(str(tmp_path), "oz", 10, 10 + SHARD_ROWS)
        ids = torch.tensor(
            [[0, 10, 10 + SHARD_ROWS - 1, 10 + SHARD_ROWS, 5, 20]], dtype=torch.long
        )
        entry.embedding = FakeEmbedding([ids])
        prep = PleDecodePrep(RecordingBuilder(object()), [entry])

        prep.prep_and_fill(torch.zeros(1, dtype=torch.long), forward_batch=None)

        got = staging.buffer[:6].view(torch.uint8)
        table = table_u8
        assert torch.equal(got[0], torch.zeros(ROW_BYTES, dtype=torch.uint8))  # 0 < 10
        assert torch.equal(got[1], table[0])  # id 10 -> local row 0
        assert torch.equal(got[2], table[SHARD_ROWS - 1])  # id 73 -> local row 63
        assert torch.equal(
            got[3], torch.zeros(ROW_BYTES, dtype=torch.uint8)
        )  # >= vocab_end
        assert torch.equal(got[4], torch.zeros(ROW_BYTES, dtype=torch.uint8))
        assert torch.equal(got[5], table[10])  # id 20 -> local row 10


class TestBoundaries:
    def test_bucket_switch_refills_and_zeroes_stale_slots(self, tmp_path):
        # bs bucket shrank then grew again: slots within each gather's width
        # must always reflect the latest ids (rows refilled, within-width
        # invalid slots zeroed). Slots beyond the current width hold stale
        # bytes, which is safe by contract: the gather kernel's grid is
        # exactly the current flat width and never reads beyond it.
        table_u8, staging, entry = _make_entry(str(tmp_path), "bs", 0, 1 << 62)
        entry.embedding = FakeEmbedding(
            [
                torch.arange(8).unsqueeze(1).expand(8, 2),
                torch.arange(40, 43).unsqueeze(1).expand(3, 2),
            ]
        )
        prep = PleDecodePrep(RecordingBuilder(object()), [entry])

        prep.prep_and_fill(torch.zeros(8, dtype=torch.long), forward_batch=None)
        prep.prep_and_fill(torch.zeros(3, dtype=torch.long), forward_batch=None)

        got = staging.buffer.view(torch.uint8)
        table = table_u8
        assert torch.equal(got[:6], table[[40, 40, 41, 41, 42, 42]])

        # Widen again with trailing out-of-range ids: slots 6..15 fall back
        # inside the gather width, so they MUST be zeroed, not stale.
        ids3 = torch.full((1, 16), 1 << 20, dtype=torch.long)
        ids3[0, 0] = 40
        ids3[0, 1] = 41
        ids3[0, 2] = 42
        entry.embedding.id_sets.append(ids3)
        prep.prep_and_fill(torch.zeros(1, dtype=torch.long), forward_batch=None)

        assert torch.equal(got[0], table[40])
        assert torch.equal(got[1], table[41])
        assert torch.equal(got[2], table[42])
        assert torch.equal(got[3:16], torch.zeros(13, ROW_BYTES, dtype=torch.uint8))

    def test_max_rows_exceeded_raises_with_hint(self, tmp_path):
        _, staging, entry = _make_entry(str(tmp_path), "mr", 0, 1 << 62, max_rows=4)
        entry.embedding = FakeEmbedding(
            [torch.arange(5).unsqueeze(1).expand(5, 2)]
        )  # 10 flat ids > 4 slots
        prep = PleDecodePrep(RecordingBuilder(object()), [entry])

        try:
            prep.prep_and_fill(torch.zeros(5, dtype=torch.long), forward_batch=None)
        except ValueError as e:
            assert "SGLANG_QWEN4_PLE_STAGED_MAX_ROWS" in str(e)
        else:
            raise AssertionError("expected ValueError for staging overflow")

    def test_layer_width_disagreement_raises(self, tmp_path):
        # Same batch must yield the same flat width on every layer; a
        # disagreement means broken wiring — fail loud, fill nothing.
        _, staging_a, entry_a = _make_entry(str(tmp_path), "wa", 0, 1 << 62)
        _, staging_b, entry_b = _make_entry(str(tmp_path), "wb", 0, 1 << 62)
        entry_a.embedding = FakeEmbedding([torch.arange(6).unsqueeze(1).expand(6, 2)])
        entry_b.embedding = FakeEmbedding([torch.arange(4).unsqueeze(1).expand(4, 2)])
        prep = PleDecodePrep(RecordingBuilder(object()), [entry_a, entry_b])

        try:
            prep.prep_and_fill(torch.zeros(6, dtype=torch.long), forward_batch=None)
        except RuntimeError as e:
            assert "width" in str(e).lower()
        else:
            raise AssertionError("expected RuntimeError for width disagreement")

    def test_empty_entries_rejected(self):
        try:
            PleDecodePrep(RecordingBuilder(object()), [])
        except ValueError as e:
            assert "entry" in str(e).lower()
        else:
            raise AssertionError("expected ValueError for empty entries")

    def test_return_value_tracks_flat_width(self, tmp_path):
        _, staging, entry = _make_entry(str(tmp_path), "rv", 0, 1 << 62)
        entry.embedding = FakeEmbedding([torch.arange(7).unsqueeze(1).expand(7, 8)])
        prep = PleDecodePrep(RecordingBuilder(object()), [entry])

        assert (
            prep.prep_and_fill(torch.zeros(7, dtype=torch.long), forward_batch=None)
            == 56
        )


class TestEnvContract:
    def test_decode_prep_defaults_on(self):
        # Section 11.8: the Phase 2 main switch defaults to enabled; False
        # falls back to Phase 1 behavior (capture refused).
        assert envs.SGLANG_QWEN4_PLE_STAGED_DECODE_PREP.get() is True

    def test_chunk_ahead_defaults_off(self):
        # Section 11.9.5: chunk-ahead prefetch is opt-in (depth 0 = off);
        # existing single-buffer double-buffering behavior is untouched.
        assert envs.SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD.get() == 0


# ---------------------------------------------------------------------------
# Chunk-ahead prefetch (docs/v100/PLE_NVME_TIER_DESIGN.md section 11.9.5)
# ---------------------------------------------------------------------------


class FakeMode:
    """Duck-typed ForwardMode subset used by the prefetch gate."""

    def __init__(self, name="extend"):
        self.name = name

    def is_extend(self):
        return self.name in ("extend", "verify")

    def is_target_verify(self):
        return self.name == "verify"


class FakePleBatch:
    def __init__(self, ngram_context, lengths=(1,), mode_name="extend"):
        self.mode = FakeMode(mode_name)
        self.ngram_context = ngram_context
        self.lengths = torch.tensor(lengths)


class FakeForwardBatch:
    def __init__(self, pool_idx):
        self.req_pool_indices = torch.tensor([pool_idx], dtype=torch.long)


class FakeModel:
    def __init__(self, entries):
        self._ple_staged_prefill_entries = entries


class FakeHost:
    """Duck-typed Qwen4ExpPinnedHostEmbedding staged fields."""

    def __init__(self, table):
        self._staged_table = table
        self._staged_serve = None
        self._staged_ahead = None


class FakeNGramEmbedding:
    """Duck-typed Qwen4ExpNGramEmbedding: canned ids for the main compute-
    ahead path plus a real deterministic hash for the ahead windows (the
    prefetch calls ``_hash_contexts`` directly for those)."""

    def __init__(self, host, ngram_size=3, heads=2):
        self.ngram_embedding = host
        self.main_ids = None
        self.ngram_size = ngram_size
        self.ngram_heads = heads
        self.gather_dp_tokens = False
        self.contexts_seen = []

    def compute_ngram_ids(self, batch) -> torch.Tensor:
        return self.main_ids

    def _hash_contexts(self, contexts: torch.Tensor, *, decode_sized=False):
        # Real contract (qwen4_exp.py): 2D [W, ngram] in, [W, heads] out.
        assert contexts.dim() == 2, f"expected 2D windows, got {tuple(contexts.shape)}"
        self.contexts_seen.append(contexts.detach().clone())
        return (
            (contexts.sum(dim=-1, keepdim=True) % 251)
            .expand(-1, self.ngram_heads)
            .to(torch.long)
        )


class FakeReq:
    def __init__(self, fill_ids, end):
        self.full_untruncated_fill_ids = array("q", fill_ids)
        # record_ple_chunk_ahead only reads .end off the real Range object.
        self.extend_range = type("R", (), {"end": end})()


def _fake_hash(contexts: torch.Tensor, heads=2) -> torch.Tensor:
    assert contexts.dim() == 2
    return (contexts.sum(dim=-1, keepdim=True) % 251).expand(-1, heads).to(torch.long)


def _make_staged_table(tmp_path, tag: str, seed: int = 3):
    table_u8 = _make_table(SHARD_ROWS, seed=seed).view(torch.uint8)
    staged = PleStagedTable(
        shape=(SHARD_ROWS, ROW_BYTES),
        dtype=torch.float8_e4m3fn,
        table_dir=str(tmp_path),
        tag=tag,
    )
    staged.materialize(table_u8.view(torch.float8_e4m3fn), reuse=False)
    return table_u8, staged


@pytest.fixture(autouse=True)
def _cpu_only_capture_check(monkeypatch):
    """The CPU test container has no CUDA driver; the prefetch's capture
    check would initialize CUDA and raise. Production always has a device;
    tests pin the check to False."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


class TestPleRowWaitAndMatch:
    def test_matches_after_fill(self, tmp_path):
        table_u8, staged = _make_staged_table(tmp_path, "wm1")
        staging = PleRowStaging(staged, max_rows=64, io_threads=1)
        ids_np = np.arange(8, dtype=np.int64)
        staging.stage_async(torch.from_numpy(ids_np), 0, 1 << 62)

        assert staging.wait_and_match(ids_np) is True

    def test_rejects_different_ids(self, tmp_path):
        _, staged = _make_staged_table(tmp_path, "wm2")
        staging = PleRowStaging(staged, max_rows=64, io_threads=1)
        ids_np = np.arange(8, dtype=np.int64)
        staging.stage_async(torch.from_numpy(ids_np), 0, 1 << 62)

        assert staging.wait_and_match(ids_np + 100) is False

    def test_false_when_nothing_staged(self, tmp_path):
        _, staged = _make_staged_table(tmp_path, "wm3")
        staging = PleRowStaging(staged, max_rows=64, io_threads=1)

        assert staging.wait_and_match(np.arange(4, dtype=np.int64)) is False


class TestPleRowWaitAndMatchCompareBeforeWait:
    """m13 (docs/v100/CODE_REVIEW_TODO_20261110.md): the staged/pending ids
    are host-side numpy, so a chunk-ahead MISS must be decided without paying
    the outstanding fill's IO wait on the critical path; only a match blocks."""

    def _make_async_fill(self, tmp_path, tag):
        table_u8, staged = _make_staged_table(tmp_path, tag)
        staging = PleRowStaging(staged, max_rows=64, io_threads=2)
        ids_np = np.arange(64, dtype=np.int64)
        staging.stage_async(torch.from_numpy(ids_np), 0, 1 << 62)
        return table_u8, staging, ids_np

    def test_miss_returns_without_waiting(self, tmp_path, monkeypatch):
        _, staging, ids_np = self._make_async_fill(tmp_path, "wm4")
        called = []
        orig = staging._wait_pending

        def spying():
            called.append(True)
            return orig()

        monkeypatch.setattr(staging, "_wait_pending", spying)

        assert staging.wait_and_match(ids_np + 100) is False
        assert called == [], "a host-side id mismatch must not wait on the fill"
        assert staging._pending is not None, "the fill kept running in background"
        staging._wait_pending()  # drain before tmp_path teardown

    def test_hit_still_waits_for_fill(self, tmp_path):
        table_u8, staging, ids_np = self._make_async_fill(tmp_path, "wm5")

        assert staging.wait_and_match(ids_np) is True
        assert staging._pending is None
        got = staging.buffer[:64]
        assert torch.equal(got, table_u8[ids_np.tolist()])


class TestPleAheadStaging:
    def _make(self, tmp_path, tag: str, depth: int, max_rows: int = 64):
        table_u8, staged = _make_staged_table(tmp_path, tag)
        return table_u8, PleAheadStaging(
            staged, depth=depth, max_rows=max_rows, io_threads=1
        )

    def test_roundtrip_hit(self, tmp_path):
        table_u8, ahead = self._make(tmp_path, "ah1", depth=1)
        ids_np = np.arange(12, dtype=np.int64)
        assert ahead.submit(7, ids_np, 0, 1 << 62) is True

        serve = ahead.consume(7, ids_np)

        assert serve is not None
        got = serve.buffer[:12].view(torch.uint8)
        assert torch.equal(got, table_u8[ids_np.tolist()])

    def test_consume_empty_queue_returns_none(self, tmp_path):
        _, ahead = self._make(tmp_path, "ah2", depth=1)

        assert ahead.consume(1, np.arange(4, dtype=np.int64)) is None

    def test_consume_pool_mismatch_drops_head(self, tmp_path):
        _, ahead = self._make(tmp_path, "ah3", depth=1)
        ids_np = np.arange(8, dtype=np.int64)
        ahead.submit(7, ids_np, 0, 1 << 62)

        assert ahead.consume(8, ids_np) is None  # different pool slot: stale
        assert ahead.consume(7, ids_np) is None  # head was dropped, queue empty

    def test_consume_ids_mismatch_drops_head(self, tmp_path):
        _, ahead = self._make(tmp_path, "ah4", depth=1)
        ids_np = np.arange(8, dtype=np.int64)
        ahead.submit(7, ids_np, 0, 1 << 62)

        assert ahead.consume(7, ids_np + 1) is None
        assert ahead.consume(7, ids_np) is None  # entry was dropped on mismatch

    def test_fifo_order_across_entries(self, tmp_path):
        table_u8, ahead = self._make(tmp_path, "ah5", depth=2)
        ids_a = np.arange(4, dtype=np.int64)
        ids_b = np.arange(10, 20, dtype=np.int64)
        ahead.submit(3, ids_a, 0, 1 << 62)
        ahead.submit(3, ids_b, 0, 1 << 62)

        serve_a = ahead.consume(3, ids_a)
        serve_b = ahead.consume(3, ids_b)

        assert serve_a is not None and serve_b is not None
        assert torch.equal(
            serve_a.buffer[:4].view(torch.uint8), table_u8[ids_a.tolist()]
        )
        assert torch.equal(
            serve_b.buffer[:10].view(torch.uint8), table_u8[ids_b.tolist()]
        )

    def test_duplicate_submit_skipped(self, tmp_path):
        _, ahead = self._make(tmp_path, "ah6", depth=2)
        ids_np = np.arange(8, dtype=np.int64)

        assert ahead.submit(1, ids_np, 0, 1 << 62) is True
        assert ahead.submit(1, ids_np, 0, 1 << 62) is False  # already queued
        assert ahead.submit(1, ids_np + 1, 0, 1 << 62) is True

    def test_no_free_buffer_drops_submission(self, tmp_path):
        # depth d keeps d+1 buffers (one is the in-flight serve) and caps the
        # FIFO at d entries; a saturated window drops further submissions.
        _, ahead = self._make(tmp_path, "ah7", depth=1)
        ids_a = np.arange(8, dtype=np.int64)
        ids_b = np.arange(20, 24, dtype=np.int64)

        assert ahead.submit(1, ids_a, 0, 1 << 62) is True
        assert ahead.submit(1, ids_b, 0, 1 << 62) is False  # queue at cap

    def test_serve_buffer_protected(self, tmp_path):
        _, ahead = self._make(tmp_path, "ah8", depth=1)
        ids_a = np.arange(4, dtype=np.int64)
        ids_c = np.arange(30, 34, dtype=np.int64)
        ids_d = np.arange(40, 44, dtype=np.int64)
        ids_e = np.arange(50, 54, dtype=np.int64)
        ahead.submit(1, ids_a, 0, 1 << 62)

        serve_a = ahead.consume(1, ids_a)
        assert serve_a is not None
        assert ahead.submit(1, ids_c, 0, 1 << 62) is True  # second buffer
        # The served buffer is being gathered this forward: never reused.
        assert ahead.submit(1, ids_d, 0, 1 << 62) is False
        # Next consume releases the previous serve (caller's D2H drained it).
        serve_c = ahead.consume(1, ids_c)
        assert serve_c is not None
        assert ahead.submit(1, ids_e, 0, 1 << 62) is True

    def test_overflow_submission_returns_false_without_raise(self, tmp_path):
        _, ahead = self._make(tmp_path, "ah9", depth=1, max_rows=4)

        assert ahead.submit(1, np.arange(8, dtype=np.int64), 0, 1 << 62) is False


class TestPleChunkAheadRecord:
    def test_records_future_chunk_tokens(self, tmp_path, monkeypatch):
        import sglang.srt.models.qwen4_exp_ple_prep as prep

        monkeypatch.setattr(prep, "_chunked_prefill_size", lambda: 4)
        monkeypatch.setenv("SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD", "2")
        _PLE_CHUNK_STASH.clear()
        req = FakeReq(fill_ids=[10] * 10, end=2)  # 2 done, 8 remain

        record_ple_chunk_ahead([req], torch.tensor([5]))

        assert _PLE_CHUNK_STASH[5] == [10] * 8
        _PLE_CHUNK_STASH.clear()

    def test_disabled_env_records_nothing(self, monkeypatch):
        import sglang.srt.models.qwen4_exp_ple_prep as prep

        monkeypatch.setattr(prep, "_chunked_prefill_size", lambda: 4)
        monkeypatch.delenv("SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD", raising=False)
        _PLE_CHUNK_STASH.clear()
        req = FakeReq(fill_ids=[10] * 10, end=2)

        record_ple_chunk_ahead([req], torch.tensor([5]))

        assert _PLE_CHUNK_STASH == {}
        _PLE_CHUNK_STASH.clear()

    def test_req_without_next_chunk_skipped(self, monkeypatch):
        import sglang.srt.models.qwen4_exp_ple_prep as prep

        monkeypatch.setattr(prep, "_chunked_prefill_size", lambda: 4)
        monkeypatch.setenv("SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD", "2")
        _PLE_CHUNK_STASH.clear()
        req = FakeReq(fill_ids=[10] * 6, end=6)  # fully prefilled

        record_ple_chunk_ahead([req], torch.tensor([5]))

        assert _PLE_CHUNK_STASH == {}
        _PLE_CHUNK_STASH.clear()

    def test_slice_capped_at_remaining_tokens(self, monkeypatch):
        import sglang.srt.models.qwen4_exp_ple_prep as prep

        monkeypatch.setattr(prep, "_chunked_prefill_size", lambda: 4)
        monkeypatch.setenv("SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD", "2")
        _PLE_CHUNK_STASH.clear()
        req = FakeReq(fill_ids=[7] * 10, end=5)  # 5 remain < depth*cs = 8

        record_ple_chunk_ahead([req], torch.tensor([5]))

        assert _PLE_CHUNK_STASH[5] == [7] * 5
        _PLE_CHUNK_STASH.clear()

    def test_latest_record_wins_per_pool_slot(self, monkeypatch):
        import sglang.srt.models.qwen4_exp_ple_prep as prep

        monkeypatch.setattr(prep, "_chunked_prefill_size", lambda: 4)
        monkeypatch.setenv("SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD", "2")
        _PLE_CHUNK_STASH.clear()
        record_ple_chunk_ahead([FakeReq(fill_ids=[1] * 10, end=2)], torch.tensor([5]))
        record_ple_chunk_ahead([FakeReq(fill_ids=[2] * 10, end=4)], torch.tensor([5]))

        assert _PLE_CHUNK_STASH[5] == [2] * 6
        _PLE_CHUNK_STASH.clear()


class TestPleChunkStashRelease:
    """m16 (docs/v100/CODE_REVIEW_TODO_20261110.md): a stash entry keyed by a
    req pool slot must not outlive the request — when the request finishes,
    aborts, or retracts, the slot returns to the free pool and the slot's next
    occupant would consume the stale entry as a wrong speculative fill
    (correctness is guarded by the consume-side ids compare, but it wastes a
    full-capacity fill wait), and multi-request-batch entries are never
    consumed at all, so they used to leak forever."""

    def test_release_pops_entry(self):
        _PLE_CHUNK_STASH[42] = [1, 2, 3]
        release_ple_chunk_stash(42)
        assert 42 not in _PLE_CHUNK_STASH

    def test_release_unknown_slot_is_noop(self):
        release_ple_chunk_stash(2**40)
        assert True  # no raise

    def test_recorded_entry_cannot_outlive_release(self, monkeypatch):
        import sglang.srt.models.qwen4_exp_ple_prep as prep

        monkeypatch.setattr(prep, "_chunked_prefill_size", lambda: 4)
        monkeypatch.setenv("SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD", "2")
        _PLE_CHUNK_STASH.clear()
        req = FakeReq(fill_ids=[10] * 10, end=2)

        record_ple_chunk_ahead([req], torch.tensor([5]))
        assert _PLE_CHUNK_STASH[5] == [10] * 8

        release_ple_chunk_stash(5)
        assert _PLE_CHUNK_STASH == {}
        _PLE_CHUNK_STASH.clear()

    def test_release_kv_cache_invalidates_stash(self, monkeypatch):
        # The wiring point: release_kv_cache is the single choke point where a
        # finished/aborted/retracted request returns its req pool slot.
        # Contract mirrors ReqToTokenPool.free (memory_pool.py): the slot row
        # lives at req.kv.req_pool_idx (ReqKvInfo) and free() sets it to None.
        # The stash invalidation must therefore capture the slot BEFORE the
        # pool free, or it reads None and crashes every release (13e1db78c7
        # follow-up: int(None) TypeError).
        from sglang.srt.mem_cache import common as mem_common

        monkeypatch.setattr(
            mem_common, "_release_overallocated_kv_indices", lambda *a, **k: None
        )
        popped = []

        class FakeKV:
            holds_kv = True
            is_kv_released = False
            cache_protected_len = 0
            kv_allocated_len = 10
            req_pool_idx = 77

            def mark_kv_released(self):
                self.is_kv_released = True

        class FakeTreeCache:
            def claim_kv_row(self, req):
                return None

            def free_kv_row(self, kv, ranges):
                pass

            def unpin(self, req):
                pass

            def on_release(self, req, inserted):
                pass

            class req_to_token_pool:
                @staticmethod
                def free(req):
                    popped.append(req.kv.req_pool_idx)
                    req.kv.req_pool_idx = None

        req = SimpleNamespace(
            kv=FakeKV(),
            owned_kv_len=lambda: 10,
            skip_radix_cache_insert=True,
            rid="fake",
        )
        _PLE_CHUNK_STASH[77] = [1, 2, 3]

        mem_common.release_kv_cache(req, FakeTreeCache(), is_insert=False)

        assert popped == [77]
        assert 77 not in _PLE_CHUNK_STASH
        _PLE_CHUNK_STASH.clear()


class TestPleChunkAheadPrefetch:
    def _wire(self, tmp_path, tag: str):
        table_u8, staged = _make_staged_table(tmp_path, tag)
        host = FakeHost(staged)
        emb = FakeNGramEmbedding(host)
        entry = PlePrepEntry(
            embedding=emb,
            staging=PleRowStaging(staged, max_rows=64, io_threads=1),
            vocab_start=0,
            vocab_end=1 << 62,
        )
        return table_u8, host, emb, entry, FakeModel([entry])

    def _fake_hash(self, contexts):
        return _fake_hash(contexts, heads=2)

    def test_first_forward_miss_then_submits_window(self, tmp_path, monkeypatch):
        import sglang.srt.models.qwen4_exp_ple_prep as prep

        monkeypatch.setattr(prep, "_chunked_prefill_size", lambda: 4)
        monkeypatch.setenv("SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD", "2")
        _PLE_CHUNK_STASH.clear()
        table_u8, host, emb, entry, model = self._wire(tmp_path, "pf1")
        ple_batch = FakePleBatch(torch.tensor([[100, 101, 1, 2, 3, 4]]))
        fb = FakeForwardBatch(5)
        emb.main_ids = torch.tensor([[11, 12]] * 4)  # chunk i's own ids
        stash = [10, 11, 12, 13, 14, 15, 16, 17]
        _PLE_CHUNK_STASH[5] = list(stash)

        prefetch_staged_ple_prefill(model, ple_batch, fb)

        # Miss on the empty queue: the main staging takes chunk i's fill.
        assert host._staged_serve is None
        assert np.array_equal(
            entry.staging._staged_ids, emb.main_ids.reshape(-1).numpy()
        )

        # The ahead windows compose from the already-built ngram_context plus
        # the stash tokens: windows [L_i : L_i + T] of the merged sequence.
        merged = torch.tensor([[100, 101, 1, 2, 3, 4] + stash])
        windows = merged.unfold(1, 3, 1)
        assert len(emb.contexts_seen) == 1
        assert torch.equal(emb.contexts_seen[0], windows[:, 4:12].squeeze(0))

        queue = host._staged_ahead._queue
        assert len(queue) == 2
        exp1 = self._fake_hash(windows[:, 4:8].squeeze(0)).reshape(-1).numpy()
        exp2 = self._fake_hash(windows[:, 8:12].squeeze(0)).reshape(-1).numpy()
        assert queue[0][0] == 5 and np.array_equal(queue[0][1], exp1)
        assert queue[1][0] == 5 and np.array_equal(queue[1][1], exp2)
        _PLE_CHUNK_STASH.clear()

    def test_window_composition_l1_edge(self, tmp_path, monkeypatch):
        # L_i = 1: the committed history's last two tokens span the old
        # history column and the single chunk token; the merged-window slice
        # must still equal the real forward's own windows for chunk i+1.
        import sglang.srt.models.qwen4_exp_ple_prep as prep

        monkeypatch.setattr(prep, "_chunked_prefill_size", lambda: 4)
        monkeypatch.setenv("SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD", "2")
        _PLE_CHUNK_STASH.clear()
        _, host, emb, _, model = self._wire(tmp_path, "pf2")
        ple_batch = FakePleBatch(torch.tensor([[100, 101, 7]]))
        fb = FakeForwardBatch(5)
        emb.main_ids = torch.tensor([[11, 12]])  # 1 window x 2 heads
        stash = [10, 11, 12, 13]
        _PLE_CHUNK_STASH[5] = list(stash)

        prefetch_staged_ple_prefill(model, ple_batch, fb)

        # Real forward for chunk i+1: cat([[101, 7], [10, 11, 12, 13]]).
        real_ctx = torch.tensor([[101, 7, 10, 11, 12, 13]])
        assert torch.equal(emb.contexts_seen[0], real_ctx.unfold(1, 3, 1).squeeze(0))
        _PLE_CHUNK_STASH.clear()

    def test_second_forward_consumes_serve(self, tmp_path, monkeypatch):
        import sglang.srt.models.qwen4_exp_ple_prep as prep

        monkeypatch.setattr(prep, "_chunked_prefill_size", lambda: 4)
        monkeypatch.setenv("SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD", "2")
        _PLE_CHUNK_STASH.clear()
        table_u8, host, emb, entry, model = self._wire(tmp_path, "pf3")
        ple_batch = FakePleBatch(torch.tensor([[100, 101, 1, 2, 3, 4]]))
        fb = FakeForwardBatch(5)
        emb.main_ids = torch.tensor([[11, 12]] * 4)
        _PLE_CHUNK_STASH[5] = [10, 11, 12, 13, 14, 15, 16, 17]
        prefetch_staged_ple_prefill(model, ple_batch, fb)
        chunk1_ids = host._staged_ahead._queue[0][1]

        # Forward i+1: the ple_batch advances (history = chunk i's last two
        # tokens), the main ids are exactly the speculatively staged chunk
        # i+1 ids -> serve hit, main staging untouched.
        emb.main_ids = torch.from_numpy(chunk1_ids.reshape(4, 2))
        ple_batch.ngram_context = torch.tensor([[3, 4, 10, 11, 12, 13]])
        emb.contexts_seen.clear()
        _PLE_CHUNK_STASH[5] = [20, 21, 22, 23, 24, 25, 26, 27]
        prefetch_staged_ple_prefill(model, ple_batch, fb)

        assert host._staged_serve is not None
        got = host._staged_serve.buffer[:8].view(torch.uint8)
        assert torch.equal(got, table_u8[chunk1_ids.tolist()])
        # Main staging was skipped this forward: still holding forward i's ids.
        assert np.array_equal(
            entry.staging._staged_ids, [11, 12, 11, 12, 11, 12, 11, 12]
        )
        # Sliding window: one entry consumed, the next chunk admitted.
        assert len(host._staged_ahead._queue) == 2
        _PLE_CHUNK_STASH.clear()

    def test_ids_mismatch_falls_back_to_main_stage(self, tmp_path, monkeypatch):
        import sglang.srt.models.qwen4_exp_ple_prep as prep

        monkeypatch.setattr(prep, "_chunked_prefill_size", lambda: 4)
        monkeypatch.setenv("SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD", "2")
        _PLE_CHUNK_STASH.clear()
        table_u8, host, emb, entry, model = self._wire(tmp_path, "pf4")
        ple_batch = FakePleBatch(torch.tensor([[100, 101, 1, 2, 3, 4]]))
        fb = FakeForwardBatch(5)
        emb.main_ids = torch.tensor([[11, 12]] * 4)
        _PLE_CHUNK_STASH[5] = [10, 11, 12, 13, 14, 15, 16, 17]
        prefetch_staged_ple_prefill(model, ple_batch, fb)

        # The speculative fill cannot be trusted: the forward's own ids
        # differ from the staged head -> serve stays None, the main staging
        # takes the fill, and the stale head is dropped.
        emb.main_ids = torch.tensor([[90, 91]] * 4)
        _PLE_CHUNK_STASH[5] = [20, 21, 22, 23, 24, 25, 26, 27]
        prefetch_staged_ple_prefill(model, ple_batch, fb)

        assert host._staged_serve is None
        assert np.array_equal(
            entry.staging._staged_ids, [90, 91, 90, 91, 90, 91, 90, 91]
        )
        _PLE_CHUNK_STASH.clear()

    def test_multi_sequence_batch_skips_ahead(self, tmp_path, monkeypatch):
        import sglang.srt.models.qwen4_exp_ple_prep as prep

        monkeypatch.setattr(prep, "_chunked_prefill_size", lambda: 4)
        monkeypatch.setenv("SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD", "2")
        _PLE_CHUNK_STASH.clear()
        _, host, emb, entry, model = self._wire(tmp_path, "pf5")
        ple_batch = FakePleBatch(
            torch.tensor([[100, 101, 1, 2], [102, 103, 3, 4]]), lengths=(2, 2)
        )
        fb = FakeForwardBatch(5)
        emb.main_ids = torch.tensor([[11, 12], [13, 14]])
        stash = [10, 11, 12, 13]
        _PLE_CHUNK_STASH[5] = list(stash)

        prefetch_staged_ple_prefill(model, ple_batch, fb)

        # Only the shape the speculative fill can serve exactly gets ahead
        # fills; everything else keeps the synchronous double buffer.
        assert host._staged_ahead is None
        assert host._staged_serve is None
        assert _PLE_CHUNK_STASH[5] == stash  # stash untouched
        assert np.array_equal(entry.staging._staged_ids, [11, 12, 13, 14])
        _PLE_CHUNK_STASH.clear()

    def test_verify_mode_skips_ahead(self, tmp_path, monkeypatch):
        import sglang.srt.models.qwen4_exp_ple_prep as prep

        monkeypatch.setattr(prep, "_chunked_prefill_size", lambda: 4)
        monkeypatch.setenv("SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD", "2")
        _PLE_CHUNK_STASH.clear()
        _, host, emb, _, model = self._wire(tmp_path, "pf6")
        ple_batch = FakePleBatch(torch.tensor([[100, 101, 1, 2]]), mode_name="verify")
        fb = FakeForwardBatch(5)
        emb.main_ids = torch.tensor([[11, 12]])
        _PLE_CHUNK_STASH[5] = [10, 11, 12, 13]

        prefetch_staged_ple_prefill(model, ple_batch, fb)

        assert host._staged_ahead is None
        assert host._staged_serve is None
        _PLE_CHUNK_STASH.clear()

    def test_final_partial_chunk_never_serves(self, tmp_path, monkeypatch):
        # The scheduler records no stash once a request is fully prefilled
        # (record_ple_chunk_ahead skips end >= len(fill_ids)), so the final
        # chunk — partial or exactly full — never pops a stash and never
        # consults the ahead queue: it always gathers from the main staging,
        # even when its speculative fill (submitted by the previous forward)
        # is bit-exact. The stale fill lingers in the queue until a later
        # consume drains it (see test_stale_entries_harmless_across_requests).
        import sglang.srt.models.qwen4_exp_ple_prep as prep

        monkeypatch.setattr(prep, "_chunked_prefill_size", lambda: 4)
        monkeypatch.setenv("SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD", "2")
        _PLE_CHUNK_STASH.clear()
        table_u8, host, emb, entry, model = self._wire(tmp_path, "pf8")
        fb = FakeForwardBatch(5)

        # Chunk 1 of 3 (fill [1..10] -> chunks 4+4+2): queue empty, so the
        # main staging takes chunk 1's fill and chunks 2+3 are submitted.
        emb.main_ids = _fake_hash(
            torch.tensor([[100, 101, 1, 2, 3, 4]]).unfold(1, 3, 1).squeeze(0)
        )
        _PLE_CHUNK_STASH[5] = [5, 6, 7, 8, 9, 10]
        prefetch_staged_ple_prefill(
            model, FakePleBatch(torch.tensor([[100, 101, 1, 2, 3, 4]])), fb
        )
        assert host._staged_serve is None
        assert len(host._staged_ahead._queue) == 2

        # Chunk 2 (full): its speculative fill is bit-exact -> serve hit.
        ctx2 = torch.tensor([[3, 4, 5, 6, 7, 8]])
        emb.main_ids = _fake_hash(ctx2.unfold(1, 3, 1).squeeze(0))
        _PLE_CHUNK_STASH[5] = [9, 10]
        prefetch_staged_ple_prefill(model, FakePleBatch(ctx2), fb)
        serve = host._staged_serve
        assert serve is not None
        serve_ids = emb.main_ids.reshape(-1).numpy()
        assert np.array_equal(serve._staged_ids, serve_ids)
        assert torch.equal(
            serve.buffer[:8].view(torch.uint8), table_u8[serve_ids.tolist()]
        )
        # The gather consumes the serve once (consume-once semantics).
        host._staged_serve = None

        # Final partial chunk (2 tokens < cs): no stash recorded, no consume
        # — main path even though the queued fill for it is bit-exact.
        ctx3 = torch.tensor([[7, 8, 9, 10]])
        emb.main_ids = _fake_hash(ctx3.unfold(1, 3, 1).squeeze(0))
        prefetch_staged_ple_prefill(model, FakePleBatch(ctx3), fb)
        assert host._staged_serve is None
        assert np.array_equal(entry.staging._staged_ids, [24, 24, 27, 27])
        stale_ids = host._staged_ahead._queue[0][1]
        assert np.array_equal(stale_ids, entry.staging._staged_ids)
        _PLE_CHUNK_STASH.clear()

    def test_stale_entries_harmless_across_requests(self, tmp_path, monkeypatch):
        # A finished request leaves its unconsumed speculative fills in the
        # queue (its final chunk never consumes). When a new request reuses
        # the pool slot, the first consume drops the stale head on the ids
        # mismatch and the fresh request serves normally afterwards — stale
        # entries cost at most one dropped consume, never wrong rows.
        import sglang.srt.models.qwen4_exp_ple_prep as prep

        monkeypatch.setattr(prep, "_chunked_prefill_size", lambda: 4)
        monkeypatch.setenv("SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD", "2")
        _PLE_CHUNK_STASH.clear()
        _, host, emb, entry, model = self._wire(tmp_path, "pf9")
        fb = FakeForwardBatch(5)

        # Request 1: fill [1..6] -> chunks 4+2 (final partial). Its chunk-2
        # speculative fill is never consumed.
        ctx1 = torch.tensor([[100, 101, 1, 2, 3, 4]])
        emb.main_ids = _fake_hash(ctx1.unfold(1, 3, 1).squeeze(0))
        _PLE_CHUNK_STASH[5] = [5, 6]
        prefetch_staged_ple_prefill(model, FakePleBatch(ctx1), fb)
        assert len(host._staged_ahead._queue) == 1

        ctx2 = torch.tensor([[3, 4, 5, 6]])
        emb.main_ids = _fake_hash(ctx2.unfold(1, 3, 1).squeeze(0))
        prefetch_staged_ple_prefill(model, FakePleBatch(ctx2), fb)
        assert host._staged_serve is None  # final chunk: main path
        assert np.array_equal(entry.staging._staged_ids, [12, 12, 15, 15])
        assert len(host._staged_ahead._queue) == 1  # stale fill lingers

        # Request 2 reuses pool slot 5: fill [20..29] -> chunks 4+4+2. The
        # first consume drops request 1's stale head (ids mismatch) and the
        # main staging takes the fill.
        ctx3 = torch.tensor([[200, 201, 20, 21, 22, 23]])
        emb.main_ids = _fake_hash(ctx3.unfold(1, 3, 1).squeeze(0))
        _PLE_CHUNK_STASH[5] = [24, 25, 26, 27, 28, 29]
        prefetch_staged_ple_prefill(model, FakePleBatch(ctx3), fb)
        assert host._staged_serve is None  # stale head dropped -> sync path
        assert np.array_equal(
            entry.staging._staged_ids, emb.main_ids.reshape(-1).numpy()
        )
        assert len(host._staged_ahead._queue) == 2  # only request-2 entries

        # Request 2's chunk 2: its speculative fill is bit-exact -> serve
        # hit; request 1's staleness never leaked into it.
        ctx4 = torch.tensor([[22, 23, 24, 25, 26, 27]])
        emb.main_ids = _fake_hash(ctx4.unfold(1, 3, 1).squeeze(0))
        _PLE_CHUNK_STASH[5] = [28, 29]
        prefetch_staged_ple_prefill(model, FakePleBatch(ctx4), fb)
        serve = host._staged_serve
        assert serve is not None
        assert np.array_equal(serve._staged_ids, emb.main_ids.reshape(-1).numpy())
        _PLE_CHUNK_STASH.clear()

    def test_disabled_env_keeps_legacy_behavior(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD", raising=False)
        _PLE_CHUNK_STASH.clear()
        _, host, emb, entry, model = self._wire(tmp_path, "pf7")
        ple_batch = FakePleBatch(torch.tensor([[100, 101, 1, 2, 3, 4]]))
        fb = FakeForwardBatch(5)
        emb.main_ids = torch.tensor([[11, 12]] * 4)
        _PLE_CHUNK_STASH[5] = [10, 11, 12, 13, 14, 15, 16, 17]

        prefetch_staged_ple_prefill(model, ple_batch, fb)

        # Depth 0: no ahead pool, no stash pop — exactly the phase-2 double
        # buffering behavior before this feature.
        assert host._staged_ahead is None
        assert _PLE_CHUNK_STASH[5] == [10, 11, 12, 13, 14, 15, 16, 17]
        assert np.array_equal(
            entry.staging._staged_ids, [11, 12, 11, 12, 11, 12, 11, 12]
        )
        _PLE_CHUNK_STASH.clear()


class _RecordingGatherKernel:
    """Triton-style stand-in for _gather_ple_embedding_from_pinned_kernel."""

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


class TestGatherStagedServeWidthGuard:
    """Consume-side width guard for the chunk-ahead serve path (review M5,
    2026-11-10).

    The serve buffer is filled from the previous forward's ple_batch, whose
    width is the LOCAL rank's token count; ``_gather_staged`` consumes
    ``flat_ids`` whose width is the GLOBAL gathered width under DP attention.
    A width mismatch makes the gather read the wrong pinned rows (n <=
    max_rows) or index past the buffer end with a wild
    ``buffer.data_ptr()+slot*row_bytes`` dereference (n > max_rows) — both
    silent, and the prep-side bit-exact id match cannot catch it because its
    two compared sides are both local-width. The consume side must refuse to
    serve when the staged width differs from the live ids width.
    """

    def _make_host(self):
        from sglang.srt.models.qwen4_exp import Qwen4ExpPinnedHostEmbedding

        host = object.__new__(Qwen4ExpPinnedHostEmbedding)
        host.embedding_dim = 40
        host._emit_dtype = torch.float16
        host._block_d = 32
        host.weight = torch.zeros(1, dtype=torch.float8_e4m3fn)
        host._staged_staging = object()  # non-None: skips PleRowStaging build
        host._staged_serve = None
        host._staged_ahead_hits = 0
        host._staged_stage_ms = 0.0
        host._staged_stage_n = 0
        host._staged_debug_ids = None
        return host

    @staticmethod
    def _make_serve(staged_ids):
        from types import SimpleNamespace

        # Production _staged_ids is ALWAYS a numpy array (PleRowStaging
        # _to_ids_np); a torch stub hides dtype-dependent bugs in the guard
        # itself (crash: numpy has no .numel — 2026-11-11 production SIGQUIT).
        return SimpleNamespace(
            _staged_ids=np.asarray(staged_ids, dtype=np.int64),
            buffer=SimpleNamespace(data_ptr=lambda: 0),
            slot_arange=lambda device: torch.arange(8, device=device, dtype=torch.long),
        )

    def _gather(self, host, serve, n):
        host._staged_serve = serve
        flat_ids = torch.arange(n, dtype=torch.long)
        output = torch.empty(n, host.embedding_dim, dtype=host._emit_dtype)
        import sglang.srt.models.qwen4_exp as q4

        kernel = _RecordingGatherKernel()
        monkeypatch_guard = pytest.MonkeyPatch()
        monkeypatch_guard.setattr(
            torch.cuda, "is_current_stream_capturing", lambda: False
        )
        monkeypatch_guard.setattr(
            q4, "_gather_ple_embedding_from_pinned_kernel", kernel
        )
        try:
            return host._gather_staged(flat_ids, output), kernel
        finally:
            monkeypatch_guard.undo()

    def test_width_mismatch_refuses_to_serve(self):
        # The DP shape: staged from a local 5-token chunk, consume gets the
        # 8-token global flat batch. The gather must fail loud before the
        # kernel launch — reading the buffer would be wrong rows.
        host = self._make_host()
        serve = self._make_serve([10, 11, 12, 13, 14])
        with pytest.raises(RuntimeError, match="chunk-ahead"):
            self._gather(host, serve, 8)
        assert host._staged_serve is None  # consumed once, never re-served

    def test_width_mismatch_refuses_even_under_capacity(self):
        # n <= max_rows still reads the wrong rows; the guard must not
        # degrade into "fits in the buffer, serve anyway".
        host = self._make_host()
        serve = self._make_serve([10, 11, 12])
        with pytest.raises(RuntimeError, match="chunk-ahead"):
            self._gather(host, serve, 5)

    def test_width_match_serves_normally(self):
        # Healthy non-DP operation: prep-side bit-exact match implies equal
        # width; the gather proceeds with the recorded kernel.
        host = self._make_host()
        serve = self._make_serve([10, 11, 12, 13])
        output, kernel = self._gather(host, serve, 4)
        assert output.shape == (4, host.embedding_dim)
        assert len(kernel.calls) == 1
        assert kernel.calls[0][0] == (4,)
        assert host._staged_serve is None


class TestStagedWeightUpdateGuard:
    """update_weights_from_disk vs the staged backend (review m11,
    2026-11-10).

    After materialize, _release_loading_table swaps the full pinned loading
    table for a 1-row dummy parameter that carries no weight_loader, so an
    online weight update falls into default_weight_loader and dies on a
    cryptic shape mismatch while the staged NVMe shard keeps serving stale
    rows. The loading table is already freed (12+ GiB pinned), so an
    in-place re-materialize cannot be supported; the placeholder must
    reject updates with a self-explanatory error instead.
    """

    def _make_host(self):
        import torch.nn as nn
        from types import SimpleNamespace

        from sglang.srt.models.qwen4_exp import Qwen4ExpPinnedHostEmbedding

        host = object.__new__(Qwen4ExpPinnedHostEmbedding)
        nn.Module.__init__(host)
        host.embedding_dim = 8
        host.weight = nn.Parameter(
            torch.empty(4, 8, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        host._staged_table = SimpleNamespace(path="p", shard_rows=4, row_bytes=8)
        return host

    def test_released_placeholder_rejects_weight_update(self):
        host = self._make_host()
        host._release_loading_table()
        dummy = host.get_parameter("weight")
        assert dummy.shape == (1, host.embedding_dim)
        loader = getattr(dummy, "weight_loader", None)
        assert loader is not None, "dummy must carry an explicit guard"
        with pytest.raises(RuntimeError, match="staged"):
            loader(dummy, torch.empty(4, 8, dtype=torch.float8_e4m3fn))

    def test_untouched_loading_table_still_loadable(self):
        # Non-materialized staged table (or pinned/file backend): the real
        # table stays installed, no guard is attached, the generic loader
        # keeps working.
        host = self._make_host()
        host._staged_table = None
        host._release_loading_table()
        assert host.get_parameter("weight").shape == (4, host.embedding_dim)
        loader = getattr(host.get_parameter("weight"), "weight_loader", None)
        assert loader is None
