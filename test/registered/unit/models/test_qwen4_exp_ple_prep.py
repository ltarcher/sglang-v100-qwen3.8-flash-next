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

import torch

from sglang.srt.environ import envs
from sglang.srt.models.qwen4_exp_ple_prep import (
    PleDecodePrep,
    PlePrepEntry,
)
from sglang.srt.models.qwen4_exp_ple_staged import (
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
