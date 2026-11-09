"""Compute-ahead decode prep gate for the staged PLE backend (Phase 2).

Revised per design doc section 11.6.1: sampling runs eagerly OUTSIDE the
decode CUDA graph, so right before a decode graph replay we can re-run the
exact id computation the captured graph will perform — same
``_prepare_ple_batch``/``compute_ngram_ids`` functions, same stream-ordered
n-gram history (committed by the previous step's graph), same static input
buffers. The result fills each PLE layer's pinned staging; the in-graph
staged gather then reads staging via its slot arange with zero host
participation. This is "compute-ahead", not "predictive": no mismatch
bitmap, no invalid markers, no pinned prep request buffer.

Hash multipliers are per PLE layer (``ple_layer_index`` feeds the seed), so
every layer gathers different rows of its table slice and needs its own
staging fill. One ``PlePrepEntry`` wires one layer; ``PleDecodePrep`` builds
the ``_PLEBatch`` exactly once per step and fans it out.

This module has no Triton or CUDA-kernel imports (CPU-testable contract of
``qwen4_exp_ple_staged.py``). Production wiring passes ``_prepare_ple_batch``
as the ``batch_builder``; the real per-layer hash semantics are covered on
hardware by the Phase 2 bit-exact oracle.
"""

import functools
import logging
import time

import torch

from sglang.srt.environ import envs
from sglang.srt.models.qwen4_exp_ple_staged import (
    PleAheadStaging,
    PleRowStaging,
)

logger = logging.getLogger(__name__)


class PlePrepEntry:
    """Per-PLE-layer prep wiring: id computation + staging + vocab range.

    ``embedding`` is duck-typed (``compute_ngram_ids(batch) -> device ids``)
    so tests can inject fakes; production passes the layer's
    ``Qwen4ExpNGramEmbedding``.
    """

    __slots__ = ("embedding", "staging", "vocab_start", "vocab_end")

    def __init__(
        self,
        embedding,
        staging: PleRowStaging,
        vocab_start: int,
        vocab_end: int,
    ):
        self.embedding = embedding
        self.staging = staging
        self.vocab_start = vocab_start
        self.vocab_end = vocab_end


class PleDecodePrep:
    """Launch-gate prep: fill every layer's staging for the batch about to
    replay, then the caller launches the decode graph immediately.

    ``batch_builder(input_ids, forward_batch) -> batch`` must be the exact
    callable the captured graph uses (``_prepare_ple_batch``) so gate ids and
    in-graph ids agree by construction.
    """

    def __init__(self, batch_builder, entries: list[PlePrepEntry]):
        if not entries:
            raise ValueError(
                "PleDecodePrep needs at least one PlePrepEntry (one per PLE layer)"
            )
        self.batch_builder = batch_builder
        self.entries = list(entries)
        # Oracle debugging state (SGLANG_QWEN4_PLE_STAGED_PREP_DEBUG=1):
        # gate ids of the previous decode step per entry, plus that step's
        # request set, so in-graph ids can be compared against gate ids.
        self._debug_last_flat: list = [None] * len(self.entries)
        self._debug_last_reqs: list = [None] * len(self.entries)
        self._debug_step = 0
        # Per-phase wall time accumulation (seconds): build, ids+D2H sync,
        # staging fill. Logged as a per-step mean every _TIMING_LOG_STEPS.
        self._timing = [0.0, 0.0, 0.0]
        self._timing_steps = 0

    _TIMING_LOG_STEPS = 500

    def prep_and_fill(self, input_ids: torch.Tensor, forward_batch) -> int:
        """Fill staging for all layers; returns the per-layer flat id count.

        Blocking on the host by design: this runs in the launch gate between
        the previous step's sample and this step's replay enqueue, where the
        D2H sync is already accounted for by the design doc's cost model.
        """
        t0 = time.perf_counter()
        batch = self.batch_builder(input_ids, forward_batch)
        build_s = time.perf_counter() - t0
        cursor = t0 + build_s
        n = -1
        flats = []
        for entry in self.entries:
            ids = entry.embedding.compute_ngram_ids(batch)
            flat = ids.detach().reshape(-1).to(torch.long).cpu()
            ta = time.perf_counter()
            flats.append(flat)
            if n < 0:
                n = int(flat.numel())
            elif int(flat.numel()) != n:
                raise RuntimeError(
                    "PLE layers disagree on the decode gather width: "
                    f"layer 0 produced {n} ids, a later layer produced "
                    f"{int(flat.numel())} — check per-layer embedding wiring"
                )
            entry.staging.stage(flat, entry.vocab_start, entry.vocab_end)
            tb = time.perf_counter()
            self._timing[1] += ta - cursor
            self._timing[2] += tb - ta
            cursor = tb
        self._debug_compare(forward_batch, flats)
        self._timing[0] += build_s
        self._timing_steps += 1
        if self._timing_steps % self._TIMING_LOG_STEPS == 0:
            steps = self._timing_steps
            build, sync, fill = (v / steps * 1e6 for v in self._timing)
            logger.info(
                "PLE prep timing over %d steps (per step): build %.0fus, "
                "ids+D2H sync %.0fus, staging fill %.0fus",
                steps,
                build,
                sync,
                fill,
            )
        return n

    def _debug_compare(self, forward_batch, flats: list) -> None:
        """PREP_DEBUG oracle: the in-graph gather recorded the ids it was
        captured to ignore; the first ``flat.cpu()`` of this step already
        synced the previous replay, so the buffer holds the previous step's
        in-graph ids — compare them against what this gate staged then."""
        if not envs.SGLANG_QWEN4_PLE_STAGED_PREP_DEBUG.get():
            return
        reqs = forward_batch.req_pool_indices.detach().cpu().tolist()
        for i, entry in enumerate(self.entries):
            last = self._debug_last_flat[i]
            last_reqs = self._debug_last_reqs[i]
            self._debug_last_flat[i] = flats[i]
            self._debug_last_reqs[i] = reqs
            if last is None or last_reqs != reqs:
                continue  # first decode step or a new request set: skip
            debug_ids = entry.embedding.ngram_embedding._staged_debug_ids
            if debug_ids is None:
                logger.error(
                    "PLE prep-debug step %d entry %d: no debug buffer wired; "
                    "comparison skipped",
                    self._debug_step,
                    i,
                )
                continue
            graph = debug_ids[: last.numel()].cpu()
            if not torch.equal(graph, last):
                diff = (graph != last).nonzero()[0].item()
                logger.error(
                    "PLE prep-debug step %d entry %d: gate ids != in-graph "
                    "ids (n=%d, first diff at %d: gate=%d graph=%d)",
                    self._debug_step,
                    i,
                    last.numel(),
                    diff,
                    int(last[diff]),
                    int(graph[diff]),
                )
            elif i == 0:
                # Positive confirmation that comparisons actually run.
                logger.info(
                    "PLE prep-debug step %d entry %d: gate ids == in-graph ids (n=%d)",
                    self._debug_step,
                    i,
                    last.numel(),
                )
        self._debug_step += 1


def _discover_staged_ple_entries(model) -> list:
    """Walk the model's decoder layers and return one ``PlePrepEntry`` per
    staged PLE embedding (creating its staging lazily). Shared by the decode
    prep gate and the prefill compute-ahead prefetch."""
    from sglang.srt.models.qwen4_exp import (
        Qwen4ExpAttentionDecoderLayer,
        Qwen4ExpForConditionalGeneration,
        Qwen4ExpLinearDecoderLayer,
        Qwen4ExpModel,
        Qwen4ExpPLELayer,
        Qwen4ExpPinnedHostEmbedding,
    )

    # The served instance is the VL wrapper (EntryClass); the PLE layers and
    # n-gram config live on its text tower (Qwen4ExpVLModel ⊂ Qwen4ExpModel).
    if isinstance(model, Qwen4ExpForConditionalGeneration):
        model = model.model
    if not isinstance(model, Qwen4ExpModel) or not model.has_ple:
        return []
    entries = []
    for layer in model.layers:
        if not isinstance(
            layer, (Qwen4ExpLinearDecoderLayer, Qwen4ExpAttentionDecoderLayer)
        ):
            continue
        ple = layer.ple
        if not isinstance(ple, Qwen4ExpPLELayer):
            continue
        ngram_embedding = ple.ple_embedding
        host_embedding = ngram_embedding.ngram_embedding
        if (
            not isinstance(host_embedding, Qwen4ExpPinnedHostEmbedding)
            or host_embedding._staged_table is None
        ):
            continue
        staging = host_embedding._staged_staging
        if staging is None:
            staging = PleRowStaging(
                table=host_embedding._staged_table,
                max_rows=envs.SGLANG_QWEN4_PLE_STAGED_MAX_ROWS.get(),
            )
            host_embedding._staged_staging = staging
        entries.append(
            PlePrepEntry(
                embedding=ngram_embedding,
                staging=staging,
                vocab_start=host_embedding.shard_indices.org_vocab_start_index,
                vocab_end=host_embedding.shard_indices.org_vocab_end_index,
            )
        )
    return entries


def _chunked_prefill_size() -> int:
    """The scheduler's chunked-prefill size, for chunk-ahead slicing (0 when
    unusable). Indirected so tests can monkeypatch it without pulling in the
    scheduler stack."""
    from sglang.srt.managers.schedule_batch import get_schedule

    return get_schedule().chunked_prefill_size or 0


# Chunk-ahead stash (docs/v100/PLE_NVME_TIER_DESIGN.md §11.9.5): future chunk
# tokens keyed by req pool slot, recorded by the scheduler at every
# prepare_for_extend and popped at the extend forward's start. Stale or
# overwritten entries are harmless — the consume side verifies the staged ids
# bit-exactly against the forward's own ids before trusting anything.
_PLE_CHUNK_STASH: dict[int, list] = {}


def record_ple_chunk_ahead(reqs, pool_indices_cpu) -> None:
    """Scheduler-side stash for the chunk-ahead prefetch: the future chunk
    tokens are pure scheduler state (``full_untruncated_fill_ids`` beyond the
    current ``extend_range``), so record them here keyed by the req pool slot.
    The model's extend-forward start pops the entry and speculatively stages
    chunks ``i+1..i+depth`` while chunk ``i`` computes."""
    depth = envs.SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD.get()
    if depth <= 0:
        return
    chunk_size = _chunked_prefill_size()
    if chunk_size <= 0:
        return
    for i, req in enumerate(reqs):
        end = req.extend_range.end
        fill_ids = req.full_untruncated_fill_ids
        if end >= len(fill_ids):
            continue
        _PLE_CHUNK_STASH[int(pool_indices_cpu[i])] = list(
            fill_ids[end : end + depth * chunk_size]
        )


def _pop_ple_chunk_ahead(ple_batch, forward_batch) -> tuple:
    """Gate + stash pop for the chunk-ahead window. Returns ``(pool_idx,
    tokens)`` only for the shape the speculative fill can serve exactly —
    one extend sequence in the batch (no verify strides, no multi-request
    rows) with a recorded stash; everything else falls back to the existing
    synchronous double buffer."""
    if ple_batch.ngram_context is None or ple_batch.mode.is_target_verify():
        return None, None
    if not ple_batch.mode.is_extend() or ple_batch.lengths.shape[0] != 1:
        return None, None
    pool_idx = int(forward_batch.req_pool_indices.reshape(-1)[0].item())
    tokens = _PLE_CHUNK_STASH.pop(pool_idx, None)
    if not tokens:
        return None, None
    return pool_idx, tokens


def _submit_ple_chunk_ahead(
    entries: list, ple_batch, stash_tokens: list, pool_idx: int, depth: int
) -> None:
    """Speculatively stage the next ``depth`` chunks' rows (§11.9.5).

    Chunk ``i+1``'s gather ids are pure input-side state: its ngram context is
    ``cat([history, chunk_i+1])`` where ``history`` is chunk ``i``'s last two
    tokens, and ``ple_batch.ngram_context`` is already ``cat([history, chunk_i])``
    — so windows ``[L_i + j*cs : L_i + (j+1)*cs]`` of
    ``cat([ple_batch.ngram_context, next_tokens])`` equal chunk ``i+1+j``'s own
    forward windows for any ``L_i >= 1`` (including the ``L_i == 1`` boundary
    where the history spans the old history column). Hashing these windows on
    the GPU with each layer's real ``_hash_contexts`` and slicing per chunk
    keeps the speculative ids bit-identical to what the future forward will
    compute; mismatches (chunk boundaries off ``cs``, scheduler budget splits,
    stale stash) are caught at consume time and fall back cleanly.
    """
    chunk_size = _chunked_prefill_size()
    if chunk_size <= 0:
        return
    embedding0 = entries[0].embedding
    ngram_context = ple_batch.ngram_context
    l_i = ngram_context.shape[1] - (embedding0.ngram_size - 1)
    next_tokens = torch.tensor(
        stash_tokens, dtype=torch.long, device=ngram_context.device
    ).unsqueeze(0)
    merged = torch.cat([ngram_context, next_tokens], dim=1)
    windows = merged.unfold(1, embedding0.ngram_size, 1)[:, l_i:]
    total_windows = windows.shape[1]
    for entry in entries:
        embedding = entry.embedding
        ahead = embedding.ngram_embedding._staged_ahead
        if ahead is None:
            continue
        # One hash + one D2H per layer for the whole window: chunk slices are
        # contiguous row ranges of the flattened [W, heads] id tensor.
        ids = embedding._hash_contexts(windows, decode_sized=False)
        ids_np = ids.reshape(-1).to(torch.long).cpu().numpy()
        heads = embedding.ngram_heads
        for j in range(depth):
            lo, hi = j * chunk_size, min((j + 1) * chunk_size, total_windows)
            if hi <= lo:
                break
            ahead.submit(
                pool_idx,
                ids_np[lo * heads : hi * heads],
                vocab_start=entry.vocab_start,
                vocab_end=entry.vocab_end,
            )


def prefetch_staged_ple_prefill(model, ple_batch, forward_batch) -> None:
    """Phase 2 prefill double buffering (docs/v100/PLE_NVME_TIER_DESIGN.md
    §4.2/§11.9.4) plus chunk-ahead prefetch (§11.9.5): stage every PLE
    layer's rows once at extend-forward start.

    At this point the current stream holds only batch-prep kernels, so the
    per-layer ids D2H is cheap; the background fills then overlap early-layer
    compute instead of blocking the host between layers (each fill would
    otherwise sit on the critical path right where the next layer's launch is
    due). Each layer's gather later finds its staging pending via
    ``fill_pending`` and skips its own D2H and refill entirely.

    With ``SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD=d > 0``, chunk ``i``'s rows are
    served from the speculative fill submitted at forward ``i-1`` (bit-exact
    id match at consume), the main staging is skipped, and chunks
    ``i+1..i+d`` are submitted for the next forwards — the fill tail wait
    moves entirely into earlier chunks' compute windows.

    No-op for other backends, non-extend batches, capture (captured decode /
    verify graphs are filled by the launch gate), and when disabled via
    ``SGLANG_QWEN4_PLE_STAGED_PREFILL_PREFETCH=0``."""
    if ple_batch is None or not envs.SGLANG_QWEN4_PLE_STAGED_PREFILL_PREFETCH.get():
        return
    if torch.cuda.is_current_stream_capturing():
        return
    entries = model._ple_staged_prefill_entries
    if entries is None:
        entries = _discover_staged_ple_entries(model)
        model._ple_staged_prefill_entries = entries
        if entries:
            logger.info(
                "PLE staged prefill prefetch wired: %d PLE layers", len(entries)
            )
    if not entries:
        return
    depth = envs.SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD.get()
    pool_idx, stash_tokens = None, None
    if depth > 0:
        pool_idx, stash_tokens = _pop_ple_chunk_ahead(ple_batch, forward_batch)
    for entry in entries:
        ids = entry.embedding.compute_ngram_ids(ple_batch)
        ids_flat = ids.detach().reshape(-1)
        host = entry.embedding.ngram_embedding
        serve = None
        if stash_tokens is not None:
            # One D2H here (cheap at extend-forward start) also proves the
            # previous forward's gathers finished, so the buffer served then
            # is safe to refill by the submits below.
            ids_np = ids_flat.to(torch.long).cpu().numpy()
            if host._staged_ahead is None:
                host._staged_ahead = PleAheadStaging(
                    table=host._staged_table,
                    depth=depth,
                    max_rows=envs.SGLANG_QWEN4_PLE_STAGED_MAX_ROWS.get(),
                )
                logger.info(
                    "PLE chunk-ahead prefetch wired: depth=%d, buffers=%d, max_rows=%d",
                    depth,
                    depth + 1,
                    envs.SGLANG_QWEN4_PLE_STAGED_MAX_ROWS.get(),
                )
            serve = host._staged_ahead.consume(pool_idx, ids_np)
            if serve is not None:
                # The gather reads this buffer directly; no main fill.
                host._staged_serve = serve
            else:
                entry.staging.stage_async(
                    torch.from_numpy(ids_np),
                    vocab_start=entry.vocab_start,
                    vocab_end=entry.vocab_end,
                )
        else:
            entry.staging.stage_async(
                ids_flat,
                vocab_start=entry.vocab_start,
                vocab_end=entry.vocab_end,
            )
    if stash_tokens is not None:
        _submit_ple_chunk_ahead(entries, ple_batch, stash_tokens, pool_idx, depth)


def build_qwen4_exp_ple_decode_prep(model):
    """Wire the compute-ahead decode gate for a staged-PLE Qwen4-Exp model.

    Called by the decode CUDA graph runner before capture (production wiring;
    no-op for every other model or backend). The runner gates both plain-DECODE
    and MTP TARGET_VERIFY replays through this prep — the MTP draft model
    carries no PLE layers (``ple_layer_ids=[]``) so the draft needs nothing:
      - one shared ``PleRowStaging`` per staged embedding, assigned to
        ``_staged_staging`` so the in-graph capture-safe gather reads the very
        buffer the gate fills (zero-filled first so capture warmup sees
        deterministic content, with the slot arange pre-created on device);
      - ``_prepare_ple_batch`` (the exact callable the captured graph runs)
        bound as the gate's ``batch_builder`` — its TARGET_VERIFY branch
        reproduces the captured verify layout when the runner passes the
        static buffers plus ``extend_seq_lens=None``.

    Returns None when the model carries no staged PLE table.
    """
    from sglang.srt.models.qwen4_exp import (
        Qwen4ExpForConditionalGeneration,
        Qwen4ExpModel,
        _prepare_ple_batch,
    )

    if isinstance(model, Qwen4ExpForConditionalGeneration):
        model = model.model
    entries = _discover_staged_ple_entries(model)
    if not entries or not isinstance(model, Qwen4ExpModel):
        return None
    for entry in entries:
        staging = entry.staging
        staging.buffer.zero_()
        staging.slot_arange(torch.device("cuda", torch.cuda.current_device()))
        if envs.SGLANG_QWEN4_PLE_STAGED_PREP_DEBUG.get():
            # Allocated before capture so the captured debug write does not
            # land in the graph's private memory pool.
            debug_ids = torch.zeros(
                envs.SGLANG_QWEN4_PLE_STAGED_MAX_ROWS.get(),
                dtype=torch.int64,
                device=torch.device("cuda", torch.cuda.current_device()),
            )
            entry.embedding.ngram_embedding._staged_debug_ids = debug_ids
    logger.info(
        "PLE staged decode prep gate wired: %d PLE layers, max_rows=%d",
        len(entries),
        envs.SGLANG_QWEN4_PLE_STAGED_MAX_ROWS.get(),
    )
    return PleDecodePrep(
        batch_builder=functools.partial(
            _prepare_ple_batch,
            ngram_size=model.ple_ngram_size,
            ngram_eos_token_id=model.ple_ngram_eos_token_id,
        ),
        entries=entries,
    )
