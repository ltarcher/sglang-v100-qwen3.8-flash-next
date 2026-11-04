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
from sglang.srt.models.qwen4_exp_ple_staged import PleRowStaging

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


def prefetch_staged_ple_prefill(model, ple_batch) -> None:
    """Phase 2 prefill double buffering (docs/v100/PLE_NVME_TIER_DESIGN.md
    §4.2/§11.9.4): stage every PLE layer's rows once at extend-forward start.

    At this point the current stream holds only batch-prep kernels, so the
    per-layer ids D2H is cheap; the background fills then overlap early-layer
    compute instead of blocking the host between layers (each fill would
    otherwise sit on the critical path right where the next layer's launch is
    due). Each layer's gather later finds its staging pending via
    ``fill_pending`` and skips its own D2H and refill entirely.

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
    for entry in entries:
        ids = entry.embedding.compute_ngram_ids(ple_batch)
        entry.staging.stage_async(
            ids.detach().reshape(-1),
            vocab_start=entry.vocab_start,
            vocab_end=entry.vocab_end,
        )


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
