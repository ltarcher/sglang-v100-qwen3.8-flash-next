"""Adaptive mixed-prefill budget: learn a prefill token budget from completed
GPU steps so a mixed step (prefill chunks + resident decode tails) stays near
a latency target instead of stalling the residents for a whole chunk.

Port of 1Cat vLLM v1.5.2 (Apache-2.0), PR #860:
``vllm/v1/core/sched/mixed_prefill.py`` (controller) and
``vllm/v1/worker/mixed_prefill.py`` (nonblocking GPU timing). The algorithm is
kept as-is; the timing plumbing is sglang-shaped — and sglang-shaped means
*rank-deterministic*: every TP/PP rank runs its own scheduler and they only
stay in lockstep while every scheduling input is bit-identical across ranks.
A per-rank CUDA-event timing is not (GPUs resolve events microseconds apart,
and the quantized budget map amplifies that into divergent chunk caps —
observed as TP ranks scheduling different forwards and hanging in allreduce).
So exactly like DynamicChunkSizer, the timing is measured once on the leader
(world rank 0) with two CUDA events around ``forward_batch_generation`` and
broadcast over the CPU group; every rank feeds its local budget with the same
sample, so all ranks converge to identical budgets. One tiny broadcast per
completed mixed step (~750 ms apart) is free next to the step itself.

Two 1Cat simplifications apply here:
- sglang mixed steps always run eager (no extend CUDA graphs in this fork), so
  the graph/eager split fit collapses to the eager fit only;
- the scheduler caps ``chunked_prefill_size`` before ``PrefillAdder``, and the
  adder already deducts the mixed decode rows from that budget, so the cap is
  handed over as ``budget.tokens + num_mixed_decode`` to keep the prefill part
  at ``budget.tokens``.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.utils import broadcast_pyobj

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

logger = logging.getLogger(__name__)

# Broadcast protocol on the CPU group, one message per pending-head attempt:
# None = head events not ready yet (retry later), float = resolved elapsed,
# negative float = give up on this sample (do not feed it to the fit).
_BROADCAST_DISCARD = -1.0


@dataclass
class MixedPrefillStep:
    """One controlled mixed step. On the leader it carries two CUDA events on
    the forward stream; ``try_resolve`` never blocks. Followers create the
    sample event-free and learn the elapsed from the leader's broadcast."""

    prefill_tokens: int
    decode_tokens: int
    budget_tokens: int
    start_event: Optional[object] = None
    end_event: Optional[object] = None
    elapsed_ms: Optional[float] = None
    resolve_attempts: int = field(default=0)

    def try_resolve(self) -> bool:
        if self.elapsed_ms is not None:
            return True
        if self.start_event is None or self.end_event is None:
            return False
        if not (self.start_event.query() and self.end_event.query()):
            return False
        self.elapsed_ms = self.start_event.elapsed_time(self.end_event)
        return True


class MixedPrefillBudget:
    """Learn a token budget from completed GPU steps, without host-side waits.

    The first mixed step uses a conservative seed. Subsequent samples adapt to
    the actual model, device, TP size and context. This is a latency target,
    not a hard deadline: even a decode-only step can exceed the target.
    """

    # Weight of earlier samples in the step-cost fit.
    DECAY = 0.75

    def __init__(
        self,
        max_tokens: int,
        target_ms: float,
        min_tokens: int = 0,
        align: int = 1,
    ) -> None:
        self.max_tokens = max_tokens
        self.target_ms = target_ms
        self.min_tokens = min(min_tokens, max_tokens)
        # Budget tokens must land on ``align`` multiples: the KV pool's page
        # granularity is the alignment anchor of every chunk boundary (the QSA
        # compress write plan asserts prefix_lens % compress_ratio == 0 and
        # page_size % compress_ratio == 0, so page-aligned caps keep every
        # non-final chunk of a request ratio-aligned). The legacy uncapped
        # path only ever handed over the chunked_prefill_size, which is
        # page-aligned; an arbitrary learned budget would break that.
        self.align = max(1, min(align, max_tokens))
        self.tokens = self._align(
            max(min(512, max_tokens), self.min_tokens, self.align)
        )
        self.ms_per_token: Optional[float] = None
        # Steps whose prefill part is below this contribute no fit sample: at
        # that size the step cost is the per-step fixed overhead (kernel
        # launches, the sparse-attention per-step floor, decode rows), which
        # would only teach the fit that tokens are more expensive than they are.
        self.min_sample_tokens = 2 * min(16, max_tokens)
        # Decayed sums for the fit elapsed = fixed + per_token * tokens:
        # weight, x, y, x*x and x*y.
        self._fit = [0.0] * 5
        # Unreachable-target detection: see update().
        self.gave_up = False
        self._full_over_target = 0

    def _align(self, tokens: int) -> int:
        """Floor to the page alignment, never below one aligned chunk."""
        floored = tokens // self.align * self.align
        return max(floored, self.align)

    def _affine_budget(self) -> Optional[float]:
        """Budget from the fixed-plus-linear step cost, once chunks differ.

        A mixed step pays a fixed cost (the decode rows plus any launches
        that do not scale with prefill) on top of the prefill work. Charging
        all of it to prefill tokens shrinks the budget whenever the fixed
        part alone reaches the target, and the smaller chunks then make each
        token look even more expensive. Fitting both terms removes that
        feedback. A chunk smaller than the prefill that costs as much as the
        fixed part spends most of its step on overhead and only multiplies the
        number of mixed steps, so the budget never drops below that amount.
        Without this floor a fixed cost just under the target would leave a
        budget of a few tokens.
        """
        w, sx, sy, sxx, sxy = self._fit
        spread = w * sxx - sx * sx
        # Require a relative spread of the chunk sizes of about 10%.
        if w < 2 or spread <= 0.01 * sx * sx:
            return None
        per_token = (w * sxy - sx * sy) / spread
        fixed = max(0.0, (sy - per_token * sx) / w)
        if per_token <= 0:
            return float(self.max_tokens)
        return max(self.target_ms - fixed, fixed) / per_token

    def update(
        self,
        prefill_tokens: int,
        decode_tokens: int,
        budget_tokens: int,
        elapsed_ms: float,
    ) -> None:
        if (
            self.target_ms <= 0
            or prefill_tokens <= 0
            or decode_tokens <= 0
            or not math.isfinite(elapsed_ms)
            or elapsed_ms <= 0
        ):
            return
        if self.gave_up:
            return
        # A tiny boundary/tail chunk is dominated by fixed decode overhead.
        # It must not teach the controller that a full chunk is equally costly.
        if prefill_tokens < budget_tokens * 0.75:
            return
        if prefill_tokens < self.min_sample_tokens:
            return
        # Unreachable-target guard: with the budget already clamped to a small
        # fraction of max_tokens, a full-budget step that still blows past the
        # target means the per-step fixed cost alone exceeds it — no chunk size
        # can meet the target, and further shrinking only destroys TTFT.
        # Give up and leave the chunk uncapped.
        if elapsed_ms > self.target_ms and budget_tokens <= max(
            self.max_tokens // 8, self.min_sample_tokens
        ):
            self._full_over_target += 1
            if self._full_over_target >= 8:
                self.gave_up = True
                self.tokens = self._align(self.max_tokens)
                return
        else:
            self._full_over_target = 0
        x, y = float(prefill_tokens), float(elapsed_ms)
        w, sx, sy, sxx, sxy = (v * self.DECAY for v in self._fit)
        self._fit = [w + 1, sx + x, sy + y, sxx + x * x, sxy + x * y]
        cost = elapsed_ms / prefill_tokens
        self.ms_per_token = (
            cost if self.ms_per_token is None else 0.5 * self.ms_per_token + 0.5 * cost
        )
        affine = self._affine_budget()
        if affine is not None:
            desired = affine
        elif self.ms_per_token is None:
            return
        else:
            desired = self.target_ms / self.ms_per_token
        quantum = min(16, self.max_tokens)
        # Grow gradually, by at least one quantum so that a budget at the
        # quantum can recover. React promptly to a long-context stall.
        growth = 2.0
        # Proportional damping: how much the budget may grow is bounded by
        # how far under target the last full step actually ran. Step cost on
        # this hardware is noisy (dense prefill routing vs the sparse
        # per-step floor); without this, a run of cheap-looking small chunks
        # catapults the budget into multi-second stalls. A 2x margin is
        # required for a 2x jump, so the budget tracks evidence.
        if elapsed_ms < self.target_ms:
            growth = min(growth, self.target_ms / elapsed_ms)
        desired = min(desired, max(self.tokens * growth, self.tokens + quantum))
        self.tokens = self._align(
            max(
                quantum,
                self.min_tokens,
                self.align,
                min(self.max_tokens, int(desired) // quantum * quantum),
            )
        )


class MixedPrefillBudgetControl:
    """Scheduler-facing collaborator: cap the prefill chunk, time the mixed
    step on the forward stream (leader only), and feed the same resolved
    sample to every rank's budget over the CPU group.

    Keyed by ``batch.forward_iter`` because the overlap loop hands
    ``process_batch_result`` a snapshot copy of the batch; the sample object
    must not travel on it.

    Rank-determinism: every rank runs an identical controller driven by an
    identical sample stream. The leader resolves the head-of-line sample from
    its CUDA events; each ``process_batch_result`` call with a non-empty
    pending queue exchanges exactly one broadcast (value, "not ready", or
    "discard"), so followers pop/feed in lockstep with the leader. All other
    state mutations (inflight eviction, pending overflow) are pure functions
    of the identical batch sequence.
    """

    # Samples whose events are not ready at result-processing time wait here
    # (bounded) and are retried on later results, like 1Cat's pending deque.
    MAX_PENDING_STEPS = 8
    MAX_INFLIGHT_STEPS = 16
    # Head-of-line guard: if the leader cannot resolve the oldest sample's
    # events after this many result-processing rounds, drop it (identically
    # on every rank) instead of stalling the sample stream forever.
    MAX_RESOLVE_ATTEMPTS = 32

    def __init__(
        self,
        *,
        chunked_prefill_size: int,
        target_ms: float,
        min_tokens: int,
        align: int = 1,
        world_group=None,
    ):
        self.budget = MixedPrefillBudget(
            max_tokens=chunked_prefill_size,
            target_ms=target_ms,
            min_tokens=min_tokens,
            align=align,
        )
        self._world_group = world_group
        self._is_leader = world_group is None or world_group.rank == 0
        self._inflight: dict = {}
        self._pending: deque = deque()
        self._num_samples = 0
        self._logged_give_up = False

    def _exchange_head_sample(self) -> Optional[float]:
        """Resolve the pending head, leader-measured; return the elapsed the
        rank must feed (None = keep the head pending, DISCARD = drop it).

        The leader broadcasts its decision so every rank takes the same one;
        callers on every rank either pop or keep the head alike.
        """
        head = self._pending[0]
        if self._is_leader:
            resolved = head.elapsed_ms if head.try_resolve() else None
            if resolved is None:
                head.resolve_attempts += 1
                if head.resolve_attempts < self.MAX_RESOLVE_ATTEMPTS:
                    decision = None
                else:
                    decision = _BROADCAST_DISCARD
            else:
                decision = resolved
            if self._world_group is not None:
                broadcast_pyobj(
                    [decision],
                    self._world_group.rank,
                    self._world_group.cpu_group,
                    src=0,
                )
        else:
            (decision,) = broadcast_pyobj(
                [None], self._world_group.rank, self._world_group.cpu_group, src=0
            )
        if decision is None:
            return None
        if decision <= 0:
            return _BROADCAST_DISCARD
        return decision

    def maybe_cap_chunked_prefill_size(
        self,
        chunked_prefill_size: Optional[int],
        running_batch: ScheduleBatch,
        waiting_queue: list,
        chunked_reqs: list,
        is_mixed_chunk: bool,
    ) -> Optional[int]:
        """Cap the pass chunk so the prefill part of a mixed step fits the
        budget. Returns the (possibly unchanged) size, or None to keep the
        caller's value untouched.

        The gate mirrors the mix conditions in Scheduler.get_new_batch_prefill:
        a pass that will not actually mix (logprob / beam / input-embeds
        requests) must not inherit a capped chunk, because it would pay the
        TTFT cost without producing a timing sample for the budget.
        """
        if (
            chunked_prefill_size is None
            or not is_mixed_chunk
            or running_batch is None
            or running_batch.is_empty()
            or running_batch.return_logprob
        ):
            return None

        def _unmixable(req: Req) -> bool:
            return (
                req.return_logprob
                or req.beam_group is not None
                or req.input_embeds is not None
            )

        if any(_unmixable(r) for r in running_batch.reqs):
            return None
        if any(_unmixable(r) for r in waiting_queue):
            return None
        if any(_unmixable(r) for r in chunked_reqs):
            return None

        num_mixed_decode = len(running_batch.reqs)
        # PrefillAdder deducts the decode rows from rem_chunk_tokens; add them
        # back so the prefill part lands on budget.tokens.
        return min(chunked_prefill_size, self.budget.tokens + num_mixed_decode)

    def begin_step(self, batch: ScheduleBatch) -> Optional[MixedPrefillStep]:
        """Create the sample for a controlled mixed step. On the leader this
        also records the start event; must run on the forward stream, right
        before the forward is enqueued."""
        if not batch.forward_mode.is_mixed():
            return None
        decode_tokens = len(batch.decoding_reqs or ())
        if decode_tokens <= 0:
            return None
        if len(self._inflight) >= self.MAX_INFLIGHT_STEPS:
            self._inflight.clear()
        step = MixedPrefillStep(
            prefill_tokens=batch.extend_num_tokens - decode_tokens,
            decode_tokens=decode_tokens,
            budget_tokens=self.budget.tokens,
        )
        if self._is_leader:
            step.start_event = torch.cuda.Event(enable_timing=True)
            step.start_event.record()
        self._inflight[batch.forward_iter] = step
        return step

    def end_step(self, step: Optional[MixedPrefillStep]) -> None:
        """Record the end event after the forward is enqueued (leader only)."""
        if step is not None and self._is_leader:
            step.end_event = torch.cuda.Event(enable_timing=True)
            step.end_event.record()

    def process_batch_result(self, batch: ScheduleBatch) -> None:
        """Hand the step's sample to the budget once the leader resolves it.

        At most one head-of-line exchange per call, in strict FIFO order, so
        every rank keeps the same pending queue and the same sample stream.
        """
        step = self._inflight.pop(batch.forward_iter, None)
        if step is not None:
            self._pending.append(step)
            if len(self._pending) > self.MAX_PENDING_STEPS:
                self._pending.popleft()
        if not self._pending:
            return
        decision = self._exchange_head_sample()
        if decision is None:
            return
        head = self._pending.popleft()
        if decision != _BROADCAST_DISCARD:
            head.elapsed_ms = decision
            self._feed(head)

    def _feed(self, step: MixedPrefillStep) -> None:
        before = self.budget.tokens
        self.budget.update(
            prefill_tokens=step.prefill_tokens,
            decode_tokens=step.decode_tokens,
            budget_tokens=step.budget_tokens,
            elapsed_ms=step.elapsed_ms,
        )
        self._num_samples += 1
        if not self._is_leader:
            return
        if self.budget.gave_up and not self._logged_give_up:
            self._logged_give_up = True
            logger.warning(
                "Mixed prefill budget: target %.0fms is below the per-step "
                "fixed cost (full-budget steps still take %.1fms at %d "
                "tokens). Giving up; mixed steps run uncapped.",
                self.budget.target_ms,
                step.elapsed_ms,
                step.prefill_tokens,
            )
            return
        if self._num_samples <= 3 or self._num_samples % 50 == 0:
            logger.info(
                "Mixed prefill budget sample #%d: prefill=%d decode=%d "
                "elapsed=%.1fms budget %d -> %d tokens (target=%.0fms)",
                self._num_samples,
                step.prefill_tokens,
                step.decode_tokens,
                step.elapsed_ms,
                before,
                self.budget.tokens,
                self.budget.target_ms,
            )
        else:
            logger.debug(
                "Mixed prefill budget sample #%d: prefill=%d decode=%d "
                "elapsed=%.1fms budget %d -> %d tokens",
                self._num_samples,
                step.prefill_tokens,
                step.decode_tokens,
                step.elapsed_ms,
                before,
                self.budget.tokens,
            )
