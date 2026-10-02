"""Adaptive speculative decoding parameters.

Adjusts speculative_num_steps at runtime based on observed acceptance lengths.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
from functools import cached_property
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.overrides import (
    resolved_view,
    resolving_view,
)
from sglang.srt.environ import envs
from sglang.srt.utils import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

DEFAULT_ADAPTIVE_CONFIG: dict[str, dict] = {
    "1": {
        "candidate_steps": [1, 3, 5, 7],
        "up_hysteresis": 0.0,
        "down_hysteresis": -0.25,
        "ceiling_coeff": 0,
    },
    "8": {
        "candidate_steps": [0, 1, 3],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
    "32": {
        "candidate_steps": [0, 1],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
    "64": {
        "candidate_steps": [0],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
}


def adaptive_unsupported_reason(server_args: ServerArgs) -> str | None:
    """Return why adaptive spec cannot run under the given server args, or None if supported."""

    cfg = resolving_view(server_args)

    # NEXTN aliases to the EAGLE worker, so it takes the same adaptive path.
    if cfg.speculative_algorithm not in ("EAGLE", "EAGLE3", "NEXTN"):
        return (
            f"speculative_algorithm={cfg.speculative_algorithm} "
            "(only EAGLE/EAGLE3/NEXTN are supported)"
        )
    if cfg.speculative_eagle_topk is not None and cfg.speculative_eagle_topk != 1:
        return (
            f"speculative_eagle_topk={cfg.speculative_eagle_topk} "
            "(only topk=1 is supported)"
        )
    if resolved_view(server_args).enable_dp_attention:
        return (
            "enable_dp_attention=True is not supported "
            "(adaptive tier decisions are not synchronized across DP ranks)"
        )
    if resolved_view(server_args).enable_multi_layer_eagle:
        return (
            "enable_multi_layer_eagle=True is not supported "
            "(MultiLayerEagleWorkerV2 does not implement adaptive)"
        )
    if cfg.enable_two_batch_overlap:
        return (
            "enable_two_batch_overlap=True is not supported "
            "(adaptive state swap would discard the TboAttnBackend wrapper)"
        )
    if cfg.enable_pdmux:
        return (
            "enable_pdmux=True is not supported "
            "(adaptive state swap does not update decode_attn_backend_group)"
        )
    return None


def _load_adaptive_config(
    cfg_path: str | None,
) -> tuple[dict, dict[int, dict]]:
    """Load and validate adaptive config.

    Uses ``DEFAULT_ADAPTIVE_CONFIG`` when *cfg_path* is ``None``.
    """
    if cfg_path is not None:
        with open(cfg_path) as f:
            cfg = json.load(f)
    else:
        cfg = DEFAULT_ADAPTIVE_CONFIG

    bs_entries: dict[int, dict] = {}
    for key, entry in cfg.items():
        if not key.isdigit():
            continue

        steps = entry.get("candidate_steps")
        if (
            not isinstance(steps, list)
            or not steps
            or not all(isinstance(s, int) and s >= 0 for s in steps)
        ):
            raise ValueError(
                f"BS {key}: candidate_steps must be a list of non-negative ints, "
                f"got {steps!r}"
            )
        bs_entries[int(key)] = entry

    if not bs_entries:
        raise ValueError(
            "speculative_adaptive_config must contain at least one integer-string "
            'BS key, e.g. {"1": {"candidate_steps": [1,3,7]}}. '
            f"Got keys: {list(cfg.keys())}"
        )
    return cfg, bs_entries


def resolve_candidate_steps_from_config(
    cfg_path: str | None = None,
) -> list[int]:
    """Union of every BS slot's candidate steps; sizes the runtime buffers."""
    _, bs_entries = _load_adaptive_config(cfg_path)
    all_steps: set[int] = set()
    for entry in bs_entries.values():
        all_steps.update(entry["candidate_steps"])
    return sorted(all_steps)


class AdaptiveStepSlot:
    """Tracks acceptance rate via EMA and adapts num_steps accordingly.

    The core idea: if drafts are consistently accepted, try more steps;
    if drafts are consistently rejected early, reduce steps to avoid waste.

    Formula: target_steps = clamp(round(ema_accept_len) + 1, min_steps, max_steps)
    - Probes one step beyond observed acceptance
    - EMA smoothing prevents oscillation
    - Only updates every `update_interval` batches for stability
    - num_steps can be selected from different candidate sets on different batch_sizes
    """

    def __init__(self, initial_steps: int, cfg: dict):
        candidates = sorted(set(cfg["candidate_steps"]))
        assert len(candidates) >= 1, "candidate_steps must have at least 1 value"
        self.candidate_steps = candidates

        self.ema_alpha = cfg.get("ema_alpha", 0.2)
        self.update_interval = cfg.get("update_interval", 5)
        self.warmup_batches = cfg.get("warmup_batches", 10)
        self.down_hysteresis = cfg.get("down_hysteresis", -0.25)
        self.up_hysteresis = cfg.get("up_hysteresis", 0.0)
        self.ceiling_coeff = cfg.get("ceiling_coeff", 0)

        if initial_steps in self.candidate_steps:
            self.current_steps = initial_steps
        else:
            self.current_steps = self.candidate_steps[len(self.candidate_steps) // 2]

        # Initialize EMA at current steps - 1 (neutral starting point)
        self.ema_accept_len = float(self.current_steps - 1)
        self._batch_count = 0

    def update(self, num_correct_drafts_per_req: list[int]) -> bool:
        """Update EMA with observed accept lengths. Returns True if params changed.

        Args:
            num_correct_drafts_per_req: Per-request accepted draft token counts from last verify.
        """
        if not num_correct_drafts_per_req:
            return False

        if self.current_steps > 0:
            batch_avg = sum(num_correct_drafts_per_req) / len(
                num_correct_drafts_per_req
            )
            self.ema_accept_len = (
                1 - self.ema_alpha
            ) * self.ema_accept_len + self.ema_alpha * batch_avg

        self._batch_count += 1
        if self._batch_count <= self.warmup_batches:
            return False

        if (self._batch_count - self.warmup_batches) % self.update_interval != 0:
            return False

        return self._recompute_params()

    def _recompute_params(self) -> bool:
        """Recompute steps from EMA. Returns True if params changed."""
        old_steps = self.current_steps
        current_idx = self.candidate_steps.index(old_steps)
        old_idx = current_idx

        # Probe the smallest positive step after a zero-step nospec interval.
        if old_steps == 0:
            current_idx = min(current_idx + 1, len(self.candidate_steps) - 1)
            target = self.candidate_steps[current_idx]
            if target > 0 and self.ema_accept_len < 0:
                # A slot initialized at steps=0 has no draft acceptance history;
                # start the first positive-step probe from that step's neutral EMA.
                self.ema_accept_len = float(target - 1)
            return self._apply_target_steps(old_steps, target)

        # TODO: Consider limiting step changes to avoid overshooting.
        while current_idx > 0:
            prev_step = self.candidate_steps[current_idx - 1]
            # A zero-step candidate disables drafting. Treat zero accepted drafts
            # as low enough to reach it when it is the floor candidate.
            drop_threshold = 0.5 if prev_step == 0 else prev_step - 0.5
            drop_threshold += self.down_hysteresis
            if self.ema_accept_len <= drop_threshold:
                current_idx -= 1
            else:
                break

        moved_down = current_idx < old_idx
        if not moved_down:
            while current_idx < len(self.candidate_steps) - 1:
                current_step = self.candidate_steps[current_idx]
                rise_threshold = current_step - 0.5 + self.up_hysteresis
                if self.ema_accept_len > rise_threshold:
                    current_idx += 1
                else:
                    break

        target = self.candidate_steps[current_idx]
        # EMA ceiling: only caps downward — never blocks step-ups, so the
        # system can explore higher steps and let the EMA catch up.
        if self.ceiling_coeff > 0:
            ceiling = max(1, math.ceil(self.ema_accept_len * self.ceiling_coeff))
            if target > ceiling and target <= old_steps:
                while current_idx > 0 and self.candidate_steps[current_idx] > ceiling:
                    current_idx -= 1
                target = self.candidate_steps[current_idx]

        return self._apply_target_steps(old_steps, target)

    def _apply_target_steps(self, old_steps: int, target: int) -> bool:
        if target != old_steps:
            self.current_steps = target
            log_info_on_rank0(
                logger,
                f"Adaptive spec params updated: steps {old_steps} -> {target} "
                f"(ema_accept_len={self.ema_accept_len:.2f})",
            )
            return True
        return False


# Strata DraftPolicy constants (src/spec/draft_policy.cpp), minus the n-gram
# lookup machinery this port does not carry.
_COST_ALPHA = 0.1  # EMA weight of a new measured round time
_TOK_ALPHA = 0.05  # EMA weight of a new measured accept_length
_PROBE_CT = 3.0  # rounds a guessed width is tried before its data is trusted
_MARGIN = 0.03  # a width switch must beat the incumbent by this much
# Per-round decay of the measurement counts: a width the workload drifted
# away from loses its measurements (and reverts to the prior) after roughly
# 130 idle rounds, so a regime shift can be re-probed instead of being
# blocked forever by stale-low estimates.
_CT_DECAY = 0.98
# Round cost by step count relative to steps=1, used only for widths never
# measured yet (measured round times replace it). Strata's measured curve;
# index 0 (steps=0, no drafting) is this port's guess.
_COST_SHAPE = (0.9, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6)


def _cost_shape(steps: int) -> float:
    if steps < len(_COST_SHAPE):
        return _COST_SHAPE[steps]
    # Linear extension beyond the tabulated range.
    return _COST_SHAPE[-1] + 0.35 * (steps - (len(_COST_SHAPE) - 1))


class CostAwareStepSlot:
    """Picks the draft width with the best tokens per millisecond, one BS slot.

    Strata DraftPolicy semantics without the lookup window: per step count the
    slot keeps EMAs of the measured verify-round wall time and of accept_length
    (tokens per round, bonus included), and selects
    argmax(accept_length / round_ms). Unmeasured widths are scored from the
    shape prior, but every candidate is still measured for ``_PROBE_CT`` rounds
    before argmax decides (cold-start sweep), so a mis-calibrated prior never
    permanently hides a width. Measurement counts decay every round and a width
    whose counts reach zero is wiped back to the prior, so workload shifts can
    re-probe widths whose stale estimates would otherwise block their own
    re-measurement.
    """

    def __init__(self, initial_steps: int, cfg: dict):
        candidates = sorted(set(cfg["candidate_steps"]))
        assert len(candidates) >= 1, "candidate_steps must have at least 1 value"
        self.candidate_steps = candidates

        self.cost_alpha = cfg.get("cost_alpha", _COST_ALPHA)
        self.tok_alpha = cfg.get("tok_alpha", _TOK_ALPHA)
        self.margin = cfg.get("margin", _MARGIN)
        self.ct_decay = cfg.get("ct_decay", _CT_DECAY)
        self.update_interval = cfg.get("update_interval", 4)
        self.warmup_batches = cfg.get("warmup_batches", 10)

        if initial_steps in self.candidate_steps:
            self.current_steps = initial_steps
        else:
            self.current_steps = self.candidate_steps[len(self.candidate_steps) // 2]

        # Per-step EMAs; the *_ct sample counts are floats so the cost scaling
        # can cap how much one width's run length outweighs another's.
        self._cost_ms: dict[int, float] = {}
        self._cost_ct: dict[int, float] = {}
        self._accept_len: dict[int, float] = {}
        self._accept_ct: dict[int, float] = {}
        self._round_ct = 0

    def update(
        self, num_correct_drafts_per_req: list[int], round_ms: float | None
    ) -> bool:
        """Fold one verify round's outcome in. True if the width changed."""
        if not num_correct_drafts_per_req:
            return False

        steps = self.current_steps
        if round_ms is not None and round_ms > 0:
            prev = self._cost_ms.get(steps)
            self._cost_ms[steps] = (
                round_ms
                if prev is None
                else (1 - self.cost_alpha) * prev + self.cost_alpha * round_ms
            )
            self._cost_ct[steps] = self._cost_ct.get(steps, 0.0) * self.ct_decay + 1.0

        # accept_length (bonus included): mean over the batch's verified rows.
        n = len(num_correct_drafts_per_req)
        got = (sum(num_correct_drafts_per_req) + n) / n
        prev_len = self._accept_len.get(steps)
        self._accept_len[steps] = (
            got
            if prev_len is None
            else (1 - self.tok_alpha) * prev_len + self.tok_alpha * got
        )
        # Fold the count with the decay already applied: the +1 offsets this
        # round's decay, so a freshly probed width's first sample (ct == 1.0)
        # survives the wipe threshold instead of being erased at birth.
        self._accept_ct[steps] = self._accept_ct.get(steps, 0.0) * self.ct_decay + 1.0

        self._round_ct += 1
        self._age_measurements(keep=steps)
        if self._round_ct <= self.warmup_batches:
            return False
        if (self._round_ct - self.warmup_batches) % self.update_interval != 0:
            return False

        target = self._choose()
        if target != self.current_steps:
            self.current_steps = target
            log_info_on_rank0(
                logger,
                f"Cost-aware adaptive spec switched: steps -> {target} "
                f"(cost_ms={self._cost_ms_est(target):.2f}, "
                f"accept_len={self._tokens(target):.2f}, "
                f"scores={self._format_scores()})",
            )
            return True
        return False

    def _choose(self) -> int:
        scores = {
            t: self._tokens(t) / self._cost_ms_est(t) for t in self.candidate_steps
        }
        current = self.current_steps
        # Cold-start sweep: the accept prior can rank an unmeasured width
        # below the incumbent forever even when its real score wins (GLM
        # width 5 on easy text: prior accept 3.8 vs measured 5.9), so every
        # candidate gets measured up to _PROBE_CT rounds before argmax
        # decides. Finish the current width's probe first, then sweep the
        # rest best-prior-first; measured rounds are the probe's whole cost.
        need_probe = [
            t for t in self.candidate_steps if self._accept_ct.get(t, 0.0) < _PROBE_CT
        ]
        if need_probe:
            if current in need_probe:
                return current
            return max(need_probe, key=lambda t: scores[t])
        best = max(scores, key=scores.get)
        if best == current:
            return current
        # Stickiness: leaving the incumbent requires beating it by the margin.
        if scores[best] > scores[current] * (1.0 + self.margin):
            return best
        return current

    def _age_measurements(self, keep: int) -> None:
        """Decay every width's measurement counts; wipe the fully stale ones.

        The width measured this round is exempt (its fold already applied the
        decay), so the incumbent sits at the 1/(1-decay) equilibrium and is
        never wiped, while an idle width's count decays toward the threshold.
        """
        decay = self.ct_decay
        for steps in list(self._cost_ct):
            if steps == keep:
                continue
            ct = self._cost_ct[steps] * decay
            if ct < 1.0:
                del self._cost_ct[steps]
                del self._cost_ms[steps]
            else:
                self._cost_ct[steps] = ct
        for steps in list(self._accept_ct):
            if steps == keep:
                continue
            ct = self._accept_ct[steps] * decay
            if ct < 1.0:
                del self._accept_ct[steps]
                del self._accept_len[steps]
            else:
                self._accept_ct[steps] = ct

    def _tokens(self, steps: int) -> float:
        if self._accept_ct.get(steps, 0.0) > 0:
            return self._accept_len[steps]
        if steps == 0:
            return 1.0  # no drafts: every round emits exactly the root
        # Before any round at this width: a typical acceptance.
        return 1.0 + 0.7 * (steps - 1)

    def _cost_ms_est(self, steps: int) -> float:
        if self._cost_ct.get(steps, 0.0) > 0:
            return self._cost_ms[steps]
        # Scale from the measured widths, each weighted by how often it ran
        # (capped so one width's long run cannot dominate the estimate).
        num = den = 0.0
        for other, ct in self._cost_ct.items():
            if ct > 0:
                w = min(ct, 20.0)
                num += (
                    w * self._cost_ms[other] * _cost_shape(steps) / _cost_shape(other)
                )
                den += w
        return num / den if den > 0 else _cost_shape(steps)

    def _format_scores(self) -> str:
        return ",".join(
            f"{t}:{self._tokens(t) / self._cost_ms_est(t):.3f}"
            for t in self.candidate_steps
        )


class AdaptiveSpeculativeParams:
    """Routes ``batch_size`` to the correct per-BS slot.

    A slot is a per-BS configuration of adaptive step selection.
    """

    def __init__(
        self,
        initial_steps: int,
        cfg_path: str | None = None,
    ):
        cfg, bs_entries = _load_adaptive_config(cfg_path)
        self._bs_list: list[int] = sorted(bs_entries)
        self._slots: dict[int, AdaptiveStepSlot] = {}
        self._cuda_graph_bs: list[int] | None = None

        for bs, entry in sorted(bs_entries.items()):
            self._slots[bs] = self._make_slot(
                initial_steps=initial_steps,
                cfg={**cfg, **entry},
            )

        first_slot = self._slots[self._bs_list[0]]
        log_info_on_rank0(
            logger,
            f"AdaptiveSpeculativeParams initialized: "
            f"steps={first_slot.current_steps}, "
            f"candidate_steps={first_slot.candidate_steps}",
        )

    @cached_property
    def candidate_steps(self) -> list[int]:
        """Union of all BS slots' candidate steps."""
        return sorted({s for p in self._slots.values() for s in p.candidate_steps})

    def _make_slot(self, initial_steps: int, cfg: dict) -> AdaptiveStepSlot:
        return AdaptiveStepSlot(initial_steps=initial_steps, cfg=cfg)

    def set_cuda_graph_bs(self, cuda_graph_bs: list[int] | None) -> None:
        self._cuda_graph_bs = sorted(cuda_graph_bs) if cuda_graph_bs else None

    def get_steps_for_batch(self, batch_size: int) -> int:
        return self._route(batch_size).current_steps

    def on_verify_complete(
        self,
        num_correct_drafts_per_req: list[int],
        batch_size: int,
        round_ms: float | None = None,
    ) -> int | None:
        """Feed verify results to the matching BS slot's EMA.

        ``round_ms`` is the verify round's measured wall time; the
        acceptance-driven policy ignores it, the cost-aware one consumes it.

        Returns the new step if a switch is warranted, else ``None``.
        """
        params = self._route(batch_size)
        if params.update(num_correct_drafts_per_req):
            return params.current_steps
        return None

    def cuda_graph_bs_for_step(self, step: int) -> list[int] | None:
        """Return cuda_graph_bs values that can reach *step* at runtime.

        Returns ``None`` when CUDA graphs are disabled (``set_cuda_graph_bs``
        was never called or was called with ``None``).
        """
        if self._cuda_graph_bs is None:
            return None
        return [
            v
            for v in self._cuda_graph_bs
            if step in self._slots[self._find_closest_bs(v)].candidate_steps
        ]

    def _route(self, batch_size: int) -> AdaptiveStepSlot:
        """Map *batch_size* → pad to CUDA-graph BS → closest slot."""
        return self._slots[
            self._find_closest_bs(self._pad_to_cuda_graph_bs(batch_size))
        ]

    def _pad_to_cuda_graph_bs(self, batch_size: int) -> int:
        if self._cuda_graph_bs is None:
            return batch_size
        idx = bisect.bisect_left(self._cuda_graph_bs, batch_size)
        return (
            self._cuda_graph_bs[idx] if idx < len(self._cuda_graph_bs) else batch_size
        )

    def _find_closest_bs(self, target: int) -> int:
        idx = bisect.bisect_right(self._bs_list, target) - 1
        return self._bs_list[max(0, idx)]


class CostAwareSpeculativeParams(AdaptiveSpeculativeParams):
    """BS routing of the base class with the cost-aware width policy.

    Selected by ``SGLANG_ADAPTIVE_SPEC_COST_AWARE``; see
    :class:`CostAwareStepSlot` for the objective.
    """

    def _make_slot(self, initial_steps: int, cfg: dict) -> CostAwareStepSlot:
        return CostAwareStepSlot(initial_steps=initial_steps, cfg=cfg)

    def on_verify_complete(
        self,
        num_correct_drafts_per_req: list[int],
        batch_size: int,
        round_ms: float | None = None,
    ) -> int | None:
        params = self._route(batch_size)
        if params.update(num_correct_drafts_per_req, round_ms):
            return params.current_steps
        return None


def make_adaptive_policy(
    initial_steps: int,
    cfg_path: str | None = None,
) -> AdaptiveSpeculativeParams:
    """Build the adaptive policy: cost-aware when SGLANG_ADAPTIVE_SPEC_COST_AWARE."""
    cls = (
        CostAwareSpeculativeParams
        if envs.SGLANG_ADAPTIVE_SPEC_COST_AWARE.get()
        else AdaptiveSpeculativeParams
    )
    return cls(initial_steps=initial_steps, cfg_path=cfg_path)
