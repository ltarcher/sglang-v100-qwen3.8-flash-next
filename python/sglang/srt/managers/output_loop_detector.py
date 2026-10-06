"""Periodic-loop detection over generated token streams.

A runaway decode (repetition attractor) burns the whole context window with a
repeating phrase; SGLANG_MAX_NEW_TOKENS_LIMIT only caps the blast radius, this
is what stops the loop mid-flight. The scheduler calls `detect_periodic_loop`
once per finish-check; it is a pure-Python tail scan over token id lists with
no engine state, so it is safe from any finish path and trivially cheap at
single-stream decode rates.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

# Phrase-length periods only: short periods fire on legitimate output
# (punctuation runs, "...", "————"), runaway loops repeat multi-token blocks.
MIN_PERIOD = 4
MAX_PERIOD = 32
MIN_REPEATS = 3


def detect_periodic_loop(
    token_ids: List[int],
    min_period: int = MIN_PERIOD,
    max_period: int = MAX_PERIOD,
    min_repeats: int = MIN_REPEATS,
) -> Optional[Tuple[int, int]]:
    """Return ``(period, repeats)`` if the tail of ``token_ids`` ends with
    ``min_repeats`` consecutive identical blocks of some period in
    ``[min_period, max_period]``, else ``None``. Periods are scanned ascending,
    so the tightest description wins and the shortest loop fires first.
    """
    n = len(token_ids)
    if n < min_period * min_repeats:
        return None
    for period in range(min_period, max_period + 1):
        if n < period * min_repeats:
            break
        block = token_ids[n - period :]
        if block != token_ids[n - 2 * period : n - period]:
            continue
        if block != token_ids[n - 3 * period : n - 2 * period]:
            continue
        repeats = min_repeats
        while (
            n - (repeats + 1) * period >= 0
            and token_ids[n - (repeats + 1) * period : n - repeats * period] == block
        ):
            repeats += 1
        return period, repeats
    return None
