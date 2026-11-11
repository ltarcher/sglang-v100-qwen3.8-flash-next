# 32k poison repro scripts

Scripts reproducing the radix-cache poisoning found by `bench_quality_v100.py`
(see README "Flush the radix cache before every run"). Each script drives the
running server over HTTP; run them with `SGLANG_V100_PORT=8110` set. Every
script self-contains its imports (`importlib` loads the bench lib) and flushes
the cache first unless stated otherwise.

## Established facts (2026-11-11, chunk-ahead image, production flags)

| experiment | sequence | gsm8k discriminator | verdict |
| --- | --- | --- | --- |
| `ab_cache_state.py empty` | flush -> 32k row (128 decode) -> 32q | 30/32 (93.8%), 1 flood | clean band |
| `ab_cache_state.py warm` | flush -> 12q warm -> 32k row -> 32q | **11/32 (34.4%), 15 floods** | poisoned, 2/2 reproducible |
| `control_no_row.py` | flush -> 12q -> 12q (no row) | 10-12/12 | baseline noise |
| `post_flush_test.py` | warm -> 32k row -> **flush** -> 32q | 26/32 (81.2%), 5 floods | **flush does not fully heal** |
| `salt_test.py` | warm -> 32k row w/ cache_salt -> 32q | 81.2%, 0 floods | tree-merge contributes the flood signature; residual damage remains |
| `decode_sweep.py 8..96` | flush -> 32k row (N decode) -> 12q | 12/12 | clean |
| `decode_sweep.py 128` (early session) | same | 9/12, floods | poisoned then; **5/5 clean late session** — nondeterministic on empty cache |
| length sweep (`row_test.py` variants) | flush -> row(8k/16k/20k/24k/28k) -> 12q | 12/12 | clean |
| length 29696/28672 (8/7 chunks) | same | 11-12/12 | clean; chunk-count hypothesis dead |
| 31735 (row0) vs 31909 (row1) | same | row0 poisoned 3x early, clean 5x late | content/length not the axis |

## Mechanism shape

A >=31.7k chunked-prefill request arriving on a **non-empty** radix tree
(warm short-prefix traffic) intermittently corrupts subsequent greedy
answers: silent numeric errors + '!' floods (spec accept 4.0, draft
perfectly predicts '!' tokens). `flush_cache` resets tree + pools +
req_to_token but does **not** fully restore accuracy, so the damage
extends past what flush resets — prime suspect: the mamba extra_buffer
stash/rebind path under the overlap scheduler
(`UnifiedRadixCache.checkpoint`, python/sglang/srt/mem_cache/unified_radix_cache.py:1057),
same family as the near-full-pool lesion (stash free/rebind use-after-free)
and the 262k dual-condition lesion.

## Not yet narrowed (needs server restart, blocked on production traffic)

- `--mamba-radix-cache-strategy no_buffer` vs `extra_buffer`
- overlap scheduler on/off
- `--speculative-algorithm` off (is MTP decode required for the poison?)
- chunked-prefill size dependence

## Discriminator convention

12-question gsm8k (fast, noisy 10-12/12) for triage; 32-question for
verdicts. Flood = completion starts with '!!!'. Baseline greedy accuracy
on this server: 30-31/32.
