"""Prefix-cache KV consistency oracle guards (pure CPU; no server, no weights).

Why this file exists
--------------------
The dangerous failure mode of prefix caching is: a request hits the WRONG KV
slots, the service still answers normally, and the output silently degrades
(AGENTS.md:84).  A green "the service replied" signal proves nothing.  What is
asserted here instead is the *identity of the KV slots* the answer depends on:
``MatchResult.device_indices`` must be bit-for-bit equal to the indices a full
(uncached) token-by-token prefill would have written, and the reported cache-hit
metric must be derived from that very same match result.

Naming clarification: three names, one quantity
----------------------------------------------
* ``AGENTS.md:84`` / ``scripts/p0_trace_probe.py`` say ``cached_prompt_tokens``
  -- this is the *semantic* name used in the oracle prose.
* The server ``meta_info`` field is ``cached_tokens``
  (``python/sglang/srt/entrypoints/openai/usage_processor.py:42``).
* The OpenAI protocol output is ``UsageInfo.prompt_tokens_details.cached_tokens``
  (``.../openai/protocol.py:192``); the Anthropic protocol output is
  ``AnthropicUsage.cache_read_input_tokens``
  (``.../anthropic/serving.py:129``).

All three denote the SAME quantity: the number of prompt tokens served from the
prefix cache.  This test binds that quantity to ``len(device_indices)`` of the
actual cache hit; it does NOT hardcode it.  The verified real chain is
``cached_tokens`` -> ``UsageInfo.prompt_tokens_details`` -> Anthropic output.

Three-layer oracle
------------------
* L1  : ``match_prefix(...).device_indices.tolist() == E.tolist()`` where
        ``E = [base + i for i in range(n)]`` is a globally unique, disjoint,
        position-encoded index space per sequence (bases 10000 apart).  Any
        cross-talk, misalignment, out-of-page tail or stale slot injects another
        base segment and FAILs the bit-wise comparison, turning a silent
        degradation into a loud failure carrying full-sequence diagnostics.
        The same assertion distinguishes "no hit" (length differs) from "hit the
        wrong segment" (values differ).
* L1b : ``FakeTrackingAllocator`` records a generation per index and increments
        it on ``free_segment``; every index returned by a match must carry its
        insertion-time generation.  Guards against "indices look right but the
        KV slot was freed and reused" (the most hidden silent degradation).
* L2  : ``H = len(device_indices)`` is fed into ``UsageProcessor`` and then the
        Anthropic usage layer; protocol outputs must equal ``H`` and
        ``0 <= H <= P``.  Prevents "metric green while KV wrong" fake green.

Scope / wording (STRICT)
------------------------
These unit tests guard the **cache-layer indices / metric contract** only.  They
MUST NOT be read as "output consistency has been verified".  Real output
consistency (same-prefix two-request text/logprobs alignment, MTP accept ~2.1,
HiCache L3 file-backend load-back) requires 4xV100 E2E and is NOT covered here.
TODO(E2E): HiCache file-backend (L3) KV content and MTP/Eagle speculative output
belong to a separate multi-GPU E2E task (design doc 8.4 / 8.5), not here.

Target-machine verification (NOT executed on this Windows dev host)
-------------------------------------------------------------------
    PYTHON="$HOME/miniconda3/envs/sglang-v100/bin/python"
    F=test/registered/unit/mem_cache/test_prefix_kv_consistency_oracle.py
    $PYTHON "$F" -f
    $PYTHON -m pytest "$F" -v
    $PYTHON -m pytest test/registered/unit/mem_cache/ -v
    pre-commit run --all-files
"""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# Exactly one CPU CI registry, with constant literals (parsed via AST).
register_cpu_ci(est_time=10, suite="base-a-test-cpu")


# --------------------------------------------------------------------------- #
# Helpers / fakes -- all inlined in this one file on purpose.  A separate helper
# .py would be globbed by test/run_suite.py and rejected by ci_register.py for
# having no registry.  See design doc 4.4.
# --------------------------------------------------------------------------- #

# Distinct, non-overlapping position-encoded index spaces.  Sequence lengths in
# this file are <= 16, so a 10000 step guarantees no base segment can overlap.
INDEX_BASE_STEP = 10000


def _tokens(base, n):
    """A token-id sequence; distinct bases keep sequences disjoint."""
    return array("q", [base + i for i in range(n)])


def _indices(base, n):
    """E: the indices a full (uncached) prefill would write, token by token."""
    return torch.tensor([base + i for i in range(n)], dtype=torch.int64)


def _empty_indices():
    return torch.empty((0,), dtype=torch.int64)


def _key(tokens, *, extra_key=None, cache_salt=None, limit=None):
    """Always build a FRESH RadixKey: maybe_to_bigram_view flips is_bigram in
    place, so reusing one object across insert/match would pollute the test."""
    return RadixKey(
        array("q", list(tokens)),
        extra_key=extra_key,
        cache_salt=cache_salt,
        limit=limit,
    )


def _response(prompt_tokens, cached_tokens, completion_tokens=1):
    """A service ``meta_info`` shaped response for the usage layer."""
    return {
        "meta_info": {
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "completion_tokens": completion_tokens,
        }
    }


def _assert_indices_equal(tc, actual, expected, ctx=""):
    """L1: bit-wise equality with complete-sequence diagnostics."""
    tc.assertEqual(
        actual.dtype,
        torch.int64,
        f"{ctx}: device_indices dtype must be int64, got {actual.dtype}",
    )
    tc.assertEqual(
        actual.device.type,
        "cpu",
        f"{ctx}: device_indices must be a CPU tensor, got {actual.device}",
    )
    exp_list = torch.as_tensor(expected, dtype=torch.int64).tolist()
    act_list = actual.tolist()
    tc.assertEqual(
        act_list,
        exp_list,
        f"{ctx}: KV indices mismatch -> the cache hit points at the WRONG KV "
        f"(silent-degradation oracle).\n"
        f"  expected (full prefill) : {exp_list}\n"
        f"  actual   (cache hit)    : {act_list}\n"
        f"  len(expected)={len(exp_list)} len(actual)={len(act_list)}",
    )


def _assert_no_tail_leak(tc, actual, expected_full, hit_len, ctx=""):
    """No index belonging to the un-cached (page-tail / beyond-limit / bigram)
    portion of E may appear in a hit result."""
    tail = set(expected_full.tolist()[hit_len:])
    leaked = [int(v) for v in actual.tolist() if int(v) in tail]
    tc.assertEqual(
        leaked,
        [],
        f"{ctx}: tail indices leaked into the hit result: {leaked}; "
        f"expected only E[:{hit_len}] = {expected_full.tolist()[:hit_len]}",
    )


class FakeTrackingAllocator:
    """Minimal allocator fake with an index-generation ledger (L1b).

    Satisfies the interface ``RadixCache`` needs on CPU:
    ``device`` (str / torch.device, see radix_cache.py:297-304),
    ``free_segment(indices, start_pos=0)`` (evict at :568), and
    ``available_size()`` (used by diagnostics paths).  Unlike a Mock, it does
    not swallow ``free_segment`` -- it records the actual state transition.
    """

    def __init__(self):
        self.device = torch.device("cpu")
        self._next = 1
        self._gen = {}  # index -> generation
        self._freed = set()
        self.free_calls = []  # (indices.tolist(), start_pos); auxiliary only

    def alloc(self, n, base=None):
        """Return n globally unique indices; optionally position-encoded at
        ``base`` so different sequences occupy disjoint, identifiable ranges."""
        if base is None:
            idxs = list(range(self._next, self._next + n))
            self._next += n
        else:
            idxs = [base + i for i in range(n)]
        for idx in idxs:
            self._gen.setdefault(idx, 0)
            self._freed.discard(idx)
        return torch.tensor(idxs, dtype=torch.int64)

    def free_segment(self, indices, start_pos=0):
        idxs = torch.as_tensor(indices).reshape(-1).tolist()
        for idx in idxs:
            idx = int(idx)
            self._gen[idx] = self._gen.get(idx, 0) + 1
            self._freed.add(idx)
        self.free_calls.append((idxs, start_pos))

    def generation_of(self, idx):
        return self._gen.get(int(idx), 0)

    def is_stale(self, idx, expected_gen):
        return self.generation_of(idx) != expected_gen

    def available_size(self):
        return len(self._freed)


def _generation_snapshot(alloc, tensor):
    return {int(i): alloc.generation_of(i) for i in tensor.tolist()}


def _assert_generation_fresh(tc, alloc, indices, snapshot, ctx=""):
    """L1b: every returned index must still be the one inserted at that slot."""
    for raw in indices.tolist():
        idx = int(raw)
        tc.assertIn(
            idx,
            snapshot,
            f"{ctx}: match returned an index ({idx}) that was never inserted",
        )
        tc.assertFalse(
            alloc.is_stale(idx, snapshot[idx]),
            f"{ctx}: index {idx} is STALE -- generation "
            f"{alloc.generation_of(idx)} != insertion generation "
            f"{snapshot[idx]}; the KV slot was freed/reused but still returned",
        )


def _make_cache(allocator, *, page_size=1, is_eagle=False, enable_events=False):
    """Non-eagle uses create_simulated; eagle MUST use CacheInitParams because
    create_simulated has no is_eagle parameter (radix_cache.py:313-329,
    cache_init_params.py:24)."""
    if is_eagle:
        return RadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=allocator,
                page_size=page_size,
                is_eagle=True,
                enable_kv_cache_events=enable_events,
            )
        )
    return RadixCache.create_simulated(
        mock_allocator=allocator,
        page_size=page_size,
        enable_kv_cache_events=enable_events,
    )


def _match(cache, tokens, **kwargs):
    return cache.match_prefix(MatchPrefixParams(key=_key(tokens, **kwargs)))


def _insert(cache, tokens, value, **kwargs):
    return cache.insert(InsertParams(key=_key(tokens, **kwargs), value=value))


def _walk_non_root(root):
    for child in root.children.values():
        yield child
        yield from _walk_non_root(child)


class FakeReqToTokenPool:
    """Row storage for the checkpoint test (mirrors radix_cache.py:447/520)."""

    def __init__(self, row):
        self.req_to_token = row.reshape(1, -1).clone()

    def write(self, indices, values):
        self.req_to_token[indices] = values


class TestPrefixKVConsistencyOracle(CustomTestCase):
    """Consistency-oracle guards, one level above the existing RadixKey/TreeNode
    data-structure tests: every case asserts indices content, a state
    transition, a protocol output, or an error -- never "mock was called".
    """

    # ------------------------------------------------------------------ #
    # Group A -- oracle main scenario, differential baseline, metric linkage
    # ------------------------------------------------------------------ #

    def test_same_prefix_second_request_matches_full_prefill_indices(self):
        """A1 (P0): the AGENTS.md:84 oracle as one executable scenario.

        Covered cause: wrong-KV hit / no-hit.  Assertions: L1 indices,
        generation (L1b), hit length, and cached_tokens linkage (L2).
        """
        page_size = 1
        tokens = _tokens(7, 10)  # [7 .. 16]
        expected = _indices(1000, 10)
        alloc = FakeTrackingAllocator()
        cache = _make_cache(alloc, page_size=page_size)

        # 1) Empty tree: an identical prefix is NOT a hit yet.
        first = _match(cache, tokens)
        self.assertEqual(
            first.device_indices.tolist(),
            [],
            "empty tree must report no cached prefix (H=0 baseline)",
        )

        # 2) Full prefill: insert the token-by-token expected indices.
        gen_at_insert = _generation_snapshot(alloc, expected)
        _insert(cache, tokens, expected)

        # 3) Same prefix, second request: must hit and return exactly E.
        second = _match(cache, tokens)
        _assert_indices_equal(self, second.device_indices, expected, ctx="A1/L1")
        hit_len = len(second.device_indices)
        self.assertEqual(hit_len, len(expected), "A1: full prefix must be cached")
        self.assertGreater(hit_len, 0, "A1: second request must show a cache hit")

        # L1b: the hit must not point at a freed/reused slot.
        _assert_generation_fresh(
            self, alloc, second.device_indices, gen_at_insert, ctx="A1/L1b"
        )

        # L2: the reported metric is derived from THIS match, not hardcoded.
        cached_tokens = len(second.device_indices)
        usage = UsageProcessor.calculate_response_usage(
            [_response(len(tokens), cached_tokens)],
            n_choices=1,
            enable_cache_report=True,
        )
        self.assertEqual(usage.prompt_tokens, len(tokens))
        self.assertIsNotNone(usage.prompt_tokens_details)
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, hit_len)
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, cached_tokens)

    def test_hit_path_equals_full_prefill_baseline(self):
        """A2 (P0): hit-path vs. an independently constructed full-prefill
        baseline, parameterized over page_size 1 / 2 / 4.

        Covered cause: topology-dependent misalignment / cross-sequence
        cross-talk.  Assertions: L1 indices, page alignment, generation.
        """
        n = 12
        seq_a = _tokens(100, n)
        seq_b = _tokens(300, n)  # disjoint first token -> disjoint root child
        exp_a = _indices(2 * INDEX_BASE_STEP, n)
        exp_b = _indices(5 * INDEX_BASE_STEP, n)

        for page_size in (1, 2, 4):
            with self.subTest(page_size=page_size):
                alloc = FakeTrackingAllocator()
                cache = _make_cache(alloc, page_size=page_size)
                _insert(cache, seq_a, exp_a)
                _insert(cache, seq_b, exp_b)
                gen = _generation_snapshot(alloc, torch.cat([exp_a, exp_b]))
                aligned_len = (n // page_size) * page_size

                for name, tokens, expected in (
                    ("A", seq_a, exp_a),
                    ("B", seq_b, exp_b),
                ):
                    with self.subTest(page_size=page_size, sequence=name):
                        match = _match(cache, tokens)
                        baseline = expected[:aligned_len]
                        _assert_indices_equal(
                            self,
                            match.device_indices,
                            baseline,
                            ctx=f"A2/L1 page={page_size} seq={name}",
                        )
                        self.assertEqual(
                            len(match.device_indices),
                            aligned_len,
                            "A2: hit length must equal the page-aligned prefill",
                        )
                        self.assertEqual(
                            len(match.device_indices) % page_size,
                            0,
                            "A2: hit length must be page aligned",
                        )
                        _assert_generation_fresh(
                            self,
                            alloc,
                            match.device_indices,
                            gen,
                            ctx=f"A2/L1b page={page_size} seq={name}",
                        )

                # A shorter query splits a node; the returned segment must
                # still be exactly the first aligned chunk of E (no wrong page).
                partial_len = max(page_size, aligned_len // 2)
                partial_aligned = (partial_len // page_size) * page_size
                if 0 < partial_aligned < aligned_len:
                    partial = _match(cache, seq_a[:partial_len])
                    _assert_indices_equal(
                        self,
                        partial.device_indices,
                        exp_a[:partial_aligned],
                        ctx=f"A2/L1 page={page_size} partial split",
                    )
                    _assert_no_tail_leak(
                        self,
                        partial.device_indices,
                        exp_a[:aligned_len],
                        partial_aligned,
                        ctx=f"A2 page={page_size} partial split",
                    )

    def test_cached_prompt_tokens_equals_actual_hit_length(self):
        """A3 (P0): cross-layer metric linkage (L2), including H=0 and H=P.

        Covered cause: metric decoupled from the real hit ("green metric,
        wrong KV").  Assertions: L1 indices, then UsageInfo / Anthropic
        protocol outputs derived from ``len(device_indices)``.

        The Anthropic helpers are imported lazily so their heavier server-side
        import chain is confined to this one L2 test.
        """
        from sglang.srt.entrypoints.anthropic.serving import (
            _anthropic_usage_from_openai,
            _cached_prompt_tokens,
        )

        scenarios = []

        # (1) full hit: H == P
        t_full = _tokens(10, 8)
        e_full = _indices(10 * INDEX_BASE_STEP, 8)
        cache_full = _make_cache(FakeTrackingAllocator(), page_size=1)
        _insert(cache_full, t_full, e_full)
        scenarios.append(
            ("full_hit", 8, _match(cache_full, t_full), e_full[:8])
        )

        # (2) partial hit: page=4, n=10 -> aligned 8; P stays 10
        t_part = _tokens(20, 10)
        e_part = _indices(20 * INDEX_BASE_STEP, 10)
        cache_part = _make_cache(FakeTrackingAllocator(), page_size=4)
        _insert(cache_part, t_part, e_part)
        scenarios.append(
            ("partial_hit", 10, _match(cache_part, t_part[:9]), e_part[:8])
        )

        # (3) zero hit: empty tree
        t_zero = _tokens(30, 8)
        cache_zero = _make_cache(FakeTrackingAllocator(), page_size=1)
        scenarios.append(
            ("zero_hit", 8, _match(cache_zero, t_zero), _empty_indices())
        )

        # (4) namespace hit: two namespaces, query owns only its own E
        t_ns = _tokens(40, 8)
        e_na = _indices(30 * INDEX_BASE_STEP, 8)
        e_nb = _indices(40 * INDEX_BASE_STEP, 8)
        ns_a = dict(extra_key="lora-a", cache_salt="tenant-a")
        ns_b = dict(extra_key="lora-b", cache_salt="tenant-b")
        cache_ns = _make_cache(FakeTrackingAllocator(), page_size=2)
        _insert(cache_ns, t_ns, e_na, **ns_a)
        _insert(cache_ns, t_ns, e_nb, **ns_b)
        match_ns = _match(cache_ns, t_ns, **ns_b)
        scenarios.append(("namespace_hit", 8, match_ns, e_nb[:8]))

        for name, prompt_tokens, match, expected_indices in scenarios:
            with self.subTest(scenario=name):
                # L1 first: if the KV is wrong, the metric can never be trusted.
                _assert_indices_equal(
                    self, match.device_indices, expected_indices, ctx=f"A3/L1 {name}"
                )
                if name == "namespace_hit":
                    for value in match.device_indices.tolist():
                        self.assertNotIn(
                            int(value),
                            e_na.tolist(),
                            "A3: cross-namespace KV index leaked into the hit",
                        )

                # H is derived from the SAME match whose indices were asserted.
                hit_len = len(match.device_indices)
                self.assertEqual(hit_len, len(expected_indices))
                self.assertGreaterEqual(hit_len, 0)
                self.assertLessEqual(hit_len, prompt_tokens)

                usage = UsageProcessor.calculate_response_usage(
                    [_response(prompt_tokens, hit_len)],
                    n_choices=1,
                    enable_cache_report=True,
                )
                self.assertEqual(usage.prompt_tokens, prompt_tokens)
                self.assertEqual(usage.completion_tokens, 1)
                if hit_len > 0:
                    self.assertIsNotNone(usage.prompt_tokens_details)
                    self.assertEqual(
                        usage.prompt_tokens_details.cached_tokens,
                        hit_len,
                        "A3: cached_tokens must equal the real hit length",
                    )
                else:
                    self.assertIsNone(
                        usage.prompt_tokens_details,
                        "A3: H=0 must keep the JSON slim (details is None)",
                    )
                self.assertEqual(_cached_prompt_tokens(usage), hit_len)

                anthropic_usage = _anthropic_usage_from_openai(
                    usage, include_input=True, include_output=True
                )
                self.assertEqual(anthropic_usage.input_tokens, prompt_tokens - hit_len)
                if hit_len > 0:
                    self.assertEqual(
                        anthropic_usage.cache_read_input_tokens,
                        hit_len,
                        "A3: Anthropic cache_read_input_tokens must equal H",
                    )
                else:
                    self.assertIsNone(anthropic_usage.cache_read_input_tokens)
                self.assertEqual(anthropic_usage.output_tokens, 1)

    # ------------------------------------------------------------------ #
    # Group B -- the six causes of hitting the wrong KV
    # ------------------------------------------------------------------ #

    def test_namespace_isolation_returns_own_indices_after_split(self):
        """B1 (P0): extra_key / cache_salt isolation AFTER a _split_node.

        Covered cause: namespace cross-talk.  Assertions: L1 indices per
        namespace, index-space guard, correct post-split segmentation, and
        the empty result for a mixed namespace.
        """
        page_size = 4
        tokens = _tokens(500, 12)
        branch = array("q", list(tokens[:8]) + [9000, 9001, 9002, 9003])
        exp_a = _indices(1 * INDEX_BASE_STEP, 12)
        exp_b = _indices(2 * INDEX_BASE_STEP, 12)
        exp_c = _indices(3 * INDEX_BASE_STEP, 12)
        ns_a = dict(extra_key="lora-a", cache_salt="tenant-a")
        ns_b = dict(extra_key="lora-b", cache_salt="tenant-b")
        alloc = FakeTrackingAllocator()
        cache = _make_cache(alloc, page_size=page_size)

        _insert(cache, tokens, exp_a, **ns_a)
        _insert(cache, tokens, exp_b, **ns_b)
        # A diverging sequence in namespace A forces a split of A's node.
        _insert(cache, branch, exp_c, **ns_a)
        self.assertGreaterEqual(
            len(list(_walk_non_root(cache.root_node))),
            4,
            "B1: the diverging insert must have split at least one node",
        )

        # Two-way cross validation: each namespace returns only its own E.
        for name, key_ns, expected, lo, hi in (
            ("A", ns_a, exp_a, 1 * INDEX_BASE_STEP, 2 * INDEX_BASE_STEP),
            ("B", ns_b, exp_b, 2 * INDEX_BASE_STEP, 3 * INDEX_BASE_STEP),
        ):
            with self.subTest(namespace=name):
                match = _match(cache, tokens, **key_ns)
                _assert_indices_equal(
                    self, match.device_indices, expected, ctx=f"B1/L1 ns={name}"
                )
                for value in match.device_indices.tolist():
                    self.assertTrue(
                        lo <= int(value) < hi,
                        f"B1: index {value} escaped namespace {name} range "
                        f"[{lo}, {hi})",
                    )

        # Post-split segmentation: the branch keeps A's shared prefix and its
        # own tail -- no wrong segment, no cross-namespace value.
        match_branch = _match(cache, branch, **ns_a)
        expected_branch = torch.cat([exp_a[:8], exp_c[8:12]])
        _assert_indices_equal(
            self, match_branch.device_indices, expected_branch, ctx="B1/L1 branch"
        )
        # A's full match is unchanged by the split.
        _assert_indices_equal(
            self,
            _match(cache, tokens, **ns_a).device_indices,
            exp_a,
            ctx="B1/L1 A after split",
        )
        # A mixed (half-A/half-B) namespace must hit nothing.
        self.assertEqual(
            _match(
                cache, tokens, extra_key="lora-a", cache_salt="tenant-b"
            ).device_indices.tolist(),
            [],
            "B1: a mixed namespace must never hit either stored segment",
        )

    def test_partial_page_tail_is_never_reported_as_hit(self):
        """B2 (P0): a non-page-aligned tail is never counted as a hit.

        Covered cause: page-alignment boundary.  Assertions: L1 indices of
        the aligned part plus explicit absence of the tail indices.
        """
        page_size = 4
        n = 10
        tokens = _tokens(700, n)
        expected = _indices(3 * INDEX_BASE_STEP, n)
        aligned_len = (n // page_size) * page_size  # 8
        self.assertLess(aligned_len, n, "B2: fixture must have a partial tail")

        alloc = FakeTrackingAllocator()
        cache = _make_cache(alloc, page_size=page_size)
        _insert(cache, tokens, expected)

        queries = (
            ("full", tokens),
            ("shorter", tokens[:9]),
            ("extended", array("q", list(tokens) + [9999])),
        )
        for name, query in queries:
            with self.subTest(query=name):
                match = _match(cache, query)
                self.assertEqual(
                    len(match.device_indices),
                    aligned_len,
                    f"B2: {name} query must hit exactly the aligned prefix",
                )
                _assert_indices_equal(
                    self,
                    match.device_indices,
                    expected[:aligned_len],
                    ctx=f"B2/L1 {name}",
                )
                _assert_no_tail_leak(
                    self,
                    match.device_indices,
                    expected,
                    aligned_len,
                    ctx=f"B2 {name}",
                )
                for leaked in (int(expected[8]), int(expected[9])):
                    self.assertNotIn(
                        leaked,
                        match.device_indices.tolist(),
                        f"B2: tail index {leaked} was reported as a hit",
                    )

    def test_eagle_bigram_view_does_not_leak_extra_kv_index(self):
        """B3 (P0): is_eagle bigram view must not leak an extra KV index.

        Covered cause: maybe_to_bigram_view in-place flip + value truncation
        (radix_cache.py:153-164, :424-427).  Assertions: L1 indices, explicit
        absence of E[n-1], in-place is_bigram state transition, and the
        eagle / non-eagle length difference.
        """
        n = 8
        tokens = _tokens(400, n)
        expected = _indices(4 * INDEX_BASE_STEP, n)  # deliberately 1 longer
        # than the bigram view, so a missing truncation leaks E[7].

        for page_size in (1, 4):
            with self.subTest(page_size=page_size):
                alloc = FakeTrackingAllocator()
                cache = _make_cache(alloc, page_size=page_size, is_eagle=True)

                insert_key = _key(tokens)
                self.assertFalse(
                    insert_key.is_bigram, "B3: fixture key starts non-bigram"
                )
                cache.insert(InsertParams(key=insert_key, value=expected))
                self.assertTrue(
                    insert_key.is_bigram,
                    "B3: maybe_to_bigram_view must flip is_bigram in place",
                )

                match = _match(cache, tokens)
                hit_len = ((n - 1) // page_size) * page_size
                self.assertEqual(
                    len(match.device_indices),
                    hit_len,
                    "B3: bigram hit length must be (n-1) page-aligned",
                )
                _assert_indices_equal(
                    self,
                    match.device_indices,
                    expected[:hit_len],
                    ctx=f"B3/L1 page={page_size}",
                )
                _assert_no_tail_leak(
                    self,
                    match.device_indices,
                    expected,
                    hit_len,
                    ctx=f"B3 page={page_size}",
                )
                self.assertNotIn(
                    int(expected[n - 1]),
                    match.device_indices.tolist(),
                    "B3: the extra (n-th) KV index must never be leaked",
                )

        # Non-eagle control keeps all n indices; eagle keeps n-1.  Assert both
        # contents and the length delta so a missing truncation cannot hide.
        control = _make_cache(FakeTrackingAllocator(), page_size=1)
        _insert(control, tokens, expected)
        control_match = _match(control, tokens)
        _assert_indices_equal(
            self, control_match.device_indices, expected, ctx="B3/L1 control"
        )

        eagle_control = _make_cache(
            FakeTrackingAllocator(), page_size=1, is_eagle=True
        )
        _insert(eagle_control, tokens, expected)
        eagle_match = _match(eagle_control, tokens)
        _assert_indices_equal(
            self,
            eagle_match.device_indices,
            expected[: n - 1],
            ctx="B3/L1 eagle control",
        )
        self.assertEqual(len(control_match.device_indices), n)
        self.assertEqual(len(eagle_match.device_indices), n - 1)
        self.assertEqual(
            len(control_match.device_indices) - len(eagle_match.device_indices),
            1,
            "B3: eagle must cache exactly one fewer KV index than non-eagle",
        )

    def test_split_keeps_value_key_hash_length_invariants(self):
        """B4 (P0): _split_node keeps value / key / hash_value /
        event_hash_value lengths consistent, and match still returns the
        correct segments.

        Covered cause: node split.  Assertions: per-node invariants,
        parent+child hash concatenation, and L1 indices for both branches.
        """
        page_size = 2
        cache_salt = "tenant-b4"
        seq_a = _tokens(1, 8)
        seq_b = array("q", [1, 2, 3, 4, 90, 91, 92, 93])  # fork at 4
        exp_a = _indices(10 * INDEX_BASE_STEP, 8)
        exp_b = _indices(20 * INDEX_BASE_STEP, 8)
        alloc = FakeTrackingAllocator()
        cache = _make_cache(alloc, page_size=page_size, enable_events=True)

        _insert(cache, seq_a, exp_a, cache_salt=cache_salt)
        cache.take_events()  # ensure hash_value / event_hash_value are computed
        first = _match(cache, seq_a, cache_salt=cache_salt)
        original_hash = list(first.last_device_node.hash_value)
        original_event_hash = list(first.last_device_node.event_hash_value)
        self.assertEqual(len(original_hash), len(seq_a) // page_size)

        _insert(cache, seq_b, exp_b, cache_salt=cache_salt)
        cache.take_events()
        nodes = list(_walk_non_root(cache.root_node))

        # The invariants must be checked on real (non-None) hashes, not vacuously.
        self.assertTrue(any(n.hash_value is not None for n in nodes))
        self.assertTrue(any(n.event_hash_value is not None for n in nodes))
        for node in nodes:
            with self.subTest(node_id=node.id, key_len=len(node.key)):
                self.assertEqual(
                    len(node.value),
                    len(node.key),
                    "B4: value length must track key length after a split",
                )
                if node.hash_value is not None:
                    self.assertEqual(
                        len(node.hash_value),
                        len(node.key) // page_size,
                        "B4: hash_value pages must track key length",
                    )
                if node.event_hash_value is not None:
                    self.assertEqual(
                        len(node.event_hash_value),
                        len(node.key) // page_size,
                        "B4: event_hash_value pages must track key length",
                    )

        match_a = _match(cache, seq_a, cache_salt=cache_salt)
        match_b = _match(cache, seq_b, cache_salt=cache_salt)
        split_parent = match_a.last_device_node.parent
        self.assertEqual(
            split_parent.hash_value + match_a.last_device_node.hash_value,
            original_hash,
            "B4: parent + child hash_value must reconstruct the original",
        )
        self.assertEqual(
            split_parent.event_hash_value
            + match_a.last_device_node.event_hash_value,
            original_event_hash,
            "B4: parent + child event_hash_value must reconstruct the original",
        )
        _assert_indices_equal(
            self, match_a.device_indices, exp_a, ctx="B4/L1 seq A"
        )
        # The branch shares A's first 4 indices, then its own tail: correct
        # segmentation, not the wrong segment.
        expected_b = torch.cat([exp_a[:4], exp_b[4:8]])
        _assert_indices_equal(
            self, match_b.device_indices, expected_b, ctx="B4/L1 seq B"
        )

    def test_evicted_indices_are_never_returned_by_later_match(self):
        """B5 (P0): evicted indices must never be returned again.

        Covered cause: stale / reused KV slots after eviction.  Assertions:
        generation state transition (L1b), the exact freed index set, empty
        match results, and total_size drop.  It never asserts that a mock was
        merely invoked; every assertion is on a result or a state transition.
        """
        page_size = 1
        seq_a = _tokens(1, 4)
        seq_b = _tokens(11, 4)  # disjoint first token -> two leaves
        alloc = FakeTrackingAllocator()
        exp_a = alloc.alloc(4, base=7 * INDEX_BASE_STEP)
        exp_b = alloc.alloc(4, base=8 * INDEX_BASE_STEP)
        gen_at_insert = _generation_snapshot(alloc, torch.cat([exp_a, exp_b]))
        cache = _make_cache(alloc, page_size=page_size)
        _insert(cache, seq_a, exp_a)
        _insert(cache, seq_b, exp_b)
        self.assertEqual(cache.total_size(), 8)

        result = cache.evict(EvictParams(num_tokens=8))
        self.assertEqual(
            result.num_tokens_evicted,
            8,
            "B5: full eviction must free every evictable token",
        )
        # L1b as a state transition: every owned index's generation advanced.
        for idx in exp_a.tolist() + exp_b.tolist():
            with self.subTest(index=int(idx)):
                self.assertEqual(
                    alloc.generation_of(idx),
                    gen_at_insert[int(idx)] + 1,
                    "B5: evict must free (advance generation of) the index",
                )
        freed = sorted(
            int(i) for call in alloc.free_calls for i in call[0]
        )
        self.assertEqual(
            freed,
            sorted(exp_a.tolist() + exp_b.tolist()),
            "B5: the freed index set must be exactly the two stored segments",
        )
        self.assertEqual(cache.total_size(), 0, "B5: tree must be empty")

        for name, tokens in (("A", seq_a), ("B", seq_b)):
            with self.subTest(sequence=name):
                match = _match(cache, tokens)
                self.assertEqual(
                    match.device_indices.tolist(),
                    [],
                    "B5: an evicted sequence must not hit any stale index",
                )
                _assert_generation_fresh(
                    self, alloc, match.device_indices, gen_at_insert, ctx="B5/L1b"
                )

    def test_locked_node_survives_eviction_with_unchanged_indices(self):
        """B6 (P0): a locked node survives full eviction pressure unchanged.

        Covered cause: lock_ref / eviction race.  Assertions: L1 indices and
        generation for the survivor, empty result for the evicted peer, and
        protected/evictable size transitions (before and after dec_lock_ref).
        """
        page_size = 1
        seq_a = _tokens(1, 4)
        seq_b = _tokens(11, 4)
        alloc = FakeTrackingAllocator()
        exp_a = alloc.alloc(4, base=11 * INDEX_BASE_STEP)
        exp_b = alloc.alloc(4, base=12 * INDEX_BASE_STEP)
        gen_at_insert = _generation_snapshot(alloc, torch.cat([exp_a, exp_b]))
        cache = _make_cache(alloc, page_size=page_size)
        _insert(cache, seq_a, exp_a)
        _insert(cache, seq_b, exp_b)

        before = _match(cache, seq_a)
        _assert_indices_equal(
            self, before.device_indices, exp_a, ctx="B6/L1 before lock"
        )
        cache.inc_lock_ref(before.last_device_node)
        self.assertGreaterEqual(
            cache.protected_size(),
            len(exp_a),
            "B6: locking must protect the whole A segment",
        )

        cache.evict(EvictParams(num_tokens=8))

        survivor = _match(cache, seq_a)
        _assert_indices_equal(
            self, survivor.device_indices, exp_a, ctx="B6/L1 after eviction"
        )
        _assert_generation_fresh(
            self, alloc, survivor.device_indices, gen_at_insert, ctx="B6/L1b"
        )
        for value in survivor.device_indices.tolist():
            self.assertNotIn(
                int(value),
                exp_b.tolist(),
                "B6: the evicted peer's indices leaked into the locked hit",
            )
        self.assertEqual(
            _match(cache, seq_b).device_indices.tolist(),
            [],
            "B6: the unlocked peer must have been evicted",
        )
        for idx in exp_b.tolist():
            self.assertEqual(
                alloc.generation_of(idx),
                gen_at_insert[int(idx)] + 1,
                "B6: the unlocked peer's indices must have been freed",
            )
        self.assertEqual(
            cache.evictable_size(),
            0,
            "B6: after eviction nothing evictable remains, only the lock",
        )

        # Releasing the lock restores the accounting (state transition).
        cache.dec_lock_ref(before.last_device_node)
        self.assertEqual(cache.protected_size(), 0)
        self.assertEqual(cache.evictable_size(), len(exp_a))

    # ------------------------------------------------------------------ #
    # Group C -- boundary
    # ------------------------------------------------------------------ #

    def test_limited_key_never_matches_beyond_limit(self):
        """C1 (P1): a ``limit``-capped key never matches past its limit.

        Covered cause: zero-copy ``limit`` slice / child_key_at view.
        Assertions: L1 indices, absence of E[limit:], child_key view equality,
        and the out-of-bounds IndexError from child_key_at.
        """
        page_size = 4
        n = 12
        limit = 6
        tokens = _tokens(600, n)
        expected = _indices(6 * INDEX_BASE_STEP, n)
        aligned_len = (limit // page_size) * page_size  # 4

        limited = _key(tokens, limit=limit)
        self.assertEqual(len(limited), limit)
        self.assertEqual(len(limited.raw_token_ids()), limit)
        self.assertEqual(
            limited.child_key(page_size),
            limited[:4].child_key(page_size),
            "C1: child_key_at must agree with the materialized slice view",
        )
        with self.assertRaises(IndexError):
            # offset + page_size (8) > len(self) (6): a capped key must not
            # expose child keys beyond its limit.
            limited.child_key_at(offset=4, page_size=page_size)

        alloc = FakeTrackingAllocator()
        cache = _make_cache(alloc, page_size=page_size)
        cache.insert(InsertParams(key=_key(tokens, limit=limit), value=expected))

        # Both the capped key and the full 12-token key may only hit E[:4].
        for name, query in (
            ("capped", _key(tokens, limit=limit)),
            ("full", _key(tokens)),
        ):
            with self.subTest(query=name):
                match = cache.match_prefix(MatchPrefixParams(key=query))
                self.assertLessEqual(len(match.device_indices), limit)
                self.assertEqual(len(match.device_indices), aligned_len)
                _assert_indices_equal(
                    self,
                    match.device_indices,
                    expected[:aligned_len],
                    ctx=f"C1/L1 {name}",
                )
                _assert_no_tail_leak(
                    self,
                    match.device_indices,
                    expected,
                    aligned_len,
                    ctx=f"C1 {name}",
                )
                for leaked in expected[limit:].tolist():
                    self.assertNotIn(
                        int(leaked),
                        match.device_indices.tolist(),
                        "C1: an index beyond the limit was returned",
                    )

    # ------------------------------------------------------------------ #
    # Group D -- checkpoint (P2).  Self-contained fakes: no ReqKvInfo import,
    # no real allocator, no server.  See design doc 5.4.
    # ------------------------------------------------------------------ #

    def test_checkpoint_repoints_row_to_own_indices(self):
        """D1 (P2): checkpoint repoints the req_to_token row and prefix indices
        onto the tree's own E, with no stale index.

        Covered cause: checkpoint row re-pointing.  Assertions: L1 indices on
        req.prefix_indices and the row, generation (L1b), cache_protected_len
        and last_node state transitions.
        """
        page_size = 1
        n = 6
        tokens = _tokens(800, n)
        expected = _indices(80 * INDEX_BASE_STEP, n)
        alloc = FakeTrackingAllocator()
        gen_at_insert = _generation_snapshot(alloc, expected)
        cache = _make_cache(alloc, page_size=page_size)
        cache.req_to_token_pool = FakeReqToTokenPool(expected)

        kv = SimpleNamespace(
            req_pool_idx=0,
            cache_protected_len=0,
            cache_inserted_len=0,
            swa_evict_floor=0,
        )
        req = SimpleNamespace(
            kv=kv,
            extra_key=None,
            cache_salt=None,
            priority=0,
            origin_input_ids=tokens,
            full_untruncated_fill_ids=tokens,
            output_ids=array("q"),
            last_node=cache.root_node,
            prefix_indices=None,
        )

        cache.checkpoint(req, up_to=n)

        _assert_indices_equal(
            self,
            req.prefix_indices,
            expected,
            ctx="D1/L1 prefix_indices",
        )
        _assert_generation_fresh(
            self, alloc, req.prefix_indices, gen_at_insert, ctx="D1/L1b"
        )
        self.assertEqual(
            cache.req_to_token_pool.req_to_token[0].tolist(),
            expected.tolist(),
            "D1: the req_to_token row must be repointed onto the tree's E",
        )
        _assert_indices_equal(
            self,
            _match(cache, tokens).device_indices,
            expected,
            ctx="D1/L1 tree match",
        )
        self.assertEqual(kv.cache_protected_len, len(expected))
        self.assertIsNot(
            req.last_node,
            cache.root_node,
            "D1: last_node must be re-anchored on the inserted node",
        )


if __name__ == "__main__":
    unittest.main()
