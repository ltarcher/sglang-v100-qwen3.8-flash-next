from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
    _sm70_dense_prefill_max_tokens,
    use_sm70_qsa_dense_prefill,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class _FakeTensor:
    def __init__(self, shape, dtype, *, is_cuda=True):
        self.shape = torch.Size(shape)
        self.dtype = dtype
        self.is_cuda = is_cuda
        self.device = torch.device("cuda:0")

    @property
    def ndim(self):
        return len(self.shape)


@pytest.mark.parametrize(
    "forward_mode",
    [
        ForwardMode.DECODE,
        ForwardMode.TARGET_VERIFY,
        ForwardMode.DRAFT_EXTEND_V2,
    ],
)
def test_sm70_qsa_decode_accepts_paged_decode_modes(monkeypatch, forward_mode):
    rows = 4
    q = _FakeTensor((rows, 6, 256), torch.float16)
    k = _FakeTensor((4096, 1, 256), torch.float8_e5m2)
    v = _FakeTensor((4096, 1, 256), torch.float8_e5m2)
    topk_indices = _FakeTensor((rows, 2048), torch.int32)
    metadata = SimpleNamespace(
        sequence_lengths=SimpleNamespace(numel=lambda: rows),
    )
    forward_batch = SimpleNamespace(forward_mode=forward_mode)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 0))

    assert QwenSparseAttnBackend._can_use_sm70_sparse_decode(
        q,
        k,
        v,
        forward_batch,
        metadata,
        topk_indices,
    )


def test_sm70_qsa_decode_rejects_ordinary_extend(monkeypatch):
    rows = 4
    q = _FakeTensor((rows, 6, 256), torch.float16)
    k = _FakeTensor((4096, 1, 256), torch.float8_e5m2)
    v = _FakeTensor((4096, 1, 256), torch.float8_e5m2)
    topk_indices = _FakeTensor((rows, 2048), torch.int32)
    metadata = SimpleNamespace(
        sequence_lengths=SimpleNamespace(numel=lambda: rows),
    )
    forward_batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 0))

    assert not QwenSparseAttnBackend._can_use_sm70_sparse_decode(
        q,
        k,
        v,
        forward_batch,
        metadata,
        topk_indices,
    )


def test_sm70_qsa_dense_prefill_uses_packaged_kernel_per_sequence(monkeypatch):
    from sglang.srt.layers.attention.tilelang_fa_v100 import _kernels_dense_d256

    calls = []

    def fake_get_kernel(heads, heads_kv):
        assert (heads, heads_kv) == (6, 1)

        def fake_kernel(q, k, v, prefix_len, softmax_scale):
            calls.append((q.shape[0], k.shape[0], v.shape[0], prefix_len))
            assert softmax_scale == pytest.approx(0.0625)
            return q + len(calls)

        return fake_kernel

    monkeypatch.setattr(
        _kernels_dense_d256, "get_dense_prefix_d256_kernel", fake_get_kernel
    )
    q = torch.zeros((5, 6, 256), dtype=torch.float16)
    k = torch.zeros((5, 1, 256), dtype=torch.float16)
    v = torch.zeros_like(k)

    output = QwenSparseAttnBackend._forward_sm70_dense_prefill(q, k, v, [2, 3], 0.0625)

    assert calls == [(2, 2, 2, 0), (3, 3, 3, 0)]
    torch.testing.assert_close(output[:2], torch.ones_like(output[:2]))
    torch.testing.assert_close(output[2:], torch.full_like(output[2:], 2))


def test_sm70_qsa_dense_prefill_rejects_mismatched_packing():
    q = torch.zeros((5, 6, 256), dtype=torch.float16)
    k = torch.zeros((5, 1, 256), dtype=torch.float16)
    v = torch.zeros_like(k)

    with pytest.raises(ValueError, match="packed rows"):
        QwenSparseAttnBackend._forward_sm70_dense_prefill(q, k, v, [2, 2], 0.0625)


@pytest.mark.parametrize(
    "lengths,extend_lengths,mode,expected",
    [
        ([1000], [1000], ForwardMode.EXTEND, True),
        ([8192, 1000], [8192, 1000], ForwardMode.EXTEND, True),
        ([8193], [8193], ForwardMode.EXTEND, False),
        ([8192], [4096], ForwardMode.EXTEND, False),
        ([1000], [1], ForwardMode.DECODE, False),
        ([4], [4], ForwardMode.TARGET_VERIFY, False),
        ([4], [4], ForwardMode.DRAFT_EXTEND_V2, False),
    ],
)
def test_sm70_qsa_dense_selection_matches_attention_route(
    monkeypatch, lengths, extend_lengths, mode, expected
):
    monkeypatch.setenv("SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS", "8192")
    _sm70_dense_prefill_max_tokens.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 0))
    batch = SimpleNamespace(
        seq_lens_cpu=lengths, extend_seq_lens_cpu=extend_lengths, forward_mode=mode
    )
    assert use_sm70_qsa_dense_prefill(batch, torch.device("cuda")) is expected
    batch._original_forward_mode = ForwardMode.TARGET_VERIFY
    assert not use_sm70_qsa_dense_prefill(batch, torch.device("cuda"))
    _sm70_dense_prefill_max_tokens.cache_clear()


def test_sm70_qsa_dense_selection_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS", "0")
    _sm70_dense_prefill_max_tokens.cache_clear()
    batch = SimpleNamespace(
        seq_lens_cpu=[1000], extend_seq_lens_cpu=[1000], forward_mode=ForwardMode.EXTEND
    )
    assert not use_sm70_qsa_dense_prefill(batch, torch.device("cuda"))
    _sm70_dense_prefill_max_tokens.cache_clear()


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_dense_prefill_retains_mtp_seed_indices(monkeypatch, capture, compressed):
    from sglang.srt.layers.attention import qwen_sparse_attn_backend
    from sglang.srt.layers.attention.qsa import glue
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
    from sglang.srt.models import qwen4_exp

    calls = []

    class RecordingIndexer(QSAIndexer if compressed else torch.nn.Module):
        def __init__(self):
            torch.nn.Module.__init__(self)

        def forward(self, hidden, positions, batch, metadata, **kwargs):
            skip = kwargs.get("skip_prefill_selection", False)
            calls.append(skip)
            return torch.zeros((3, 0 if skip else 2051), dtype=torch.int32)

    captured = []
    backend = SimpleNamespace(
        should_capture_mtp_sparse_indices=lambda batch: capture,
        capture_mtp_sparse_indices=lambda indices, *args, **kwargs: captured.append(
            indices
        ),
    )
    monkeypatch.setattr(qwen4_exp, "get_attn_backend", lambda: backend)
    monkeypatch.setattr(glue, "resolve_qsa_sparse_backend", lambda backend: backend)
    monkeypatch.setattr(glue, "get_qsa_indexer_metadata", lambda *args: object())
    monkeypatch.setattr(
        qwen_sparse_attn_backend, "use_sm70_qsa_dense_prefill", lambda *args: True
    )
    layer = SimpleNamespace(indexer=RecordingIndexer(), layer_id=2)
    output = qwen4_exp.Qwen4ExpAttentionDecoderLayer._compute_qsa_topk_indices_eager(
        layer, torch.zeros((3, 2560)), torch.arange(3), SimpleNamespace()
    )
    assert calls == [compressed and not capture]
    if capture:
        assert captured == [output]
        assert output.shape == (3, 2051)
    else:
        assert not captured


def test_dense_prefill_selection_skip_still_updates_keys():
    from unittest.mock import Mock

    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

    positions = torch.arange(5)
    hidden = torch.zeros((5, 2560))
    token_k = torch.randn((3, 1, 128))
    slots = torch.arange(3)
    metadata = SimpleNamespace(
        get_token_to_batch_idx=lambda: torch.zeros(3, dtype=torch.int32),
        pending_ring_slots=slots,
        token_to_kv_pool=object(),
    )
    indexer = SimpleNamespace(
        index_n_heads=4,
        project_qk=Mock(return_value=(torch.zeros((3, 4, 128)), token_k, True)),
        update_key_state_and_compress=Mock(),
    )
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND, positions=positions)
    output = QSAIndexer._forward_impl(
        indexer, hidden, positions, batch, metadata, skip_prefill_selection=True
    )
    assert output.shape == (3, 0)  # Physical DP padding does not create semantic rows.
    assert output.dtype == torch.int32
    projected = indexer.project_qk.call_args.args
    assert projected[0].shape == (3, 2560)
    args = indexer.update_key_state_and_compress.call_args
    assert args.args[0] is token_k
    torch.testing.assert_close(args.args[1], positions[:3])
    assert args.kwargs == {"state_slots": slots, "state_stored": True}


@pytest.mark.parametrize(
    "mode", [ForwardMode.DECODE, ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2]
)
def test_selection_skip_rejects_decode_and_speculation(mode):
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

    with pytest.raises(ValueError, match="ordinary prefill"):
        QSAIndexer._forward_impl(
            object(),
            None,
            None,
            SimpleNamespace(forward_mode=mode),
            None,
            skip_prefill_selection=True,
        )


class _GuardSentinelBackend:
    """Stub backend whose ``req_to_token`` access proves the mixed-batch
    guard let a batch through to the grouped planner."""

    _kv_host_tier = None

    @property
    def req_to_token(self):
        raise AssertionError("grouped-prefill guard must not block this batch")


def _eligible_grouped_prefill_args():
    rows = 256
    q = torch.randn(rows, 6, 256, dtype=torch.float16)
    kv = torch.randn(rows, 1, 256, dtype=torch.float16)
    topk = torch.zeros(rows, 64, dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, rows], dtype=torch.int32)
    return q, kv, kv, topk, cu_seqlens_q, 0.0625


class _RecordingTier:
    """Fake KV host tier recording flush() calls (m8: replay flush guard)."""

    phase3 = False
    pair_split = False

    def __init__(self, phase3):
        self.phase3 = phase3
        self.pagein = False
        self.flush_calls = []

    def flush(self, out_cache_loc):
        self.flush_calls.append(out_cache_loc)


class _PairSplitTier(_RecordingTier):
    """phase3 + PAIR_PARITY tier stub recording pair_peer_kv lookups."""

    pair_split = True

    def __init__(self):
        super().__init__(phase3=True)
        self.layer_slot = {5: 0}
        self.peer_calls = []

    def pair_peer_kv(self, tier_layer):
        self.peer_calls.append(tier_layer)
        return (1 << 30, "peer_k_buf", "peer_v_buf")


def _make_replay_backend(tier, gpu_kernel):
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend._kv_host_tier = tier
    backend._require_chain_speculation = lambda *a, **k: None
    backend._is_speculative_paged_mode = lambda mode: False
    backend._update_qsa_cuda_graph_metadata = lambda *a, **k: None
    backend._tier3_refresh_graph_tables = lambda *a, **k: None
    if gpu_kernel:
        backend._can_replay_with_gpu_kernels = lambda metadata, seq_lens: True
        backend._replay_cuda_graph_metadata_gpu = lambda *a, **k: None
    else:
        backend._can_replay_with_gpu_kernels = lambda metadata, seq_lens: False
    metadata = SimpleNamespace(
        sequence_lengths=torch.zeros(4, dtype=torch.int32),
        row_req_pool_indices=torch.zeros(4, dtype=torch.int32),
        indexer_metadata=SimpleNamespace(
            graph_prefix_lengths=torch.zeros(4, dtype=torch.int32)
        ),
    )
    backend._cuda_graph_metadata = {(ForwardMode.DECODE, 4): metadata}
    backend.forward_metadata = None
    return backend


@pytest.mark.parametrize("gpu_kernel", [False, True])
def test_replay_flush_skipped_under_phase3(gpu_kernel):
    # m8 (docs/v100/CODE_REVIEW_TODO_20261110.md): phase3 bumps step_seq in
    # on_build_phase3 (also reached via _tier3_refresh_graph_tables on
    # replay); an unguarded flush(None) would bump it a second time -- the
    # replay path must mirror the eager build's `if not phase3` guard.
    tier = _RecordingTier(phase3=True)
    backend = _make_replay_backend(tier, gpu_kernel)
    backend._replay_cuda_graph_metadata(
        bs=4,
        req_pool_indices=torch.zeros(4, dtype=torch.int64),
        seq_lens=torch.tensor([10, 11, 12, 13]),
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
        seq_lens_cpu=None,
        out_cache_loc=None,
    )
    assert tier.flush_calls == []


@pytest.mark.parametrize("gpu_kernel", [False, True])
def test_replay_flush_still_runs_without_phase3(gpu_kernel):
    tier = _RecordingTier(phase3=False)
    backend = _make_replay_backend(tier, gpu_kernel)
    backend._replay_cuda_graph_metadata(
        bs=4,
        req_pool_indices=torch.zeros(4, dtype=torch.int64),
        seq_lens=torch.tensor([10, 11, 12, 13]),
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
        seq_lens_cpu=None,
        out_cache_loc=None,
    )
    assert tier.flush_calls == [None]


def test_grouped_prefill_guard_blocks_rewritten_mixed_batch():
    # M3 (docs/v100/CODE_REVIEW_TODO_20261110.md): eager_runner rewrites
    # MIXED -> EXTEND before metadata init, so the mixed identity only
    # survives in _original_forward_mode. The guard must read it, or
    # mixed-chunk batches reach the grouped planner and corrupt decode
    # tails.
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        _original_forward_mode=ForwardMode.MIXED,
    )
    q, k, v, topk, cu, scale = _eligible_grouped_prefill_args()
    result = QwenSparseAttnBackend._forward_sm70_grouped_sparse_prefill(
        _GuardSentinelBackend(), q, k, v, forward_batch, topk, cu, scale
    )
    assert result is None


def test_grouped_prefill_guard_blocks_direct_mixed_batch():
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.MIXED,
        _original_forward_mode=None,
    )
    q, k, v, topk, cu, scale = _eligible_grouped_prefill_args()
    result = QwenSparseAttnBackend._forward_sm70_grouped_sparse_prefill(
        _GuardSentinelBackend(), q, k, v, forward_batch, topk, cu, scale
    )
    assert result is None


def test_grouped_prefill_guard_does_not_block_pure_extend():
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        _original_forward_mode=None,
    )
    q, k, v, topk, cu, scale = _eligible_grouped_prefill_args()
    with pytest.raises(AssertionError, match="must not block"):
        QwenSparseAttnBackend._forward_sm70_grouped_sparse_prefill(
            _GuardSentinelBackend(), q, k, v, forward_batch, topk, cu, scale
        )


class TestMixedTailPrefixAlignmentAssert:
    """m17 (docs/v100/CODE_REVIEW_TODO_20261110.md): on a mixed-chunk batch
    only the exempt decode tails may carry a ratio-misaligned prefix — a
    misaligned prefill row would leave a shared group half-written. The
    alignment invariant must stay asserted (device-side, zero sync) for the
    non-tail rows instead of being skipped wholesale for mixed batches."""

    RATIO = 2

    def _backend(self):
        backend = object.__new__(QwenSparseAttnBackend)
        backend.token_to_kv_pool = SimpleNamespace(qsa_compress_ratio=self.RATIO)
        return backend

    def _forward_batch(self, extend_lens):
        return SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int32),
            input_ids=torch.zeros(int(sum(extend_lens)), dtype=torch.int64),
        )

    def _write_plan(self, backend, sequence_lengths, extend_lens, mix_tails):
        width = 16
        token_slot_table = torch.arange(
            len(sequence_lengths) * width, dtype=torch.int64
        ).reshape(len(sequence_lengths), width)
        return backend._qsa_build_write_plan(
            forward_batch=self._forward_batch(extend_lens),
            speculative_paged=False,
            token_slot_table=token_slot_table,
            sequence_lengths=torch.tensor(sequence_lengths, dtype=torch.int64),
            mix_tails=mix_tails,
        )

    def test_mixed_batch_aligned_prefill_and_exempt_tail_passes(self):
        # Row 0: prefill with a ratio-aligned shared prefix (4). Row 1: decode
        # tail resuming at its own length (prefix 3, misaligned, exempt).
        backend = self._backend()
        plan = self._write_plan(
            backend,
            sequence_lengths=[8, 4],
            extend_lens=[4, 1],
            mix_tails=torch.tensor([False, True]),
        )
        assert plan is not None

    def test_mixed_batch_misaligned_prefill_row_fails_loud(self):
        # Row 1 is NOT a tail but carries a misaligned prefix (3): the
        # invariant is violated and must be asserted even on a mixed batch.
        backend = self._backend()
        with pytest.raises(RuntimeError):
            self._write_plan(
                backend,
                sequence_lengths=[8, 4],
                extend_lens=[4, 1],
                mix_tails=torch.tensor([False, False]),
            )

    def test_non_mixed_batch_misaligned_row_still_fails_loud(self):
        backend = self._backend()
        with pytest.raises(RuntimeError):
            self._write_plan(
                backend,
                sequence_lengths=[8, 4],
                extend_lens=[4, 1],
                mix_tails=None,
            )

    def test_mixed_batch_all_tails_passes(self):
        # Every row is an exempt tail: the non-tail subset is empty and the
        # vacuous assert passes without a sync.
        backend = self._backend()
        plan = self._write_plan(
            backend,
            sequence_lengths=[4, 5],
            extend_lens=[1, 1],
            mix_tails=torch.tensor([True, True]),
        )
        assert plan is not None


def _tier3_phase3(monkeypatch, phase3):
    tier = _RecordingTier(phase3=phase3)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 0))
    return tier


def test_fused_sm70_prefill_gate_rejects_kv_tier_phase3(monkeypatch):
    # Phase3 remaps device rows (mod-picked slots; PAIR halves them again),
    # so a prefill kernel that indexes the pool through the LOGICAL
    # req_to_token table reads the wrong physical rows. The direct-read
    # fused single-request route must decline and fall through to the
    # tier-aware row-by-row gather.
    _tier3_phase3(monkeypatch, phase3=False)
    q = _FakeTensor((100, 6, 256), torch.float16)
    k = _FakeTensor((4096, 1, 256), torch.float8_e5m2)
    v = _FakeTensor((4096, 1, 256), torch.float8_e5m2)
    topk = torch.zeros(100, 64, dtype=torch.int32)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        req_pool_indices=torch.zeros(1, dtype=torch.int64),
        seq_lens=torch.tensor([1000]),
    )
    assert QwenSparseAttnBackend._can_use_sm70_sparse_prefill(q, k, v, batch, topk)
    assert not QwenSparseAttnBackend._can_use_sm70_sparse_prefill(
        q, k, v, batch, topk, kv_tier_phase3=True
    )


def test_batched_sm70_prefill_plan_rejects_kv_tier_phase3(monkeypatch):
    # Same contract for the multi-request per-sequence chunk-prefill plan.
    _tier3_phase3(monkeypatch, phase3=False)
    q = _FakeTensor((100, 6, 256), torch.float16)
    k = _FakeTensor((4096, 1, 256), torch.float8_e5m2)
    v = _FakeTensor((4096, 1, 256), torch.float8_e5m2)
    topk = torch.zeros(100, 64, dtype=torch.int32)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        req_pool_indices=torch.zeros(2, dtype=torch.int64),
        seq_lens=torch.tensor([1000, 1000]),
        extend_seq_lens_cpu=[50, 50],
    )
    plan = QwenSparseAttnBackend._plan_sm70_sparse_prefill_batched(q, k, v, batch, topk)
    assert plan == [(0, 0, 50), (1, 50, 100)]
    assert (
        QwenSparseAttnBackend._plan_sm70_sparse_prefill_batched(
            q, k, v, batch, topk, kv_tier_phase3=True
        )
        is None
    )


def test_grouped_prefill_guard_blocks_kv_tier_phase3_non_pair(monkeypatch):
    # Phase3 remaps device rows (mod-picked slots; PAIR halves them), so a
    # prefill kernel that indexes the pool through the LOGICAL req_to_token
    # table reads the wrong physical rows. Without PAIR there is no
    # device-resident read path at all (host-mirror fallback + extend-write
    # aliasing), so the grouped route must still decline under phase3.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 0))
    tier = _RecordingTier(phase3=True)
    backend = _GuardSentinelBackend()
    backend._kv_host_tier = tier
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        _original_forward_mode=None,
    )
    q, k, v, topk, cu, scale = _eligible_grouped_prefill_args()
    with patch(
        "sglang.srt.layers.attention.tilelang_fa_v100."
        "_kernels_grouped_sparse_prefill.build_grouped_union_plan"
    ) as build_plan:
        result = QwenSparseAttnBackend._forward_sm70_grouped_sparse_prefill(
            backend, q, k, v, forward_batch, topk, cu, scale
        )
    assert result is None
    build_plan.assert_not_called()


def test_grouped_prefill_phase3_pair_routes_through_planner(monkeypatch):
    # Route B (docs/v100/PREFILL_LEVERS_PLAN.md §8.3): under phase3+PAIR the
    # planner resolves slots through the tier's batch-local physical table
    # (owner-bit dual pointer, same contract as the decode read path) and the
    # kernel reads the pair member's imported buffers, so the phase3 gate no
    # longer blocks the grouped route.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 0))
    tier = _PairSplitTier()
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend._kv_host_tier = tier
    backend.max_context_len = 2048
    backend._sm70_grouped_prefill_logged = True
    phys_table = torch.zeros(1, 32, dtype=torch.int32)
    backend.forward_metadata = SimpleNamespace(phys_token_slot_table=phys_table)
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        _original_forward_mode=None,
        seq_lens=torch.tensor([256], dtype=torch.int64),
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
    )
    layer = SimpleNamespace(layer_id=5)
    plan = (torch.zeros(4, dtype=torch.int32), None, 64, None)
    kernel_output = torch.zeros(256, 6 * 256, dtype=torch.float16)
    fake_peer = torch.zeros(256, 1, 256, dtype=torch.float16)
    q, k, v, topk, cu, scale = _eligible_grouped_prefill_args()
    with (
        patch(
            "sglang.srt.layers.attention.qwen_sparse_attn_backend."
            "is_in_breakable_cuda_graph",
            return_value=False,
        ),
        patch(
            "sglang.srt.mem_cache.pair_split_ipc.imported_as_tensor",
            return_value=fake_peer,
        ) as as_tensor,
        patch(
            "sglang.srt.layers.attention.tilelang_fa_v100."
            "_kernels_grouped_sparse_prefill.build_grouped_union_plan",
            return_value=plan,
        ) as build_plan,
        patch(
            "sglang.srt.layers.attention.tilelang_fa_v100."
            "_kernels_grouped_sparse_prefill.run_grouped_sparse_prefill",
            return_value=kernel_output,
        ) as run_kernel,
    ):
        result = QwenSparseAttnBackend._forward_sm70_grouped_sparse_prefill(
            backend, q, k, v, forward_batch, topk, cu, scale, layer=layer
        )
    assert result is kernel_output
    assert tier.peer_calls == [0]
    assert as_tensor.call_count == 2
    assert as_tensor.call_args_list[0].args[1] == (256, 1, 256)
    build_plan.assert_called_once_with(
        topk,
        cu,
        forward_batch.seq_lens,
        forward_batch.req_pool_indices,
        None,
        2048,
        slot_table=phys_table,
        owner_mask=1 << 30,
    )
    run_kernel.assert_called_once_with(
        q, k, v, plan, scale, peer_kv=(fake_peer, fake_peer)
    )
