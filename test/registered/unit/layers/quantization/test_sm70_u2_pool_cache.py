"""Unit tests for the persistent u2 expert pool cache (sm70_u2_pool)."""

import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization import sm70_u2_pool
from sglang.srt.layers.quantization.sm70_u2_pool import (
    _U2LayerCache,
    _u2_cache_fingerprint,
    _u2_cache_prune,
    _u2_pool_layout_ok,
    convert_moe_layer_to_u2,
    u2_checkpoint_reader,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _fake_layer(layer_id: int = 3) -> SimpleNamespace:
    return SimpleNamespace(layer_id=layer_id)


def _pools(e: int = 4, n13: int = 8, hidden: int = 16, group: int = 8):
    return {
        "w13": torch.arange(e * (hidden // 16) * n13, dtype=torch.int32).view(
            e, hidden // 16, n13
        ),
        "w2": torch.arange(e * (hidden // 16) * hidden, dtype=torch.int32).view(
            e, hidden // 16, hidden
        ),
        "s13": torch.randn(e, hidden // group, n13).to(torch.float16),
        "s2": torch.randn(e, hidden // group, hidden).to(torch.float16),
    }


def _shapes(pools: dict) -> dict:
    return {k: tuple(v.shape) for k, v in pools.items()}


class TestU2CacheFingerprint(CustomTestCase):
    def test_weight_file_change_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.safetensors")
            with open(path, "wb") as f:
                f.write(b"x" * 16)
            fp1 = _u2_cache_fingerprint(d, tp_size=4)
            os.utime(path, ns=(1_000_000_000, 1_000_000_000))
            os.utime(path, ns=(2_000_000_000, 2_000_000_000))
            fp2 = _u2_cache_fingerprint(d, tp_size=4)
            self.assertNotEqual(fp1, fp2)

    def test_tp_change_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertNotEqual(
                _u2_cache_fingerprint(d, tp_size=1),
                _u2_cache_fingerprint(d, tp_size=4),
            )


class TestU2LayerCache(CustomTestCase):
    def test_store_load_roundtrip_is_bit_exact(self):
        layer = _fake_layer()
        src = _pools()
        with tempfile.TemporaryDirectory() as d:
            cache = _U2LayerCache(cache_dir=d, rank=2)
            cache.store_layer(layer, group_size=128, macro13=256, macro2=256, pools=src)
            got = cache.load_layer(
                layer,
                group_size=128,
                macro13=256,
                macro2=256,
                device=torch.device("cpu"),
            )
            self.assertIsNotNone(got)
            for key in src:
                self.assertEqual(got[key].dtype, src[key].dtype)
                self.assertTrue(torch.equal(got[key], src[key]))
            self.assertEqual((cache.hits, cache.stores), (1, 1))

    def test_publish_leaves_no_tmp_files(self):
        with tempfile.TemporaryDirectory() as d:
            cache = _U2LayerCache(cache_dir=d, rank=0)
            cache.store_layer(
                _fake_layer(7),
                group_size=128,
                macro13=256,
                macro2=256,
                pools=_pools(),
            )
            self.assertEqual([f for f in os.listdir(d) if ".tmp" in f], [])

    def test_meta_mismatch_is_a_miss(self):
        layer = _fake_layer()
        pools = _pools()
        with tempfile.TemporaryDirectory() as d:
            cache = _U2LayerCache(cache_dir=d, rank=0)
            cache.store_layer(
                layer, group_size=128, macro13=256, macro2=256, pools=pools
            )
            self.assertIsNone(
                cache.load_layer(
                    layer,
                    group_size=64,
                    macro13=256,
                    macro2=256,
                    device=torch.device("cpu"),
                )
            )
            # The stale file is kept so the following store overwrites it.
            self.assertFalse(cache.dead)
            self.assertEqual(len(os.listdir(d)), 1)

    def test_layout_mismatch_rejected(self):
        pools = _pools()
        shapes = _shapes(pools)
        self.assertTrue(_u2_pool_layout_ok(pools, shapes))
        wrong_shape = dict(pools, w13=pools["w13"][:, :-1, :])
        self.assertFalse(_u2_pool_layout_ok(wrong_shape, shapes))
        wrong_dtype = dict(pools, s2=pools["s2"].to(torch.float32))
        self.assertFalse(_u2_pool_layout_ok(wrong_dtype, shapes))

    def test_corrupt_payload_disables_the_cache(self):
        layer = _fake_layer()
        with tempfile.TemporaryDirectory() as d:
            cache = _U2LayerCache(cache_dir=d, rank=0)
            cache.store_layer(
                layer, group_size=128, macro13=256, macro2=256, pools=_pools()
            )
            path = cache._layer_path(layer)
            with open(path, "wb") as f:
                f.write(b"garbage")
            self.assertIsNone(
                cache.load_layer(
                    layer,
                    group_size=128,
                    macro13=256,
                    macro2=256,
                    device=torch.device("cpu"),
                )
            )
            self.assertTrue(cache.dead)

    def test_disabled_handle_never_touches_disk(self):
        with tempfile.TemporaryDirectory() as d:
            cache = _U2LayerCache(cache_dir=d, rank=0)
            cache._disable("test", RuntimeError("boom"))
            cache.store_layer(
                _fake_layer(),
                group_size=128,
                macro13=256,
                macro2=256,
                pools=_pools(),
            )
            self.assertEqual(os.listdir(d), [])


class TestU2CachePrune(CustomTestCase):
    def test_keeps_current_and_newest_other_only(self):
        with tempfile.TemporaryDirectory() as d:
            keep = os.path.join(d, "current")
            gen_a = os.path.join(d, "aaaa")
            gen_b = os.path.join(d, "bbbb")
            for path in (keep, gen_a, gen_b):
                os.makedirs(path)
            os.utime(gen_a, (1_000, 1_000))
            os.utime(gen_b, (2_000, 2_000))
            _u2_cache_prune(d, keep=keep)
            # The newest other generation survives; the older one is reclaimed.
            self.assertEqual(sorted(os.listdir(d)), ["bbbb", "current"])


# Minimal stand-ins for the classes u2_checkpoint_reader type-narrows on, so
# the reader's decision table is testable without importing the FusedMoE /
# modelopt modules (which drag in the kernel stack). The reader only needs
# isinstance to succeed and `quant_method.sm70_u2_pool` to be readable.
class _FakeFusedMoE(torch.nn.Module):
    pass


class _FakeMoEMethod:
    def __init__(self):
        self.sm70_u2_pool = True


class _RecordingHandle:
    """Safetensors handle double: records pass-through reads."""

    def __init__(self):
        self.names = []

    def get_tensor(self, name):
        self.names.append(name)
        return "TENSOR"


# One FusedMoE staging layout small enough for every group-size fallback to
# resolve: hidden=64 (H/2=32 packed), inter=32 (I/2=16 packed), E=4.
_E, _N13, _W2_COLS = 4, 64, 16


def _fake_pool_layer(layer_id: int) -> _FakeFusedMoE:
    mod = _FakeFusedMoE()
    mod.layer_id = layer_id
    mod.quant_method = _FakeMoEMethod()
    mod.moe_runner_config = SimpleNamespace(is_gated=True)
    mod.w13_weight = torch.empty(_E, _N13, 32, dtype=torch.uint8)
    mod.w2_weight = torch.empty(_E, 64, _W2_COLS, dtype=torch.uint8)
    return mod


def _fake_model(num_fused_shared_experts: int, layer_ids=(3,)) -> torch.nn.Module:
    root = torch.nn.Module()
    root.num_fused_shared_experts = num_fused_shared_experts
    for layer_id in layer_ids:
        root.add_module(f"moe{layer_id}", _fake_pool_layer(layer_id))
    return root


def _fake_modules():
    fused = types.ModuleType("sglang.srt.layers.moe.fused_moe_triton.layer")
    fused.FusedMoE = _FakeFusedMoE
    quant = types.ModuleType("sglang.srt.layers.quantization.modelopt_quant")
    quant.ModelOptNvFp4FusedMoEMethod = _FakeMoEMethod
    return {
        fused.__name__: fused,
        quant.__name__: quant,
    }


class _ReaderContext:
    """Env + hardware + cache + module-identity stubs around the reader."""

    def __init__(self, cache):
        self._patches = [
            envs.SGLANG_SM70_U2_CACHE.override(True),
            patch.object(sm70_u2_pool, "sm70_u2_expert_pool_enabled", lambda: True),
            patch.object(sm70_u2_pool, "_u2_cache_open", lambda: cache),
            patch.dict(sys.modules, _fake_modules()),
        ]

    def __enter__(self):
        for p in self._patches:
            p.__enter__()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.__exit__(*exc)
        return False


class TestU2CheckpointReader(CustomTestCase):
    def setUp(self):
        self._armed = patch.object(
            sm70_u2_pool, "_U2_SKIP_ARMED_LAYER_IDS", frozenset()
        )
        self._armed.start()
        self.addCleanup(self._armed.stop)

    def _armed_cache(self, d, layer_ids=(3,)):
        """A cache dir holding a full-size payload for each requested layer."""
        cache = _U2LayerCache(cache_dir=d, rank=0)
        layout = sm70_u2_pool._layer_pool_layout(_fake_pool_layer(3))
        shapes = layout["pool_shapes"]
        pools = {
            "w13": torch.zeros(shapes["w13"], dtype=torch.int32),
            "w2": torch.zeros(shapes["w2"], dtype=torch.int32),
            "s13": torch.zeros(shapes["s13"], dtype=torch.float16),
            "s2": torch.zeros(shapes["s2"], dtype=torch.float16),
        }
        for layer_id in layer_ids:
            layer = _fake_pool_layer(layer_id)
            cache.store_layer(
                layer,
                group_size=layout["group_size"],
                macro13=layout["macro13"],
                macro2=layout["macro2"],
                pools=pools,
            )
        return cache

    def test_disabled_when_cache_env_off(self):
        model = _fake_model(num_fused_shared_experts=4)
        with envs.SGLANG_SM70_U2_CACHE.override(False):
            self.assertIsNone(u2_checkpoint_reader(model))
        self.assertEqual(sm70_u2_pool._U2_SKIP_ARMED_LAYER_IDS, frozenset())

    def test_disabled_when_cache_dead(self):
        model = _fake_model(num_fused_shared_experts=4)
        with tempfile.TemporaryDirectory() as d:
            cache = _U2LayerCache(cache_dir=d, rank=0)
            cache._disable("test", RuntimeError("boom"))
            with _ReaderContext(cache):
                self.assertIsNone(u2_checkpoint_reader(model))
        self.assertEqual(sm70_u2_pool._U2_SKIP_ARMED_LAYER_IDS, frozenset())

    def test_no_cached_layer_promises_nothing(self):
        model = _fake_model(num_fused_shared_experts=4)
        with tempfile.TemporaryDirectory() as d:
            with _ReaderContext(_U2LayerCache(cache_dir=d, rank=0)):
                self.assertIsNone(u2_checkpoint_reader(model))
        self.assertEqual(sm70_u2_pool._U2_SKIP_ARMED_LAYER_IDS, frozenset())

    def test_expert_bytes_skipped_only_for_verified_layers(self):
        model = _fake_model(num_fused_shared_experts=4, layer_ids=(3, 7))
        with tempfile.TemporaryDirectory() as d:
            with _ReaderContext(self._armed_cache(d, layer_ids=(3,))):
                read = u2_checkpoint_reader(model)
        self.assertIsNotNone(read)
        self.assertEqual(sm70_u2_pool._U2_SKIP_ARMED_LAYER_IDS, frozenset({3}))

        handle = _RecordingHandle()
        skipped = [
            "model.language_model.layers.3.mlp.experts.5.gate_proj.weight",
            "model.language_model.layers.3.mlp.experts.5.up_proj.weight_scale",
            "model.language_model.layers.3.mlp.shared_experts.down_proj.weight",
        ]
        kept = [
            "model.language_model.layers.3.mlp.experts.5.gate_proj.weight_scale_2",
            "model.language_model.layers.3.mlp.experts.5.gate_proj.input_scale",
            "model.language_model.layers.7.mlp.experts.0.gate_proj.weight",
            "model.language_model.layers.3.mlp.self_attn.qkv_proj.weight",
            "model.embed_tokens.weight",
        ]
        for name in skipped:
            self.assertIsNone(read(name, handle), name)
        for name in kept:
            self.assertEqual(read(name, handle), "TENSOR", name)
        self.assertEqual(sorted(handle.names), sorted(kept))

    def test_shared_experts_kept_when_not_fused(self):
        model = _fake_model(num_fused_shared_experts=0)
        with tempfile.TemporaryDirectory() as d:
            with _ReaderContext(self._armed_cache(d)):
                read = u2_checkpoint_reader(model)
        handle = _RecordingHandle()
        name = "model.language_model.layers.3.mlp.shared_experts.down_proj.weight"
        self.assertEqual(read(name, handle), "TENSOR")
        self.assertEqual(handle.names, [name])


class TestU2SkipArmedHardFail(CustomTestCase):
    """A broken cache promise must fail the load, not emit zero experts."""

    def test_armed_layer_with_dead_cache_raises(self):
        layer = _fake_pool_layer(3)
        layer.w13_weight_scale_2 = torch.zeros(_E, 2)
        layer.w2_weight_scale_2 = torch.zeros(_E)
        # Cache disabled -> the restore attempt finds nothing; the armed
        # layer id must raise before the (empty-staging) requant runs.
        with envs.SGLANG_SM70_U2_CACHE.override(False):
            with patch.object(sm70_u2_pool, "_U2_SKIP_ARMED_LAYER_IDS", frozenset({3})):
                with self.assertRaisesRegex(RuntimeError, "skipped checkpoint expert"):
                    convert_moe_layer_to_u2(layer)

    def test_unarmed_layer_is_not_blocked(self):
        layer = _fake_pool_layer(3)
        layer.w13_weight_scale_2 = torch.zeros(_E, 2)
        layer.w2_weight_scale_2 = torch.zeros(_E)
        with envs.SGLANG_SM70_U2_CACHE.override(False):
            with patch.object(sm70_u2_pool, "_U2_SKIP_ARMED_LAYER_IDS", frozenset({9})):
                # Layer 3 is not armed: the guard must not fire. The CPU
                # staging cannot run the CUDA requant, so expect any error
                # EXCEPT the cache-promise RuntimeError.
                try:
                    convert_moe_layer_to_u2(layer)
                except RuntimeError as e:
                    self.assertNotIn("skipped checkpoint expert", str(e))
                except Exception:  # noqa: BLE001 -- CPU fallback path, any error
                    pass


if __name__ == "__main__":
    unittest.main(verbosity=3)
