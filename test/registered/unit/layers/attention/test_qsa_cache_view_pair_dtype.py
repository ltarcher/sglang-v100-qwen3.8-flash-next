"""Unit test (CPU): ``_qsa_cache_view`` wraps pair-split peer pools with the
same dtype semantics as the local-tensor branch.

The CUDA op reads the paged KV cache through a raw pointer: an E5M2 pool must
be reinterpreted as uint8 (one byte per element), an FP16 pool stays half.
Peer pools arrive as ``ImportedDeviceBuffer`` pointer wrappers, so the byte
view has to be applied to the dtype handed to ``sm70_as_tensor`` — review m22
(docs/v100/CODE_REVIEW_TODO_20261110.md): the peer branch used to pass the
raw ``cache.dtype`` through, so an E5M2 peer pool reached the op with a
float8 dtype instead of the byte view the local branch produces.
"""

from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import sglang.srt.layers.attention.tilelang_fa_v100._decode_cuda as decode_cuda
from sglang.test.test_utils import CustomTestCase


class _FakeImportedBuffer:
    """Duck-typed stand-in for the pybind ImportedDeviceBuffer wrapper."""

    def __init__(self, dtype):
        self._data_ptr = 0x1234
        self.numel = 8
        self.dtype = dtype

    def data_ptr(self):
        return self._data_ptr


class _RecordingExt:
    def __init__(self):
        self.calls = []

    def sm70_as_tensor(self, data_ptr, shape, dtype):
        self.calls.append((data_ptr, list(shape), dtype))
        return object()  # opaque stand-in for the wrapped tensor


class TestQsaCacheViewPeerDtype(CustomTestCase):
    def _view(self, dtype, like):
        ext = _RecordingExt()
        buf = _FakeImportedBuffer(dtype)
        with (
            mock.patch.object(decode_cuda, "_load_sm70_cuda_decode_ops", lambda: ext),
            mock.patch.object(decode_cuda, "ImportedDeviceBuffer", _FakeImportedBuffer),
        ):
            view = decode_cuda._qsa_cache_view(buf, like=like)
        return ext, view

    def test_peer_e5m2_wrapped_as_uint8_byte_view(self):
        ext, _ = self._view(torch.float8_e5m2, like=torch.zeros(4, 16))
        ((ptr, shape, dtype),) = ext.calls
        self.assertEqual(dtype, torch.uint8)
        self.assertEqual(shape, [4, 16])
        self.assertEqual(ptr, 0x1234)

    def test_peer_fp16_keeps_half(self):
        ext, _ = self._view(torch.float16, like=torch.zeros(4, 16))
        ((_, shape, dtype),) = ext.calls
        self.assertEqual(dtype, torch.float16)
        self.assertEqual(shape, [4, 16])

    def test_local_e5m2_tensor_viewed_as_uint8(self):
        cache = torch.zeros(4, dtype=torch.float8_e5m2)
        view = decode_cuda._qsa_cache_view(cache)
        self.assertEqual(view.dtype, torch.uint8)
        self.assertEqual(view.numel(), 4)

    def test_local_fp16_tensor_passthrough(self):
        cache = torch.zeros(4, dtype=torch.float16)
        view = decode_cuda._qsa_cache_view(cache)
        self.assertEqual(view.dtype, torch.float16)

    def test_local_e4m3_tensor_viewed_as_uint8(self):
        # E4M3 is also a one-byte-per-element pool and the op only accepts
        # byte or half caches; it must reach the op through the same uint8
        # byte view as E5M2 (with the dequant LUT supplied separately).
        cache = torch.zeros(4, dtype=torch.float8_e4m3fn)
        view = decode_cuda._qsa_cache_view(cache)
        self.assertEqual(view.dtype, torch.uint8)
        self.assertEqual(view.numel(), 4)

    def test_peer_e4m3_wrapped_as_uint8_byte_view(self):
        ext, _ = self._view(torch.float8_e4m3fn, like=torch.zeros(4, 16))
        ((ptr, shape, dtype),) = ext.calls
        self.assertEqual(dtype, torch.uint8)
        self.assertEqual(shape, [4, 16])
        self.assertEqual(ptr, 0x1234)


class TestQsaFp8Lut(CustomTestCase):
    def test_e4m3_builds_256_entry_fp16_table(self):
        lut = decode_cuda._qsa_fp8_lut(torch.float8_e4m3fn)
        self.assertEqual(lut.dtype, torch.float16)
        self.assertEqual(lut.numel(), 256)
        # 0x38: sign 0, exp 0111 (bias 7), man 000 -> 1.0.
        self.assertEqual(lut[0x38].item(), 1.0)
        # 0xBC: sign 1, exp 0111, man 100 -> -1.5.
        self.assertEqual(lut[0xBC].item(), -1.5)
        # 0x7E: exp 1111, man 110 -> 2**8 * 1.75 = 448 (e4m3 max normal).
        self.assertEqual(lut[0x7E].item(), 448.0)
        self.assertEqual(lut[0x00].item(), 0.0)
        # 0x7F is the E4M3 NaN pattern.
        self.assertTrue(lut[0x7F].isnan())

    def test_e5m2_and_fp16_need_no_lut(self):
        # E5M2 shares the FP16 exponent layout, so the kernel keeps its
        # built-in bit-shift conversion when the LUT pointer is null; FP16
        # pools are never byte-dequantized at all.
        self.assertIsNone(decode_cuda._qsa_fp8_lut(torch.float8_e5m2))
        self.assertIsNone(decode_cuda._qsa_fp8_lut(torch.float16))


class _RecordingPrefillExt:
    def __init__(self):
        self.lut = "not-called"
        self.kv_dtype = None

    def sm70_qsa_prefill(
        self, q, k, v, req_to_token, req_indices, indices, seq_lens, scale, lut, out
    ):
        self.lut = lut
        self.kv_dtype = k.dtype
        out.fill_(0)
        return out


class TestQsaPrefillLutWiring(CustomTestCase):
    """The prefill wrapper must hand the op a dequant LUT for E4M3 pools.

    Regression (ff52c061e8 audit, P1): the SM70 gates admitted
    ``float8_e4m3fn`` pools while the wrapper passed ``lut=None``, so every
    byte went through the kernel's E5M2 bit-shift path and prefill silently
    dequantized wrong (decode, by contrast, TORCH_CHECK-failed on the raw
    float8 dtype -- loudly).
    """

    def _run(self, kv_dtype):
        ext = _RecordingPrefillExt()
        q = torch.zeros(2, 6, 256, dtype=torch.float16)
        k = torch.zeros(16, 1, 256, dtype=kv_dtype)
        v = torch.zeros(16, 1, 256, dtype=kv_dtype)
        with mock.patch.object(decode_cuda, "_load_sm70_cuda_decode_ops", lambda: ext):
            out = decode_cuda.sm70_cuda_qsa_prefill(
                q,
                k,
                v,
                torch.zeros(1, 16, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                torch.zeros(2, 4, dtype=torch.int32),
                torch.tensor([16], dtype=torch.int32),
                0.0625,
            )
        return ext, out

    def test_e4m3_pool_passes_dequant_lut(self):
        ext, _ = self._run(torch.float8_e4m3fn)
        self.assertEqual(ext.kv_dtype, torch.uint8)
        self.assertEqual(ext.lut.dtype, torch.float16)
        self.assertEqual(ext.lut.numel(), 256)
        self.assertEqual(ext.lut[0x38].item(), 1.0)

    def test_e5m2_pool_passes_no_lut(self):
        ext, _ = self._run(torch.float8_e5m2)
        self.assertEqual(ext.kv_dtype, torch.uint8)
        self.assertIsNone(ext.lut)


if __name__ == "__main__":
    import unittest

    unittest.main()
