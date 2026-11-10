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


if __name__ == "__main__":
    import unittest

    unittest.main()
