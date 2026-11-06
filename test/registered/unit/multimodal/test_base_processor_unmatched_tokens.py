"""Unit tests for unmatched multimodal placeholder tokens in the legacy path.

When the prompt contains more image placeholder tokens than the request
attaches image items, ``legacy_load_mm_data`` must keep raising by default
(upstream behavior). With ``SGLANG_ENABLE_MM_UNMATCHED_TOKEN_DROP=1`` the
unmatched tokens are dropped and the request proceeds instead of failing
with a 500 (agent clients echo ``<image>`` in text without the media).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import asyncio
import concurrent.futures
import unittest

from PIL import Image

from sglang.srt.environ import envs
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.test.test_utils import CustomTestCase


class _StubProcessor(BaseMultimodalProcessor):
    # CPU PIL decode so the test needs no GPU; __init__ is bypassed via
    # __new__ and only the attributes used by legacy_load_mm_data are set.
    gpu_image_decode = False

    async def process_mm_data_async(self, *args, **kwargs):  # abstract slot
        raise NotImplementedError


def _make_processor() -> _StubProcessor:
    processor = object.__new__(_StubProcessor)
    processor.io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    return processor


def _red_image() -> Image.Image:
    return Image.new("RGB", (4, 4), color=(255, 0, 0))


def _tokens() -> MultimodalSpecialTokens:
    return MultimodalSpecialTokens(image_token="<image>").build(processor=None)


def _run(processor, prompt, image_data):
    return asyncio.run(
        processor.legacy_load_mm_data(
            prompt=prompt,
            multimodal_tokens=_tokens(),
            image_data=image_data,
        )
    )


class TestUnmatchedTokenDefaultRaises(CustomTestCase):
    """Default: more tokens than data must keep failing (upstream behavior)."""

    def test_two_tokens_one_image_raises(self):
        processor = _make_processor()
        with self.assertRaisesRegex(RuntimeError, "loading multimodal data"):
            _run(processor, "a <image> b <image> c", [_red_image()])

    def test_token_without_any_image_raises(self):
        # Upstream: no data iterator at all -> ValueError, also a 500.
        processor = _make_processor()
        with self.assertRaisesRegex(ValueError, "No data iterator found"):
            _run(processor, "a <image> b", None)


class TestUnmatchedTokenDropEnabled(CustomTestCase):
    """Flag on: unmatched tokens are dropped, matched data still loads."""

    def test_two_tokens_one_image_drops_extra_token(self):
        processor = _make_processor()
        with envs.SGLANG_ENABLE_MM_UNMATCHED_TOKEN_DROP.override(True):
            output = _run(processor, "a <image> b <image> c", [_red_image()])
        self.assertEqual(len(output.images), 1)
        # The unmatched second placeholder must be removed from the text.
        self.assertEqual(output.input_text, "a <image> b  c")

    def test_token_without_any_image_drops_token(self):
        processor = _make_processor()
        with envs.SGLANG_ENABLE_MM_UNMATCHED_TOKEN_DROP.override(True):
            output = _run(processor, "a <image> b", None)
        self.assertEqual(len(output.images), 0)
        self.assertEqual(output.input_text, "a  b")

    def test_leading_and_trailing_unmatched_tokens(self):
        # Boundary: unmatched tokens at both ends of the prompt. The matched
        # placeholder is re-emitted as the canonical token, the trailing
        # unmatched one is dropped.
        processor = _make_processor()
        with envs.SGLANG_ENABLE_MM_UNMATCHED_TOKEN_DROP.override(True):
            output = _run(processor, "<image> a <image>", [_red_image()])
        self.assertEqual(len(output.images), 1)
        self.assertEqual(output.input_text, "<image> a ")

    def test_matched_counts_unchanged(self):
        # Boundary: token count == data count must behave exactly as before.
        processor = _make_processor()
        with envs.SGLANG_ENABLE_MM_UNMATCHED_TOKEN_DROP.override(True):
            output = _run(processor, "a <image> b", [_red_image()])
        self.assertEqual(len(output.images), 1)
        self.assertEqual(output.input_text, "a <image> b")

    def test_more_data_than_tokens_still_ok(self):
        # Boundary: extra data beyond tokens keeps the existing
        # warn-and-ignore behavior (no raise) with the flag on.
        processor = _make_processor()
        with envs.SGLANG_ENABLE_MM_UNMATCHED_TOKEN_DROP.override(True):
            output = _run(processor, "a <image> b", [_red_image(), _red_image()])
        self.assertEqual(len(output.images), 1)
        self.assertEqual(output.input_text, "a <image> b")


if __name__ == "__main__":
    unittest.main()
