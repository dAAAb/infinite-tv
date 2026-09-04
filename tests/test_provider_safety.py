import os
import unittest
from unittest.mock import patch

from streaming_pipeline.video_generation.video_generator import (
    _require_fal_video_opt_in,
)


class FalVideoSafetyTests(unittest.TestCase):
    def test_fal_video_is_disabled_by_default_even_when_key_exists(self):
        with patch.dict(
            os.environ,
            {"FAL_KEY": "test-only-placeholder"},
            clear=False,
        ):
            os.environ.pop("ENABLE_FAL_VIDEO", None)
            with self.assertRaisesRegex(RuntimeError, "ENABLE_FAL_VIDEO=true"):
                _require_fal_video_opt_in("h3-max")

    def test_fal_video_requires_key_after_explicit_opt_in(self):
        with patch.dict(
            os.environ,
            {"ENABLE_FAL_VIDEO": "true"},
            clear=False,
        ):
            os.environ.pop("FAL_KEY", None)
            with self.assertRaisesRegex(RuntimeError, "requires FAL_KEY"):
                _require_fal_video_opt_in("ltx-2.3")

    def test_fal_video_allows_explicit_opt_in_with_key(self):
        with patch.dict(
            os.environ,
            {
                "ENABLE_FAL_VIDEO": "true",
                "FAL_KEY": "test-only-placeholder",
            },
            clear=False,
        ):
            _require_fal_video_opt_in("h3-max")


if __name__ == "__main__":
    unittest.main()
