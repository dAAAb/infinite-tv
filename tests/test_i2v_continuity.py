import base64
import io
import os
import tempfile
import unittest
from queue import Queue
from types import SimpleNamespace

import numpy as np
from PIL import Image

from streaming_pipeline.core.streaming_engine import RealtimeVideoStreamer
from streaming_pipeline.models import LTXVideoResponseWithFrames, StreamingState
from streaming_pipeline.models import TwitchComment
from streaming_pipeline.prompt_generation.prompt_generator import PromptResult
from streaming_pipeline.video_generation.comfy_ltx25_backend import (
    _build_prompt_i2v_video_only,
    _prepare_handoff_frame,
    _trim_generated_frames,
)
from streaming_pipeline.output.rtmp_streamer import FFmpegRTMPStreamer
from streaming_pipeline.postprocessing.text_overlay import TextOverlay


def _streamer_for_quality(reference: Image.Image) -> RealtimeVideoStreamer:
    streamer = RealtimeVideoStreamer.__new__(RealtimeVideoStreamer)
    streamer._quality_reference = streamer._frame_quality(reference)
    return streamer


class QualityGateTests(unittest.TestCase):
    def test_rejects_binary_posterization_collapse(self):
        x = np.linspace(30, 225, 256, dtype=np.uint8)
        healthy = Image.fromarray(np.dstack([
            np.tile(x, (144, 1)),
            np.tile(np.roll(x, 31), (144, 1)),
            np.tile(np.roll(x, 67), (144, 1)),
        ]), "RGB")
        collapsed = Image.fromarray(
            np.where(np.indices((144, 256)).sum(axis=0)[..., None] % 12 < 6, 255, 0)
            .repeat(3, axis=2)
            .astype(np.uint8),
            "RGB",
        )
        streamer = _streamer_for_quality(healthy)

        self.assertFalse(streamer._frame_is_corrupt(healthy))
        self.assertTrue(streamer._frame_is_corrupt(collapsed))

    def test_clip_rejects_a_bad_tail(self):
        healthy = Image.new("RGB", (64, 36), (90, 120, 150))
        # Add enough structure to avoid the flat-frame guard.
        array = np.asarray(healthy).copy()
        array[:, ::2, 0] = 180
        healthy = Image.fromarray(array)
        collapsed = Image.new("RGB", (64, 36), (255, 255, 255))
        streamer = _streamer_for_quality(healthy)

        self.assertTrue(streamer._clip_is_corrupt([healthy] * 8 + [collapsed]))

    def test_rejects_a_growing_black_picture_frame(self):
        y, x = np.indices((144, 256))
        base = np.dstack([
            60 + (x % 150),
            50 + (y % 140),
            70 + ((x + y) % 130),
        ]).astype(np.uint8)
        healthy = Image.fromarray(base, "RGB")
        bordered = base.copy()
        bordered[:14, :, :] = 0
        bordered[-14:, :, :] = 0
        bordered[:, :14, :] = 0
        bordered[:, -14:, :] = 0
        bordered = Image.fromarray(bordered, "RGB")
        streamer = _streamer_for_quality(healthy)

        self.assertTrue(streamer._border_guard_needed(bordered))
        self.assertTrue(streamer._frame_is_corrupt(bordered))

    def test_outer_rectangular_line_triggers_border_guard(self):
        y, x = np.indices((144, 256))
        base = np.dstack([
            60 + x * 0.45,
            70 + y * 0.75,
            80 + (x + y) * 0.30,
        ]).clip(0, 255).astype(np.uint8)
        healthy = Image.fromarray(base, "RGB")
        framed = base.copy()
        framed[13:18, 18:-18] = 245
        framed[-18:-13, 18:-18] = 245
        framed[13:-13, 18:23] = 245
        framed[13:-13, -23:-18] = 245
        framed = Image.fromarray(framed, "RGB")
        streamer = _streamer_for_quality(healthy)

        self.assertTrue(streamer._border_guard_needed(framed))

    def test_soft_saturated_halo_triggers_guard_and_stronger_repair(self):
        y, x = np.indices((144, 256))
        base = np.dstack([
            70 + x * 0.25,
            85 + y * 0.45,
            95 + (x + y) * 0.18,
        ]).clip(0, 255).astype(np.uint8)
        healthy = Image.fromarray(base, "RGB")
        halo = base.astype(np.float32)
        distance = np.minimum.reduce([x, 255 - x, y, 143 - y]).astype(np.float32)
        alpha = np.clip((18 - distance) / 18, 0, 1)[..., None]
        saturated_edge = np.zeros_like(halo)
        saturated_edge[..., 0] = 210
        saturated_edge[..., 1] = 25
        saturated_edge[..., 2] = 190
        halo = (halo * (1 - alpha) + saturated_edge * alpha).astype(np.uint8)
        halo = Image.fromarray(halo, "RGB")
        streamer = _streamer_for_quality(healthy)

        self.assertTrue(streamer._border_guard_needed(halo))
        self.assertTrue(streamer._border_repair_needed(halo))
        self.assertGreaterEqual(streamer._border_repair_ratio(halo), 0.13)

    def test_progressive_border_crop_preserves_exact_seam_and_repairs_tail(self):
        y, x = np.indices((72, 128))
        scene = np.dstack([
            50 + (x % 150),
            70 + (y % 130),
            80 + ((x + y) % 120),
        ]).astype(np.uint8)
        scene[:8, :, :] = 0
        scene[-8:, :, :] = 0
        scene[:, :8, :] = 0
        scene[:, -8:, :] = 0
        framed = Image.fromarray(scene, "RGB")
        streamer = _streamer_for_quality(Image.fromarray(scene[8:-8, 8:-8], "RGB"))

        repaired = streamer._progressive_full_bleed_crop([framed.copy() for _ in range(9)], ratio=0.14)

        self.assertEqual(framed.tobytes(), repaired[0].tobytes())
        self.assertEqual(framed.size, repaired[-1].size)
        self.assertGreater(np.asarray(repaired[-1])[0].mean(), 10)
        self.assertGreater(np.asarray(repaired[-1])[:, 0].mean(), 10)

    def test_local_recovery_clip_keeps_join_and_clears_border_early(self):
        y, x = np.indices((72, 128))
        scene = np.dstack([
            55 + (x % 145),
            65 + (y % 135),
            75 + ((x + y) % 125),
        ]).astype(np.uint8)
        healthy = Image.fromarray(scene, "RGB")
        bordered_array = scene.copy()
        bordered_array[:8, :, :] = 0
        bordered_array[-8:, :, :] = 0
        bordered_array[:, :8, :] = 0
        bordered_array[:, -8:, :] = 0
        bordered = Image.fromarray(bordered_array, "RGB")
        streamer = _streamer_for_quality(healthy)

        recovered = streamer._make_local_recovery_clip(bordered, 121)

        self.assertEqual(121, len(recovered))
        self.assertEqual(bordered.tobytes(), recovered[0].tobytes())
        self.assertGreater(np.asarray(recovered[30])[0].mean(), 10)
        self.assertGreater(np.asarray(recovered[-1])[:, 0].mean(), 10)

    def test_adaptive_repair_can_rescue_border_without_changing_first_frame(self):
        y, x = np.indices((72, 128))
        scene = np.dstack([
            50 + (x % 150),
            70 + (y % 130),
            80 + ((x + y) % 120),
        ]).astype(np.uint8)
        healthy = Image.fromarray(scene, "RGB")
        bordered_array = scene.copy()
        bordered_array[:8, :, :] = 0
        bordered_array[-8:, :, :] = 0
        bordered_array[:, :8, :] = 0
        bordered_array[:, -8:, :] = 0
        bordered = Image.fromarray(bordered_array, "RGB")
        streamer = _streamer_for_quality(healthy)

        repaired, accepted = streamer._try_adaptive_clip_repair(
            [bordered.copy() for _ in range(121)]
        )

        self.assertTrue(accepted)
        self.assertEqual(bordered.tobytes(), repaired[0].tobytes())
        self.assertFalse(streamer._clip_is_corrupt(repaired))


class HandoffFrameTests(unittest.TestCase):
    def test_same_size_png_handoff_is_pixel_exact(self):
        array = np.arange(64 * 36 * 3, dtype=np.uint8).reshape((36, 64, 3))
        image = Image.fromarray(array, "RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

        prepared = _prepare_handoff_frame(encoded, 64, 36)

        self.assertEqual(image.tobytes(), prepared.tobytes())

    def test_temporal_padding_is_not_streamed_or_chained(self):
        frames = [Image.new("RGB", (8, 8), (value, 0, 0)) for value in range(129)]

        trimmed = _trim_generated_frames(frames, 121)

        self.assertEqual(121, len(trimmed))
        self.assertEqual((120, 0, 0), trimmed[-1].getpixel((0, 0)))

    def test_video_only_i2v_graph_has_no_audio_latent_or_av_concat(self):
        graph = _build_prompt_i2v_video_only(
            "first.png", "continue", "", 512, 288, 121, 9.0, 123, 1.0
        )
        classes = {node["class_type"] for node in graph.values()}

        self.assertIn("LTXVAddGuide", classes)
        self.assertNotIn("LTXVEmptyLatentAudio", classes)
        self.assertNotIn("LTXVConcatAVLatent", classes)
        self.assertNotIn("LTXVSeparateAVLatent", classes)

    def test_committed_handoff_snapshot_is_exact_and_atomic(self):
        array = np.arange(64 * 36 * 3, dtype=np.uint8).reshape((36, 64, 3))
        frame = Image.fromarray(array, "RGB")
        streamer = RealtimeVideoStreamer.__new__(RealtimeVideoStreamer)
        streamer._handoff_snapshot_path = ""
        with tempfile.TemporaryDirectory() as temp_dir:
            target = os.path.join(temp_dir, "tail.png")
            old_value = os.environ.get("LTX25_HANDOFF_SNAPSHOT")
            os.environ["LTX25_HANDOFF_SNAPSHOT"] = target
            try:
                streamer._persist_committed_handoff(frame)
            finally:
                if old_value is None:
                    os.environ.pop("LTX25_HANDOFF_SNAPSHOT", None)
                else:
                    os.environ["LTX25_HANDOFF_SNAPSHOT"] = old_value

            restored = Image.open(target).convert("RGB")
            self.assertEqual(frame.tobytes(), restored.tobytes())
            self.assertFalse(os.path.exists(target + ".tmp.png"))


class RTMPBatchTimingTests(unittest.TestCase):
    def test_temporal_resampling_spreads_duplicates_across_clip(self):
        streamer = FFmpegRTMPStreamer.__new__(FFmpegRTMPStreamer)
        streamer.is_streaming = True
        streamer.fps = 10
        streamer.width = 8
        streamer.height = 8
        streamer.frames_dropped = 0
        streamer.frame_queue = Queue(maxsize=100)
        frames = [Image.new("RGB", (8, 8), (i, 0, 0)) for i in range(10)]

        accepted = streamer.add_frame_batch(frames, playback_seconds=2.0)
        queued = [streamer.frame_queue.get_nowait()[0, 0, 0] for _ in range(20)]

        self.assertEqual(10, accepted)
        self.assertEqual(20, len(queued))
        self.assertEqual(0, queued[0])
        self.assertEqual(9, queued[-1])
        self.assertGreater(len(set(queued[:10])), 2)


class TextOverlayTests(unittest.TestCase):
    def test_comment_is_visible_longer_and_never_contaminates_raw_handoff(self):
        overlay = TextOverlay(width=128, height=72)
        raw = [Image.new("RGB", (128, 72), (40, 80, 120)) for _ in range(121)]
        raw_tail = raw[-1].tobytes()

        overlay.set_comment("鏡頭拉遠 貓咪變黑貓！", "daaab")
        streamed = overlay.apply_overlay_batch(raw)
        status = overlay.get_status()

        self.assertTrue(status["last_overlay_verified"])
        self.assertEqual("comment", status["last_kind_rendered"])
        self.assertGreater(status["last_visible_frames"], 90)
        self.assertNotEqual(raw[0].tobytes(), streamed[0].tobytes())
        self.assertEqual(raw_tail, raw[-1].tobytes())
        self.assertEqual(raw_tail, streamed[-1].tobytes())

    def test_prompt_keeps_shorter_caption_window(self):
        overlay = TextOverlay(width=128, height=72)
        frames = [Image.new("RGB", (128, 72), (40, 80, 120)) for _ in range(121)]

        overlay.set_prompt("continue the scene")
        overlay.apply_overlay_batch(frames)

        self.assertLess(overlay.get_status()["last_visible_frames"], 60)


class RecoveryOrchestrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_viewer_clip_is_suppressed_then_retried_with_weaker_guide(self):
        rng = np.random.default_rng(5090)
        healthy_array = rng.integers(45, 205, size=(36, 64, 3), dtype=np.uint8)
        healthy = Image.fromarray(healthy_array, "RGB")
        frames = [healthy.copy()]
        frames.extend(
            Image.fromarray(np.roll(healthy_array, index, axis=1), "RGB")
            for index in range(1, 9)
        )
        comment = TwitchComment(
            username="daaab",
            message="鏡頭拉遠 這個生物戴上眼鏡",
            timestamp=1.0,
        )

        class Generator:
            def __init__(self):
                self.requests = []

            def generate_video_from_image(self, request):
                self.requests.append(request)
                return LTXVideoResponseWithFrames(
                    frames=[frame.copy() for frame in frames]
                )

        class RTMP:
            def __init__(self):
                self.batches = []

            def add_frame_batch(self, batch, playback_seconds=None):
                self.batches.append(batch)
                return len(batch)

        class Overlay:
            def set_prompt(self, _prompt):
                pass

            def set_comment(self, _message, _username):
                pass

            def apply_overlay_batch(self, batch):
                return batch

            def get_status(self):
                return {"last_overlay_verified": True, "last_visible_frames": 8}

        class PromptGenerator:
            def __init__(self):
                self.calls = 0

            def verify_comment_adherence(self, _comment, _before, _frames):
                self.calls += 1
                if self.calls < 3:
                    return {
                        "satisfied": False,
                        "progressing": True,
                        "missing": ["生物尚未戴上眼鏡"],
                        "summary": "眼鏡只出現在附近",
                    }
                return {
                    "satisfied": True,
                    "progressing": False,
                    "missing": [],
                    "summary": "生物已戴上眼鏡",
                }

        generator = Generator()
        rtmp = RTMP()
        streamer = RealtimeVideoStreamer(
            twitch_listener=SimpleNamespace(get_recent_comments=lambda _count: []),
            prompt_generator=PromptGenerator(),
            realtime_generator=generator,
            rtmp_streamer=rtmp,
            text_overlay=Overlay(),
            initial_prompt="old story",
        )
        streamer.state = StreamingState(
            is_running=True,
            current_prompt="old story",
            previous_prompts=["old story"],
        )
        streamer.ltx_config = streamer.ltx_config.copy(update={
            "model_type": "ltx25-comfy",
            "width": 64,
            "height": 36,
            "num_frames": 9,
            "frame_rate": 9.0,
        })
        streamer._ltx25_continuity_prefix = ""
        streamer.state.current_frame_base64 = streamer._frame_to_base64(healthy)
        streamer._quality_reference = streamer._frame_quality(healthy)
        streamer._retry_prompt_result = PromptResult(
            selected_comment=comment,
            prompt=(
                'Viewer director command -- execute every clause literally and finish '
                'the visible result in this clip: "鏡頭拉遠 這個生物戴上眼鏡".'
            ),
            reasoning="test",
        )

        await streamer._generate_next_video(use_initial_prompt=False)

        self.assertEqual(0.30, generator.requests[0].strength)
        self.assertEqual(1.0, generator.requests[0].guidance_scale)
        self.assertEqual(0, streamer.state.generation_count)
        self.assertEqual([], rtmp.batches)
        self.assertEqual(1, streamer._comment_adherence_retries)
        self.assertEqual(comment, streamer._retry_prompt_result.selected_comment)
        self.assertIn("生物尚未戴上眼鏡", streamer._retry_prompt_result.prompt)
        self.assertEqual(["old story"], streamer.state.previous_prompts)

        await streamer._generate_next_video(use_initial_prompt=False)

        self.assertEqual(0.10, generator.requests[1].strength)
        self.assertFalse(generator.requests[1].force_t2v)
        self.assertEqual(0, streamer.state.generation_count)
        self.assertEqual([], rtmp.batches)

        await streamer._generate_next_video(use_initial_prompt=False)

        self.assertEqual(0.0, generator.requests[2].strength)
        self.assertTrue(generator.requests[2].force_t2v)
        self.assertEqual(1, streamer.state.generation_count)
        self.assertEqual(1, len(rtmp.batches))
        self.assertEqual(healthy.tobytes(), rtmp.batches[0][0].tobytes())
        self.assertEqual(1, streamer._comment_adherence_successes)
        self.assertEqual(2, streamer._comment_preflight_rejections)
        self.assertEqual(2, streamer._comment_adherence_retries)
        self.assertIn("PRIOR ATTEMPT WAS NOT SHOWN", streamer.state.current_prompt)

    async def test_two_poisoned_clips_commit_local_recovery_and_reuse_prompt(self):
        y, x = np.indices((36, 64))
        healthy_array = np.dstack([
            55 + (x * 2),
            65 + (y * 3),
            75 + ((x + y) * 2),
        ]).clip(0, 255).astype(np.uint8)
        healthy = Image.fromarray(healthy_array, "RGB")
        # Solid white remains corrupt after any crop, forcing the bounded retry
        # path rather than being recoverable through geometric post-processing.
        collapsed = Image.new("RGB", (64, 36), (255, 255, 255))

        class Generator:
            def __init__(self):
                self.calls = 0

            def generate_video_from_image(self, _request):
                self.calls += 1
                return LTXVideoResponseWithFrames(frames=[collapsed.copy() for _ in range(9)])

        class RTMP:
            def __init__(self):
                self.batches = []

            def add_frame_batch(self, frames, playback_seconds=None):
                self.batches.append(frames)
                return len(frames)

        class Overlay:
            def set_prompt(self, _prompt):
                pass

            def set_comment(self, _message, _username):
                pass

            def apply_overlay_batch(self, frames):
                return frames

        generator = Generator()
        rtmp = RTMP()
        streamer = RealtimeVideoStreamer(
            twitch_listener=SimpleNamespace(get_recent_comments=lambda _count: []),
            prompt_generator=SimpleNamespace(),
            realtime_generator=generator,
            rtmp_streamer=rtmp,
            text_overlay=Overlay(),
            initial_prompt="old story",
        )
        streamer.state = StreamingState(
            is_running=True,
            current_prompt="old story",
            previous_prompts=["old story"],
        )
        streamer.ltx_config = streamer.ltx_config.copy(update={
            "model_type": "ltx25-comfy",
            "width": 64,
            "height": 36,
            "num_frames": 9,
            "frame_rate": 9.0,
        })
        streamer._ltx25_continuity_prefix = ""
        streamer.state.current_frame_base64 = streamer._frame_to_base64(healthy)
        streamer._quality_reference = streamer._frame_quality(healthy)
        prompt_result = SimpleNamespace(
            prompt="continue the same story",
            selected_comment=None,
            reasoning="test",
        )
        streamer._retry_prompt_result = prompt_result

        await streamer._generate_next_video(use_initial_prompt=False)
        self.assertEqual(1, streamer._consecutive_corrupt_rejections)
        self.assertIs(prompt_result, streamer._retry_prompt_result)
        self.assertEqual([], rtmp.batches)

        await streamer._generate_next_video(use_initial_prompt=False)
        self.assertEqual(2, generator.calls)
        self.assertEqual(1, streamer._recovery_segments)
        self.assertEqual(1, streamer.state.generation_count)
        self.assertEqual(["old story"], streamer.state.previous_prompts)
        self.assertIs(prompt_result, streamer._retry_prompt_result)
        self.assertEqual(1, len(rtmp.batches))
        self.assertEqual(healthy.tobytes(), rtmp.batches[0][0].tobytes())
        self.assertNotEqual(healthy.tobytes(), rtmp.batches[0][-1].tobytes())


if __name__ == "__main__":
    unittest.main()
