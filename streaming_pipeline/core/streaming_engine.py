import asyncio
import base64
import hashlib
import os
import time
from typing import Dict, Any
from io import BytesIO
from PIL import Image
import requests

from streaming_pipeline.utils.logger_config import generation_log


from streaming_pipeline.models import LTXVideoRequestI2V, StreamingState, Monitorable, UserCommentParams



class RealtimeVideoStreamer(Monitorable):

    def __init__(self, 
                 twitch_listener,       
                 prompt_generator,     
                 realtime_generator,  
                 rtmp_streamer,        
                 text_overlay,          
                 comments_lookback: int = 5,
                 initial_prompt: str = None,
                 initial_image_url: str = None):
        
       
        self.twitch_listener = twitch_listener
        self.prompt_generator = prompt_generator
        self.realtime_generator = realtime_generator
        self.rtmp_streamer = rtmp_streamer
        self.text_overlay = text_overlay
        self.comments_lookback = comments_lookback
        

        self.state = StreamingState()
  
        
        # Use provided values or defaults
        self.initial_prompt = initial_prompt
        self.initial_image_url = initial_image_url
        self.next_prompt_ready = None  # Pre-generated prompt
        self.prompt_generation_task = None
        # Cached base64 of the *original* initial image (never overwritten).
        # Used by the h3-max anchor-reset mechanism to prevent style drift into
        # photorealism as H3 Max Turbo tends to ignore image style reference.
        self.initial_image_base64_cached: str = ""
        # Rolling cache of the most recent NON-corrupt last-frame, used to recover
        # chaining after a collapsed clip without resetting the story to origin.
        self._last_good_frame_base64: str = ""
        # Optional style prefix. Disabled by default: repeating high-contrast style
        # tokens on every autoregressive clip can amplify line/posterization artifacts.
        self._h3_max_style_prefix: str = os.getenv("H3_MAX_STYLE_PREFIX", "")
        self._ltx25_continuity_prefix: str = os.getenv(
            "LTX25_CONTINUITY_PREFIX",
            "Continue directly from the provided first frame without a cut. "
            "Keep a full-bleed image with scene content extending beyond all four edges; "
            "no frame, border, vignette, letterbox, mat, or picture-in-picture. ",
        )
        # Every N generations, re-anchor h3-max to the original initial image.
        # 0 or very large = disabled (natural frame-to-frame continuity).
        # Style is kept via the prompt prefix alone, which is less jarring
        # than resetting to the keyframe every clip.
        # Small positive value = periodic hard re-anchor if drift returns.
        self._h3_max_anchor_interval: int = int(os.getenv("H3_MAX_ANCHOR_INTERVAL", "0"))
        # A clip becomes part of the story only after it passes the visual gate and
        # every generated frame is accepted by the output streamer.
        self._quality_reference: Dict[str, float] = {}
        self._rejected_clips: int = 0
        self._committed_handoff_sha256: str = ""
        self._handoff_snapshot_path: str = ""
        self._border_guard_activations: int = 0
        self._border_repairs: int = 0
        # A strict visual gate must not be allowed to deadlock a live channel.
        # Reuse the same story prompt for bounded retries, then stream a local
        # deterministic push-in that removes the poisoned perimeter and gives
        # ComfyUI a clean tail to continue from.
        self._retry_prompt_result = None
        self._consecutive_corrupt_rejections: int = 0
        self._adaptive_repairs: int = 0
        self._recovery_segments: int = 0
        self._queue_backpressure_waits: int = 0
        self._queue_backpressure_seconds: float = 0.0
        # A displayed comment is not considered fulfilled until a vision audit
        # confirms that every clause became visible. Unfinished commands remain
        # sticky for a bounded number of clips while later chat stays queued.
        self._active_comment_key: str = ""
        self._comment_adherence_attempts: int = 0
        self._comment_adherence_retries: int = 0
        self._comment_adherence_failures: int = 0
        self._comment_adherence_successes: int = 0
        self._comment_preflight_rejections: int = 0
        self._last_comment_adherence: Dict[str, Any] = {}
        self._terminal_blur_trims: int = 0
        
        # Track generation parameters history (for metrics)
        self.generation_params_history = []
        
        # Current LTX configuration (starts with defaults, updated from start request)
        self.ltx_config = LTXVideoRequestI2V(
            prompt="",  # Will be set per generation
            image_base64=""  # Will be set per generation
        )
        

    

    
    def update_ltx_config(self, **kwargs):
        """Update LTX configuration with new parameters"""
        # Create new config with updated values
        current_dict = self.ltx_config.dict()
        current_dict.update(kwargs)
        self.ltx_config = LTXVideoRequestI2V(**current_dict)
        print(f"Updated LTX config: {', '.join(f'{k}={v}' for k, v in kwargs.items())}")
    
    def start_rtmp_stream(self):
        """Start the injected RTMP stream"""
        if self.rtmp_streamer and not self.rtmp_streamer.is_streaming:
            self.rtmp_streamer.start_stream()
            generation_log.info("✅ RTMP stream started")
    
    def stop_rtmp_stream(self):
        """Stop the injected RTMP stream"""
        if self.rtmp_streamer and self.rtmp_streamer.is_streaming:
            self.rtmp_streamer.stop_stream()
            generation_log.info("✅ RTMP stream stopped")
    
    def _url_to_base64(self, image_url: str) -> str:
        """Convert image URL to base64 (or return as-is if already base64)"""
        try:
            # Check if input is already base64 data URL
            if image_url.startswith('data:image'):
                print("🎯 Input is already base64 data URL, extracting base64 part...")
                # Extract base64 part after the comma
                base64_part = image_url.split(',')[1] if ',' in image_url else image_url
                
                # Validate and resize the base64 image
                try:
                    image_data = base64.b64decode(base64_part)
                    img = Image.open(BytesIO(image_data)).convert("RGB")
                    
                    # Normalize to this stream's configured generation dimensions.
                    width, height = self.ltx_config.width, self.ltx_config.height
                    
                    img = self._normalize_initial_image(img, width, height)
                    
                    # Re-encode to ensure consistent format
                    buffer = BytesIO()
                    img.save(buffer, format='PNG')
                    buffer.seek(0)
                    
                    return base64.b64encode(buffer.read()).decode('utf-8')
                    
                except Exception as e:
                    print(f"❌ Failed to process base64 image: {e}")
                    raise ValueError(f"Invalid base64 image data: {e}")
            
            # Check if input is raw base64 (without data URL prefix)
            elif self._is_base64_string(image_url):
                print("🎯 Input appears to be raw base64, processing...")
                try:
                    image_data = base64.b64decode(image_url)
                    img = Image.open(BytesIO(image_data)).convert("RGB")
                    
                    # Normalize to this stream's configured generation dimensions.
                    width, height = self.ltx_config.width, self.ltx_config.height
                    img = self._normalize_initial_image(img, width, height)
                    
                    # Re-encode
                    buffer = BytesIO()
                    img.save(buffer, format='PNG')
                    buffer.seek(0)
                    
                    return base64.b64encode(buffer.read()).decode('utf-8')
                    
                except Exception as e:
                    print(f"❌ Failed to process raw base64: {e}")
                    # Fall back to treating as URL
            
            # Regular URL - download and convert
            print(f"🌐 Downloading image from URL: {image_url[:100]}...")
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # Normalize to this stream's configured generation dimensions.
            width, height = self.ltx_config.width, self.ltx_config.height
            
            # Open image and resize to match generation dimensions
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img = self._normalize_initial_image(img, width, height)
            
            # Convert to base64
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            
            return base64.b64encode(buffer.read()).decode('utf-8')
            
        except Exception as e:
            raise ValueError(f"Failed to load initial image from {image_url}: {e}")

    def _normalize_initial_image(self, image: Image.Image, width: int, height: int) -> Image.Image:
        """Normalize the first keyframe and remove any tiny source-edge vignette.

        A one-pixel dark edge is harmless in a still image but becomes a learned
        picture frame after dozens of autoregressive I2V handoffs.  The inset is
        applied only once to the original keyframe, never to later handoffs.
        """
        ratio = 0.0
        if self.ltx_config.model_type == "ltx25-comfy":
            ratio = max(0.0, min(0.25, float(os.getenv("LTX25_INITIAL_INSET_RATIO", "0.15"))))
        if ratio:
            inset_x = max(1, int(round(image.width * ratio)))
            inset_y = max(1, int(round(image.height * ratio)))
            if image.width > inset_x * 2 and image.height > inset_y * 2:
                image = image.crop((inset_x, inset_y, image.width - inset_x, image.height - inset_y))
                print(f"🔎 Initial full-bleed crop: {ratio:.1%} per edge")
        return image.resize((width, height), Image.Resampling.LANCZOS)
    
    def _is_base64_string(self, s: str) -> bool:
        """Check if a string is valid base64"""
        try:
            # Basic checks
            if len(s) < 100:  # Too short to be a meaningful image
                return False
            if not s.replace('+', '').replace('/', '').replace('=', '').isalnum():
                return False
            
            # Try to decode
            decoded = base64.b64decode(s, validate=True)
            
            # Check if it starts with common image file signatures
            image_signatures = [
                b'\xff\xd8\xff',  # JPEG
                b'\x89PNG\r\n\x1a\n',  # PNG
                b'GIF87a',  # GIF87a
                b'GIF89a',  # GIF89a
                b'RIFF',  # WEBP (starts with RIFF)
            ]
            
            return any(decoded.startswith(sig) for sig in image_signatures)
            
        except Exception:
            return False
    
    def _frame_to_base64(self, frame: Image.Image) -> str:
        """Convert PIL Image frame to base64. Lossless PNG for chaining so we don't
        accumulate JPEG artifacts clip-over-clip (a source of degradation/collapse)."""
        buffer = BytesIO()
        frame.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    def _base64_to_frame(self, image_base64: str) -> Image.Image:
        raw = base64.b64decode(image_base64.split(",")[-1])
        return Image.open(BytesIO(raw)).convert("RGB")

    def _frame_digest(self, frame: Image.Image) -> str:
        """Digest decoded pixels, independent of PNG/JPEG container metadata."""
        return hashlib.sha256(frame.convert("RGB").tobytes()).hexdigest()

    def _persist_committed_handoff(self, frame: Image.Image) -> None:
        """Atomically persist the exact clean tail used by the next I2V clip.

        ComfyUI output files contain pre-repair pixels, so selecting its newest
        PNG during a restart can roll the story back or restore a halo. This
        snapshot is written only after RTMP accepts the complete clip and never
        contains the stream-only subtitle overlay.
        """
        configured = os.getenv(
            "LTX25_HANDOFF_SNAPSHOT",
            os.path.join("logs", "last_committed_handoff.png"),
        ).strip()
        if not configured:
            return
        try:
            target = os.path.abspath(configured)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            temporary = target + ".tmp.png"
            frame.convert("RGB").save(temporary, format="PNG")
            os.replace(temporary, target)
            self._handoff_snapshot_path = target
        except Exception as exc:
            generation_log.warning(f"⚠️ Could not persist committed handoff: {exc}")

    def _frame_quality(self, frame: Image.Image) -> Dict[str, float]:
        """Cheap structural metrics for detecting feedback-loop collapse."""
        import numpy as np

        rgb = np.asarray(frame.convert("RGB").resize((256, 144)), dtype=np.float32)
        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        hsv = np.asarray(frame.convert("HSV").resize((256, 144)), dtype=np.float32)
        hist = np.histogram(gray, bins=64, range=(0, 256))[0].astype(np.float64)
        probabilities = hist / max(1.0, hist.sum())
        nonzero = probabilities[probabilities > 0]
        entropy = float(-(nonzero * np.log2(nonzero)).sum())
        band = max(2, int(round(min(gray.shape) * 0.06)))
        edge_mask = np.zeros(gray.shape, dtype=bool)
        edge_mask[:band, :] = True
        edge_mask[-band:, :] = True
        edge_mask[:, :band] = True
        edge_mask[:, -band:] = True
        edge = gray[edge_mask]
        inner = gray[band:-band, band:-band]
        saturation = hsv[:, :, 1]
        edge_saturation = saturation[edge_mask]
        inner_saturation = saturation[band:-band, band:-band]

        # Hard black/white bars are easy to detect. The long-running LTX chain
        # more often produces a *soft* rounded halo: each perimeter strip shifts
        # hue/saturation or luminance relative to the immediately adjacent strip,
        # without a single sharp line. Compare all four sides locally so a dark
        # subject in the centre is not mistaken for a vignette.
        side_pairs = (
            (rgb[:band, :, :], rgb[band:band * 2, :, :], gray[:band, :], gray[band:band * 2, :], saturation[:band, :], saturation[band:band * 2, :]),
            (rgb[-band:, :, :], rgb[-band * 2:-band, :, :], gray[-band:, :], gray[-band * 2:-band, :], saturation[-band:, :], saturation[-band * 2:-band, :]),
            (rgb[:, :band, :], rgb[:, band:band * 2, :], gray[:, :band], gray[:, band:band * 2], saturation[:, :band], saturation[:, band:band * 2]),
            (rgb[:, -band:, :], rgb[:, -band * 2:-band, :], gray[:, -band:], gray[:, -band * 2:-band], saturation[:, -band:], saturation[:, -band * 2:-band]),
        )
        side_rgb_distances = []
        side_luma_deltas = []
        side_saturation_deltas = []
        for outer_rgb, adjacent_rgb, outer_gray, adjacent_gray, outer_sat, adjacent_sat in side_pairs:
            side_rgb_distances.append(float(
                np.linalg.norm(outer_rgb.mean(axis=(0, 1)) - adjacent_rgb.mean(axis=(0, 1)))
            ))
            side_luma_deltas.append(float(outer_gray.mean() - adjacent_gray.mean()))
            side_saturation_deltas.append(float(outer_sat.mean() - adjacent_sat.mean()))
        soft_border_sides = sum(
            distance > 30
            or (distance > 20 and (abs(luma) > 9 or abs(sat_delta) > 15))
            for distance, luma, sat_delta in zip(
                side_rgb_distances,
                side_luma_deltas,
                side_saturation_deltas,
            )
        )
        soft_luma_sides = sum(
            distance > 18 and abs(luma) > 14
            for distance, luma in zip(side_rgb_distances, side_luma_deltas)
        )
        soft_chroma_sides = sum(
            distance > 22 and sat_delta > 15
            for distance, sat_delta in zip(side_rgb_distances, side_saturation_deltas)
        )
        horizontal_delta = np.abs(np.diff(gray, axis=0))
        vertical_delta = np.abs(np.diff(gray, axis=1))
        outer_y = max(2, int(round(gray.shape[0] * 0.22)))
        outer_x = max(2, int(round(gray.shape[1] * 0.22)))
        outer_rows = np.r_[0:outer_y, gray.shape[0] - 1 - outer_y:gray.shape[0] - 1]
        outer_cols = np.r_[0:outer_x, gray.shape[1] - 1 - outer_x:gray.shape[1] - 1]
        row_line_coverage = (horizontal_delta > 25).mean(axis=1)
        col_line_coverage = (vertical_delta > 25).mean(axis=0)
        row_line_gradient = horizontal_delta.mean(axis=1)
        col_line_gradient = vertical_delta.mean(axis=0)
        return {
            "mean_sat": float(hsv[:, :, 1].mean()),
            "variance": float(rgb.var(axis=(0, 1)).sum()),
            "mean_val": float(rgb.mean()),
            "extreme_fraction": float(((gray < 20) | (gray > 235)).mean()),
            "entropy": entropy,
            "edge_dark_fraction": float((edge < 24).mean()),
            "edge_bright_fraction": float((edge > 232).mean()),
            "edge_extreme_fraction": float(((edge < 24) | (edge > 232)).mean()),
            "edge_mean": float(edge.mean()),
            "inner_mean": float(inner.mean()),
            "edge_saturation_mean": float(edge_saturation.mean()),
            "inner_saturation_mean": float(inner_saturation.mean()),
            "perimeter_saturation_delta": float(edge_saturation.mean() - inner_saturation.mean()),
            "perimeter_luma_delta": float(edge.mean() - inner.mean()),
            "side_rgb_distance_mean": float(np.mean(side_rgb_distances)),
            "side_rgb_distance_max": float(np.max(side_rgb_distances)),
            "soft_border_sides": float(soft_border_sides),
            "soft_luma_sides": float(soft_luma_sides),
            "soft_chroma_sides": float(soft_chroma_sides),
            "outer_line_coverage": float(max(
                row_line_coverage[outer_rows].max(),
                col_line_coverage[outer_cols].max(),
            )),
            "outer_line_gradient": float(max(
                row_line_gradient[outer_rows].max(),
                col_line_gradient[outer_cols].max(),
            )),
        }

    def _border_guard_needed(self, frame: Image.Image) -> bool:
        """Detect early picture-frame growth while it can still be zoomed away."""
        quality = self._frame_quality(frame)
        reference = self._quality_reference or quality
        dark_warning = max(0.38, reference.get("edge_dark_fraction", 0.0) + 0.12)
        bright_warning = max(0.30, reference.get("edge_bright_fraction", 0.0) + 0.12)
        extreme_warning = max(0.56, reference.get("edge_extreme_fraction", 0.0) + 0.16)
        line_warning = max(0.58, reference.get("outer_line_coverage", 0.0) + 0.10)
        dark_border = (
            quality["edge_dark_fraction"] > dark_warning
            and quality["edge_mean"] < quality["inner_mean"] * 0.72
        )
        bright_border = (
            quality["edge_bright_fraction"] > bright_warning
            and quality["edge_mean"] > quality["inner_mean"] + 45
        )
        chromatic_frame = (
            quality["edge_extreme_fraction"] > extreme_warning
            and abs(quality["edge_mean"] - quality["inner_mean"]) > 35
        )
        rectangular_frame = (
            quality["outer_line_coverage"] >= line_warning
            and quality["outer_line_gradient"] > 12
        )
        soft_chromatic_halo = (
            quality["soft_border_sides"] >= 2
            and quality["perimeter_saturation_delta"] > 16
        )
        soft_vignette = (
            quality["soft_luma_sides"] >= 3
            and abs(quality["perimeter_luma_delta"]) > 10
        )
        return (
            dark_border
            or bright_border
            or chromatic_frame
            or rectangular_frame
            or soft_chromatic_halo
            or soft_vignette
        )

    def _border_repair_needed(self, frame: Image.Image) -> bool:
        """Require stronger evidence before altering pixels with a smooth zoom.

        Prompt-only guidance can turn a growing frame translucent without actually
        removing it.  This stricter test is reserved for geometric remediation so
        ordinary high-contrast scene edges do not cause unnecessary camera pushes.
        """
        quality = self._frame_quality(frame)
        reference = self._quality_reference or quality
        mean_delta = quality["edge_mean"] - quality["inner_mean"]
        dark_limit = max(0.34, reference.get("edge_dark_fraction", 0.0) + 0.15)
        bright_limit = max(0.28, reference.get("edge_bright_fraction", 0.0) + 0.11)
        extreme_limit = max(0.50, reference.get("edge_extreme_fraction", 0.0) + 0.15)
        dark_frame = quality["edge_dark_fraction"] > dark_limit and mean_delta < -25
        bright_frame = quality["edge_bright_fraction"] > bright_limit and mean_delta > 35
        chromatic_frame = quality["edge_extreme_fraction"] > extreme_limit and abs(mean_delta) > 32
        strong_rectangle = (
            quality["outer_line_coverage"] > 0.72
            and quality["outer_line_gradient"] > 28
            and abs(mean_delta) > 30
        )
        soft_halo = (
            quality["soft_border_sides"] >= 2
            and quality["perimeter_saturation_delta"] > 18
        ) or (
            quality["soft_luma_sides"] >= 3
            and abs(quality["perimeter_luma_delta"]) > 12
        )
        return dark_frame or bright_frame or chromatic_frame or strong_rectangle or soft_halo

    def _border_repair_ratio(self, frame: Image.Image) -> float:
        """Choose enough crop to remove a soft halo in one clip, not dozens."""
        quality = self._frame_quality(frame)
        sat_delta = quality["perimeter_saturation_delta"]
        sides = quality["soft_border_sides"]
        if sat_delta > 55 or sides >= 4:
            return 0.16
        if sat_delta > 35 or sides >= 3:
            return 0.13
        if sat_delta > 18 or sides >= 2:
            return 0.10
        return float(os.getenv("LTX25_BORDER_REPAIR_INSET_RATIO", "0.08"))

    def _progressive_full_bleed_crop(
        self,
        frames,
        ratio: float | None = None,
        settle_fraction: float = 1.0,
    ):
        """Remove a feedback-loop border without introducing a clip-boundary cut.

        Frame zero is left byte-for-byte untouched, so it remains identical to the
        last streamed frame of the preceding clip.  The crop then eases in over the
        shot; the cropped tail becomes the exact I2V handoff for the next clip.
        """
        if not frames:
            return frames
        if ratio is None:
            ratio = float(os.getenv("LTX25_BORDER_REPAIR_INSET_RATIO", "0.08"))
        ratio = max(0.0, min(0.20, ratio))
        if ratio == 0 or len(frames) == 1:
            return frames

        settle_fraction = max(0.05, min(1.0, settle_fraction))
        repaired = []
        last = len(frames) - 1
        for index, source in enumerate(frames):
            if index == 0:
                repaired.append(source)
                continue
            # Recovery clips need to clear the perimeter early instead of leaving
            # several sampled frames corrupt. Normal preventative crops still ease
            # over the whole shot with the default settle_fraction=1.0.
            t = min(1.0, index / max(1.0, last * settle_fraction))
            eased = t * t * (3.0 - 2.0 * t)
            inset_x = int(round(source.width * ratio * eased))
            inset_y = int(round(source.height * ratio * eased))
            if inset_x < 1 and inset_y < 1:
                repaired.append(source)
                continue
            cropped = source.crop((
                inset_x,
                inset_y,
                source.width - inset_x,
                source.height - inset_y,
            ))
            repaired.append(cropped.resize(source.size, Image.Resampling.LANCZOS))
        return repaired

    def _try_adaptive_clip_repair(self, frames):
        """Try increasingly strong local crops before rejecting generated work.

        This is deterministic post-processing only: frame zero remains exact, so
        the join to the preceding streamed segment is still pixel-identical.
        """
        if not frames:
            return frames, False
        for ratio in (0.12, 0.16, 0.20):
            candidate = self._progressive_full_bleed_crop(
                frames,
                ratio=ratio,
                settle_fraction=0.35,
            )
            if not self._clip_is_corrupt(candidate):
                return candidate, True
        return frames, False

    def _make_local_recovery_clip(self, frame: Image.Image, num_frames: int):
        """Create a seamless local motion segment that cannot consume cloud GPU.

        Infinite TV keeps its output queue alive by repeating the last frame. We
        preserve that liveness invariant but turn the repeat into a smooth push-in,
        which also removes autoregressive border feedback before the next I2V call.
        """
        count = max(2, int(num_frames))
        ratio = float(os.getenv("LTX25_RECOVERY_INSET_RATIO", "0.18"))
        sources = [frame.copy() for _ in range(count)]
        return self._progressive_full_bleed_crop(
            sources,
            ratio=ratio,
            settle_fraction=0.25,
        )

    def _frame_is_corrupt(self, frame: Image.Image) -> bool:
        """Reject frames that would poison an autoregressive I2V chain.

        Besides rainbow/flat/exposure failures, detect the black/white posterization
        collapse seen after repeated LTX I2V chaining. Thresholds are relative to the
        clean initial keyframe so stylized but healthy source images remain valid.
        """
        try:
            quality = self._frame_quality(frame)
            mean_sat = quality["mean_sat"]
            variance = quality["variance"]
            mean_val = quality["mean_val"]
            extreme_fraction = quality["extreme_fraction"]
            entropy = quality["entropy"]
            sat_th = float(os.getenv("CORRUPT_SAT_THRESHOLD", "140"))
            var_th = float(os.getenv("CORRUPT_VARIANCE_THRESHOLD", "20000"))
            flat_floor = float(os.getenv("FLAT_VARIANCE_FLOOR", "60"))
            rainbow = (mean_sat > sat_th) and (variance > var_th)
            flat = variance < flat_floor
            exposure = mean_val > 240 or mean_val < 12
            reference_extreme = self._quality_reference.get("extreme_fraction", extreme_fraction)
            reference_entropy = self._quality_reference.get("entropy", entropy)
            extreme_limit = float(os.getenv(
                "CORRUPT_EXTREME_FRACTION",
                str(min(0.78, max(0.62, reference_extreme + 0.30))),
            ))
            entropy_limit = float(os.getenv(
                "CORRUPT_ENTROPY_FLOOR",
                str(max(3.8, min(4.6, reference_entropy - 0.9))),
            ))
            posterized = extreme_fraction > extreme_limit and entropy < entropy_limit
            reference_dark = self._quality_reference.get("edge_dark_fraction", quality["edge_dark_fraction"])
            reference_bright = self._quality_reference.get("edge_bright_fraction", quality["edge_bright_fraction"])
            reference_edge_extreme = self._quality_reference.get("edge_extreme_fraction", quality["edge_extreme_fraction"])
            dark_limit = min(0.82, max(0.52, reference_dark + 0.22))
            bright_limit = min(0.78, max(0.46, reference_bright + 0.22))
            edge_extreme_limit = min(0.88, max(0.68, reference_edge_extreme + 0.25))
            line_limit = max(
                0.82,
                self._quality_reference.get("outer_line_coverage", quality["outer_line_coverage"]) + 0.28,
            )
            border = (
                quality["edge_dark_fraction"] > dark_limit
                and quality["edge_mean"] < quality["inner_mean"] * 0.55
            ) or (
                quality["edge_bright_fraction"] > bright_limit
                and quality["edge_mean"] > quality["inner_mean"] + 60
            ) or (
                quality["edge_extreme_fraction"] > edge_extreme_limit
                and abs(quality["edge_mean"] - quality["inner_mean"]) > 50
            ) or (
                quality["outer_line_coverage"] > line_limit
                and quality["outer_line_gradient"] > 20
            )
            soft_border = (
                quality["soft_border_sides"] >= 3
                and quality["perimeter_saturation_delta"] > 24
            ) or (
                quality["soft_luma_sides"] >= 3
                and abs(quality["perimeter_luma_delta"]) > 16
            )
            border = border or soft_border
            verdict = (
                "RAINBOW" if rainbow else
                "FLAT/DEAD" if flat else
                "EXPOSURE" if exposure else
                "POSTERIZED" if posterized else
                "BORDER" if border else
                "ok"
            )
            print(
                f"   check: sat={mean_sat:.0f} var={variance:.0f} val={mean_val:.0f} "
                f"ext={extreme_fraction:.3f}/{extreme_limit:.3f} "
                f"entropy={entropy:.2f}/{entropy_limit:.2f} "
                f"edge(d/b/x)={quality['edge_dark_fraction']:.2f}/"
                f"{quality['edge_bright_fraction']:.2f}/"
                f"{quality['edge_extreme_fraction']:.2f} "
                f"line={quality['outer_line_coverage']:.2f} "
                f"halo(sides/sat/luma)={quality['soft_border_sides']:.0f}/"
                f"{quality['perimeter_saturation_delta']:.0f}/"
                f"{quality['perimeter_luma_delta']:.0f} → {verdict}"
            )
            return rainbow or flat or exposure or posterized or border
        except Exception as e:
            print(f"   ⚠️ corruption check failed (skipping): {e}")
            return False

    def _clip_is_corrupt(self, frames) -> bool:
        """Sample the whole clip; reject a bad tail or two bad interior samples."""
        if not frames:
            return True
        last = len(frames) - 1
        indices = sorted({0, last // 4, last // 2, (3 * last) // 4, last})
        bad = [self._frame_is_corrupt(frames[index]) for index in indices]
        return bad[-1] or sum(bad) >= 2

    def start_streaming(self):
        """Start the realtime streaming process"""
        if self.state.is_running:
            print("⚠️ Already running")
            return
        
        # Auto-set initial state if not already set
        if not self.state.current_frame_base64:
            print(f"🖼️ Loading initial image from: {self.initial_image_url}")
            initial_image_base64 = self._url_to_base64(self.initial_image_url)
            self.state.current_frame_base64 = initial_image_base64
            # Cache the *original* for h3-max style-anchor re-injection.
            self.initial_image_base64_cached = initial_image_base64
            self.state.current_prompt = self.initial_prompt
            self.state.previous_prompts = [self.initial_prompt]

        # The starting keyframe is a valid recovery point even if the very first
        # generated clip collapses.
        self._last_good_frame_base64 = self.state.current_frame_base64
        initial_frame = self._base64_to_frame(self.state.current_frame_base64)
        self._quality_reference = self._frame_quality(initial_frame)
        self._committed_handoff_sha256 = self._frame_digest(initial_frame)
        self._rejected_clips = 0
        self._border_guard_activations = 0
        self._border_repairs = 0
        self._retry_prompt_result = None
        self._consecutive_corrupt_rejections = 0
        self._adaptive_repairs = 0
        self._recovery_segments = 0
        self._queue_backpressure_waits = 0
        self._queue_backpressure_seconds = 0.0
        self._active_comment_key = ""
        self._comment_adherence_attempts = 0
        self._comment_adherence_retries = 0
        self._comment_adherence_failures = 0
        self._comment_adherence_successes = 0
        self._comment_preflight_rejections = 0
        self._last_comment_adherence = {}
        self._terminal_blur_trims = 0
        generation_log.info(
            "🧭 Quality reference: "
            f"ext={self._quality_reference['extreme_fraction']:.3f}, "
            f"entropy={self._quality_reference['entropy']:.2f}, "
            f"edge-dark={self._quality_reference['edge_dark_fraction']:.3f}, "
            f"handoff={self._committed_handoff_sha256[:12]}"
        )
        
        generation_log.info(f"🎬 Starting realtime video streaming...")
        generation_log.info(f"📺 Twitch channel: #{self.twitch_listener.channel_name}")
        
        # Start RTMP stream first
        self.start_rtmp_stream()
        
        self.state.is_running = True
        self.twitch_listener.start_listening()
        
        # Start the generation loop in a separate thread
        import threading
        self.generation_thread = threading.Thread(target=self._run_generation_loop)  # Change back to this
        self.generation_thread.daemon = True
        self.generation_thread.start()

    def stop_streaming(self):
        """Stop the realtime streaming process"""
        if not self.state.is_running:
            print("⚠️ Already stopped")
            return
        
        print("🛑 Stopping realtime video streaming...")
        
        # Stop the streaming state
        self.state.is_running = False
        
        # Stop Twitch listener
        if self.twitch_listener:
            self.twitch_listener.stop_listening()
        
        # Stop RTMP stream
        self.stop_rtmp_stream()
        
        # Wait for generation thread to finish (with timeout)
        if hasattr(self, 'generation_thread') and self.generation_thread.is_alive():
            print("⏳ Waiting for generation thread to finish...")
            self.generation_thread.join(timeout=2.0)  # Shorter timeout to avoid blocking
            if self.generation_thread.is_alive():
                print("⚠️ Generation thread did not stop gracefully")
        
        # Clear all context and state for fresh restart
        print("🧹 Clearing context and state...")
        self.state.current_frame_base64 = ""
        self.state.current_prompt = ""
        self.state.generation_count = 0
        self.state.previous_prompts = []
        self.next_prompt_ready = None
        self.prompt_generation_task = None
        self._retry_prompt_result = None
        self._consecutive_corrupt_rejections = 0
        self._active_comment_key = ""
        self._comment_adherence_attempts = 0
        
        # Reset metrics on all components
        if hasattr(self.prompt_generator, 'reset_metrics'):
            self.prompt_generator.reset_metrics()
        if hasattr(self.realtime_generator, 'reset_metrics'):
            self.realtime_generator.reset_metrics()
        if hasattr(self.text_overlay, 'reset_metrics'):
            self.text_overlay.reset_metrics()
        # Note: RTMP streamer resets itself in stop_stream()
        
        generation_log.info("✅ Realtime video streaming stopped and context cleared")

    def _run_generation_loop(self):
        """Run the async generation loop in a new event loop"""
        import asyncio
        asyncio.run(self._generation_loop())  # This properly runs the async function
    
    async def _generation_loop(self):
        """Continuous generation loop with no gaps.

        Prompt generation must run AFTER video generation completes, because
        the prompt generator's vision LLM analyzes the most recent frame to
        steer the next clip.  Overlapping them would feed the LLM a stale
        frame from the previous video.
        """
        first_generation = True

        while self.state.is_running:
            try:
                # Keep Twitch close to the generation head. Without backpressure,
                # a generator that is only slightly faster than 9 FPS can build a
                # multi-minute queue, making viewer comments appear "missing" even
                # though their caption is already burned into a later batch.
                await self._wait_for_output_latency_budget()

                # For first generation, don't start prompt generation task
                if not first_generation:
                    # Start prompt generation for NEXT video (sequential with current frame)
                    if not self.prompt_generation_task and self._retry_prompt_result is None:
                        self.prompt_generation_task = asyncio.create_task(
                            self._prepare_next_prompt()
                        )

                # Generate current video
                await self._generate_next_video(use_initial_prompt=first_generation)

                first_generation = False  # After first generation, use normal flow

                # No sleep! Immediately continue to next generation

            except Exception as e:
                generation_log.error(f"❌ Generation error: {e}")
                generation_log.info(f"🔄 Continuing with next generation attempt...")

                # Important: Still mark first_generation as False even if it failed
                # This prevents getting stuck in initial prompt mode
                first_generation = False

                # Cancel any pending prompt generation task to start fresh
                if self.prompt_generation_task:
                    self.prompt_generation_task.cancel()
                    self.prompt_generation_task = None

                await asyncio.sleep(1)  # Brief pause on error only

    async def _wait_for_output_latency_budget(self):
        """Pause generation while the RTMP queue is ahead of real time."""
        streamer = self.rtmp_streamer
        frame_queue = getattr(streamer, "frame_queue", None)
        fps = max(1.0, float(getattr(streamer, "fps", 1.0) or 1.0))
        target_seconds = max(1.0, float(os.getenv("RTMP_TARGET_QUEUE_SECONDS", "18")))
        if frame_queue is None:
            return

        wait_started = None
        while self.state.is_running and getattr(streamer, "is_streaming", True):
            queue_seconds = frame_queue.qsize() / fps
            if queue_seconds <= target_seconds:
                break
            if wait_started is None:
                wait_started = time.time()
                self._queue_backpressure_waits += 1
                generation_log.info(
                    f"⏳ RTMP backpressure: {queue_seconds:.1f}s queued; "
                    f"waiting for <= {target_seconds:.1f}s"
                )
            await asyncio.sleep(min(1.0, max(0.1, queue_seconds - target_seconds)))

        if wait_started is not None:
            waited = time.time() - wait_started
            self._queue_backpressure_seconds += waited
            generation_log.info(f"▶️ RTMP queue caught up after {waited:.1f}s")

    @staticmethod
    def _comment_key(comment) -> str:
        return f"{comment.username}\n{comment.message}" if comment else ""

    def _comment_attempt_number(self, comment) -> int:
        key = self._comment_key(comment)
        if key and key == self._active_comment_key:
            return self._comment_adherence_attempts + 1
        return 1

    @staticmethod
    def _comment_strength_for_attempt(attempt: int) -> float:
        """Progressively release the first-frame guide without abandoning the set.

        LTXVAddGuide strength controls both guide noise and reference attention.
        Start at the official LTX I2V value for scene continuity, loosen once for
        action compliance, then use a first/last-frame scene bridge rather than
        removing image conditioning entirely.
        """
        raw = os.getenv("COMMENT_I2V_STRENGTH_SCHEDULE", "0.70,0.45,0.20")
        try:
            values = [min(1.0, max(0.0, float(item.strip()))) for item in raw.split(",")]
            values = [value for value in values if value >= 0.0]
        except ValueError:
            values = []
        if not values:
            values = [0.70, 0.45, 0.20]
        return values[min(max(1, attempt) - 1, len(values) - 1)]

    @staticmethod
    def _soften_prompt_first_seam(frames, handoff_frame, transition_frames: int = 8):
        """Keep frame zero exact and ease into a prompt-dominant final retry."""
        if not frames:
            return frames
        source = handoff_frame.convert("RGB").resize(frames[0].size, Image.Resampling.LANCZOS)
        softened = list(frames)
        softened[0] = source.copy()
        count = min(max(1, transition_frames), len(softened) - 1)
        for index in range(1, count + 1):
            t = index / count
            smooth = t * t * (3.0 - 2.0 * t)
            softened[index] = Image.blend(source, softened[index].convert("RGB"), smooth)
        return softened

    @staticmethod
    def _frame_sharpness(frame: Image.Image) -> float:
        """Return a cheap edge-energy score; low terminal values indicate motion blur."""
        import numpy as np

        rgb = np.asarray(frame.convert("RGB").resize((256, 144)), dtype=np.float32)
        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        dx = np.diff(gray, axis=1)
        dy = np.diff(gray, axis=0)
        return float((dx * dx).mean() + (dy * dy).mean())

    def _stabilize_terminal_blur(self, frames):
        """End on a recent sharp generated frame instead of chaining motion blur.

        LTX often accelerates motion in its final few decoded frames. Feeding that
        smeared terminal image into the next I2V call makes a blur pulse recur at
        every join. Select the latest acceptably sharp frame from the final second,
        then distribute the retained frames back to the original count without
        pixel blending. The selected image is still generated and streamed, and it
        becomes the exact next handoff.
        """
        if len(frames) < 9:
            return frames
        import numpy as np

        original_count = len(frames)
        start = max(0, original_count - 9)
        scores = [self._frame_sharpness(frame) for frame in frames[start:]]
        reference = float(np.median(scores[:max(1, len(scores) - 3)]))
        threshold = max(1.0, reference * 0.82)
        chosen = original_count - 1
        for offset in range(len(scores) - 1, -1, -1):
            if scores[offset] >= threshold:
                chosen = start + offset
                break
        if chosen >= original_count - 1:
            return frames

        retained = frames[:chosen + 1]
        indices = np.linspace(0, len(retained) - 1, original_count).round().astype(int)
        stabilized = [retained[index] for index in indices]
        self._terminal_blur_trims += 1
        generation_log.info(
            f"🎯 Terminal blur guard: ended on generated frame {chosen + 1}/"
            f"{original_count} (sharpness {scores[chosen - start]:.1f} vs "
            f"{scores[-1]:.1f}); kept {original_count} streamed frames"
        )
        return stabilized
    
    async def _prepare_next_prompt(self):
        """Generate the next prompt while current video is generating - WITH VISUAL CONTEXT"""
        # Get recent comments
        # Consume one command per clip in FIFO order. Pulling five at once caused
        # four unselected comments to disappear from the queue permanently.
        comments = self.twitch_listener.get_recent_comments(1)
        
        # Log LLM input details
        print(f"\n🤖 LLM INPUT for next generation:")
        print(f"   📝 Previous prompts: {len(self.state.previous_prompts)}")
        if self.state.previous_prompts:
            print(f"   📝 Last prompt: {self.state.previous_prompts[-1]}")
        print(f"   🖼️ Visual context: {'✅ Available' if self.state.current_frame_base64 else '❌ None'}")
        print(f"   💬 Recent comments: {len(comments)}")
        for i, comment in enumerate(comments):
            print(f"   💬 [{comment.username}]: {comment.message}")
        
        # Generate prompt with visual context (pass state directly)
        prompt_result = await asyncio.to_thread(
            self.prompt_generator.generate_prompt, 
            comments, 
            self.state  # Pass unified state instead of separate context
        )
        
        # Log LLM output details
        print(f"🤖 LLM OUTPUT:")
        if prompt_result.selected_comment:
            print(f"   ✅ Selected comment: [{prompt_result.selected_comment.username}] {prompt_result.selected_comment.message}")
        else:
            print(f"   🌱 Evolution mode (no suitable comments)")
        print(f"   🧠 LLM Reasoning: {prompt_result.reasoning}")
        print(f"   📝 Generated Prompt: {prompt_result.prompt}")
        
        self.next_prompt_ready = prompt_result
        return prompt_result
    
    async def _generate_next_video(self, use_initial_prompt=False):
        """Generate video using pre-prepared prompt or initial prompt for first generation"""
        cycle_start_time = time.time()
        prompt_result = None
        comment_adherence = None
        comment_attempt_number = 0
        comment_strength = None

        # Track whether a user comment was used for dynamic parameter adjustment
        used_comment = False

        if use_initial_prompt:
            # For the FIRST generation, use the initial prompt directly
            generation_log.info(f"🎬 Generation #1 (INITIAL)")
            generation_log.info(f"📝 Using initial prompt: {self.state.current_prompt}")

            prompt_to_use = self.state.current_prompt
            selected_comment = None
            used_comment = False  # Initial prompt is not a user comment

            # Show initial prompt on overlay
            self.text_overlay.set_prompt(prompt_to_use)

        else:
            # For subsequent generations, use the normal prompt generation process
            # Wait for prompt to be ready (should already be done)
            if self._retry_prompt_result is not None:
                prompt_result = self._retry_prompt_result
                self._retry_prompt_result = None
                generation_log.info("♻️ Reusing rejected clip prompt; no new LLM charge")
            elif self.prompt_generation_task:
                prompt_result = await self.prompt_generation_task
                self.prompt_generation_task = None  # Reset for next iteration
            else:
                # Fallback if no pre-generated prompt
                comments = self.twitch_listener.get_recent_comments(1)

                # Log fallback LLM input
                print(f"\n🤖 FALLBACK LLM INPUT:")
                print(f"   💬 Recent comments: {len(comments)}")
                for comment in comments:
                    print(f"   💬 [{comment.username}]: {comment.message}")

                prompt_result = self.prompt_generator.generate_prompt(comments, self.state)

                # Log fallback LLM output
                print(f"🤖 FALLBACK LLM OUTPUT:")
                print(f"   🧠 LLM Reasoning: {prompt_result.reasoning}")
                print(f"   📝 Generated Prompt: {prompt_result.prompt}")

            generation_log.info(f"🎬 Generation #{self.state.generation_count + 1}")
            if prompt_result.selected_comment:
                generation_log.info(f"💬 Selected: [{prompt_result.selected_comment.username}] {prompt_result.selected_comment.message}")
            else:
                generation_log.info(f"🌱 Evolution: {prompt_result.reasoning}")
            generation_log.info(f"📝 Prompt: {prompt_result.prompt}")

            prompt_to_use = prompt_result.prompt
            selected_comment = prompt_result.selected_comment
            used_comment = selected_comment is not None  # Track if comment was used

            # Set the overlay text (RealtimeVideoStreamer controls presentation)
            if selected_comment:
                # Show the selected comment
                self.text_overlay.set_comment(
                    selected_comment.message,
                    selected_comment.username
                )
            else:
                # Show the AI-generated prompt when no comment is selected
                self.text_overlay.set_prompt(prompt_to_use)

        # Generate video (same for both initial and subsequent generations)
        try:
            # ─── H3 Max Turbo style-anchor patch ─────────────────────────────
            # H3 Max Turbo is heavily biased toward photorealism and quickly
            # drifts away from cartoon/stylized reference images if we let it
            # chain generation-to-generation. Re-anchor to the original image
            # every N generations, and always prepend the style prefix to the
            # LLM-generated prompt so the visual DNA is re-asserted each cycle.
            frame_for_this_gen = self.state.current_frame_base64
            prompt_for_this_gen = prompt_to_use
            border_guard_active = False
            border_repair_active = False
            if self.ltx_config.model_type in ("h3-max", "ltx25-comfy"):
                # Only re-anchor to the keyframe if interval > 0 (opt-in).
                # Default OFF because it breaks visual continuity between clips.
                interval = self._h3_max_anchor_interval
                if interval > 0 and self.initial_image_base64_cached and self.state.generation_count % interval == 0:
                    frame_for_this_gen = self.initial_image_base64_cached
                    print(f"🔒 h3-max anchor: re-injecting original image (interval={interval}, gen#{self.state.generation_count})")
                # Style prefix on every prompt keeps the LLM plot text in the
                # right visual DNA without touching the video reference frame.
                if self._h3_max_style_prefix and not prompt_for_this_gen.startswith(self._h3_max_style_prefix.strip()[:30]):
                    prompt_for_this_gen = self._h3_max_style_prefix + prompt_for_this_gen
                    print(f"🎨 h3-max style prefix injected")
            if self.ltx_config.model_type == "ltx25-comfy" and self._ltx25_continuity_prefix:
                prefix = self._ltx25_continuity_prefix
                if not prompt_for_this_gen.startswith(prefix.strip()[:30]):
                    prompt_for_this_gen = prefix + prompt_for_this_gen
                    print("🔗 ltx25 continuity instruction injected")
                if frame_for_this_gen:
                    handoff_frame = self._base64_to_frame(frame_for_this_gen)
                    border_guard_active = self._border_guard_needed(handoff_frame)
                    border_repair_active = self._border_repair_needed(handoff_frame)
                    if border_guard_active:
                        self._border_guard_activations += 1
                        print(
                            "🔎 Border guard: reserving deterministic full-bleed repair "
                            "without overriding the story/camera prompt"
                        )
            # ─────────────────────────────────────────────────────────────────

            current_frame_preview = frame_for_this_gen[:50] + "..." if frame_for_this_gen else "None"
            print(f"🎬 Using input frame: {current_frame_preview}")

            request_dict = self.ltx_config.dict()
            request_dict.update({
                "prompt": prompt_for_this_gen,
                "image_base64": frame_for_this_gen,
            })

            # Scene continuity is a global invariant, not only a comment-retry
            # concern. The visual planner describes the current set from the real
            # streamed handoff; keep that set recognizable for ordinary evolution
            # too. Only an explicit viewer relocation command may release it.
            scene_description = (
                getattr(prompt_result, "visual_description", "")
                if prompt_result is not None
                else ""
            )
            scene_change_requested = bool(
                selected_comment
                and getattr(prompt_result, "scene_change_requested", False)
            )
            if scene_description and not scene_change_requested:
                prompt_for_this_gen += (
                    " SCENE LOCK: keep this same physical location, background layout, "
                    f"lighting, and camera axis recognizable: {scene_description}. "
                    "Advance the story through action inside this set; do not replace it."
                )
            request_dict.update({
                "prompt": prompt_for_this_gen,
                "scene_description": scene_description,
                "preserve_scene": not scene_change_requested,
            })

            # Pass character references for the condition pipeline
            if self.ltx_config.model_type == "ltx-2.3-condition" and self.state.character_refs:
                request_dict["character_refs"] = self.state.character_refs

            if used_comment:
                comment_params = UserCommentParams()
                comment_attempt_number = self._comment_attempt_number(selected_comment)
                comment_strength = self._comment_strength_for_attempt(comment_attempt_number)
                request_dict.update({
                    "guidance_scale": comment_params.guidance_scale,
                    "strength": comment_strength,
                })
                if comment_attempt_number >= 3:
                    # A pure T2V fallback obeys radical commands but invents a new
                    # set. Build a local target, then use the official first/last
                    # frame conditioning pattern so the real streamed tail remains
                    # a model keyframe throughout the transition.
                    request_dict["scene_bridge"] = True
                request_dict["negative_prompt"] = (
                    f"{request_dict.get('negative_prompt', '')}, unchanged subject, "
                    "ignored action, incomplete transformation"
                ).strip(", ")
                print(
                    f"🎯 Using COMMENT mode: attempt={comment_attempt_number}, "
                    f"guidance={comment_params.guidance_scale}, strength={comment_strength}"
                )
            elif prompt_result is not None and getattr(prompt_result, "forced_novelty", False):
                novelty_strength = float(os.getenv("LTX25_NOVELTY_STRENGTH", "0.78"))
                request_dict["strength"] = min(
                    float(request_dict["strength"]),
                    novelty_strength,
                )
                print(
                    f"🔀 Using ANTI-STALL mode: strength={request_dict['strength']} "
                    "to let the new beat visibly change the scene"
                )
            else:
                print(f"🌱 Using EVOLUTION mode: guidance={request_dict['guidance_scale']}, strength={request_dict['strength']}")

            request = LTXVideoRequestI2V(**request_dict)

            print(f"🎛️ LTX Request Parameters:")
            print(f"   📝 prompt: {request.prompt}")
            print(f"   📝 negative_prompt: {request.negative_prompt}")
            print(f"   📏 dimensions: {request.width}x{request.height}")
            print(f"   🎞️ num_frames: {request.num_frames}")
            print(f"   💪 strength: {request.strength}")
            print(f"   🎯 guidance_scale: {request.guidance_scale}")
            print(f"   ⏱️ timesteps: {request.timesteps}")

            generation_params = {
                "timestamp": time.time(),
                "generation_id": self.state.generation_count + 1,
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "width": request.width,
                "height": request.height,
                "num_frames": request.num_frames,
                "strength": request.strength,
                "guidance_scale": request.guidance_scale,
                "timesteps": request.timesteps,
                "force_t2v": request.force_t2v,
                "scene_bridge": request.scene_bridge,
                "preserve_scene": request.preserve_scene,
            }
            self.generation_params_history.append(generation_params)
            self.generation_params_history = self.generation_params_history[-10:]

            # Run video generation on the GPU thread.  The prompt task for
            # the NEXT clip is created by the outer _generation_loop AFTER
            # this returns and the state is updated, so it can use the just-
            # produced last frame as visual context.
            generation_start_time = time.time()
            if os.getenv("RUN_GENERATION_INLINE", "true").lower() == "true":
                video_result = self.realtime_generator.generate_video_from_image(request)
            else:
                video_result = await asyncio.to_thread(
                    self.realtime_generator.generate_video_from_image,
                    request
                )
            generation_duration = time.time() - generation_start_time
            cycle_duration = time.time() - cycle_start_time

            if used_comment and request.scene_bridge and video_result.frames:
                generation_log.info(
                    "🎞️ Local scene bridge final comment attempt: streamed tail used as "
                    "a true first-frame guide, with a scene-locked target at the final frame"
                )

            if border_guard_active and video_result.frames:
                repair_ratio = self._border_repair_ratio(handoff_frame)
                video_result.frames = self._progressive_full_bleed_crop(
                    video_result.frames,
                    ratio=repair_ratio,
                    settle_fraction=0.30,
                )
                self._border_repairs += 1
                print(
                    f"🔎 Border repair: applied {repair_ratio:.0%} early-settling "
                    "full-bleed crop"
                )

            # Reject a poisoned clip before it reaches either Twitch or the next
            # generation. Sampling the whole clip also catches gradual posterization.
            clip_is_corrupt = self._clip_is_corrupt(video_result.frames)
            recovery_segment = False
            if clip_is_corrupt and video_result.frames:
                repaired_frames, repaired = self._try_adaptive_clip_repair(video_result.frames)
                if repaired:
                    video_result.frames = repaired_frames
                    clip_is_corrupt = False
                    self._adaptive_repairs += 1
                    generation_log.info(
                        "🩹 Adaptive repair rescued the generated clip with an exact-seam push-in"
                    )

            if not clip_is_corrupt and video_result.frames:
                video_result.frames = self._stabilize_terminal_blur(video_result.frames)

            if clip_is_corrupt:
                self._consecutive_corrupt_rejections += 1
                max_retries = max(1, int(os.getenv("LTX25_MAX_CORRUPT_RETRIES", "2")))
                if prompt_result is not None:
                    # Keep story intent stable and avoid another OpenAI call while
                    # retrying from the same handoff.
                    self._retry_prompt_result = prompt_result

                if self._consecutive_corrupt_rejections >= max_retries and frame_for_this_gen:
                    poisoned_count = len(video_result.frames) or request.num_frames
                    recovery_source = self._base64_to_frame(frame_for_this_gen)
                    video_result.frames = self._make_local_recovery_clip(
                        recovery_source,
                        poisoned_count,
                    )
                    clip_is_corrupt = False
                    recovery_segment = True
                    self._recovery_segments += 1
                    self._rejected_clips += 1
                    generation_log.warning(
                        "🚑 Corrupt retry limit reached; replacing the poisoned clip "
                        "with a seamless LOCAL recovery push-in"
                    )
            # Viewer clips are audited before they can be captioned, streamed,
            # committed as the next I2V handoff, or written into story history.
            # This prevents the UI from claiming an action that never appeared and
            # stops a failed fish frame from teaching later prompts that a cat exists.
            if (
                selected_comment
                and not clip_is_corrupt
                and not recovery_segment
                and video_result.frames
                and hasattr(self.prompt_generator, "verify_comment_adherence")
            ):
                try:
                    before_frame = self._base64_to_frame(frame_for_this_gen)
                    comment_adherence = await asyncio.to_thread(
                        self.prompt_generator.verify_comment_adherence,
                        selected_comment,
                        before_frame,
                        video_result.frames,
                    )
                except Exception as exc:
                    comment_adherence = {
                        "satisfied": None,
                        "skipped": True,
                        "reason": str(exc)[:200],
                    }
                    generation_log.warning(f"⚠️ Comment preflight audit skipped: {exc}")
                self._last_comment_adherence = comment_adherence or {}
                generation_log.info(
                    "🎯 COMMENT PREFLIGHT AUDIT: "
                    f"satisfied={comment_adherence.get('satisfied')} "
                    f"progressing={comment_adherence.get('progressing')} "
                    f"missing={comment_adherence.get('missing', [])} "
                    f"summary={comment_adherence.get('summary', '')}"
                )

                comment_key = self._comment_key(selected_comment)
                if comment_key != self._active_comment_key:
                    self._active_comment_key = comment_key
                    self._comment_adherence_attempts = 0
                self._comment_adherence_attempts = comment_attempt_number
                if comment_adherence.get("satisfied") is not True:
                    self._comment_preflight_rejections += 1
                    self._rejected_clips += 1
                    max_attempts = max(
                        1,
                        int(os.getenv("COMMENT_MAX_ADHERENCE_ATTEMPTS", "3")),
                    )
                    missing = comment_adherence.get("missing") or [selected_comment.message]
                    if not isinstance(missing, list):
                        missing = [str(missing)]
                    missing_text = "; ".join(str(item) for item in missing)[:400]
                    if comment_attempt_number < max_attempts:
                        retry_prompt = (
                            f"{prompt_result.prompt} PRIOR ATTEMPT WAS NOT SHOWN. "
                            f"Correct only these missing requirements now: {missing_text}. "
                            "Make the requested result large, central, and unmistakable."
                        )
                        self._retry_prompt_result = type(prompt_result)(
                            selected_comment=selected_comment,
                            prompt=retry_prompt,
                            reasoning="Preflight kept the original viewer command active",
                            visual_description=getattr(prompt_result, "visual_description", ""),
                            scene_change_requested=bool(
                                getattr(prompt_result, "scene_change_requested", False)
                            ),
                        )
                        self._comment_adherence_retries += 1
                        generation_log.warning(
                            f"🛑 Suppressed failed viewer clip before RTMP; retry "
                            f"{comment_attempt_number + 1}/{max_attempts} will use a weaker "
                            f"image guide. Missing: {missing_text}"
                        )
                    else:
                        self._comment_adherence_failures += 1
                        self._active_comment_key = ""
                        self._comment_adherence_attempts = 0
                        generation_log.error(
                            f"❌ Viewer command failed {max_attempts} preflight attempts; "
                            "no misleading clip or subtitle was streamed, and story state "
                            "was not advanced"
                        )
                    return

            clip_committed = False

            # Stream frames to external streamer if available - USE BATCH PROCESSING
            # Check if still running before sending frames
            if not self.state.is_running:
                generation_log.info("🛑 Stopping detected - skipping frame streaming")
                return
                
            if self.rtmp_streamer and video_result.frames:
                if clip_is_corrupt:
                    generation_log.warning(
                        f"🛡️ Suppressing corrupt clip ({len(video_result.frames)} frames); "
                        "holding the last healthy frame"
                    )
                else:
                    if recovery_segment and selected_comment:
                        # A geometric liveness rescue is not the requested scene.
                        # Keep the command cached for the next generated attempt,
                        # but never burn its caption onto the recovery push-in.
                        self.text_overlay.set_comment("")
                        generation_log.info(
                            "🧹 Suppressed viewer caption on local recovery segment"
                        )
                    generation_log.info(f"📺 PROCESSING {len(video_result.frames)} frames with overlay...")

                    # Apply text overlay to all frames using batch processing
                    overlaid_frames = self.text_overlay.apply_overlay_batch(video_result.frames)

                    if selected_comment and hasattr(self.text_overlay, "get_status"):
                        overlay_status = self.text_overlay.get_status()
                        verified = bool(overlay_status.get("last_overlay_verified"))
                        log_method = generation_log.info if verified else generation_log.error
                        log_method(
                            f"💬 COMMENT OVERLAY {'VERIFIED' if verified else 'FAILED'}: "
                            f"{overlay_status.get('last_visible_frames', 0)}/{len(overlaid_frames)} "
                            f"source frames, @{selected_comment.username}: "
                            f"{selected_comment.message}"
                        )

                    generation_log.info(f"📺 SENDING {len(overlaid_frames)} frames to RTMP streamer...")
                    cycle_duration = time.time() - cycle_start_time
                    processed_count = self.rtmp_streamer.add_frame_batch(
                        overlaid_frames,
                        playback_seconds=cycle_duration,
                    )
                    generation_log.info(f"📺 RTMP processed: {processed_count}/{len(overlaid_frames)} frames")
                    clip_committed = processed_count >= len(overlaid_frames)

                    # Forward the matching audio chunk (no-op if audio disabled or
                    # if the model didn't produce audio this cycle).
                    audio_pcm = getattr(video_result, "audio_pcm", None)
                    if audio_pcm:
                        self.rtmp_streamer.add_audio_chunk(audio_pcm)

            elif not self.rtmp_streamer:
                generation_log.error("❌ NO FRAME STREAMER SET!")
            elif not video_result.frames:
                generation_log.error("❌ NO FRAMES IN VIDEO RESULT!")
            else:
                generation_log.error("❌ Unknown frame streaming issue")

            # Transaction boundary: a rejected/unaccepted clip must not change either
            # the I2V handoff frame or the story history.
            if not clip_committed:
                if prompt_result is not None:
                    self._retry_prompt_result = prompt_result
                self._rejected_clips += 1
                generation_log.warning(
                    f"↩️ Clip not committed (rejected total={self._rejected_clips}); "
                    "retrying from the same streamed handoff and story state"
                )
                return
            
            # Update state with last frame (extract from frames on-demand)
            # Check if still running before updating state
            if not self.state.is_running:
                generation_log.info("🛑 Stopping detected - skipping state update")
                return
                
            # Extract last frame as base64 only when needed
            if video_result.frames:
                last_pil = video_result.frames[-1]
                last_frame_base64 = self._frame_to_base64(last_pil)
                self._last_good_frame_base64 = last_frame_base64
                self._committed_handoff_sha256 = self._frame_digest(last_pil)
                self._persist_committed_handoff(last_pil)
            else:
                generation_log.error("❌ No frames in video result for state update")
                return

            print(f"🔄 Updating state with new frame from generation #{self.state.generation_count + 1}")
            self.state.current_frame_base64 = last_frame_base64
            self.state.generation_count += 1
            self._consecutive_corrupt_rejections = 0
            if not recovery_segment:
                self.state.current_prompt = prompt_to_use
                self.state.previous_prompts.append(prompt_to_use)

            if selected_comment and comment_adherence is not None:
                if comment_adherence.get("satisfied") is True:
                    self._comment_adherence_successes += 1
                self._active_comment_key = ""
                self._comment_adherence_attempts = 0
            generation_log.info(
                f"🔗 Committed streamed tail as next I2V first frame: "
                f"{self._committed_handoff_sha256[:12]}"
            )
            if recovery_segment:
                generation_log.info(
                    "🚑 Local recovery committed without advancing story history; "
                    "the same prompt will continue from its repaired tail"
                )
            
            generation_log.info(f"✅ Generated video #{self.state.generation_count}")
            # No delay - immediately ready for next generation!
            
        except Exception as e:
            generation_log.error(f"❌ Video generation failed: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get video generation orchestration status"""
        return {
            "is_running": self.state.is_running,
            "generation_count": self.state.generation_count,
            "current_prompt": self.state.current_prompt[:50] + "..." if len(self.state.current_prompt) > 50 else self.state.current_prompt,
            "generation_params_history": self.generation_params_history,
            "rejected_clips": self._rejected_clips,
            "committed_handoff_sha256": self._committed_handoff_sha256,
            "handoff_snapshot_ready": bool(self._handoff_snapshot_path),
            "quality_reference": self._quality_reference,
            "border_guard_activations": self._border_guard_activations,
            "border_repairs": self._border_repairs,
            "adaptive_repairs": self._adaptive_repairs,
            "recovery_segments": self._recovery_segments,
            "consecutive_corrupt_rejections": self._consecutive_corrupt_rejections,
            "prompt_retry_cached": self._retry_prompt_result is not None,
            "active_comment": bool(self._active_comment_key),
            "comment_adherence_attempts": self._comment_adherence_attempts,
            "comment_adherence_retries": self._comment_adherence_retries,
            "comment_adherence_failures": self._comment_adherence_failures,
            "comment_adherence_successes": self._comment_adherence_successes,
            "comment_preflight_rejections": self._comment_preflight_rejections,
            "last_comment_adherence": self._last_comment_adherence,
            "terminal_blur_trims": self._terminal_blur_trims,
            "queue_backpressure_waits": self._queue_backpressure_waits,
            "queue_backpressure_seconds": round(self._queue_backpressure_seconds, 1),
        }




