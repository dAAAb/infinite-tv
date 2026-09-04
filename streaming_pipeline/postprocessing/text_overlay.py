from PIL import Image, ImageChops, ImageDraw, ImageFont
from typing import Optional, Dict, Any, List
import time
from streaming_pipeline.models import Monitorable


class TextOverlay(Monitorable):
    """
    Handles text overlay rendering for video frames.
    
    Separated from streaming logic for better separation of concerns.
    """
    
    # Common system font search path; first hit wins.
    # Windows CJK fonts listed FIRST so 中文 / 日文 chat renders correctly on
    # local (Windows) deployments — DejaVuSans has no CJK glyphs and would
    # otherwise draw tofu boxes for Han characters.
    _FONT_CANDIDATES = (
        # Windows — Traditional Chinese (JhengHei UI Bold, covers Trad + Simp + JP kana)
        "C:/Windows/Fonts/msjh.ttc",
        # Windows — Simplified Chinese (YaHei UI, wider coverage)
        "C:/Windows/Fonts/msyh.ttc",
        # Windows — MingLiU-ExtB (very wide Han coverage, fallback)
        "C:/Windows/Fonts/mingliub.ttc",
        # macOS system CJK (Traditional Chinese)
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        # Linux — Noto Sans CJK (best cross-platform CJK if installed)
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        # macOS Latin fallback
        "/System/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        # Debian/Ubuntu Latin fallback (no CJK — used only if none above hit)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    )

    def __init__(self, width: int, height: int):
        # `width` / `height` are now just hints used when no frame is available
        # (e.g. for the initial font cache).  All actual overlay sizing reads
        # from the real frame inside apply_overlay so output scales when the
        # user picks a different stream resolution mid-session.
        self.width = width
        self.height = height

        # Current overlay state
        self.current_text = None
        self.current_kind = None

        # Font cache keyed by pixel size so we don't reload TTF on every frame.
        self._font_cache: Dict[int, Any] = {}

        # Pre-rendered overlay bitmap cache.  Rebuilt only when the text or
        # the target frame size changes, so per-frame work drops from ~9
        # truetype rasterizations to a single alpha paste.
        self._overlay_cache: Optional[Image.Image] = None
        self._overlay_y: int = 0
        self._cache_key: Optional[tuple] = None  # (text, frame_w, frame_h)

        # Performance tracking for monitoring
        self.total_frames_processed = 0
        self.total_processing_time = 0.0
        self.last_batch_size = 0
        self.last_batch_time = 0.0
        self.last_text_rendered = None
        self.last_kind_rendered = None
        self.last_visible_frames = 0
        self.last_overlay_verified = False

    def set_comment(self, comment_text: str, username: str = None):
        """Set comment to overlay on frames"""
        if comment_text:
            self.current_text = f"@{username}: {comment_text}" if username else comment_text
            self.current_kind = "comment"
        else:
            self.current_text = None
            self.current_kind = None
        self._invalidate_cache()

    def set_prompt(self, prompt_text: str):
        """Set AI prompt to overlay on frames"""
        if prompt_text:
            self.current_text = f"AI: {prompt_text}"
            self.current_kind = "prompt"
        else:
            self.current_text = None
            self.current_kind = None
        self._invalidate_cache()

    def _invalidate_cache(self):
        """Drop the cached overlay bitmap; the next frame will rebuild it."""
        self._overlay_cache = None
        self._cache_key = None

    def _get_font(self, font_size: int):
        """Return a cached truetype font at the given pixel size."""
        cached = self._font_cache.get(font_size)
        if cached is not None:
            return cached
        for path in self._FONT_CANDIDATES:
            try:
                font = ImageFont.truetype(path, font_size)
                self._font_cache[font_size] = font
                return font
            except (OSError, IOError):
                continue
        # Last resort: PIL's bitmap default font (does not honor size, but draws).
        try:
            font = ImageFont.load_default()
            self._font_cache[font_size] = font
            return font
        except Exception:
            self._font_cache[font_size] = None
            return None

    @staticmethod
    def _text_pixel_width(font, text: str) -> int:
        """Measure the pixel width of `text` rendered with `font`.

        Uses Pillow's modern API (`getlength`) when available; falls back to
        `getbbox` and finally to character-count estimation for the default
        bitmap font (which exposes neither).
        """
        try:
            return int(font.getlength(text))
        except (AttributeError, TypeError):
            pass
        try:
            l, _, r, _ = font.getbbox(text)
            return int(r - l)
        except (AttributeError, TypeError):
            return len(text) * 6  # crude fallback for bitmap default font

    def _wrap_text_to_width(self, text: str, font, max_width: int) -> list[str]:
        """Greedy word-wrap `text` so each line fits within `max_width` pixels.

        Words longer than `max_width` are placed on their own line and clipped
        by the renderer; the alternative (mid-word break) tends to produce
        worse-looking output for chat overlays.
        """
        words = text.split()
        if not words:
            return [text]

        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if self._text_pixel_width(font, candidate) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def _build_overlay(self, text: str, frame_w: int, frame_h: int):
        """Render the text + outline into a small RGBA bitmap once.

        Returns (overlay_image, paste_y).  The overlay bitmap is sized to fit
        as many wrapped lines as `text` requires, capped to `frame_h // 2`.
        """
        # Initial sizing: ~5% of frame width, clamped to a usable range.
        font_size = max(12, min(48, frame_w // 20))
        padding = max(6, frame_w // 40)
        max_text_width = max(1, frame_w - 2 * padding)

        # If even the longest single word doesn't fit, shrink the font until
        # it does (down to a 10px floor).  This prevents the rendered text
        # from spilling past the right edge on very narrow frames.
        font = self._get_font(font_size)
        longest_word = max(text.split() or [text], key=len)
        while font_size > 10 and self._text_pixel_width(font, longest_word) > max_text_width:
            font_size -= 1
            font = self._get_font(font_size)

        # Word-wrap the (possibly shrunk) text and compute layout.
        lines = self._wrap_text_to_width(text, font, max_text_width)
        line_height = int(font_size * 1.2)
        text_block_h = line_height * len(lines)

        # Cap the overlay at half the frame height so it never covers the
        # whole picture if the user pastes a paragraph.
        max_overlay_h = max(line_height + 2 * padding, frame_h // 2)
        overlay_h = min(text_block_h + 2 * padding, max_overlay_h)
        # If we capped, drop the trailing lines that no longer fit (visual
        # truncation; keeps the overlay readable instead of overflowing).
        max_lines = max(1, (overlay_h - 2 * padding) // line_height)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            text_block_h = line_height * len(lines)
            overlay_h = text_block_h + 2 * padding

        # Sit the text just inside the bottom edge.
        paste_y = max(0, frame_h - overlay_h)

        overlay = Image.new("RGBA", (frame_w, overlay_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Render each wrapped line.  Single PIL call per line with native
        # 1-pixel outline (Pillow >= 8.0); falls back to a manual 8-pass
        # outline if the build is older.
        for i, line in enumerate(lines):
            ly = padding + i * line_height
            try:
                draw.text(
                    (padding, ly),
                    line,
                    font=font,
                    fill=(255, 255, 255, 255),
                    stroke_width=1,
                    stroke_fill=(0, 0, 0, 255),
                )
            except TypeError:
                for adj_x in (-1, 0, 1):
                    for adj_y in (-1, 0, 1):
                        if adj_x or adj_y:
                            draw.text((padding + adj_x, ly + adj_y), line, font=font, fill=(0, 0, 0, 255))
                draw.text((padding, ly), line, font=font, fill=(255, 255, 255, 255))

        return overlay, paste_y

    def apply_overlay(self, frame: Image.Image) -> Image.Image:
        """Apply text overlay to frame.  Mutates `frame` in place and returns it.

        Uses a cached overlay bitmap keyed by (text, frame_w, frame_h), so the
        truetype rasterization happens once per text-change, not per frame.
        """
        if not self.current_text:
            return frame

        frame_w, frame_h = frame.size
        key = (self.current_text, frame_w, frame_h)
        if key != self._cache_key or self._overlay_cache is None:
            self._overlay_cache, self._overlay_y = self._build_overlay(self.current_text, frame_w, frame_h)
            self._cache_key = key

        # Alpha-paste the cached overlay onto the frame in place.  No copy:
        # the caller (RealtimeVideoStreamer) owns these frames and discards
        # them after streaming, so mutation is safe.
        frame.paste(self._overlay_cache, (0, self._overlay_y), self._overlay_cache)
        return frame
    
    # Fraction of a batch's frames that show the caption (the rest are clean,
    # creating a visual gap before the next caption arrives).
    CAPTION_VISIBLE_RATIO = 0.4
    # Viewer commands are the interactive part of the show. Keep them on screen
    # much longer than autonomous AI prompts so a queued Twitch segment cannot
    # flash the requested comment too briefly to notice.
    COMMENT_VISIBLE_RATIO = 0.85
    # Always leave a clean tail. The generator chains from the raw tail, and the
    # visible stream should also settle back to that clean image before the join.
    MIN_CLEAN_TAIL_FRAMES = 1
    # Number of frames over which the caption fades out at the end of the
    # visible window.  Keeps the transition smooth instead of a hard cut.
    FADE_OUT_FRAMES = 6

    def apply_overlay_batch(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Apply overlay to the first portion of a batch, with a fade-out.

        Only the first ~40% of frames carry the caption; the remainder play
        clean.  This creates a natural pause before the next generation's
        caption appears, preventing the "clashing captions" look.
        """
        if not frames:
            return frames

        start_time = time.time()

        n = len(frames)
        visible_ratio = (
            self.COMMENT_VISIBLE_RATIO
            if self.current_kind == "comment"
            else self.CAPTION_VISIBLE_RATIO
        )
        clean_tail = max(self.MIN_CLEAN_TAIL_FRAMES, n // 10)
        visible_end = max(1, min(n - clean_tail, int(n * visible_ratio)))
        fade_start = max(0, visible_end - self.FADE_OUT_FRAMES)

        # Never mutate the generator-owned frames. The raw final frame is the
        # transactional I2V handoff; captions exist only on the RTMP copies.
        overlaid_frames = list(frames)
        overlay_verified = False

        for i in range(visible_end):
            target = frames[i].copy()
            if i < fade_start:
                self.apply_overlay(target)
            else:
                # Fade-out region: paste with decreasing alpha.
                self._apply_overlay_with_alpha(
                    target,
                    alpha=(visible_end - 1 - i) / max(1, visible_end - fade_start),
                )
            if not overlay_verified:
                overlay_verified = ImageChops.difference(
                    frames[i].convert("RGB"), target.convert("RGB")
                ).getbbox() is not None
            overlaid_frames[i] = target

        self.last_batch_time = time.time() - start_time
        self.last_batch_size = n
        self.total_frames_processed += n
        self.total_processing_time += self.last_batch_time
        self.last_text_rendered = self.current_text
        self.last_kind_rendered = self.current_kind
        self.last_visible_frames = visible_end
        self.last_overlay_verified = overlay_verified

        return overlaid_frames

    def _apply_overlay_with_alpha(self, frame: Image.Image, alpha: float) -> Image.Image:
        """Like apply_overlay but blends the cached bitmap at reduced opacity."""
        if not self.current_text or alpha <= 0:
            return frame

        frame_w, frame_h = frame.size
        key = (self.current_text, frame_w, frame_h)
        if key != self._cache_key or self._overlay_cache is None:
            self._overlay_cache, self._overlay_y = self._build_overlay(self.current_text, frame_w, frame_h)
            self._cache_key = key

        if alpha >= 1.0:
            frame.paste(self._overlay_cache, (0, self._overlay_y), self._overlay_cache)
        else:
            # Scale the alpha channel of the cached overlay for the fade.
            faded = self._overlay_cache.copy()
            a = faded.split()[3]
            a = a.point(lambda p: int(p * alpha))
            faded.putalpha(a)
            frame.paste(faded, (0, self._overlay_y), faded)

        return frame
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.total_frames_processed = 0
        self.total_processing_time = 0.0
        self.last_batch_size = 0
        self.last_batch_time = 0.0
        self.current_text = None  # Clear overlay text too
        self.current_kind = None
        self.last_text_rendered = None
        self.last_kind_rendered = None
        self.last_visible_frames = 0
        self.last_overlay_verified = False
        self._invalidate_cache()  # Drop the rendered overlay bitmap as well
        # Keep _font_cache - no need to reload TTF
        print("🧹 Text overlay metrics reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get text overlay performance metrics"""
        avg_time_per_frame = self.total_processing_time / max(1, self.total_frames_processed)
        # Calculate average time per frame for last batch
        last_avg_time = self.last_batch_time / max(1, self.last_batch_size) if self.last_batch_size > 0 else 0
        
        return {
            "frames_processed": self.total_frames_processed,
            "avg_time_per_frame": round(avg_time_per_frame, 4),
            "last_batch_size": self.last_batch_size,
            "last_batch_time": round(self.last_batch_time, 3),
            "last_batch_avg_per_frame": round(last_avg_time, 4),
            "has_overlay": self.current_text is not None,
            "current_kind": self.current_kind,
            "last_text_rendered": self.last_text_rendered,
            "last_kind_rendered": self.last_kind_rendered,
            "last_visible_frames": self.last_visible_frames,
            "last_overlay_verified": self.last_overlay_verified,
        }
