from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Dict, Any, List
import time
from streaming_pipeline.models import Monitorable


class TextOverlay(Monitorable):
    """
    Handles text overlay rendering for video frames.
    
    Separated from streaming logic for better separation of concerns.
    """
    
    # Common system font search path; first hit wins.
    _FONT_CANDIDATES = (
        "/System/Library/Fonts/Arial.ttf",          # macOS
        "/System/Library/Fonts/Helvetica.ttc",      # macOS fallback
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Debian/Ubuntu (fal runners)
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

        # Font cache keyed by pixel size so we don't reload TTF on every frame.
        self._font_cache: Dict[int, Any] = {}

        # Performance tracking for monitoring
        self.total_frames_processed = 0
        self.total_processing_time = 0.0
        self.last_batch_size = 0
        self.last_batch_time = 0.0

    def set_comment(self, comment_text: str, username: str = None):
        """Set comment to overlay on frames"""
        if comment_text:
            self.current_text = f"@{username}: {comment_text}" if username else comment_text
        else:
            self.current_text = None

    def set_prompt(self, prompt_text: str):
        """Set AI prompt to overlay on frames"""
        if prompt_text:
            self.current_text = f"AI: {prompt_text}"
        else:
            self.current_text = None

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

    def apply_overlay(self, frame: Image.Image) -> Image.Image:
        """Apply text overlay to frame; scales font + position to actual frame size."""
        if not self.current_text:
            return frame

        frame_w, frame_h = frame.size

        # Scale font and padding off the actual frame so smaller resolutions
        # still produce a visible, on-screen overlay.
        # ~5% of frame width, clamped to a usable range.
        font_size = max(12, min(48, frame_w // 20))
        padding = max(6, frame_w // 40)
        # Rough text height estimate (truetype line height ~= 1.2 * point size).
        text_block_h = int(font_size * 1.2)

        text_x = padding
        # Sit the text just inside the bottom edge.
        text_y = max(0, frame_h - text_block_h - padding)

        font = self._get_font(font_size)

        overlay_frame = frame.copy()
        draw = ImageDraw.Draw(overlay_frame)

        # Black 1-pixel outline for readability over any background.
        for adj_x in (-1, 0, 1):
            for adj_y in (-1, 0, 1):
                if adj_x or adj_y:
                    draw.text((text_x + adj_x, text_y + adj_y), self.current_text, font=font, fill=(0, 0, 0))

        # White fill on top.
        draw.text((text_x, text_y), self.current_text, font=font, fill=(255, 255, 255))

        return overlay_frame
    
    def apply_overlay_batch(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Apply overlay to multiple frames with performance tracking"""
        if not frames:
            return frames
        
        start_time = time.time()
        
        overlaid_frames = []
        for frame in frames:
            overlaid_frame = self.apply_overlay(frame)
            overlaid_frames.append(overlaid_frame)
        
        # Track performance
        self.last_batch_time = time.time() - start_time
        self.last_batch_size = len(frames)
        self.total_frames_processed += len(frames)
        self.total_processing_time += self.last_batch_time
        
        return overlaid_frames
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.total_frames_processed = 0
        self.total_processing_time = 0.0
        self.last_batch_size = 0
        self.last_batch_time = 0.0
        self.current_text = None  # Clear overlay text too
        # Keep cached_font - no need to reload it
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
            "has_overlay": self.current_text is not None
        }
