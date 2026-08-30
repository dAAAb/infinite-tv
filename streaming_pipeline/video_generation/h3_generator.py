"""H3 Max backend for the realtime stream.

Calls `minimax/h3-max/image-to-video` through fal_client (no local weights,
no GPU), downloads the finished mp4 from the CDN and decodes it into the
(PIL frames, 44.1 kHz stereo s16le PCM) shape the rest of the pipeline
already consumes. Ported from rehan-remade/h3-realtime-stream's
h3_generator and adapted to this repo's synchronous VideoGenerator
interface (LTXVideoRequestI2V in, LTXVideoResponseWithFrames out).

H3 renders 24 fps clips with native audio. Frames are scaled to the stream
resolution at decode time; audio is resampled to 44.1 kHz and trimmed or
padded so its playback duration is exactly frame_count / frame_rate — the
RTMP streamer paces frames at frame_rate, so this is what keeps A/V locked.
Run the stream at target_fps=24 so clips play at their native speed.
"""

import os
import shutil
import subprocess
import tempfile
import time
from typing import List, Optional

import fal_client
import httpx
import numpy as np
from PIL import Image

from streaming_pipeline.models import LTXVideoRequestI2V, LTXVideoResponseWithFrames

H3_ENDPOINT = "minimax/h3-max/image-to-video"
H3_NATIVE_FPS = 24.0
AUDIO_SAMPLE_RATE = 44100  # matches LTXVideoResponseWithFrames.audio_pcm contract
AUDIO_CHANNELS = 2

_DOWNLOAD_TIMEOUT = httpx.Timeout(180.0, connect=30.0)
_DOWNLOAD_CHUNK = 1 << 16

_ffmpeg_path: Optional[str] = None


def _ffmpeg_exe() -> str:
    """Locate an ffmpeg binary: PATH first, then the imageio-ffmpeg bundle."""
    global _ffmpeg_path
    if _ffmpeg_path is None:
        found = shutil.which("ffmpeg")
        if found is None:
            from imageio_ffmpeg import get_ffmpeg_exe

            found = get_ffmpeg_exe()
        _ffmpeg_path = found
    return _ffmpeg_path


def _decode_video_frames(mp4_path: str, width: int, height: int) -> List[Image.Image]:
    """Decode the mp4 to a list of PIL frames scaled to the stream resolution."""
    cmd = [
        _ffmpeg_exe(),
        "-loglevel", "error",
        "-i", mp4_path,
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        # Fill-and-crop instead of a forced scale: if the clip's aspect ratio
        # differs from the stream's (e.g. the first clip inherits a non-16:9
        # initial image), stretching would distort - cover and crop instead.
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}",
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg video decode failed (rc={proc.returncode}): {stderr[-500:]}")

    frame_bytes = width * height * 3
    frame_count = len(proc.stdout) // frame_bytes
    if frame_count == 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg produced 0 frames from {mp4_path}: {stderr[-500:]}")

    raw = np.frombuffer(proc.stdout, dtype=np.uint8, count=frame_count * frame_bytes)
    arrays = raw.reshape(frame_count, height, width, 3)
    return [Image.fromarray(arrays[i]) for i in range(frame_count)]


def _decode_audio_samples(mp4_path: str) -> Optional[np.ndarray]:
    """Decode + resample audio to (S, 2) int16 at 44.1 kHz; None on failure.

    A missing audio track makes ffmpeg exit non-zero — reported as None so
    the caller substitutes silence rather than failing the generation.
    """
    cmd = [
        _ffmpeg_exe(),
        "-loglevel", "error",
        "-i", mp4_path,
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-ac", str(AUDIO_CHANNELS),
        "-",
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        return None
    if proc.returncode != 0 or len(proc.stdout) < 2 * AUDIO_CHANNELS:
        return None

    samples = np.frombuffer(proc.stdout, dtype=np.int16)
    usable = (samples.shape[0] // AUDIO_CHANNELS) * AUDIO_CHANNELS
    if usable == 0:
        return None
    return samples[:usable].reshape(-1, AUDIO_CHANNELS).copy()


def _fit_audio(audio: Optional[np.ndarray], target_samples: int) -> np.ndarray:
    """Trim/pad audio to exactly target_samples so A/V durations match."""
    if audio is None:
        return np.zeros((target_samples, AUDIO_CHANNELS), dtype=np.int16)
    if audio.shape[0] > target_samples:
        return np.ascontiguousarray(audio[:target_samples])
    if audio.shape[0] < target_samples:
        pad = np.zeros((target_samples - audio.shape[0], AUDIO_CHANNELS), dtype=np.int16)
        return np.concatenate([audio, pad], axis=0)
    return audio


def generate_h3_clip(request: LTXVideoRequestI2V) -> LTXVideoResponseWithFrames:
    """Generate one H3 Max clip and decode it into frames + PCM.

    Blocking (API call + download + two ffmpeg decodes) — the engine already
    runs generation via asyncio.to_thread, same as every other backend.
    """
    duration = int(request.h3_duration or 15)
    resolution = request.h3_resolution or "480P"
    expansion = request.h3_prompt_expansion_mode or "balanced"

    print(f"🎬 Starting H3 Max generation ({H3_ENDPOINT})")
    print(f"   Prompt: {request.prompt[:80]}...")
    print(f"   Duration: {duration}s · Resolution: {resolution} · Expansion: {expansion}")
    print(f"🔑 FAL_KEY set: {bool(os.getenv('FAL_KEY'))}")

    arguments = {
        "prompt": request.prompt,
        "duration": duration,
        "resolution": resolution,
        "enable_safety_checker": True,
        "prompt_expansion_mode": expansion,
        "sync_mode": False,
    }
    if expansion == "disabled":
        # The facade pins prompt_expansion_mode; this flag actually turns it off.
        arguments["enable_prompt_expansion"] = False
    if request.seed is not None:
        arguments["seed"] = request.seed

    # Last-frame conditioning: pass the previous clip's final frame as a data
    # URI (no upload roundtrip). Without an image the endpoint runs t2v.
    if request.image_base64:
        image_data = request.image_base64
        if not image_data.startswith("data:image"):
            image_data = f"data:image/jpeg;base64,{image_data}"
        arguments["image_url"] = image_data

    start_time = time.time()
    result = fal_client.subscribe(H3_ENDPOINT, arguments=arguments, with_logs=True)

    video_info = result.get("video") or {}
    video_url = video_info.get("url")
    if not video_url:
        raise RuntimeError(f"H3 result has no video url: {list(result.keys())}")
    expanded = result.get("expanded_prompt")
    if expanded:
        print(f"✍️ Expanded prompt: {expanded[:120]}...")
    print(f"📹 Video URL: {video_url[:80]}...")

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        with httpx.Client(timeout=_DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
            with client.stream("GET", video_url) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(_DOWNLOAD_CHUNK):
                    tmp.write(chunk)
        tmp.close()

        frames = _decode_video_frames(tmp.name, request.width, request.height)
        audio = _decode_audio_samples(tmp.name)

        # Fit audio to the clip's PLAYBACK duration on this stream. The RTMP
        # streamer paces frames at frame_rate (= target_fps), so audio must
        # cover frame_count / frame_rate seconds exactly, or A/V drifts a
        # little more every clip.
        frame_rate = float(request.frame_rate or H3_NATIVE_FPS)
        target_samples = round(len(frames) * AUDIO_SAMPLE_RATE / frame_rate)
        audio_pcm = _fit_audio(audio, target_samples).tobytes()

        elapsed = time.time() - start_time
        clip_seconds = len(frames) / frame_rate
        print(
            f"✅ H3 clip ready in {elapsed:.1f}s: {len(frames)} frames "
            f"({clip_seconds:.1f}s at {frame_rate:.0f} fps), "
            f"audio {'native' if audio is not None else 'silent (no track)'} "
            f"· realtime factor {clip_seconds / elapsed:.2f}x"
        )

        return LTXVideoResponseWithFrames(frames=frames, audio_pcm=audio_pcm)
    finally:
        tmp.close()
        os.unlink(tmp.name)
