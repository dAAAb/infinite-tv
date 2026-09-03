from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageOps

from streaming_pipeline.models import LTXVideoRequestI2V
from streaming_pipeline.video_generation.video_generator import RealtimeGenerator


@dataclass
class Segment:
    name: str
    prompt: str
    frames: int
    seed: int


def image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue().hex()


def image_to_b64(image: Image.Image) -> str:
    import base64

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_start_image(path: Path, width: int, height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.fit(
        image,
        (width, height),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


def write_video(frames: list[Image.Image], path: Path, fps: float) -> None:
    import imageio.v3 as iio

    iio.imwrite(path, [frame.convert("RGB") for frame in frames], fps=fps)


def write_wav(audio_pcm: bytes, path: Path, sample_rate: int = 44100) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(audio_pcm)


def mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ],
        check=True,
    )


def concat_mp4(parts: list[Path], output_path: Path) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        list_path = Path(handle.name)
        for part in parts:
            safe_path = str(part).replace("\\", "/").replace("'", "'\\''")
            handle.write(f"file '{safe_path}'\n")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                str(output_path),
            ],
            check=True,
        )
    finally:
        list_path.unlink(missing_ok=True)


def reencode_video(input_path: Path, output_path: Path, video_bitrate: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-c:v",
            "libx264",
            "-b:v",
            video_bitrate,
            "-maxrate",
            video_bitrate,
            "-bufsize",
            video_bitrate,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            str(output_path),
        ],
        check=True,
    )


def load_segments(path: Path, default_frames: int, seed: int) -> list[Segment]:
    raw_segments = json.loads(path.read_text(encoding="utf-8"))
    segments = []
    for index, item in enumerate(raw_segments):
        segments.append(
            Segment(
                name=item.get("name", f"segment-{index + 1}"),
                prompt=item["prompt"],
                frames=int(item.get("frames", default_frames)),
                seed=int(item.get("seed", seed + index)),
            )
        )
    return segments


def write_segment_clip(
    frames: list[Image.Image],
    audio_pcm: bytes | None,
    output_path: Path,
    fps: float,
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_video = Path(temp_dir) / "video.mp4"
        write_video(frames, temp_video, fps)
        if audio_pcm:
            temp_audio = Path(temp_dir) / "audio.wav"
            write_wav(audio_pcm, temp_audio)
            mux_audio(temp_video, temp_audio, output_path)
        else:
            temp_video.replace(output_path)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Generate an offline multi-prompt timeline clip.")
    parser.add_argument("--timeline-file", required=True)
    parser.add_argument("--input-image", required=True)
    parser.add_argument("--output", default="outputs/offline-timeline.mp4")
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--frames", type=int, default=25, help="Default per-segment frames; must be 8*k+1 for LTX.")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=0.12)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--video-bitrate", default=None)
    parser.add_argument("--seed", type=int, default=5092)
    args = parser.parse_args()

    load_dotenv()
    os.environ["LOAD_LTX23_PIPELINE"] = "true"
    os.environ.setdefault("LTX23_CPU_OFFLOAD", "false")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segments_dir = output_path.with_suffix("")
    segments_dir.mkdir(parents=True, exist_ok=True)

    segments = load_segments(Path(args.timeline_file), args.frames, args.seed)
    start_image = load_start_image(Path(args.input_image), args.width, args.height)
    start_image.save(output_path.with_suffix(".start.jpg"), quality=92)

    generator = RealtimeGenerator(load_ltx23_pipeline=True)
    generator.setup()

    part_paths: list[Path] = []
    current_image = start_image
    total_started_at = time.time()

    for index, segment in enumerate(segments, start=1):
        print(f"=== Segment {index}/{len(segments)}: {segment.name} ===", flush=True)
        request = LTXVideoRequestI2V(
            model_type="ltx-2.3-local",
            prompt=segment.prompt,
            image_base64=image_to_b64(current_image),
            width=args.width,
            height=args.height,
            num_frames=segment.frames,
            frame_rate=args.fps,
            guidance_scale=args.guidance,
            noise_scale=args.noise,
            num_inference_steps=args.steps,
            seed=segment.seed,
        )
        started_at = time.time()
        result = generator.generate_video_from_image(request)
        elapsed = time.time() - started_at
        if not result.frames:
            raise RuntimeError(f"Segment returned no frames: {segment.name}")

        part_path = segments_dir / f"{index:02d}-{segment.name}.mp4"
        write_segment_clip(result.frames, result.audio_pcm, part_path, args.fps)
        result.frames[-1].save(segments_dir / f"{index:02d}-{segment.name}.last.jpg", quality=92)
        part_paths.append(part_path)
        current_image = result.frames[-1].convert("RGB")
        print(f"Generated {len(result.frames)} frames in {elapsed:.2f}s -> {part_path}", flush=True)

    if args.video_bitrate:
        with tempfile.TemporaryDirectory() as temp_dir:
            concat_path = Path(temp_dir) / "concat.mp4"
            concat_mp4(part_paths, concat_path)
            reencode_video(concat_path, output_path, args.video_bitrate)
    else:
        concat_mp4(part_paths, output_path)
    print(f"Generated {len(part_paths)} segments in {time.time() - total_started_at:.2f}s")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
