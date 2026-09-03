from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
import tempfile
import time
import wave
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageOps

from streaming_pipeline.models import LTXVideoRequestI2V
from streaming_pipeline.video_generation.video_generator import RealtimeGenerator


def image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def make_start_image(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), (18, 20, 24))
    draw = ImageDraw.Draw(image)

    for y in range(height):
        ratio = y / max(1, height - 1)
        color = (
            int(18 + 28 * ratio),
            int(20 + 36 * ratio),
            int(24 + 58 * ratio),
        )
        draw.line([(0, y), (width, y)], fill=color)

    draw.ellipse(
        (width * 0.58, height * 0.16, width * 0.9, height * 0.58),
        fill=(255, 186, 76),
        outline=(255, 224, 156),
        width=3,
    )
    draw.rectangle(
        (0, int(height * 0.66), width, height),
        fill=(20, 28, 31),
    )
    draw.line(
        [(0, int(height * 0.66)), (width, int(height * 0.66))],
        fill=(84, 194, 170),
        width=2,
    )
    draw.text((24, 24), "Infinite TV offline test", fill=(238, 244, 242))
    draw.text((24, 48), "RTX 5090 local LTX 2.3", fill=(156, 220, 206))
    return image


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

    arrays = []
    for frame in frames:
        arrays.append(frame.convert("RGB"))
    iio.imwrite(path, arrays, fps=fps)


def write_wav(audio_pcm: bytes, path: Path, sample_rate: int = 44100) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(audio_pcm)


def mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    command = [
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
    ]
    subprocess.run(command, check=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Generate one offline Infinite TV clip locally.")
    parser.add_argument("--prompt", default="A cinematic neon Taipei skyline at dawn, gentle camera movement, reflective streets, optimistic sci-fi mood")
    parser.add_argument("--prompt-file", default=None, help="Read the prompt from a UTF-8 text file.")
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--frames", type=int, default=17, help="Must be 8*k+1 for LTX.")
    parser.add_argument("--fps", type=float, default=9.0)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=5090)
    parser.add_argument("--output", default="outputs/offline-test.mp4")
    parser.add_argument("--input-image", default=None, help="Use an existing image as the first I2V frame.")
    args = parser.parse_args()

    load_dotenv()
    os.environ["LOAD_LTX23_PIPELINE"] = "true"
    os.environ.setdefault("LTX23_CPU_OFFLOAD", "false")

    if args.prompt_file:
        args.prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.input_image:
        start_image = load_start_image(Path(args.input_image), args.width, args.height)
    else:
        start_image = make_start_image(args.width, args.height)
    start_image_path = output_path.with_suffix(".start.jpg")
    start_image.save(start_image_path, quality=92)

    generator = RealtimeGenerator(load_ltx23_pipeline=True)
    generator.setup()

    request = LTXVideoRequestI2V(
        model_type="ltx-2.3-local",
        prompt=args.prompt,
        image_base64=image_to_base64(start_image),
        width=args.width,
        height=args.height,
        num_frames=args.frames,
        frame_rate=args.fps,
        guidance_scale=args.guidance,
        noise_scale=args.noise,
        seed=args.seed,
    )

    started_at = time.time()
    result = generator.generate_video_from_image(request)
    elapsed = time.time() - started_at

    if not result.frames:
        raise RuntimeError("Generation returned no frames.")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_video = Path(temp_dir) / "video.mp4"
        write_video(result.frames, temp_video, args.fps)

        if result.audio_pcm:
            temp_audio = Path(temp_dir) / "audio.wav"
            write_wav(result.audio_pcm, temp_audio)
            mux_audio(temp_video, temp_audio, output_path)
        else:
            temp_video.replace(output_path)

    print(f"Generated {len(result.frames)} frames in {elapsed:.2f}s")
    print(f"Start image: {start_image_path}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
