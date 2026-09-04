"""Start a local-first LTX 2.5 stream through the running backend.

Example:
    python scripts/start-ltx25-stream.py --image .\start.png --output-mode rtmp

This script never reads or prints API keys. The backend loads secrets from its
local, gitignored .env file.
"""
from __future__ import annotations

import argparse
import base64
from pathlib import Path

import requests


DEFAULT_NEGATIVE = (
    "frame, border, vignette, letterbox, pillarbox, black bars, white bars, "
    "picture-in-picture, screen-within-screen, oversaturated rainbow edges"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--prompt",
        default=(
            "Continue directly from the provided first frame without a cut. "
            "Use readable motion and a calm, continuous camera move."
        ),
    )
    parser.add_argument("--backend", default="http://127.0.0.1:8000")
    parser.add_argument("--output-mode", choices=("rtmp", "webrtc"), default="webrtc")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = args.image.read_bytes()
    suffix = args.image.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    image_uri = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
    payload = {
        "model": "ltx25-comfy",
        "initial_prompt": args.prompt,
        "initial_image_url": image_uri,
        "negative_prompt": DEFAULT_NEGATIVE,
        "width": 512,
        "height": 288,
        "num_frames": 121,
        "frame_rate": 9.0,
        "target_fps": 9.0,
        "strength": 1.0,
        "guidance_scale": 1.0,
        "seed": None,
        "output_mode": args.output_mode,
        "enable_audio": False,
        "style_preset": "cohesive",
        "llm_temperature": 0.55,
    }
    response = requests.post(
        f"{args.backend.rstrip('/')}/start_stream",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()
