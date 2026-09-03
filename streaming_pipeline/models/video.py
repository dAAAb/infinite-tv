from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class LTXVideoRequestI2V(BaseModel):
    prompt: str = Field(description="The prompt to generate the video")
    image_base64: str = Field(description="Base64 encoded input image")
    model_type: Literal["ltxv1", "ltx-2.3", "ltx-2.3-local", "ltx-2.3-condition"] = Field(default="ltxv1", description="Which model to use for generation")
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted, static scene, frozen frame, no motion, repetitive, looping",
        description="The negative prompt",
    )
    # Defaults aligned with StartStreamRequest in api.py.  height/width must
    # be divisible by 32 and num_frames must be 8*k + 1 for LTX.
    height: int = Field(default=384, description="The height of the video (must be divisible by 32)")
    width: int = Field(default=512, description="The width of the video (must be divisible by 32)")
    num_frames: int = Field(default=121, description="The number of frames to generate (must be 8*k + 1)")
    # Drives `duration_s = num_frames / frame_rate` inside the LTX 2.3 pipeline,
    # which determines how long the generated audio is.  Should match the RTMP
    # stream's target_fps so audio playback duration aligns with video playback.
    frame_rate: float = Field(default=14.0, description="Generation frame rate; should match RTMP target_fps for in-sync audio")
    strength: float = Field(default=1.0, description="How much to follow the input image")
    guidance_scale: float = Field(default=1.0, description="Classifier-free guidance scale (1=off, 3-4 typical for distilled)")
    num_inference_steps: int = Field(default=5, description="Number of denoising steps for local LTX 2.3 generation")
    timesteps: List[float] = Field(default=[1000, 993, 987, 981, 975, 909, 725, 0.03], description="The timesteps to use")

    # LTX 2.3-local fixation-control parameters
    stg_scale: float = Field(default=0.0, description="LTX spatiotemporal guidance (0=off, 1-3 adds motion structure). Requires spatio_temporal_guidance_blocks to be set.")
    spatio_temporal_guidance_blocks: Optional[List[int]] = Field(default=None, description="Transformer block indices at which to apply STG (required when stg_scale>0)")
    noise_scale: float = Field(default=0.15, description="Noise injected into latents during denoising; breaks deterministic loops (0-0.3)")
    seed: Optional[int] = Field(default=None, description="Generation seed. If null, a random seed is used per generation")

    # LTX 2.3 Condition pipeline: character reference images
    # Each entry: {"image": "base64_or_url", "strength": 0.4, "label": "Homer Simpson"}
    character_refs: Optional[List[dict]] = Field(
        default=None,
        description="Character reference images for identity anchoring (max 4). Each: {image, strength, label}",
    )

    # LTX 2.3-specific parameters
    duration: Optional[Literal[6, 8, 10, 12, 14, 16, 18, 20]] = Field(default=None, description="Duration in seconds (>10s requires 25fps and 1080p)")
    resolution: Optional[Literal["1080p", "1440p", "2160p"]] = Field(default=None, description="Resolution for LTX 2.3")
    aspect_ratio: Optional[Literal["auto", "16:9", "9:16"]] = Field(default=None, description="Aspect ratio for LTX 2.3")

class LTXVideoResponseBase64(BaseModel):
    video_base64: str = Field(description="Base64 encoded video data")
    mime_type: str = Field(default="video/mp4", description="MIME type of the video")


class LTXVideoResponseWithLastFrame(BaseModel):
    video_base64: str = Field(description="Base64 encoded video data")
    last_frame_base64: str = Field(description="Base64 encoded last frame")
    mime_type: str = Field(default="video/mp4", description="MIME type of the video")


class LTXVideoResponseWithFrames(BaseModel):
    frames: Optional[List] = Field(default=None, description="PIL frames (when streaming)")
    # Stereo s16le PCM at 44100 Hz, ready to feed to ffmpeg.  Length is
    # exactly 4 * 44100 * duration_seconds bytes (2 channels x 2 bytes/sample).
    audio_pcm: Optional[bytes] = Field(default=None, description="Stereo s16le PCM at 44100 Hz; None if audio disabled")

    class Config:
        # `frames` holds PIL.Image and `audio_pcm` is bytes; pydantic v1
        # cannot validate them, but we never serialize this model over the
        # wire (it stays in-process), so allow arbitrary types.
        arbitrary_types_allowed = True
