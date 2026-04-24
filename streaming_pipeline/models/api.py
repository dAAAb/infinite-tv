from typing import Optional, List, Literal, Union
from pydantic import BaseModel, Field


class StartStreamRequest(BaseModel):
    # Model Selection
    model: Optional[Literal["ltxv1", "ltx-2.3", "ltx-2.3-local"]] = Field(default="ltxv1", description="Which model to use for generation")
    
    # Basic stream configuration
    initial_prompt: Optional[str] = Field(default=None, description="Custom initial prompt for the stream")
    initial_image_url: Optional[str] = Field(default=None, description="Custom initial image URL for the stream")
    
    # LTX Model Parameters (matching LTXVideoRequestI2V).
    # Defaults are tuned for the local LTX 2.3 distilled pipeline on GPU-B200:
    #   - height/width chosen to be divisible by 32 (LTX requirement) and small
    #     enough to keep transformer FLOPs low
    #   - num_frames chosen to be `8*k + 1` (LTX requirement); 121 frames @ 9
    #     output FPS is ~13s of stream content, comfortably > one gen cycle
    negative_prompt: Optional[str] = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted, static scene, frozen frame, no motion, repetitive, looping",
        description="The negative prompt",
    )
    height: Optional[int] = Field(default=384, description="The height of the video (must be divisible by 32)")
    width: Optional[int] = Field(default=512, description="The width of the video (must be divisible by 32)")
    num_frames: Optional[int] = Field(default=121, description="The number of frames to generate (must be 8*k + 1)")
    # If unset, streaming_service auto-fills this from target_fps so audio
    # length matches the RTMP stream's playback duration.
    frame_rate: Optional[float] = Field(default=None, description="Generation frame rate (defaults to target_fps to keep audio in sync with playback)")
    strength: Optional[float] = Field(default=1.0, description="How much to follow the input image")
    guidance_scale: Optional[float] = Field(default=3.0, description="Classifier-free guidance scale (1=off, 3-4 typical for distilled)")
    timesteps: Optional[List[float]] = Field(default=[1000, 981, 909, 725, 0.03], description="The timesteps to use")

    # LTX 2.3-local fixation-control parameters
    stg_scale: Optional[float] = Field(default=0.0, description="LTX spatiotemporal guidance (0=off, 1-3 adds motion structure). Requires spatio_temporal_guidance_blocks to be set.")
    spatio_temporal_guidance_blocks: Optional[List[int]] = Field(default=None, description="Transformer block indices at which to apply STG (required when stg_scale>0)")
    noise_scale: Optional[float] = Field(default=0.15, description="Noise injected into latents during denoising; breaks deterministic loops (0-0.3)")
    seed: Optional[int] = Field(default=None, description="Generation seed. If null, a random seed is used per generation (recommended to avoid scene fixation)")

    # LTX 2.3-specific parameters
    duration: Optional[Literal[6, 8, 10, 12, 14, 16, 18, 20]] = Field(default=None, description="Duration in seconds (>10s requires 25fps and 1080p)")
    resolution: Optional[Literal["1080p", "1440p", "2160p"]] = Field(default=None, description="Resolution for LTX 2.3")
    aspect_ratio: Optional[Literal["auto", "16:9", "9:16"]] = Field(default=None, description="Aspect ratio for LTX 2.3")

    # Streaming Configuration
    target_fps: Optional[float] = Field(default=9.0, description="Target streaming FPS")
    mode: Optional[str] = Field(default="regular", description="Generation mode: 'regular' or 'nightmare'")
    enable_audio: Optional[bool] = Field(default=True, description="Stream LTX 2.3's natively-generated audio instead of silent anullsrc (ltx-2.3-local only)")
    output_mode: Optional[Literal["rtmp", "webrtc"]] = Field(
        default="rtmp",
        description="Output backend: 'rtmp' pushes to Twitch via FFmpeg, 'webrtc' streams directly to browser via aiortc",
    )
