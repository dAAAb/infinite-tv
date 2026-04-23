from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from fal.toolkit.file import File


class LTXVideoRequestI2V(BaseModel):
    prompt: str = Field(description="The prompt to generate the video")
    image_base64: str = Field(description="Base64 encoded input image")
    model_type: Literal["ltxv1", "ltx-2.3", "ltx-2.3-local"] = Field(default="ltxv1", description="Which model to use for generation")
    negative_prompt: str = Field(default="worst quality, inconsistent motion, blurry, jittery, distorted", description="The negative prompt")
    height: int = Field(default=480, description="The height of the video")
    width: int = Field(default=640, description="The width of the video")
    num_frames: int = Field(default=161, description="The number of frames to generate")
    strength: float = Field(default=1.0, description="How much to follow the input image")
    guidance_scale: float = Field(default=3.0, description="The guidance scale")
    timesteps: List[float] = Field(default=[1000, 993, 987, 981, 975, 909, 725, 0.03], description="The timesteps to use")
    
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
