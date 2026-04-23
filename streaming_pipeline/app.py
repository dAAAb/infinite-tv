import fal
from fastapi import WebSocket

from streaming_pipeline.streaming_service import StreamingService
from streaming_pipeline.models import StartStreamRequest
from dotenv import load_dotenv

#load_dotenv()

# Python requirements (FFmpeg is pre-installed in fal base images).
# Torch must match the GPU-B200 driver's CUDA version (12.8), so we pull
# the +cu128 wheels from PyTorch's index. fal passes each list entry as a
# single argv to `uv pip install`, so "--extra-index-url" and its URL
# must be SEPARATE entries (not one space-joined string).
requirements = [
    "--extra-index-url",
    "https://download.pytorch.org/whl/cu128",
    "torch==2.7.1+cu128",
    "torchvision==0.22.1+cu128",
    "git+https://github.com/huggingface/diffusers.git@main",
    "transformers>=4.47.2,<4.52.0",
    "sentencepiece>=0.1.96",
    "huggingface-hub~=0.30",
    "hf_transfer",  # Activates HF_HUB_ENABLE_HF_TRANSFER=1 in video_generator.setup()
    "einops",
    "timm",
    "accelerate==1.4.0",
    "opencv-python>=4.9.0.80",
    "imageio[ffmpeg]>=2.25.0",
    "numpy>=1.21.0",
    "ffmpeg-python",
    "python-dotenv",
    "openai",
    "fastapi",
    "uvicorn",
    "pydantic",
    "av",
    "fal_client>=0.5.0",  # For LTX 2.3 API calls
    "requests>=2.28.0",   # For downloading generated videos
    "httpx>=0.27.0",      # For TTS API calls
    "pydub>=0.25.1",      # For audio processing
]

class RealtimeStreamingApp(
    fal.App,
    min_concurrency=0,
    max_concurrency=1,
    max_multiplexing=2,
    keep_alive=1000
):
    machine_type = "GPU-B200"
    requirements = requirements
    python_version = "3.11"

    def setup(self):
        """Setup with monitoring"""
        print(" Setting up complete streaming pipeline...")
        
        # Initialize the shared streaming service
        self.streaming_service = StreamingService()
        self.streaming_service.setup()
        
        print("✅ Complete streaming pipeline setup complete!")
    
    @fal.endpoint("/start_stream")
    def start_streaming(self, request: StartStreamRequest):
        """Start the complete Twitch streaming pipeline with full LTX configuration"""
        return self.streaming_service.start_streaming(request)
    
    @fal.endpoint("/stop_stream")
    def stop_streaming(self):
        """Stop the streaming pipeline"""
        return self.streaming_service.stop_streaming()
    
    @fal.endpoint("/metrics")
    def get_metrics(self):
        """Get simplified real-time streaming metrics for dashboard"""
        return self.streaming_service.get_metrics()
    
    @fal.endpoint("/metrics/ws", is_websocket=True)
    async def metrics_websocket(self, websocket: WebSocket) -> None:
        """Real-time metrics streaming via WebSocket"""
        await self.streaming_service.handle_metrics_websocket(websocket)