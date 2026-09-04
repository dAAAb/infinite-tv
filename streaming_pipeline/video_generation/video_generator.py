from diffusers import LTXConditionPipeline
from PIL import Image
from io import BytesIO
import base64
import os
import sys


from streaming_pipeline.models import LTXVideoRequestI2V, LTXVideoResponseWithFrames, Monitorable
from typing import Dict, Any, List

def safe_snapshot_download(
    repo_id: str,
    revision: str,
    **kwargs: Any,
):
    from huggingface_hub import snapshot_download

    # max_workers parallelizes the download itself; recommended by fal:
    # https://docs.fal.ai/serverless/code/your-code-data-weights
    kwargs.setdefault("max_workers", 32)

    local_only = os.getenv("HF_LOCAL_ONLY", "false").lower() == "true"
    if local_only:
        print("Loading local repo only...")
        return snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=True,
            **kwargs,
        )

    print("Downloading or completing local repo from Hugging Face Hub...")
    repo_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_files_only=False,
        **kwargs,
    )
    return repo_path


def audio_tensor_to_pcm_bytes(
    audio,
    src_sample_rate: int,
    dst_sample_rate: int = 44100,
) -> bytes:
    """Convert an LTX audio output tensor to ffmpeg-ready stereo s16le PCM.

    Accepts the tensor as returned by `LTX2ImageToVideoPipeline` -- shape
    `(batch, channels, n_samples)` or `(channels, n_samples)`, dtype float in
    [-1, 1].  Resamples to `dst_sample_rate`, forces stereo, clamps to int16,
    and returns interleaved bytes ready to write to an s16le ffmpeg input.
    """
    import torch
    import torchaudio.functional as AF

    if audio is None:
        return b""

    # Drop the batch dim if present.
    if audio.dim() == 3:
        audio = audio[0]
    # audio is now (channels, n_samples)
    audio = audio.detach().to(torch.float32).cpu()

    # Force stereo.
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]

    # Resample if needed.
    if src_sample_rate != dst_sample_rate:
        audio = AF.resample(audio, src_sample_rate, dst_sample_rate)

    # Clamp + convert to int16.
    audio = audio.clamp(-1.0, 1.0)
    pcm_int16 = (audio * 32767.0).to(torch.int16)

    # Interleave channels: shape (n_samples, 2) row-major -> bytes is L,R,L,R...
    interleaved = pcm_int16.transpose(0, 1).contiguous().numpy()
    return interleaved.tobytes()


def preload_files(directory: str, parallelism: int = 32) -> None:
    """Pre-read every file under `directory` in parallel into the OS page cache.

    fal's /data is a distributed filesystem; sequential reads (which is what
    diffusers' from_pretrained does by default) underutilize it badly. Firing
    N parallel `cat`s warms the page cache so the subsequent from_pretrained
    reads from RAM instead of the network.
    See: https://docs.fal.ai/documentation/serverless/optimizations/parallel-file-loading
    """
    if os.getenv("PRELOAD_MODEL_FILES", "false").lower() != "true":
        return

    import subprocess
    print(f"📦 Pre-reading {directory} with parallelism={parallelism}...")
    if sys.platform.startswith("win"):
        from concurrent.futures import ThreadPoolExecutor

        files = []
        for root, _, filenames in os.walk(directory):
            files.extend(os.path.join(root, name) for name in filenames)

        def _read(path: str) -> None:
            with open(path, "rb") as handle:
                while handle.read(1024 * 1024):
                    pass

        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            list(pool.map(_read, files))
    else:
        subprocess.check_call(
            f"find '{directory}' -type f | xargs -P {parallelism} -I {{}} cat {{}} > /dev/null",
            shell=True,
        )
    print("📦 Pre-read complete.")


MODEL_ID = "Lightricks/LTX-Video-0.9.8-13B-distilled"
REVISION = "main"
WEIGHTS_DIR = os.getenv(
    "LTX_WEIGHTS_DIR",
    os.path.join(os.getcwd(), "models", "ltx-video-0.9.8-13b"),
)

LTX23_MODEL_ID = "dg845/LTX-2.3-Distilled-Diffusers"
LTX23_REVISION = "main"
LTX23_WEIGHTS_DIR = os.getenv(
    "LTX23_WEIGHTS_DIR",
    os.path.join(os.getcwd(), "models", "ltx-2.3-distilled-v2"),
)


def _require_fal_video_opt_in(model_type: str) -> None:
    """Refuse billable video generation unless it was explicitly enabled.

    A configured ``FAL_KEY`` is only a credential; it must never double as
    permission to spend money.  Local launches keep ``ENABLE_FAL_VIDEO=false``
    and cloud models require both an explicit opt-in and a key.
    """
    if os.getenv("ENABLE_FAL_VIDEO", "false").lower() != "true":
        raise RuntimeError(
            f"{model_type} uses billable fal.ai video generation. "
            "Set ENABLE_FAL_VIDEO=true explicitly to enable cloud video."
        )
    if not os.getenv("FAL_KEY"):
        raise RuntimeError(
            f"{model_type} requires FAL_KEY after ENABLE_FAL_VIDEO=true."
        )


# Regular Python class for local use (non-fal.App)
class RealtimeGenerator(Monitorable):
    

    def __init__(self, load_local_pipeline: bool = False, load_ltx23_pipeline: bool = False):
        self.pipeline = None
        self.pipeline_v23 = None
        self.pipeline_condition = None  # LTX2ConditionPipeline (shared transformer with pipeline_v23)
        self.load_local_pipeline = load_local_pipeline
        self.load_ltx23_pipeline = load_ltx23_pipeline
        # Set after the LTX 2.3 vocoder loads; used by the audio PCM path.
        self.audio_sample_rate: int = 24000
        
        # Performance tracking for video generation
        self.total_videos = 0
        self.total_generation_time = 0.0
        self.last_generation_time = 0.0
        self.last_backend = "none"


    def setup(self):
        if self.load_ltx23_pipeline:
            self._setup_ltx23()
        elif self.load_local_pipeline:
            self._setup_ltxv1()
        else:
            print("⏩ Skipping local pipeline load (API-only mode)")

    def _setup_ltxv1(self):
        import os
        import torch

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        print("🚀 Loading ltxv1 pipeline (image-to-video)...")
        checkpoint_dir = safe_snapshot_download(
            repo_id=MODEL_ID,
            revision=REVISION,
            local_dir=WEIGHTS_DIR,
            local_dir_use_symlinks=True,
        )
        preload_files(checkpoint_dir, parallelism=32)
        self.pipeline = LTXConditionPipeline.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        self.pipeline.to("cuda")
        self.pipeline.vae.enable_tiling()

        print("✅ ltxv1 pipeline setup complete!")

    def _setup_ltx23(self):
        import os
        import torch
        from diffusers import LTX2ImageToVideoPipeline, AutoModel

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        print("🚀 Loading LTX 2.3 distilled FP8 pipeline...")
        checkpoint_dir = safe_snapshot_download(
            repo_id=LTX23_MODEL_ID,
            revision=LTX23_REVISION,
            local_dir=LTX23_WEIGHTS_DIR,
            local_dir_use_symlinks=True,
        )
        preload_files(checkpoint_dir, parallelism=32)

        transformer = AutoModel.from_pretrained(
            checkpoint_dir, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
        )

        # torch.compile for faster inference — requires Triton backend.
        # Uses "reduce-overhead" mode for triton-windows compatibility
        # (max-autotune hits CompiledKernel.launch_enter_hook mismatch).
        if os.getenv("LTX23_TORCH_COMPILE", "false").lower() == "true":
            try:
                import triton  # noqa: F401
                # Windows/triton note: "reduce-overhead" (cudagraphs) hits a triton-windows
                # kernel group-file bug (__grp__*.json FileNotFoundError). Default mode ("")
                # avoids cudagraphs and compiles cleanly. Override via LTX23_COMPILE_MODE.
                _mode = os.getenv("LTX23_COMPILE_MODE", "default")
                _kw = {} if _mode in ("", "default") else {"mode": _mode}
                print(f"⚡ Compiling transformer with torch.compile (mode={_mode})...")
                transformer = torch.compile(transformer, **_kw)
                print("⚡ torch.compile applied!")
            except ImportError:
                print("⚠️ torch.compile skipped: triton not available")

        self.pipeline_v23 = LTX2ImageToVideoPipeline.from_pretrained(
            checkpoint_dir, transformer=transformer, torch_dtype=torch.bfloat16
        )
        if os.getenv("LTX23_CPU_OFFLOAD", "true").lower() == "true":
            self.pipeline_v23.enable_model_cpu_offload()
            print("🧠 LTX 2.3 CPU offload enabled")
        else:
            self.pipeline_v23.to("cuda")
        # VAE tiling is intentionally NOT enabled here: it chunks the spatial
        # decode to fit small-VRAM GPUs at the cost of per-tile overhead.
        # On GPU-B200 (192 GB HBM3e) we have plenty of headroom, so the
        # full-tensor decode path is faster.

        # Cache the vocoder output sample rate so the audio path knows what
        # to resample from.  Standard LTX 2.3 vocoder = 24000 Hz; the
        # bandwidth-extension vocoder = 48000 Hz.
        try:
            self.audio_sample_rate = int(self.pipeline_v23.vocoder.config.output_sampling_rate)
        except AttributeError:
            self.audio_sample_rate = 24000
        print(f"🔊 Vocoder output sample rate: {self.audio_sample_rate} Hz")

        print("✅ LTX 2.3 distilled FP8 pipeline setup complete!")

        # Also load the condition pipeline (shares the same transformer + VAE,
        # no extra VRAM).  This enables multi-image conditioning for character
        # reference anchoring.
        if os.getenv("LOAD_LTX23_CONDITION", "false").lower() == "true":
            self._setup_ltx23_condition()

    def _setup_ltx23_condition(self):
        """Load LTX2ConditionPipeline sharing the transformer from pipeline_v23."""
        import torch

        if self.pipeline_v23 is None:
            print("⚠️ Cannot load condition pipeline: pipeline_v23 not loaded")
            return

        try:
            from diffusers import LTX2ConditionPipeline

            print("🚀 Loading LTX 2.3 condition pipeline (shared transformer)...")
            self.pipeline_condition = LTX2ConditionPipeline(
                transformer=self.pipeline_v23.transformer,
                vae=self.pipeline_v23.vae,
                text_encoder=self.pipeline_v23.text_encoder,
                tokenizer=self.pipeline_v23.tokenizer,
                connectors=self.pipeline_v23.connectors,
                scheduler=self.pipeline_v23.scheduler,
                vocoder=self.pipeline_v23.vocoder,
                audio_vae=self.pipeline_v23.audio_vae,
            )
            self.pipeline_condition.to("cuda")
            print("✅ LTX 2.3 condition pipeline ready (shared weights, no extra VRAM)")
        except Exception as e:
            print(f"⚠️ Failed to load condition pipeline: {e}")
            print("   ltx-2.3-condition model type will be unavailable")
            self.pipeline_condition = None

    def decode_base64_image(self, base64_string: str) -> Image.Image:
        """Decode base64 string to PIL Image"""
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        return Image.open(BytesIO(image_data)).convert("RGB")
    
    
    
    def frame_to_base64(self, frame: Image.Image) -> str:
        """Convert PIL Image frame to base64"""
        buffer = BytesIO()
        frame.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def download_video_frames(self, video_url: str, target_width: int = None, target_height: int = None) -> List[Image.Image]:
        """Download video from URL and extract frames
        
        Args:
            video_url: URL of the video to download
            target_width: If set, resize frames to this width
            target_height: If set, resize frames to this height
        """
        import requests
        import tempfile
        import cv2
        
        print(f"📥 Downloading video from: {video_url}")
        
        # Download video to temp file
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
        
        try:
            # Extract frames using OpenCV
            frames = []
            cap = cv2.VideoCapture(tmp_path)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_frame = Image.fromarray(frame_rgb)
                
                # Resize if target dimensions are specified
                if target_width and target_height:
                    pil_frame = pil_frame.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                frames.append(pil_frame)
            
            cap.release()
            print(f"✅ Extracted {len(frames)} frames from video")
            if target_width and target_height:
                print(f"📐 Resized frames to {target_width}x{target_height}")
            return frames
            
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
    
    def generate_video_with_fal_api(self, request: LTXVideoRequestI2V) -> LTXVideoResponseWithFrames:
        """Generate video using fal.ai LTX 2.3 Fast API"""
        _require_fal_video_opt_in("ltx-2.3")
        import time
        import traceback
        import fal_client
        
        print(f"🎬 Starting fal.ai LTX 2.3 Fast generation")
        print(f"   Prompt: {request.prompt}")
        print(f"   Duration: {request.duration}s")
        print(f"   Resolution: {request.resolution}")
        print(f"   Aspect Ratio: {request.aspect_ratio}")
        
        start_time = time.time()
        
        try:
            # Ensure we have a proper image URL or data URI
            image_data = request.image_base64
            if not image_data.startswith('data:image'):
                # Add data URI prefix if it's just base64
                image_data = f"data:image/jpeg;base64,{image_data}"
            
            fal_input = {
                "image_url": image_data,
                "prompt": request.prompt,
                "generate_audio": False,
            }
            
            if request.duration:
                fal_input["duration"] = int(request.duration)
            if request.resolution:
                fal_input["resolution"] = request.resolution
            if request.aspect_ratio:
                fal_input["aspect_ratio"] = request.aspect_ratio
            
            print(f"🚀 Calling fal.ai API...")
            print(f"🔑 FAL_KEY set: {bool(os.getenv('FAL_KEY'))}")
            print(f"📦 Input parameters:")
            print(f"   - prompt: {fal_input['prompt'][:50]}...")
            print(f"   - duration: {fal_input.get('duration', 6)}")
            print(f"   - resolution: {fal_input.get('resolution', '1080p')}")
            print(f"   - aspect_ratio: {fal_input.get('aspect_ratio', '16:9')}")
            print(f"   - image_url length: {len(image_data)}")
            
            # Call fal API with subscribe (waits for completion)
            print(f"⏳ Waiting for fal.ai to complete generation...")
            result = fal_client.subscribe(
                "fal-ai/ltx-2.3/image-to-video/fast",
                arguments=fal_input,
                with_logs=True,
            )
            
            print(f"✅ fal.ai API completed!")
            print(f"📊 Result keys: {list(result.keys())}")
            
            # Get video URL from result
            video_url = result["video"]["url"]
            print(f"📹 Video URL: {video_url}")
            
            # Download and extract frames, resizing to match target resolution
            # This ensures text overlay and RTMP streaming work correctly
            frames = self.download_video_frames(video_url, request.width, request.height)
            
            # Track generation performance
            self.last_generation_time = time.time() - start_time
            self.total_generation_time += self.last_generation_time
            self.total_videos += 1
            
            print(f"✅ Complete generation in {self.last_generation_time:.2f}s!")
            
            return LTXVideoResponseWithFrames(
                frames=frames
            )
            
        except Exception as e:
            print(f"❌ fal.ai API generation failed: {e}")
            print(f"❌ Exception type: {type(e).__name__}")
            print(f"❌ Traceback:")
            traceback.print_exc()
            raise
    

    
    def generate_video_from_image(self, request: LTXVideoRequestI2V) -> LTXVideoResponseWithFrames:
        """Main entry point - routes to appropriate backend based on model_type"""
        self.last_backend = (
            f"fal.ai:{request.model_type}"
            if request.model_type in {"ltx-2.3", "h3-max"}
            else f"local:{request.model_type}"
        )
        if request.model_type == "ltx-2.3":
            return self.generate_video_with_fal_api(request)
        if request.model_type == "ltx-2.3-condition":
            return self.generate_video_with_condition_pipeline(request)
        if request.model_type == "ltx-2.3-local":
            return self.generate_video_with_local_v23(request)
        if request.model_type == "h3-max":
            return self.generate_video_with_h3_max(request)
        if request.model_type == "ltx25-comfy":
            return self.generate_video_with_comfy_ltx25(request)
        return self.generate_video_with_local_pipeline(request)

    def generate_video_with_comfy_ltx25(self, request: LTXVideoRequestI2V) -> LTXVideoResponseWithFrames:
        """Generate via the local ComfyUI LTX-2.5 NVFP4 server.

        Every clip with an input frame uses I2V. The caller transactionally advances
        that input only after the previous clip passed validation and was accepted by
        the streamer, so silently falling back to T2V would break continuity.
        """
        import time

        from streaming_pipeline.video_generation import comfy_ltx25_backend as comfy
        mode = "i2v" if request.image_base64 else "t2v"
        print(f"🔗 ltx25-comfy: explicit {mode.upper()} mode")
        start_time = time.monotonic()
        frames, audio_pcm = comfy.generate(
            prompt=request.prompt,
            image_base64=request.image_base64,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            frame_rate=float(request.frame_rate),
            seed=request.seed,
            negative=request.negative_prompt,
            strength=float(request.strength),
            # Twitch uses the RTMP BGM mixer. Avoid building an unused LTX audio
            # latent, which adds latency and another temporal boundary.
            want_audio=False,
            mode=mode,
        )
        self.last_generation_time = time.monotonic() - start_time
        self.total_generation_time += self.last_generation_time
        self.total_videos += 1
        print(
            f"✅ Local ComfyUI LTX 2.5 generation in "
            f"{self.last_generation_time:.2f}s"
        )
        return LTXVideoResponseWithFrames(frames=frames, audio_pcm=audio_pcm)

    def generate_video_with_h3_max(self, request: LTXVideoRequestI2V) -> LTXVideoResponseWithFrames:
        """Generate video using fal.ai MiniMax H3 Max Turbo endpoint (cloud).

        H3 Max Turbo is fal's fastest variant — 2x speed of H3 Max at half price.
        Endpoint: minimax/h3-max-turbo/image-to-video. Pricing (as of Sept 2026):
        - Promo (until 2026-09-07): $0.00625/s @ 480p, $0.01/s @ 768p
        - Regular:                  $0.025/s   @ 480p, $0.04/s @ 768p
        """
        _require_fal_video_opt_in("h3-max")

        import time
        import base64
        import traceback
        import fal_client
        start_time = time.time()
        try:
            # Prepare image data as data URI
            image_data = request.image_base64
            if not image_data.startswith('data:image'):
                image_data = f"data:image/jpeg;base64,{image_data}"

            fal_input = {
                "image_url": image_data,
                "prompt": request.prompt,
            }
            if request.duration:
                fal_input["duration"] = int(request.duration)
            if request.resolution:
                # H3 Max Turbo API expects uppercase '480P' / '768P'
                fal_input["resolution"] = str(request.resolution).upper()

            print(f"🚀 Calling fal.ai H3 Max Turbo API...")
            print(f"🔑 FAL_KEY set: {bool(os.getenv('FAL_KEY'))}")
            print(f"   - prompt: {fal_input['prompt'][:50]}...")
            print(f"   - resolution: {fal_input.get('resolution', '768p')}")
            print(f"   - duration: {fal_input.get('duration', 5)}s")

            result = fal_client.subscribe(
                "minimax/h3-max-turbo/image-to-video",
                arguments=fal_input,
                with_logs=True,
            )
            print(f"✅ H3 Max Turbo completed! keys={list(result.keys())}")

            video_url = result["video"]["url"]
            frames = self.download_video_frames(video_url, request.width, request.height)

            self.last_generation_time = time.time() - start_time
            self.total_generation_time += self.last_generation_time
            self.total_videos += 1
            print(f"✅ H3 Max Turbo generation in {self.last_generation_time:.2f}s!")

            return LTXVideoResponseWithFrames(frames=frames)
        except Exception as e:
            print(f"❌ H3 Max Turbo generation failed: {e}")
            traceback.print_exc()
            raise

    def generate_video_with_condition_pipeline(self, request: LTXVideoRequestI2V) -> LTXVideoResponseWithFrames:
        """Generate video using LTX2ConditionPipeline with multi-character reference anchoring."""
        import random
        import torch
        import time
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition

        if self.pipeline_condition is None:
            raise RuntimeError("LTX 2.3 condition pipeline not loaded.")

        seed = request.seed if request.seed is not None else random.randrange(2**31)

        print(f"🎬 Starting condition pipeline generation - {request.num_frames} frames")

        input_image = self.decode_base64_image(request.image_base64)
        input_image = input_image.resize((request.width, request.height))

        # Build conditions: last frame for continuity + character references
        conditions = [
            LTX2VideoCondition(frames=input_image, index=0, strength=1.0),
        ]

        if request.character_refs:
            for i, ref in enumerate(request.character_refs[:4]):
                ref_image_data = ref.get("image", "")
                ref_strength = float(ref.get("strength", 0.4))
                ref_label = ref.get("label", f"ref_{i}")
                if not ref_image_data:
                    continue
                try:
                    ref_img = self.decode_base64_image(ref_image_data)
                    ref_img = ref_img.resize((request.width, request.height))
                    conditions.append(
                        LTX2VideoCondition(frames=ref_img, index=0, strength=ref_strength)
                    )
                    print(f"   🧑 Character ref '{ref_label}': strength={ref_strength}")
                except Exception as e:
                    print(f"   ⚠️ Failed to decode character ref '{ref_label}': {e}")

        print(f"📏 Resolution: {request.width}x{request.height}")
        print(f"📝 Prompt: {request.prompt[:80]}...")
        print(f"🎚️ {len(conditions)} conditions, seed={seed}, guidance={request.guidance_scale}, noise={request.noise_scale}")

        start_time = time.time()

        # NOTE: do NOT pass noise_scale here.  In the condition pipeline,
        # noise_scale controls how much noise is added to unconditioned regions
        # and must be high (~1.0).  When omitted, it auto-infers from sigmas[0]
        # which is the correct behavior.  The i2v pipeline uses noise_scale
        # differently (latent interpolation), so the request.noise_scale value
        # from the UI is not applicable here.
        pipeline_kwargs = dict(
            conditions=conditions,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            frame_rate=float(request.frame_rate),
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            modality_scale=1.0,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            output_type="pil",
            return_dict=False,
        )
        if request.num_inference_steps == len(DISTILLED_SIGMA_VALUES):
            pipeline_kwargs["sigmas"] = DISTILLED_SIGMA_VALUES
        elif request.num_inference_steps < len(DISTILLED_SIGMA_VALUES):
            indices = [int(i * (len(DISTILLED_SIGMA_VALUES) - 1) / (request.num_inference_steps - 1))
                       for i in range(request.num_inference_steps)]
            pipeline_kwargs["sigmas"] = [DISTILLED_SIGMA_VALUES[i] for i in indices]

        try:
            video, audio = self.pipeline_condition(**pipeline_kwargs)

            frames = video[0]

            try:
                audio_pcm = audio_tensor_to_pcm_bytes(
                    audio,
                    src_sample_rate=self.audio_sample_rate,
                    dst_sample_rate=44100,
                )
                audio_seconds = len(audio_pcm) / (44100 * 2 * 2)
                print(f"🔊 Audio PCM ready: {len(audio_pcm)} bytes ({audio_seconds:.2f}s)")
            except Exception as e:
                print(f"⚠️ Failed to convert audio: {e}")
                audio_pcm = None

            self.last_generation_time = time.time() - start_time
            self.total_generation_time += self.last_generation_time
            self.total_videos += 1

            print(f"✅ Condition pipeline generation completed in {self.last_generation_time:.2f}s!")
        except Exception as e:
            print(f"❌ Condition pipeline generation failed: {e}")
            raise

        return LTXVideoResponseWithFrames(frames=frames, audio_pcm=audio_pcm)

    def generate_video_with_local_v23(self, request: LTXVideoRequestI2V) -> LTXVideoResponseWithFrames:
        """Generate video using local LTX 2.3 distilled FP8 pipeline"""
        import random
        import torch
        import time
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

        if self.pipeline_v23 is None:
            raise RuntimeError("LTX 2.3 pipeline not loaded. Set LOAD_LTX23_PIPELINE=true.")

        # If the caller pinned a seed, use it; otherwise pick a fresh random one
        # each generation.  Without a per-call random seed the deterministic
        # noise prior + chained input frame causes scene fixation.
        seed = request.seed if request.seed is not None else random.randrange(2**31)

        print(f"🎬 Starting local LTX 2.3 generation - {request.num_frames} frames")

        input_image = self.decode_base64_image(request.image_base64)
        input_image = input_image.resize((request.width, request.height))

        print(f"📏 Resolution: {request.width}x{request.height}")
        print(f"📝 Prompt: {request.prompt[:80]}...")
        print(f"🎚️ guidance_scale={request.guidance_scale}, stg_scale={request.stg_scale}, noise_scale={request.noise_scale}, seed={seed}, frame_rate={request.frame_rate} (audio duration = {request.num_frames / request.frame_rate:.2f}s)")

        start_time = time.time()

        # Build pipeline kwargs.  Only enable Spatio-Temporal Guidance when the
        # caller asks for it AND supplies block indices -- LTX 2.3 raises if
        # stg_scale > 0 without a block list, and we don't want to guess the
        # right indices for an arbitrary model checkpoint.
        pipeline_kwargs = dict(
            image=input_image,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            # frame_rate drives `duration_s = num_frames / frame_rate` which
            # determines the generated audio length; passing the RTMP stream's
            # target_fps keeps audio length aligned with video playback.
            frame_rate=float(request.frame_rate),
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            noise_scale=request.noise_scale,
            modality_scale=1.0,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            output_type="pil",
            return_dict=False,
        )
        # Use distilled sigma schedule: full 8-step or evenly subsampled
        if request.num_inference_steps == len(DISTILLED_SIGMA_VALUES):
            pipeline_kwargs["sigmas"] = DISTILLED_SIGMA_VALUES
        elif request.num_inference_steps < len(DISTILLED_SIGMA_VALUES):
            # Subsample the distilled sigmas evenly for fewer steps
            indices = [int(i * (len(DISTILLED_SIGMA_VALUES) - 1) / (request.num_inference_steps - 1))
                       for i in range(request.num_inference_steps)]
            pipeline_kwargs["sigmas"] = [DISTILLED_SIGMA_VALUES[i] for i in indices]
            print(f"⚡ Using {request.num_inference_steps}-step subsampled sigmas: {pipeline_kwargs['sigmas']}")

        stg_blocks = getattr(request, "spatio_temporal_guidance_blocks", None)
        if request.stg_scale and request.stg_scale > 0 and stg_blocks:
            pipeline_kwargs["stg_scale"] = request.stg_scale
            pipeline_kwargs["spatio_temporal_guidance_blocks"] = stg_blocks
            print(f"🌀 STG enabled: stg_scale={request.stg_scale}, blocks={stg_blocks}")
        elif request.stg_scale and request.stg_scale > 0:
            print(f"⚠️ stg_scale={request.stg_scale} requested but no spatio_temporal_guidance_blocks supplied; STG disabled")

        try:
            video, audio = self.pipeline_v23(**pipeline_kwargs)

            frames = video[0]

            # Convert audio waveform to ffmpeg-ready PCM bytes.  Done here on
            # the GPU thread so the cost is overlapped with the next prompt
            # generation rather than blocking the event loop.
            try:
                audio_pcm = audio_tensor_to_pcm_bytes(
                    audio,
                    src_sample_rate=self.audio_sample_rate,
                    dst_sample_rate=44100,
                )
                audio_seconds = len(audio_pcm) / (44100 * 2 * 2)  # 2ch * 2 bytes
                print(f"🔊 Audio PCM ready: {len(audio_pcm)} bytes ({audio_seconds:.2f}s @ 44.1kHz stereo)")
            except Exception as e:
                print(f"⚠️ Failed to convert audio to PCM (continuing without audio): {e}")
                audio_pcm = None

            self.last_generation_time = time.time() - start_time
            self.total_generation_time += self.last_generation_time
            self.total_videos += 1

            print(f"✅ LTX 2.3 local generation completed in {self.last_generation_time:.2f}s!")
        except Exception as e:
            print(f"❌ LTX 2.3 local generation failed: {e}")
            raise

        return LTXVideoResponseWithFrames(frames=frames, audio_pcm=audio_pcm)
    
    def generate_video_with_local_pipeline(self, request: LTXVideoRequestI2V) -> LTXVideoResponseWithFrames:
        """Generate video using local HuggingFace LTX pipeline (ltxv1)"""
        import torch
        
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Make sure setup() was called successfully.")
        
        print(f"🎬 Starting local pipeline generation - {request.num_frames} frames")
        
        # Decode Base64 input image
        print("📷 Decoding input image...")
        input_image = self.decode_base64_image(request.image_base64)
        
        # Resize image to match video dimensions
        print(f"📏 Resizing image to {request.width}x{request.height}")
        input_image = input_image.resize((request.width, request.height))
        
        # Generate video using image parameter
        print("🚀 Calling pipeline for video generation...")

        
        import time
        start_time = time.time()
        
        try:
            video = self.pipeline(
                image=input_image,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                num_frames=request.num_frames,
                timesteps=request.timesteps,
                strength=request.strength,
                guidance_scale=request.guidance_scale,
                generator=torch.Generator().manual_seed(0),
                output_type="pil",
            ).frames[0]
            
            # Track generation performance
            self.last_generation_time = time.time() - start_time
            self.total_generation_time += self.last_generation_time
            self.total_videos += 1
            
            print(f"✅ Pipeline generation completed in {self.last_generation_time:.2f}s!")
        except Exception as e:
            print(f"❌ Pipeline generation failed: {e}")
            raise
        

        # Return only frames - last frame will be extracted when needed
        return LTXVideoResponseWithFrames(
            frames=video  # All frames for RTMP streaming, last frame extracted on-demand
        )
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.total_videos = 0
        self.total_generation_time = 0.0
        self.last_generation_time = 0.0
        self.last_backend = "none"
        print("🧹 Video generation metrics reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get component status for monitoring - video generation performance!"""
        avg_generation_time = self.total_generation_time / max(1, self.total_videos)
        return {
            "videos_generated": self.total_videos,
            "avg_generation_time": round(avg_generation_time, 2),
            "last_generation_time": round(self.last_generation_time, 2),
            "backend": self.last_backend,
            "uses_cloud_video": self.last_backend.startswith("fal.ai:"),
            "ready": (
                self.pipeline is not None
                or self.pipeline_v23 is not None
                or self.last_backend == "local:ltx25-comfy"
            ),
        }





        

