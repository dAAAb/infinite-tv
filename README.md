# Infinite TV — AI-Generated Live Stream

> **Fork of [alex-remade/infinite-tv](https://github.com/alex-remade/infinite-tv)** with Windows (RTX 5090 / Blackwell) support, local GPU inference, `torch.compile` acceleration, and production-tested Twitch streaming.

A real-time AI video generation system that listens to Twitch chat and streams live AI-generated videos. Built on the **LTX Video 2.3** distilled diffusion model, running entirely on a local GPU.

## What Changed in This Fork

### 🖥️ Full Windows Support
- **RTMP streamer**: Replaced Unix FIFO pipes with Windows named pipes (`CreateNamedPipeW`) for audio streaming — the original only worked on Linux/macOS.
- **Model loading**: Replaced hardcoded Linux paths (`/data/models/...`) with environment-variable-driven paths.
- **File preloading**: Replaced `find | xargs cat` with a Python `ThreadPoolExecutor` fallback on Windows.
- **Twitch listener**: Added UTF-8 encoding fixes for Windows console output.

### ⚡ torch.compile Acceleration
- Added **`torch.compile(mode="reduce-overhead")`** support for the transformer, gated by `LTX23_TORCH_COMPILE=true`.
- On Windows, this requires [`triton-windows`](https://github.com/triton-lang/triton-windows) — install with `pip install triton-windows`.
- **Performance impact**: ~150s → ~35s per generation segment on RTX 5090 (**~77% faster**).
- Note: `max-autotune` mode doesn't work with triton-windows due to `CompiledKernel.launch_enter_hook` mismatch — `reduce-overhead` is used instead.
- We also contributed a [compatibility fix (PR #53)](https://github.com/triton-lang/triton-windows/pull/53) back to triton-windows for a `triton_key` import error that blocked `torch.compile`.

### 🎛️ Configurable Inference Parameters
- **Denoising steps**: Now configurable via `num_inference_steps` (default: 5). The original hardcoded 8 steps. Fewer steps = faster generation with the distilled model.
- **Sigma schedule**: Automatically subsamples the distilled sigma values when using fewer than 8 steps.
- **CPU offload toggle**: `LTX23_CPU_OFFLOAD` env var (default: `true`) — disable for GPUs with enough VRAM to keep everything on-device.
- **Condition pipeline**: Lazy-loaded via `LOAD_LTX23_CONDITION=true` instead of always loading.

### 🎵 Background Music (BGM)
- Added `RTMP_BGM_PATH` support — loops an audio file as background music on the Twitch stream.
- Audio mixing works alongside LTX 2.3's natively generated audio.

### 📊 Other Improvements
- WebRTC output mode alongside RTMP
- Style presets (`cohesive`, `chaotic`, `nightmare`, `custom`)
- Character reference support via `ltx-2.3-condition` mode
- Hot-reloadable generation parameters via `/update_config`
- Local LLM endpoint support for prompt generation (`LLM_BASE_URL`)

---

## Quick Start (Local GPU)

### Prerequisites

- **GPU**: NVIDIA with CUDA support (tested on RTX 5090 / Blackwell)
- **Python 3.11+**
- **Node.js 18+** (for dashboard)
- **FFmpeg** installed and in PATH
- **CUDA Toolkit** installed
- Twitch account + stream key
- OpenAI API key (for prompt generation)

### 1. Clone and Setup

```bash
git clone https://github.com/dAAAb/infinite-tv.git
cd infinite-tv

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -e .

# (Optional, Windows only) Enable torch.compile acceleration
pip install triton-windows
```

### 2. Download Models

The LTX 2.3 distilled model (~15 GB) downloads automatically on first run from HuggingFace. To pre-download:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('dg845/LTX-2.3-Distilled-Diffusers', local_dir='models/ltx-2.3-distilled-v2')"
```

### 3. Environment Configuration

Copy `.env.example` to `.env` and fill in:

```env
# Model selection
LOAD_LTX23_PIPELINE=true
LOAD_LOCAL_PIPELINE=false
LOAD_LTX23_CONDITION=false

# GPU settings
LTX23_CPU_OFFLOAD=false          # Set false if you have enough VRAM (>24GB)
LTX23_TORCH_COMPILE=true         # Requires triton-windows on Windows
PRELOAD_MODEL_FILES=false

# Model paths (auto-downloaded if not present)
LTX23_WEIGHTS_DIR=./models/ltx-2.3-distilled-v2
LTX_WEIGHTS_DIR=./models/ltx-video-0.9.8-13b

# API keys
OPENAI_API_KEY=your_key_here
FAL_KEY=your_key_here             # Only needed for fal.ai cloud mode

# Twitch
TWITCH_CHANNEL=your_channel
TWITCH_STREAM_KEY=your_stream_key

# Optional
RTMP_BGM_PATH=./outputs/bgm-lofi.mp3   # Background music
RUN_GENERATION_INLINE=true

# Optional: local LLM for prompt generation
# LLM_BASE_URL=http://127.0.0.1:8080/v1
# LLM_TEXT_MODEL=local
```

### 4. Start Backend

```powershell
# Windows (PowerShell)
.\scripts\start-local.ps1
```

Or manually:

```bash
python -m uvicorn streaming_pipeline.local_app:app --host 127.0.0.1 --port 8000
```

### 5. Start Streaming

```bash
# Start stream via API
curl -X POST http://127.0.0.1:8000/start_stream \
  -H "Content-Type: application/json" \
  -d '{"output_mode": "rtmp", "model": "ltx-2.3", "initial_image_url": "https://example.com/start-image.jpg"}'
```

Or use the dashboard at `http://127.0.0.1:3000` (run `cd dashboard && npm install && npm run dev`).

### 6. Dashboard Setup

```bash
cd dashboard
npm install
npm run dev
```

The dashboard connects to `http://127.0.0.1:8000` and provides real-time monitoring, stream controls, and parameter tuning.

---

## Pitfalls & Gotchas 🕳️

### Windows-Specific Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| `torch.compile` fails with `ImportError: cannot import name 'triton_key'` | triton-windows missing compat shim for PyTorch Inductor | `pip install triton-windows` + our [triton_key shim](https://github.com/triton-lang/triton-windows/pull/53) |
| `torch.compile` fails with `launch_enter_hook` error | `max-autotune` mode incompatible with triton-windows | Use `reduce-overhead` mode (already configured) |
| RTMP audio breaks / pipe errors | Unix FIFOs (`mkfifo`) don't exist on Windows | This fork uses Windows named pipes — already fixed |
| `UnicodeEncodeError` on console output | Windows console defaults to cp950/cp1252 | Set `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` |
| `find | xargs cat` fails for model preloading | Unix command, not available on Windows | This fork uses Python `ThreadPoolExecutor` fallback |

### General Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| First generation takes 3-5 minutes | `torch.compile` warmup (compiling CUDA kernels) | Normal — subsequent generations are fast (~35s) |
| Stream freezes between segments | Generation slower than playback | Reduce `num_frames` or `num_inference_steps`; enable `torch.compile` |
| `initial_image_url` required error | No default start image | Pass a start image URL or base64 data URI in `/start_stream` |
| VRAM OOM | Model too large for GPU | Enable `LTX23_CPU_OFFLOAD=true` or reduce resolution |
| Twitch stream drops | FFmpeg RTMP timeout | Check `TWITCH_STREAM_KEY`; ensure stable network |

---

## Performance Benchmarks (RTX 5090, 24GB VRAM)

| Configuration | Gen Time | RTMP FPS | Notes |
|--------------|----------|----------|-------|
| Baseline (8 steps, 17 frames, no compile) | ~248s | ~3.8 | Original upstream settings |
| 5 steps, 9 frames, no compile | ~150s | ~3.8 | Reduced params only |
| 5 steps, 9 frames, **torch.compile** | **~35s** | **~8.8** | 🔥 Recommended |

With torch.compile at ~35s per segment and 9 frames sustaining ~54s of playback, the stream runs with **zero freeze time** between generations.

---

## API Reference

### `POST /start_stream`

Start video generation and RTMP streaming.

```json
{
  "model": "ltx-2.3",
  "output_mode": "rtmp",
  "initial_image_url": "https://... or data:image/jpeg;base64,...",
  "initial_prompt": "A cozy lo-fi study room",
  "width": 512,
  "height": 384,
  "num_frames": 9,
  "timesteps": [1000, 981, 909, 725, 0.03],
  "guidance_scale": 1.0,
  "target_fps": 9.0,
  "style_preset": "cohesive",
  "enable_audio": true
}
```

### `POST /stop_stream`

Stop the streaming pipeline.

### `POST /update_config`

Hot-reload generation parameters without stopping the stream.

```json
{
  "num_frames": 9,
  "guidance_scale": 1.5,
  "noise_scale": 0.15,
  "llm_temperature": 0.7,
  "style_preset": "chaotic"
}
```

### `GET /metrics`

Returns real-time metrics: generation stats, RTMP status, GPU memory, Twitch chat status.

### `WebSocket /metrics/ws`

Real-time metrics stream for the dashboard.

---

## Project Structure

```
infinite-tv/
├── streaming_pipeline/           # Main Python package
│   ├── local_app.py             # Local FastAPI entry point
│   ├── app.py                   # FAL serverless entry point
│   ├── streaming_service.py     # Orchestration layer
│   ├── core/
│   │   └── streaming_engine.py  # Main generation loop
│   ├── video_generation/
│   │   └── video_generator.py   # LTX model wrapper + torch.compile
│   ├── input/
│   │   └── twitch_listener.py   # Twitch chat integration
│   ├── output/
│   │   ├── rtmp_streamer.py     # RTMP via FFmpeg (Windows named pipes)
│   │   └── webrtc_streamer.py   # WebRTC browser streaming
│   ├── prompt_generation/
│   │   └── prompt_generator.py  # LLM-driven prompt generation
│   ├── postprocessing/
│   │   └── text_overlay.py      # Video text overlays
│   ├── models/
│   │   ├── api.py               # Request/response models
│   │   ├── video.py             # Video generation config
│   │   └── streaming.py         # Streaming state
│   └── utils/
│       ├── logger_config.py
│       └── monitoring.py
├── dashboard/                    # Next.js React dashboard
├── scripts/
│   └── start-local.ps1          # Windows startup script
├── models/                       # Downloaded model weights
├── outputs/                      # Generated outputs + BGM
├── .env.example                  # Environment template
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Acknowledgments

- **[alex-remade/infinite-tv](https://github.com/alex-remade/infinite-tv)** — Original project
- **[LTX Video](https://huggingface.co/Lightricks)** — Video generation model
- **[triton-windows](https://github.com/triton-lang/triton-windows)** — Triton on Windows
- **[Diffusers](https://github.com/huggingface/diffusers)** — HuggingFace diffusion library
- **FFmpeg** — Video processing and RTMP streaming

## License

MIT — see [LICENSE](LICENSE) file.
