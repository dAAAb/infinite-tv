# Infinite TV on one RTX 5090

> A local-first fork of [alex-remade/infinite-tv](https://github.com/alex-remade/infinite-tv): Twitch chat drives an evolving story, ComfyUI + LTX 2.5 generates the next clip on a local RTX 5090, and FFmpeg keeps the stream alive.

**目前穩定版已把影片生成移到單張 RTX 5090 本機執行。** 第二段開始，一般段落都使用上一段「實際送進串流」的最後一格做 Image-to-Video；只有 viewer command 連續兩次未通過視覺驗收時，才會在第三次以同一套本機 LTX 做 prompt-first fallback，再從 exact handoff 平滑銜接。劇情、留言字幕和 RTMP queue 也一起做了連續性與存活性保護。已知限制是段落交界仍偶爾有輕微動作跳動，這是下一輪優化重點。

## The important correction: local vs. fal.ai

During development we tested two billable fal.ai video routes (`ltx-2.3` and `h3-max`). An earlier prompt implementation also preferred fal's OpenRouter proxy whenever `FAL_KEY` existed. That produced real fal.ai usage and made the phrase "fully local" inaccurate for that part of the journey.

The production path documented below is different:

- video: **local ComfyUI at `127.0.0.1:8188`**, model `ltx25-comfy`;
- prompt/vision: direct OpenAI by default (`gpt-4o-mini`), or a local OpenAI-compatible endpoint;
- fal prompt routing: disabled unless `USE_FAL_OPENROUTER=true`;
- fal video: blocked unless **both** `ENABLE_FAL_VIDEO=true` and `FAL_KEY` are set;
- local launch scripts force both fal switches off.

Merely having a `FAL_KEY` is no longer permission to spend money.

| Model id | Video compute | Default availability |
|---|---|---|
| `ltx25-comfy` | Local ComfyUI / RTX GPU | **Default** |
| `ltx-2.3-local` | Local Diffusers / RTX GPU | Optional legacy path |
| `ltx-2.3-condition` | Local Diffusers / RTX GPU | Optional legacy path |
| `ltxv1` | Local Diffusers / RTX GPU | Optional legacy path |
| `ltx-2.3` | fal.ai cloud | Blocked until explicit opt-in |
| `h3-max` | fal.ai cloud | Blocked until explicit opt-in |

## What works now

- Local LTX 2.5 NVFP4 generation through ComfyUI.
- Image-to-Video chaining after the first clip.
- Pixel-exact visible seam: clip `N`'s streamed final frame equals clip `N+1`'s first streamed frame.
- The exact post-repair committed tail is atomically snapshotted for restart recovery; ComfyUI's pre-repair output is never mistaken for stream state.
- Transactional story state: a clip advances the handoff and prompt history only after all frames are accepted by RTMP.
- LTX temporal-padding trim: a 121-frame request currently decodes 129 frames; frames 122–129 are discarded.
- Border/corruption guard for hard bars and soft chromatic/vignette halos, with adaptive full-bleed repair.
- Local recovery segment after repeated bad generations, so the channel never deadlocks on one frame.
- RTMP backpressure capped around one clip instead of accumulating minutes of latency.
- Twitch comments are consumed FIFO one per clip, compiled into a literal English action while retaining the verbatim original, and vision-audited **before** captioning or RTMP commit.
- Anti-stall prompt checks catch both near-duplicate prose and recurring action/object loops, then force a concrete scene-changing beat with a looser image guide; a three-clip cooldown preserves narrative pacing.
- Windows named-pipe audio/BGM support and orphaned-FFmpeg cleanup.

## Current measured result

RTX 5090, Windows, local ComfyUI LTX 2.5 NVFP4, 512×288, 121 requested frames:

| Measurement | Result | Context |
|---|---:|---|
| Raw ComfyUI bridge generation | ~5.1–7.1 s | Controlled local tests; video-only is faster than AV retrieval |
| End-to-end live cycle | **13.53 s average** | 29 consecutive start-to-start intervals, including prompt, generation and backpressure |
| Stream payload per clip | 121 frames / 13.44 s | Dripped at 9 FPS |
| Sustained RTMP rate | **8.9–9.0 / 9 FPS** | Twitch production run |
| Snapshot | 90 clips, 10,948 frames | 0 dropped, 0 rejected, queue 9.4 s |
| Recovery activity | 3 adaptive repairs | 0 forced recovery segments in that snapshot |
| Automated continuity tests | **26 / 26 passing** | Seam, persisted handoff, padding, soft halo, recovery, queue, comment control and provider checks |

These numbers are workload- and driver-dependent. The honest production metric is the end-to-end cycle, not just the denoising kernel time.

### How the optimization journey evolved

| Stage | Result | What we learned |
|---|---:|---|
| Original local LTX 2.3 baseline | ~248 s / segment | A direct port was functional but not live-streamable |
| Fewer frames / diffusion steps | ~150 s | Easy speedup, still far behind playback |
| Earlier LTX 2.3 `torch.compile` experiment | ~35 s | Promising historical result, but not the final production path |
| Later real-model Windows compile validation | Failed on Triton cache/group files | Synthetic compile tests were not enough; do not claim this as a stable recipe |
| LTX 2.5 NVFP4 in ComfyUI | ~5–7 s raw bridge; ~13.5 s live cycle | The first practical single-5090 path for this stream |

The final win did not come from one magic flag. It came from changing the model/runtime path, then engineering the queue, handoff and recovery behavior around it.

## Architecture

```text
Twitch chat
    │
    ▼
Prompt generator ── OpenAI direct or local LLM
    │
    ▼
Streaming engine ── story transaction + quality/recovery state machine
    │
    ▼
ComfyUI HTTP API ── local LTX 2.5 NVFP4 Image-to-Video
    │
    ├── clean frame 121 ──► next clip's frame 1
    │
    └── stream-only copy ──► comment/prompt overlay ──► FFmpeg ──► Twitch
```

## Why continuous I2V is harder than it looks

### 1. The generated first frame is not automatically exact

`LTXVAddGuide` conditions frame 0, but VAE encode/decode can still alter pixels. We replace the visible first output frame with the committed previous tail. This removes a hard cut at the seam.

### 2. The last decoded frame was not the requested last frame

Our 121-frame graph produced 129 decoded frames. The last eight are temporal padding and can degrade. Feeding frame 129 into the next clip created a long autoregressive corruption loop. We now stream and chain only frames 1–121.

### 3. A perfect pixel seam does not guarantee perfect motion

Each clip is still independently denoised. Velocity and motion blur are not carried across the boundary, so a small motion jump can remain even when the joining pixels are identical. This is the main open quality issue; likely next steps are overlap-aware generation, optical-flow-assisted transition scoring, or motion-state conditioning rather than a cosmetic crossfade.

### 4. Rejecting every bad clip can freeze a live channel

The upstream project keeps output alive by replaying the last frame when its queue is empty. A strict quality gate added safety but could repeatedly reject clips from the same poisoned handoff. The current bounded state machine tries local repairs, reuses the same prompt without another LLM bill, then emits a local push-in recovery clip and continues.

### 5. A huge queue makes interaction look broken

A Twitch comment can be correctly received, selected and burned into frames yet remain invisible for minutes if those frames sit behind an oversized queue. Backpressure now targets 18 seconds. Comment text is displayed for roughly 85% of its clip and verified at the pixel level before send.

### 6. A displayed comment is not necessarily an executed command

The old prompt stage could select `鏡頭拉遠 這個生物戴上眼鏡` but silently turn it into “the creature notices glasses,” dropping both the camera move and the completed action. Comments are now authoritative FIFO commands. A dedicated compiler translates only the current comment into a literal English video action while retaining the original text verbatim. A stricter before/middle/end audit runs before RTMP: failed clips receive no caption, never reach viewers, never become the next handoff, and never enter story history. Retries progressively release the image guide (`0.30 → 0.10`); if the old subject is still locked, a third local prompt-first LTX attempt is eased from the exact streamed frame over eight frames. Only a visually verified clip is captioned and committed. Ordinary story clips remain I2V.

### 7. Soft coloured halos are different from black bars

The long chain often produced no crisp one-pixel border at all. Instead, saturation or luminance drifted gradually around several sides, forming the rounded rainbow/vignette halo visible in the Twitch captures. The detector now compares each outer strip with its adjacent inner strip on all four sides. Repair preserves the exact seam frame, then settles into a 10–16% full-bleed crop early in the clip so the repaired tail, not the contaminated perimeter, becomes the next I2V input.

## Reproduce the local RTX 5090 setup

This is the tested Windows path. Other NVIDIA GPUs may work with lower resolution or different quantization, but the published numbers are from a 32 GB RTX 5090.

### Prerequisites

- Windows 11
- NVIDIA RTX 5090 with a working CUDA/PyTorch driver stack
- Python 3.11+
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) recent enough to include the LTXV nodes used by the API graph
- FFmpeg in `PATH`
- Node.js 18+ for the optional dashboard
- Twitch account + stream key for RTMP output
- OpenAI API key, or a local OpenAI-compatible LLM endpoint, for story prompts

### 1. Clone and install the controller

```powershell
git clone https://github.com/dAAAb/infinite-tv.git
cd infinite-tv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

cd dashboard
npm install
cd ..
```

fal.ai is not installed by the local setup. If you intentionally want the optional cloud models, install `requirements-cloud.txt` and enable the runtime guard explicitly.

### 2. Install the LTX 2.5 files in ComfyUI

Download the official files from [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) and place them where ComfyUI's loaders expect them:

```text
ComfyUI/models/diffusion_models/
  ltx-2.5-22b-distilled-transformer-nvfp4.safetensors
ComfyUI/models/text_encoders/
  gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
ComfyUI/models/vae/
  ltx-2.5-video-vae-bf16.safetensors
```

The production Twitch graph is video-only; BGM is mixed by FFmpeg, so the LTX audio VAE is not required for this path.

### 3. Configure secrets locally

```powershell
Copy-Item .env.example .env
```

Edit `.env` locally:

```dotenv
OPENAI_API_KEY=your_openai_key
OPENAI_COMMENT_AUDIT_MODEL=gpt-4o
TWITCH_CHANNEL=your_channel
TWITCH_STREAM_KEY=your_stream_key

COMFY_SERVER=127.0.0.1:8188
COMFYUI_DIR=C:\path\to\ComfyUI

USE_FAL_OPENROUTER=false
ENABLE_FAL_VIDEO=false
COMMENT_I2V_STRENGTH_SCHEDULE=0.30,0.10,0.0
```

`.env`, logs, model weights, output frames, virtual environments and generated media are gitignored. Never paste keys into scripts, prompts, screenshots or Git remote URLs.

### 4. Start ComfyUI

From your ComfyUI directory:

```powershell
.\.venv\Scripts\python.exe main.py --reserve-vram 2
```

Verify `http://127.0.0.1:8188/system_stats` responds.

### 5. Start Infinite TV

```powershell
.\scripts\start-local.ps1
```

Or start only the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn streaming_pipeline.local_app:app --host 127.0.0.1 --port 8000
```

The local launcher explicitly sets:

```text
USE_FAL_OPENROUTER=false
ENABLE_FAL_VIDEO=false
LTX25_MAX_CORRUPT_RETRIES=2
LTX25_RECOVERY_INSET_RATIO=0.18
RTMP_TARGET_QUEUE_SECONDS=18
```

### 6. Start a stream from a local image

Browser/WebRTC test, with no external broadcast:

```powershell
.\.venv\Scripts\python.exe scripts\start-ltx25-stream.py --image .\start.png
```

Twitch RTMP:

```powershell
.\.venv\Scripts\python.exe scripts\start-ltx25-stream.py `
  --image .\start.png `
  --output-mode rtmp `
  --prompt "Continue directly from this frame without a cut. The character notices a new signal."
```

The dashboard is available at `http://127.0.0.1:3000`. Runtime metrics are at `http://127.0.0.1:8000/metrics`.

## Verify before going live

```powershell
.\.venv\Scripts\python.exe -m pytest -q
cd dashboard
npm run build
```

During a run, check all three kinds of liveness:

- `video.generation_count` continues increasing;
- `rtmp.current_fps` stays near target and `frames_dropped` remains zero;
- `rtmp.queue_seconds` stays bounded instead of growing for minutes.

`generator.backend` and `generator.uses_cloud_video` expose the active video route after the updated backend is running. `prompt.provider` and `prompt.model` show where story generation is billed.

## Optional fal.ai cloud mode

Cloud routes are retained for controlled comparisons, not used by the local recipe.

```powershell
pip install -r requirements-cloud.txt
$env:ENABLE_FAL_VIDEO = "true"
$env:FAL_KEY = "..."
```

Then explicitly select `ltx-2.3` or `h3-max`. Pricing and availability change; check fal.ai before each run. Keep `ENABLE_FAL_VIDEO=false` for local-only operation.

## Lessons learned

1. **Provider selection is a spending boundary.** A credential must never silently select a billable route.
2. **Measure the full loop.** Kernel time is not stream latency; prompt generation, file retrieval, overlays and queue policy matter.
3. **Commit the frame that viewers actually saw.** Generated-but-rejected or partially queued frames must not advance the I2V chain or story.
4. **Autoregressive video needs maintenance.** Borders, posterization and saturation errors compound when every tail becomes the next input.
5. **A live system needs a bounded failure mode.** Quality gates must recover instead of deadlocking.
6. **Continuity has layers.** Exact pixels, continuous motion, narrative state and interaction latency are separate problems.
7. **Displaying intent is not satisfying intent.** Viewer control needs a pre-stream visual audit and bounded retries; failed commands must not be captioned, committed, or written into story history.
8. **Redact operational URLs.** A Twitch RTMP URL contains the stream key in its path; logs record only a redacted marker.

## Project structure

```text
streaming_pipeline/
  core/streaming_engine.py                 story transaction + recovery
  video_generation/comfy_ltx25_backend.py  local ComfyUI LTX 2.5 graph
  video_generation/video_generator.py      provider routing + billing guard
  output/rtmp_streamer.py                  queue/backpressure + FFmpeg
  postprocessing/text_overlay.py           prompt/comment burn-in
  input/twitch_listener.py                 Twitch IRC listener
scripts/
  start-local.ps1                          safe local launcher
  restart-backend.ps1                      RTMP-safe backend restart
  start-ltx25-stream.py                    reproducible local start request
tests/
  test_i2v_continuity.py                   continuity/recovery/overlay tests
  test_prompt_controls.py                  exact comments + anti-stall tests
  test_provider_safety.py                  fal.ai opt-in guard tests
dashboard/                                 Next.js monitoring UI
```

## Acknowledgments

- [alex-remade/infinite-tv](https://github.com/alex-remade/infinite-tv) — original project and queue-liveness concept
- [Lightricks LTX Video](https://github.com/Lightricks/LTX-Video) — local video model
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — practical Windows LTX 2.5 runtime
- [FFmpeg](https://ffmpeg.org/) — RTMP and audio mixing

## License

MIT — see [LICENSE](LICENSE).
