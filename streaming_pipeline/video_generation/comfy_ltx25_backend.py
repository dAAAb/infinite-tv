"""Bridge: generate an LTX-2.5 NVFP4 clip (image-to-video + synced audio) by
calling a running ComfyUI server (default 127.0.0.1:8188) over its HTTP API.

Keeps the infinite-tv streaming engine unchanged except for one dispatch branch:
the engine passes an LTXVideoRequestI2V (prompt + first-frame image_base64 +
width/height/num_frames) and gets back PIL frames + s16le stereo PCM audio.

The engine host holds NO local pipeline (set LOAD_LTX23_PIPELINE=false); ComfyUI
owns the GPU. This module is pure HTTP + PIL + ffmpeg, so it runs from any venv.
"""
from __future__ import annotations
import base64, io, json, os, time, uuid, subprocess, urllib.request, urllib.error
from typing import Optional
from PIL import Image, ImageOps

COMFY = os.getenv("COMFY_SERVER", "127.0.0.1:8188")
COMFYUI_DIR = os.getenv("COMFYUI_DIR", "")
COMFY_INPUT_DIR = os.getenv(
    "COMFY_INPUT_DIR",
    os.path.join(COMFYUI_DIR, "input") if COMFYUI_DIR else "",
)
# Same-machine fast path: read generated PNG/FLAC straight off disk instead of 100+
# HTTP /view round-trips. Falls back to HTTP when the file isn't locally reachable.
COMFY_OUTPUT_DIR = os.getenv(
    "COMFY_OUTPUT_DIR",
    os.path.join(COMFYUI_DIR, "output") if COMFYUI_DIR else "",
)
_URL = f"http://{COMFY}"


def _read_output(item: dict) -> bytes:
    """Fetch a ComfyUI output file: local disk first (fast), else HTTP /view."""
    if COMFY_OUTPUT_DIR:
        sub = item.get("subfolder", "") or ""
        p = os.path.join(COMFY_OUTPUT_DIR, sub, item["filename"])
        if os.path.isfile(p):
            with open(p, "rb") as f:
                return f.read()
    q = f"/view?filename={item['filename']}&subfolder={item.get('subfolder','')}&type={item.get('type','output')}"
    return _get_bytes(q)

NVFP4 = "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"
GEMMA = "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
VVAE  = "ltx-2.5-video-vae-bf16.safetensors"
AVAE  = "ltx-2.5-audio-vae-bf16.safetensors"


def _get(path: str, timeout=120):
    return json.load(urllib.request.urlopen(_URL + path, timeout=timeout))

def _get_bytes(path: str, timeout=120) -> bytes:
    return urllib.request.urlopen(_URL + path, timeout=timeout).read()

def _post(path: str, data: dict, timeout=120):
    req = urllib.request.Request(_URL + path, data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"ComfyUI /prompt {e.code}: {e.read().decode(errors='replace')[:600]}")


def is_server_up() -> bool:
    try:
        urllib.request.urlopen(_URL + "/system_stats", timeout=3); return True
    except Exception:
        return False


def _write_first_frame(image_base64: str) -> str:
    """Decode the engine's base64 first frame into ComfyUI/input and return its filename."""
    if not COMFY_INPUT_DIR:
        raise RuntimeError(
            "Set COMFYUI_DIR or COMFY_INPUT_DIR so the local bridge can write "
            "the I2V handoff frame into ComfyUI/input."
        )
    raw = base64.b64decode(image_base64.split(",")[-1])
    name = f"iftv_first_{uuid.uuid4().hex[:12]}.png"
    os.makedirs(COMFY_INPUT_DIR, exist_ok=True)
    Image.open(io.BytesIO(raw)).convert("RGB").save(os.path.join(COMFY_INPUT_DIR, name))
    return name


def _prepare_handoff_frame(image_base64: str, width: int, height: int) -> Image.Image:
    """Match ComfyUI ImageScale(crop=center) for a pixel-exact stream seam."""
    raw = base64.b64decode(image_base64.split(",")[-1])
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return ImageOps.fit(
        image,
        (width, height),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


def _trim_generated_frames(frames: list[Image.Image], requested: int) -> list[Image.Image]:
    """Drop temporal padding emitted by the AV latent graph.

    With a requested LTX length of 121 the current graph decodes 129 frames. The
    final eight are padding and visibly collapse; they must never be streamed or
    recycled as the next I2V keyframe.
    """
    if len(frames) > requested:
        print(
            f"LTX temporal padding: trimming {len(frames)} decoded frames "
            f"to requested {requested}"
        )
        return frames[:requested]
    return frames


def _build_prompt_i2v_with_audio(first_frame_name: str, prompt: str, negative: str,
                  width: int, height: int, num_frames: int, frame_rate: float,
                  seed: int, strength: float) -> dict:
    """Original I2V + audio graph. Kept for offline / VOD usage."""
    return {
      "1":{"class_type":"UNETLoader","inputs":{"unet_name":NVFP4,"weight_dtype":"default"}},
      "2":{"class_type":"ModelSamplingLTXV","inputs":{"model":["1",0],"max_shift":2.05,"base_shift":0.95}},
      "3":{"class_type":"CLIPLoader","inputs":{"clip_name":GEMMA,"type":"ltxv"}},
      "4":{"class_type":"CLIPTextEncode","inputs":{"clip":["3",0],"text":prompt}},
      "5":{"class_type":"CLIPTextEncode","inputs":{"clip":["3",0],"text":negative}},
      "6":{"class_type":"LTXVConditioning","inputs":{"positive":["4",0],"negative":["5",0],"frame_rate":frame_rate}},
      "7":{"class_type":"VAELoader","inputs":{"vae_name":VVAE}},
      "8":{"class_type":"VAELoader","inputs":{"vae_name":AVAE}},
      "9":{"class_type":"LoadImage","inputs":{"image":first_frame_name}},
      "10":{"class_type":"ImageScale","inputs":{"image":["9",0],"upscale_method":"lanczos","width":width,"height":height,"crop":"center"}},
      "11":{"class_type":"EmptyLTXVLatentVideo","inputs":{"width":width,"height":height,"length":num_frames,"batch_size":1}},
      # strength < 1.0 lets the prompt dominate so the (possibly-drifting) chained
      # input frame is not copied verbatim → dampens the corruption feedback loop.
      "12":{"class_type":"LTXVAddGuide","inputs":{"positive":["6",0],"negative":["6",1],"vae":["7",0],"latent":["11",0],"image":["10",0],"frame_idx":0,"strength":strength}},
      "13":{"class_type":"LTXVEmptyLatentAudio","inputs":{"frames_number":num_frames,"frame_rate":frame_rate,"batch_size":1,"audio_vae":["8",0]}},
      "14":{"class_type":"LTXVConcatAVLatent","inputs":{"video_latent":["12",2],"audio_latent":["13",0]}},
      "15":{"class_type":"LTXVScheduler","inputs":{"steps":8,"max_shift":2.05,"base_shift":0.95,"stretch":True,"terminal":0.1,"latent":["14",0]}},
      "16":{"class_type":"KSamplerSelect","inputs":{"sampler_name":"euler"}},
      "17":{"class_type":"SamplerCustom","inputs":{"model":["2",0],"add_noise":True,"noise_seed":seed,"cfg":1.0,"positive":["12",0],"negative":["12",1],"sampler":["16",0],"sigmas":["15",0],"latent_image":["14",0]}},
      "18":{"class_type":"LTXVSeparateAVLatent","inputs":{"av_latent":["17",0]}},
      "19":{"class_type":"VAEDecode","inputs":{"samples":["18",0],"vae":["7",0]}},
      "20":{"class_type":"LTXVAudioVAEDecode","inputs":{"samples":["18",1],"audio_vae":["8",0]}},
      "21":{"class_type":"SaveImage","inputs":{"images":["19",0],"filename_prefix":"iftv_out"}},
      "22":{"class_type":"SaveAudio","inputs":{"audio":["20",0],"filename_prefix":"iftv_aud"}},
    }


def _build_prompt_i2v_video_only(first_frame_name: str, prompt: str, negative: str,
                  width: int, height: int, num_frames: int, frame_rate: float,
                  seed: int, strength: float) -> dict:
    """Pure video I2V graph for the Twitch/BGM path.

    Do not create or concatenate an unused audio latent.  The AV graph rounds a
    121-frame request up to 129 decoded frames; more importantly, it adds an
    unnecessary temporal boundary to every autoregressive handoff.
    """
    return {
      "1":{"class_type":"UNETLoader","inputs":{"unet_name":NVFP4,"weight_dtype":"default"}},
      "2":{"class_type":"ModelSamplingLTXV","inputs":{"model":["1",0],"max_shift":2.05,"base_shift":0.95}},
      "3":{"class_type":"CLIPLoader","inputs":{"clip_name":GEMMA,"type":"ltxv"}},
      "4":{"class_type":"CLIPTextEncode","inputs":{"clip":["3",0],"text":prompt}},
      "5":{"class_type":"CLIPTextEncode","inputs":{"clip":["3",0],"text":negative}},
      "6":{"class_type":"LTXVConditioning","inputs":{"positive":["4",0],"negative":["5",0],"frame_rate":frame_rate}},
      "7":{"class_type":"VAELoader","inputs":{"vae_name":VVAE}},
      "9":{"class_type":"LoadImage","inputs":{"image":first_frame_name}},
      "10":{"class_type":"ImageScale","inputs":{"image":["9",0],"upscale_method":"lanczos","width":width,"height":height,"crop":"center"}},
      "11":{"class_type":"EmptyLTXVLatentVideo","inputs":{"width":width,"height":height,"length":num_frames,"batch_size":1}},
      "12":{"class_type":"LTXVAddGuide","inputs":{"positive":["6",0],"negative":["6",1],"vae":["7",0],"latent":["11",0],"image":["10",0],"frame_idx":0,"strength":strength}},
      "15":{"class_type":"LTXVScheduler","inputs":{"steps":8,"max_shift":2.05,"base_shift":0.95,"stretch":True,"terminal":0.1,"latent":["12",2]}},
      "16":{"class_type":"KSamplerSelect","inputs":{"sampler_name":"euler"}},
      "17":{"class_type":"SamplerCustom","inputs":{"model":["2",0],"add_noise":True,"noise_seed":seed,"cfg":1.0,"positive":["12",0],"negative":["12",1],"sampler":["16",0],"sigmas":["15",0],"latent_image":["12",2]}},
      "19":{"class_type":"VAEDecode","inputs":{"samples":["17",0],"vae":["7",0]}},
      "21":{"class_type":"SaveImage","inputs":{"images":["19",0],"filename_prefix":"iftv_out"}},
    }


def _build_prompt_t2v_only(prompt: str, negative: str,
                  width: int, height: int, num_frames: int, frame_rate: float, seed: int) -> dict:
    """Pure text-to-video, NO image guide, NO audio. Each clip is independent →
    no chaining feedback loop → no cascading corruption. Audio comes from BGM in RTMP mixer."""
    return {
      "1":{"class_type":"UNETLoader","inputs":{"unet_name":NVFP4,"weight_dtype":"default"}},
      "2":{"class_type":"ModelSamplingLTXV","inputs":{"model":["1",0],"max_shift":2.05,"base_shift":0.95}},
      "3":{"class_type":"CLIPLoader","inputs":{"clip_name":GEMMA,"type":"ltxv"}},
      "4":{"class_type":"CLIPTextEncode","inputs":{"clip":["3",0],"text":prompt}},
      "5":{"class_type":"CLIPTextEncode","inputs":{"clip":["3",0],"text":negative}},
      "6":{"class_type":"LTXVConditioning","inputs":{"positive":["4",0],"negative":["5",0],"frame_rate":frame_rate}},
      "7":{"class_type":"VAELoader","inputs":{"vae_name":VVAE}},
      "8":{"class_type":"EmptyLTXVLatentVideo","inputs":{"width":width,"height":height,"length":num_frames,"batch_size":1}},
      "9":{"class_type":"LTXVScheduler","inputs":{"steps":8,"max_shift":2.05,"base_shift":0.95,"stretch":True,"terminal":0.1,"latent":["8",0]}},
      "10":{"class_type":"KSamplerSelect","inputs":{"sampler_name":"euler"}},
      "11":{"class_type":"SamplerCustom","inputs":{"model":["2",0],"add_noise":True,"noise_seed":seed,"cfg":1.0,"positive":["6",0],"negative":["6",1],"sampler":["10",0],"sigmas":["9",0],"latent_image":["8",0]}},
      "12":{"class_type":"VAEDecode","inputs":{"samples":["11",0],"vae":["7",0]}},
      "21":{"class_type":"SaveImage","inputs":{"images":["12",0],"filename_prefix":"iftv_t2v"}},
    }


def _flac_to_pcm_s16le(flac_bytes: bytes, sample_rate=44100) -> bytes:
    """Decode FLAC -> raw interleaved stereo s16le PCM via ffmpeg, resampled to 44.1 kHz
    (the RTMP streamer's fixed Twitch-ingest rate, AUDIO_SAMPLE_RATE=44100)."""
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "2", "-ar", str(sample_rate), "pipe:1"],
        input=flac_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.stdout


def generate(prompt: str, image_base64: str, width: int = 512, height: int = 288,
             num_frames: int = 121, frame_rate: float = 24.0, seed: Optional[int] = None,
             negative: str = "worst quality, inconsistent motion, blurry, jittery, distorted, static scene, frozen frame",
             strength: float = 0.5,
             want_audio: bool = True,
             mode: Optional[str] = None):
    """Generate one clip. Returns (frames: list[PIL.Image], audio_pcm: bytes|None).

    LTX25_MODE env var selects the graph:
      "t2v" (default) — pure text-to-video, no image guide, no audio. Each clip is
                        independent (no chaining) → no cascading corruption; audio
                        comes from BGM in the RTMP mixer.
      "i2v"           — original image-to-video + LTX-native audio (chaining-prone).
    """
    if not is_server_up():
        raise RuntimeError(f"ComfyUI server not reachable at {_URL} — start it first.")
    if seed is None:
        seed = int(uuid.uuid4().int % (2**31))
    mode = (mode or os.getenv("LTX25_MODE", "t2v")).lower()
    if mode == "i2v":
        first_name = _write_first_frame(image_base64)
        builder = _build_prompt_i2v_with_audio if want_audio else _build_prompt_i2v_video_only
        graph = builder(first_name, prompt, negative, width, height, num_frames,
                        frame_rate, seed, float(strength))
    else:
        graph = _build_prompt_t2v_only(prompt, negative, width, height, num_frames, frame_rate, seed)

    pid = _post("/prompt", {"prompt": graph})["prompt_id"]
    t0 = time.time()
    hist = None
    while time.time() - t0 < 600:
        h = _get(f"/history/{pid}")
        if pid in h:
            st = h[pid]["status"]
            if st.get("completed"):
                hist = h[pid]; break
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
                raise RuntimeError(f"ComfyUI exec error: {json.dumps(msgs)[:500]}")
        time.sleep(0.5)
    if hist is None:
        raise RuntimeError("ComfyUI generation timed out")

    outs = hist.get("outputs", {})
    # frames from SaveImage node 21
    frames = []
    for img in outs.get("21", {}).get("images", []):
        frames.append(Image.open(io.BytesIO(_read_output(img))).convert("RGB"))
    frames = _trim_generated_frames(frames, num_frames)
    # LTXVAddGuide anchors frame 0 but VAE encode/decode still changes its pixels.
    # Replace the streamed first frame with the exact committed previous tail. This
    # makes clip N's final frame and clip N+1's first frame identical on screen.
    if mode == "i2v" and image_base64 and frames:
        frames[0] = _prepare_handoff_frame(image_base64, width, height)
    # audio from SaveAudio node 22 (only exists in i2v graph)
    audio_pcm = None
    if want_audio and "22" in outs:
        auds = outs.get("22", {}).get("audio", [])
        if auds:
            audio_pcm = _flac_to_pcm_s16le(_read_output(auds[0]))
    return frames, audio_pcm
