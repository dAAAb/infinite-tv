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
from PIL import Image, ImageDraw, ImageFilter, ImageOps

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


def _write_image(image: Image.Image, prefix: str) -> str:
    """Write a prepared guide image to ComfyUI/input and return its filename."""
    if not COMFY_INPUT_DIR:
        raise RuntimeError(
            "Set COMFYUI_DIR or COMFY_INPUT_DIR so the local bridge can write "
            "guide frames into ComfyUI/input."
        )
    name = f"{prefix}_{uuid.uuid4().hex[:12]}.png"
    os.makedirs(COMFY_INPUT_DIR, exist_ok=True)
    image.convert("RGB").save(os.path.join(COMFY_INPUT_DIR, name))
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


def _prepend_exact_handoff(
    frames: list[Image.Image], source: Image.Image
) -> list[Image.Image]:
    """Insert the exact streamed tail without discarding ComfyUI's decoded frame 0.

    Replacing decoded frame 0 made the visible transition jump directly from the
    exact source to decoded frame 1. Prepending the source and dropping the most
    failure-prone terminal frame preserves the model's natural first motion step
    while keeping both the requested length and a pixel-exact seam.
    """
    if not frames:
        return frames
    source = source.convert("RGB").resize(frames[0].size, Image.Resampling.LANCZOS)
    if len(frames) == 1:
        return [source]
    return [source] + list(frames[:-1])


def _frame_sharpness(frame: Image.Image) -> float:
    """Return a small, dependency-free edge-energy score."""
    gray = frame.convert("L").resize((256, 144), Image.Resampling.BILINEAR)
    pixels = list(gray.getdata())
    width, height = gray.size
    energy = 0.0
    samples = 0
    for y in range(height - 1):
        row = y * width
        next_row = row + width
        for x in range(width - 1):
            value = pixels[row + x]
            dx = value - pixels[row + x + 1]
            dy = value - pixels[next_row + x]
            energy += dx * dx + dy * dy
            samples += 1
    return energy / max(1, samples)


def _scene_locked_target(
    source: Image.Image,
    target: Image.Image,
    preserve_scene: bool = True,
) -> Image.Image:
    """Keep the current set around a prompt-first target's changed subject.

    Pure T2V is useful for difficult transformations but invents a new location.
    Preserve most of the original frame globally and feather the completed target
    into the central action region. The subsequent first/last-frame LTX pass turns
    this still composite into a coherent transition instead of streaming it raw.
    """
    source = source.convert("RGB")
    target = ImageOps.fit(
        target.convert("RGB"),
        source.size,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    if not preserve_scene:
        return target

    width, height = source.size
    base = Image.blend(source, target, 0.18)
    mask = Image.new("L", source.size, 0)
    draw = ImageDraw.Draw(mask)
    margin_x = max(8, int(width * 0.14))
    margin_y = max(8, int(height * 0.12))
    draw.rounded_rectangle(
        (margin_x, margin_y, width - margin_x, height - margin_y),
        radius=max(8, int(min(width, height) * 0.10)),
        fill=255,
    )
    mask = mask.filter(ImageFilter.GaussianBlur(max(4, int(min(width, height) * 0.06))))
    return Image.composite(target, base, mask)


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


def _build_prompt_flf_video_only(
    first_frame_name: str,
    last_frame_name: str,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    num_frames: int,
    frame_rate: float,
    seed: int,
    strength: float = 0.7,
) -> dict:
    """Official-style first/last-frame graph for a scene-preserving bridge."""
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
      "30":{"class_type":"LoadImage","inputs":{"image":last_frame_name}},
      "31":{"class_type":"ImageScale","inputs":{"image":["30",0],"upscale_method":"lanczos","width":width,"height":height,"crop":"center"}},
      "13":{"class_type":"LTXVAddGuide","inputs":{"positive":["12",0],"negative":["12",1],"vae":["7",0],"latent":["12",2],"image":["31",0],"frame_idx":-1,"strength":strength}},
      "15":{"class_type":"LTXVScheduler","inputs":{"steps":8,"max_shift":2.05,"base_shift":0.95,"stretch":True,"terminal":0.1,"latent":["13",2]}},
      "16":{"class_type":"KSamplerSelect","inputs":{"sampler_name":"euler"}},
      "17":{"class_type":"SamplerCustom","inputs":{"model":["2",0],"add_noise":True,"noise_seed":seed,"cfg":1.0,"positive":["13",0],"negative":["13",1],"sampler":["16",0],"sigmas":["15",0],"latent_image":["13",2]}},
      "19":{"class_type":"VAEDecode","inputs":{"samples":["17",0],"vae":["7",0]}},
      "21":{"class_type":"SaveImage","inputs":{"images":["19",0],"filename_prefix":"iftv_bridge"}},
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


def _execute_graph(graph: dict) -> dict:
    """Submit one ComfyUI graph and return its completed output map."""
    pid = _post("/prompt", {"prompt": graph})["prompt_id"]
    t0 = time.time()
    hist = None
    while time.time() - t0 < 600:
        history = _get(f"/history/{pid}")
        if pid in history:
            status = history[pid]["status"]
            if status.get("completed"):
                hist = history[pid]
                break
            if status.get("status_str") == "error":
                messages = [
                    message
                    for message in status.get("messages", [])
                    if message[0] == "execution_error"
                ]
                raise RuntimeError(
                    f"ComfyUI exec error: {json.dumps(messages)[:500]}"
                )
        time.sleep(0.5)
    if hist is None:
        raise RuntimeError("ComfyUI generation timed out")
    return hist.get("outputs", {})


def _read_frames(outputs: dict, node_id: str = "21") -> list[Image.Image]:
    return [
        Image.open(io.BytesIO(_read_output(item))).convert("RGB")
        for item in outputs.get(node_id, {}).get("images", [])
    ]


def generate(prompt: str, image_base64: str, width: int = 512, height: int = 288,
             num_frames: int = 145, frame_rate: float = 24.0, seed: Optional[int] = None,
             negative: str = "worst quality, inconsistent motion, blurry, jittery, distorted, static scene, frozen frame",
             strength: float = 0.5,
             want_audio: bool = True,
             mode: Optional[str] = None,
             scene_description: str = "",
             preserve_scene: bool = True):
    """Generate one clip. Returns (frames: list[PIL.Image], audio_pcm: bytes|None).

    LTX25_MODE env var selects the graph:
      "t2v" (default) — pure text-to-video, no image guide, no audio. Each clip is
                        independent (no chaining) → no cascading corruption; audio
                        comes from BGM in the RTMP mixer.
      "i2v"           — first-frame image-to-video continuity.
      "scene_bridge"  — local prompt-first target followed by first/last-frame I2V.
    """
    if not is_server_up():
        raise RuntimeError(f"ComfyUI server not reachable at {_URL} — start it first.")
    if seed is None:
        seed = int(uuid.uuid4().int % (2**31))
    mode = (mode or os.getenv("LTX25_MODE", "t2v")).lower()
    source = None
    if mode == "scene_bridge":
        if not image_base64:
            raise ValueError("scene_bridge requires the previous streamed frame")
        source = _prepare_handoff_frame(image_base64, width, height)
        first_name = _write_image(source, "iftv_bridge_first")
        if preserve_scene:
            target_prompt = (
                f"{prompt} Completed end-state keyframe. "
                f"Keep this exact same setting recognizable: {scene_description}. "
                "Same background layout, lighting, camera axis, and existing characters; "
                "no cut, no new room, no replacement landscape."
            )
            target_negative = (
                f"{negative}, different location, replacement background, scene cut, "
                "unrelated set, changed architecture"
            )
        else:
            target_prompt = (
                f"{prompt} Completed end-state keyframe at the explicitly requested "
                "destination, with all requested actions visibly finished."
            )
            target_negative = negative
        target_graph = _build_prompt_t2v_only(
            target_prompt,
            target_negative,
            width,
            height,
            25,
            frame_rate,
            seed,
        )
        target_frames = _trim_generated_frames(_read_frames(_execute_graph(target_graph)), 25)
        if not target_frames:
            raise RuntimeError("ComfyUI scene bridge target produced no frames")
        tail = target_frames[-min(9, len(target_frames)):]
        sharpness = [_frame_sharpness(frame) for frame in tail]
        threshold = max(sharpness) * 0.70
        target = next(
            frame
            for frame, score in reversed(list(zip(tail, sharpness)))
            if score >= threshold
        )
        target = _scene_locked_target(source, target, preserve_scene=preserve_scene)
        last_name = _write_image(target, "iftv_bridge_last")
        graph = _build_prompt_flf_video_only(
            first_name,
            last_name,
            prompt,
            target_negative,
            width,
            height,
            num_frames,
            frame_rate,
            (seed + 1) % (2**31),
            strength=0.7,
        )
    elif mode == "i2v":
        first_name = _write_first_frame(image_base64)
        builder = _build_prompt_i2v_with_audio if want_audio else _build_prompt_i2v_video_only
        graph = builder(first_name, prompt, negative, width, height, num_frames,
                        frame_rate, seed, float(strength))
    else:
        graph = _build_prompt_t2v_only(prompt, negative, width, height, num_frames, frame_rate, seed)

    outs = _execute_graph(graph)
    # frames from SaveImage node 21
    frames = _read_frames(outs)
    frames = _trim_generated_frames(frames, num_frames)
    # LTXVAddGuide anchors frame 0 but VAE encode/decode still changes its pixels.
    # Prepend the exact committed tail, retain decoded frame 0 as the next motion
    # step, and discard the weakest raw terminal frame.
    if mode in {"i2v", "scene_bridge"} and image_base64 and frames:
        if source is None:
            source = _prepare_handoff_frame(image_base64, width, height)
        frames = _prepend_exact_handoff(frames, source)
    # audio from SaveAudio node 22 (only exists in i2v graph)
    audio_pcm = None
    if want_audio and "22" in outs:
        auds = outs.get("22", {}).get("audio", [])
        if auds:
            audio_pcm = _flac_to_pcm_s16le(_read_output(auds[0]))
    return frames, audio_pcm
