"""WebRTC output backend -- streams LTX-generated video + audio directly
to the browser via aiortc, bypassing RTMP/Twitch entirely.

Implements the same public interface as FFmpegRTMPStreamer so the streaming
engine can use either backend without knowing which one is active:
  - add_frame_batch(pil_frames)
  - add_audio_chunk(pcm_bytes)
  - start_stream() / stop_stream()
  - is_streaming, get_status()
  - handle_signaling(ws)   [new -- WebSocket signaling for SDP/ICE]

Signaling protocol (matches fal-demos Matrix Game):
  Server -> Client:  {"type": "ready"}
  Client -> Server:  {"type": "offer", "sdp": "..."}
  Server -> Client:  {"type": "answer", "sdp": "..."}
  Both directions:   {"type": "icecandidate", "candidate": {...} | null}
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import threading
from fractions import Fraction
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from streaming_pipeline.models import Monitorable

logger = logging.getLogger(__name__)

# WebRTC audio is packetized in 20 ms frames.
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 2
AUDIO_FRAME_DURATION_S = 0.02  # 20 ms
AUDIO_SAMPLES_PER_FRAME = int(AUDIO_SAMPLE_RATE * AUDIO_FRAME_DURATION_S)  # 882
AUDIO_BYTES_PER_FRAME = AUDIO_SAMPLES_PER_FRAME * AUDIO_CHANNELS * 2  # s16le


class LTXVideoTrack:
    """Custom aiortc VideoStreamTrack fed from an asyncio.Queue.

    Constructed lazily inside handle_signaling so the aiortc import only
    happens on the fal runner (not at module load on the user's laptop).
    """

    @staticmethod
    def create(fps: int):
        from aiortc import VideoStreamTrack
        from av import VideoFrame

        class _Track(VideoStreamTrack):
            kind = "video"

            def __init__(self, fps: int):
                super().__init__()
                self._queue: asyncio.Queue = asyncio.Queue(maxsize=960)
                self._fps = max(1, fps)
                self._last_frame: Optional[VideoFrame] = None

            def push_frame(self, frame: VideoFrame) -> None:
                """Non-blocking enqueue; drops oldest on overflow."""
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                try:
                    self._queue.put_nowait(frame)
                except asyncio.QueueFull:
                    pass

            async def recv(self):
                interval = 1.0 / self._fps
                try:
                    frame = await asyncio.wait_for(
                        self._queue.get(), timeout=interval
                    )
                    self._last_frame = frame
                except (asyncio.TimeoutError, TimeoutError):
                    if self._last_frame is not None:
                        frame = self._last_frame
                    else:
                        arr = np.zeros((480, 640, 3), dtype=np.uint8)
                        frame = VideoFrame.from_ndarray(arr, format="rgb24")

                pts, time_base = await self.next_timestamp()
                frame.pts = pts
                frame.time_base = time_base
                return frame

        return _Track(fps)


class LTXAudioTrack:
    """Custom aiortc AudioStreamTrack fed from a byte buffer.

    The generation loop pushes large PCM chunks (several seconds at a time).
    This track drips them out as 20 ms AudioFrames at 44100 Hz stereo s16.
    """

    @staticmethod
    def create():
        from aiortc import AudioStreamTrack
        import av

        class _Track(AudioStreamTrack):
            kind = "audio"

            def __init__(self):
                super().__init__()
                self._buffer = bytearray()
                self._lock = asyncio.Lock()
                self._pts = 0

            def append_pcm(self, pcm_bytes: bytes) -> None:
                """Thread-safe: called from the generation thread."""
                # bytearray.extend is thread-safe in CPython (GIL)
                self._buffer.extend(pcm_bytes)

            async def recv(self):
                needed = AUDIO_BYTES_PER_FRAME

                if len(self._buffer) >= needed:
                    chunk = bytes(self._buffer[:needed])
                    del self._buffer[:needed]
                else:
                    chunk = b"\x00" * needed

                # av.AudioFrame.from_ndarray with format="s16" and layout="stereo"
                # expects shape (channels, samples) = (2, 882).  The raw PCM is
                # interleaved L,R,L,R... so we reshape to (samples, 2) then
                # transpose to (2, samples).
                interleaved = np.frombuffer(chunk, dtype=np.int16).reshape(
                    AUDIO_SAMPLES_PER_FRAME, AUDIO_CHANNELS
                )
                planar = interleaved.T.copy()  # (2, 882), C-contiguous

                frame = av.AudioFrame.from_ndarray(
                    planar, format="s16", layout="stereo"
                )
                frame.sample_rate = AUDIO_SAMPLE_RATE
                frame.pts = self._pts
                frame.time_base = Fraction(1, AUDIO_SAMPLE_RATE)
                self._pts += AUDIO_SAMPLES_PER_FRAME
                return frame

        return _Track()


class WebRTCStreamer(Monitorable):
    """WebRTC output backend for the realtime streaming pipeline.

    Lifecycle:
      1. Constructed during streaming_service.setup()
      2. start_stream() marks ready (no-op; actual WebRTC session starts when
         a browser connects via handle_signaling)
      3. The generation loop pushes frames/audio via add_frame_batch / add_audio_chunk
      4. handle_signaling(ws) runs for the lifetime of each browser connection
      5. stop_stream() tears everything down
    """

    def __init__(self, fps: int = 14, width: int = 512, height: int = 384):
        self.fps = fps
        self.width = width
        self.height = height

        self.is_streaming = False

        # Tracks (created per-connection in handle_signaling)
        self._video_track = None
        self._audio_track = None
        self._pc = None

        # Stats
        self.frames_sent = 0
        self.frames_dropped = 0
        self.audio_chunks_sent = 0
        self.start_time: Optional[float] = None

    def start_stream(self):
        """Mark streamer as ready to accept frames.

        The actual WebRTC PeerConnection is created when a browser connects
        via handle_signaling().  This mirrors FFmpegRTMPStreamer.start_stream()
        which starts the ffmpeg process; here we just flip the flag.
        """
        if self.is_streaming:
            return
        self.is_streaming = True
        self.start_time = time.time()
        self.frames_sent = 0
        self.frames_dropped = 0
        self.audio_chunks_sent = 0
        logger.info("WebRTCStreamer: ready to accept frames (waiting for browser)")

    def stop_stream(self):
        """Tear down any active PeerConnection and reset state."""
        if not self.is_streaming:
            return
        self.is_streaming = False

        if self._pc is not None:
            # Schedule close on the event loop if one is running
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._pc.close())
                else:
                    loop.run_until_complete(self._pc.close())
            except Exception:
                pass
            self._pc = None

        self._video_track = None
        self._audio_track = None
        self.frames_sent = 0
        self.frames_dropped = 0
        self.audio_chunks_sent = 0
        self.start_time = None
        logger.info("WebRTCStreamer: stopped")

    def add_frame(self, pil_frame: Image.Image) -> None:
        """Push a single PIL frame to the WebRTC video track."""
        if not self.is_streaming or self._video_track is None:
            return
        try:
            from av import VideoFrame
            arr = np.array(pil_frame.convert("RGB"))
            vf = VideoFrame.from_ndarray(arr, format="rgb24")
            self._video_track.push_frame(vf)
            self.frames_sent += 1
        except Exception as e:
            self.frames_dropped += 1
            logger.warning(f"WebRTCStreamer: failed to push frame: {e}")

    def add_frame_batch(self, pil_frames: List[Image.Image]) -> int:
        """Push multiple PIL frames. Returns count successfully enqueued."""
        if not self.is_streaming or self._video_track is None:
            return 0
        count = 0
        for f in pil_frames:
            self.add_frame(f)
            count += 1
        return count

    def add_audio_chunk(self, pcm_bytes: bytes) -> None:
        """Append PCM bytes (stereo s16le @ 44100 Hz) to the audio buffer."""
        if not self.is_streaming or self._audio_track is None:
            return
        if not pcm_bytes:
            return
        self._audio_track.append_pcm(pcm_bytes)
        self.audio_chunks_sent += 1
        logger.info(
            f"WebRTCStreamer: queued audio chunk "
            f"{len(pcm_bytes)} bytes ({len(pcm_bytes) / (AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * 2):.2f}s)"
        )

    async def handle_signaling(self, ws) -> None:
        """Run the WebRTC signaling loop for one browser connection.

        This is called from the fal app's /webrtc WebSocket endpoint.
        It stays alive for the duration of the connection.
        """
        from aiortc import (
            RTCPeerConnection,
            RTCSessionDescription,
            RTCConfiguration,
            RTCIceServer,
        )
        from starlette.websockets import WebSocketDisconnect, WebSocketState

        await ws.accept()
        logger.info("WebRTC: signaling websocket accepted")

        # Create tracks
        self._video_track = LTXVideoTrack.create(self.fps)
        self._audio_track = LTXAudioTrack.create()

        # Create peer connection
        config = RTCConfiguration(
            iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
        )
        pc = RTCPeerConnection(configuration=config)
        self._pc = pc

        stop_event = asyncio.Event()

        async def safe_send_json(data: dict):
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_json(data)
            except Exception:
                pass

        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate:
                await safe_send_json({
                    "type": "icecandidate",
                    "candidate": {
                        "candidate": candidate.candidate,
                        "sdpMid": candidate.sdpMid,
                        "sdpMLineIndex": candidate.sdpMLineIndex,
                    },
                })
            else:
                await safe_send_json({"type": "icecandidate", "candidate": None})

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"WebRTC: connection state -> {pc.connectionState}")
            if pc.connectionState in ("failed", "closed", "disconnected"):
                stop_event.set()

        # Add our outbound tracks
        pc.addTrack(self._video_track)
        pc.addTrack(self._audio_track)

        async def handle_offer(payload: dict):
            sdp = payload.get("sdp", "")
            offer = RTCSessionDescription(sdp=sdp, type="offer")
            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            await safe_send_json({
                "type": "answer",
                "sdp": pc.localDescription.sdp,
            })
            logger.info("WebRTC: SDP answer sent")

        async def handle_icecandidate(payload: dict):
            from aiortc.sdp import candidate_from_sdp

            candidate_data = payload.get("candidate")
            if candidate_data and isinstance(candidate_data, dict):
                candidate_str = candidate_data.get("candidate", "")
                sdp_mid = candidate_data.get("sdpMid", "")
                sdp_mline_index = candidate_data.get("sdpMLineIndex", 0)
                if candidate_str:
                    try:
                        parsed = candidate_from_sdp(candidate_str)
                        parsed.sdpMid = sdp_mid
                        parsed.sdpMLineIndex = sdp_mline_index
                        await pc.addIceCandidate(parsed)
                    except Exception as e:
                        logger.warning(f"WebRTC: failed to add ICE candidate: {e}")

        # Signal ready
        await safe_send_json({"type": "ready"})

        try:
            while not stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                except (asyncio.TimeoutError, TimeoutError):
                    continue
                except WebSocketDisconnect:
                    logger.info("WebRTC: client disconnected")
                    break

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type", "")
                if msg_type == "offer":
                    await handle_offer(msg)
                elif msg_type == "icecandidate":
                    await handle_icecandidate(msg)
                else:
                    logger.debug(f"WebRTC: ignoring message type '{msg_type}'")

        except Exception as e:
            logger.error(f"WebRTC: signaling error: {e}")
        finally:
            logger.info("WebRTC: closing peer connection")
            stop_event.set()
            await pc.close()
            self._pc = None
            self._video_track = None
            self._audio_track = None

    def get_status(self) -> Dict[str, Any]:
        """Return status dict compatible with ComponentMonitor."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            "is_streaming": self.is_streaming,
            "output_mode": "webrtc",
            "frames_sent": self.frames_sent,
            "frames_dropped": self.frames_dropped,
            "audio_chunks_sent": self.audio_chunks_sent,
            "current_fps": round(self.frames_sent / max(1, elapsed), 1),
            "target_fps": self.fps,
            "peer_connection_state": self._pc.connectionState if self._pc else "none",
        }
