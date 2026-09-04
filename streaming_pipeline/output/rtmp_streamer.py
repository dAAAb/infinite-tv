import ffmpeg
import os
import sys
import tempfile
import threading
import time
import numpy as np
from queue import Queue, Empty
from PIL import Image, ImageDraw, ImageFont
import cv2
from streaming_pipeline.utils.logger_config import queue_log
from streaming_pipeline.models import Monitorable

# Windows named-pipe helpers (used when enable_audio=True on Windows).
_IS_WINDOWS = sys.platform == "win32"
if _IS_WINDOWS:
    import ctypes
    import ctypes.wintypes
    import msvcrt
    _kernel32 = ctypes.windll.kernel32


# Audio output is fixed at stereo s16le @ 44.1 kHz (Twitch ingest spec).
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 2
AUDIO_BYTES_PER_SAMPLE = 2  # int16
AUDIO_BYTES_PER_SECOND = AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_BYTES_PER_SAMPLE


class FFmpegRTMPStreamer(Monitorable):
    def __init__(self, stream_key: str, fps: int = 24, width: int = 640, height: int = 480, enable_audio: bool = False, bgm_path: str | None = None):
        self.stream_key = stream_key
        self.fps = fps
        self.width = width
        self.height = height
        self.enable_audio = enable_audio
        self.bgm_path = bgm_path  # Optional background music file (loops forever)
        # Fix: Use correct Twitch RTMP path
        self.rtmp_url = f"rtmp://live.twitch.tv/app/{stream_key}"

        # Stream state
        self.is_streaming = False
        self.ffmpeg_process = None
        self.stream_thread = None
        self.monitor_thread = None

        # Frame management. Keep enough room to spread a generated clip across
        # several minutes of real-time playback without dropping early frames.
        self.frame_queue = Queue(maxsize=max(1000, int(self.fps * 600)))

        # Audio plumbing (only used when enable_audio=True)
        self.audio_queue: "Queue[bytes]" = Queue(maxsize=64)  # ~64 clips of audio
        self.audio_thread: threading.Thread | None = None
        self.audio_fifo_path: str | None = None
        self.audio_fifo_fd: int | None = None
        self._win_pipe_handle = None  # Windows named pipe HANDLE
        # Bytes streamed; used to throttle silence injection so we don't
        # outpace ffmpeg's audio decoder when the queue is empty.
        self.audio_bytes_written = 0
        self.audio_start_time: float | None = None

        # Statistics
        self.frames_sent = 0
        self.frames_dropped = 0
        self.frames_added_total = 0
        self.frames_added_last_second = 0
        self.frames_dropped_last_second = 0
        self.start_time = None
        self.placeholder_frame = None

    def start_stream(self):
        """Start FFmpeg RTMP stream to Twitch"""
        if self.is_streaming:
            print("⚠️ Stream already running")
            return
        
        try:
            print(f"🔗 Starting FFmpeg RTMP stream to Twitch...")
            print(f"   Resolution: {self.width}x{self.height}")
            print(f"   FPS: {self.fps}")
            print(f"   Audio: {'native (LTX 2.3 vocoder)' if self.enable_audio else 'silent (anullsrc)'}")
            # The path is the Twitch stream key. Never emit even a prefix of it.
            print("   RTMP URL: configured (<stream-key redacted>)")
            self.placeholder_frame = self._load_placeholder_frame()

            video_in = ffmpeg.input(
                'pipe:',
                format='rawvideo',
                pix_fmt='rgb24',
                s=f'{self.width}x{self.height}',
                framerate=self.fps,  # Use 'framerate' instead of 'r' for raw pipe
            )

            # Build the audio input: native PCM via pipe if enabled, else silent.
            if self.enable_audio:
                # Create a platform-appropriate named pipe so ffmpeg can read
                # raw PCM while we write from _audio_loop.
                if _IS_WINDOWS:
                    pipe_name = rf"\\.\pipe\ltx_audio_{os.getpid()}"
                    PIPE_ACCESS_OUTBOUND = 0x00000002
                    PIPE_TYPE_BYTE = 0x00000000
                    PIPE_WAIT = 0x00000000
                    INVALID_HANDLE = ctypes.wintypes.HANDLE(-1).value
                    buf_sz = 1 << 16  # 64 KiB

                    handle = _kernel32.CreateNamedPipeW(
                        pipe_name,
                        PIPE_ACCESS_OUTBOUND,
                        PIPE_TYPE_BYTE | PIPE_WAIT,
                        1,       # max instances
                        buf_sz,  # out buffer
                        buf_sz,  # in buffer
                        0,       # default timeout
                        None,    # security attrs
                    )
                    if handle == INVALID_HANDLE:
                        raise OSError(f"CreateNamedPipeW failed (err={ctypes.GetLastError()})")
                    self._win_pipe_handle = handle
                    self.audio_fifo_path = pipe_name
                    queue_log.info(f"🔊 Created Windows named pipe at {pipe_name}")
                else:
                    # Unix: classic FIFO
                    fifo_dir = tempfile.mkdtemp(prefix="ltx_audio_")
                    self.audio_fifo_path = os.path.join(fifo_dir, "audio.pcm")
                    os.mkfifo(self.audio_fifo_path)
                    queue_log.info(f"🔊 Created Unix FIFO at {self.audio_fifo_path}")

                audio_in = ffmpeg.input(
                    self.audio_fifo_path,
                    format='s16le',
                    ar=AUDIO_SAMPLE_RATE,
                    ac=AUDIO_CHANNELS,
                )
            else:
                audio_in = ffmpeg.input(
                    'anullsrc=channel_layout=stereo:sample_rate=44100',
                    f='lavfi',
                )

            # Mix in background music if a BGM file is provided.
            if self.bgm_path and os.path.isfile(self.bgm_path):
                bgm_in = ffmpeg.input(self.bgm_path, stream_loop=-1)
                # BGM is the reliable bed (louder); the LTX-generated audio is still
                # unstable, so it sits underneath. normalize=0 keeps the literal
                # 0.70 / 0.30 mix instead of amix's default 1/n auto-scaling.
                bgm_v = float(os.getenv("RTMP_BGM_VOLUME", "0.70"))
                gen_v = float(os.getenv("RTMP_GEN_AUDIO_VOLUME", "0.30"))
                bgm_loud = bgm_in.filter('volume', bgm_v)
                gen_quiet = audio_in.filter('volume', gen_v)
                audio_in = ffmpeg.filter([gen_quiet, bgm_loud], 'amix',
                                         inputs=2, duration='longest',
                                         dropout_transition=0, normalize=0)
                queue_log.info(f"🎵 BGM mixed in: {self.bgm_path} (BGM={bgm_v}, gen_audio={gen_v}, loop)")

            self.ffmpeg_process = (
                ffmpeg
                .output(
                    video_in, audio_in, self.rtmp_url,
                    vcodec='libx264',
                    pix_fmt='yuv420p',
                    preset='faster',
                    tune='zerolatency',
                    g=self.fps,
                    maxrate='1500k',
                    bufsize='3000k',
                    **{'b:v': '1500k'},
                    acodec='aac',
                    **{'b:a': '128k'},
                    ar='44100',
                    ac='2',
                    f='flv',
                    flvflags='no_duration_filesize',
                )
                .global_args('-loglevel', 'warning')
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stderr=True)
            )

            self.is_streaming = True
            self.start_time = time.time()

            # Start the streaming loop
            queue_log.info("📺 Starting continuous frame streaming loop...")
            self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
            self.stream_thread.start()

            # Start the audio loop if native audio is enabled.
            if self.enable_audio:
                self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
                self.audio_thread.start()
                queue_log.info("🔊 Audio FIFO writer thread started")

            # Start a stderr monitor so FFmpeg errors are visible in logs.
            self.monitor_thread = threading.Thread(target=self._monitor_ffmpeg, daemon=True)
            self.monitor_thread.start()

            queue_log.info(f"✅ FFmpeg RTMP stream started - NOW LIVE ON TWITCH!")
            queue_log.info("🔗 RTMP URL: configured (<stream-key redacted>)")
            queue_log.info(f"📐 Resolution: {self.width}x{self.height} @ {self.fps}fps")

        except Exception as e:
            queue_log.error(f"❌ Failed to start FFmpeg RTMP stream: {e}")
            self.is_streaming = False
            self._cleanup_audio_fifo()

    def stop_stream(self):
        """Stop FFmpeg RTMP stream"""
        if not self.is_streaming:
            return

        print("🛑 Stopping FFmpeg RTMP stream...")
        self.is_streaming = False

        # Close FFmpeg process
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.wait(timeout=5)
            except:
                self.ffmpeg_process.kill()
            self.ffmpeg_process = None

        # Tear down the audio FIFO + writer fd.
        self._cleanup_audio_fifo()

        # Clear queue and reset metrics when stopped
        self._reset_metrics()
        print("✅ FFmpeg RTMP stream stopped")

    def _cleanup_audio_fifo(self):
        """Close the audio pipe/FIFO and clean up resources."""
        if self.audio_fifo_fd is not None:
            try:
                os.close(self.audio_fifo_fd)
            except OSError:
                pass
            self.audio_fifo_fd = None
        if _IS_WINDOWS and self._win_pipe_handle is not None:
            try:
                _kernel32.CloseHandle(self._win_pipe_handle)
            except Exception:
                pass
            self._win_pipe_handle = None
        elif self.audio_fifo_path and not _IS_WINDOWS:
            try:
                os.unlink(self.audio_fifo_path)
            except OSError:
                pass
            try:
                os.rmdir(os.path.dirname(self.audio_fifo_path))
            except OSError:
                pass
        self.audio_fifo_path = None

    def _reset_metrics(self):
        """Reset all metrics and clear queue when stream stops"""
        # Clear the frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break

        # Clear the audio queue too.
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break

        # Reset counters
        self.frames_sent = 0
        self.frames_dropped = 0
        self.frames_added_total = 0
        self.frames_added_last_second = 0
        self.frames_dropped_last_second = 0
        self.audio_bytes_written = 0
        self.audio_start_time = None
        self.start_time = None
        print("🧹 RTMP metrics and queue cleared")

    def add_audio_chunk(self, pcm_bytes: bytes) -> None:
        """Enqueue stereo s16le @ 44.1 kHz PCM bytes for streaming.

        Safe to call when audio is disabled -- it just becomes a no-op.
        """
        if not self.enable_audio or not pcm_bytes:
            return
        if not self.is_streaming:
            return
        try:
            self.audio_queue.put_nowait(pcm_bytes)
            queue_log.info(f"🔊 Queued audio chunk: {len(pcm_bytes)} bytes ({len(pcm_bytes) / AUDIO_BYTES_PER_SECOND:.2f}s)")
        except Exception as e:
            queue_log.warning(f"⚠️ Failed to enqueue audio chunk: {e}")

    def _audio_loop(self):
        """Background thread that writes audio PCM to the FIFO at real-time rate.

        Opens the FIFO for writing (blocks until ffmpeg opens the read end),
        then drains audio_queue and pads with silence to keep ffmpeg's audio
        decoder fed at 44100 stereo samples/sec.  If we ever fall behind we
        let ffmpeg's PTS handling sort it out -- it's better to overshoot
        slightly than to underrun and stall the whole pipeline.
        """
        try:
            if _IS_WINDOWS and self._win_pipe_handle is not None:
                # Wait for ffmpeg to connect to the named pipe (blocking).
                queue_log.info(f"🔊 Waiting for FFmpeg to connect to named pipe: {self.audio_fifo_path}")
                connected = _kernel32.ConnectNamedPipe(self._win_pipe_handle, None)
                if not connected and ctypes.GetLastError() != 535:  # ERROR_PIPE_CONNECTED
                    raise OSError(f"ConnectNamedPipe failed (err={ctypes.GetLastError()})")
                # Convert Windows HANDLE → C runtime fd → Python can os.write()
                self.audio_fifo_fd = msvcrt.open_osfhandle(self._win_pipe_handle, 0)
                queue_log.info(f"🔊 Windows named pipe connected; streaming PCM at {AUDIO_SAMPLE_RATE} Hz stereo")
            else:
                # Unix: blocking open returns once ffmpeg has opened the FIFO for reading.
                queue_log.info(f"🔊 Opening audio FIFO for writing: {self.audio_fifo_path}")
                self.audio_fifo_fd = os.open(self.audio_fifo_path, os.O_WRONLY)
                queue_log.info(f"🔊 Audio FIFO writer connected; streaming PCM at {AUDIO_SAMPLE_RATE} Hz stereo")
        except Exception as e:
            queue_log.error(f"❌ Could not open audio pipe: {e}")
            return

        # Pace silence padding so we don't write faster than wall-clock.
        # When the queue has data we write as fast as we can (ffmpeg buffers).
        silence_chunk_seconds = 0.1
        silence_chunk = b"\x00" * int(AUDIO_BYTES_PER_SECOND * silence_chunk_seconds)
        self.audio_start_time = time.time()

        while self.is_streaming:
            try:
                try:
                    pcm = self.audio_queue.get(timeout=0.05)
                    os.write(self.audio_fifo_fd, pcm)
                    self.audio_bytes_written += len(pcm)
                    continue
                except Empty:
                    pass

                # Underrun: pad with silence, but only if we've fallen behind
                # wall clock.  This keeps ffmpeg's audio decoder fed without
                # spamming gigabytes when generation is comfortably ahead.
                wall_seconds = time.time() - self.audio_start_time
                target_bytes = int(wall_seconds * AUDIO_BYTES_PER_SECOND)
                if self.audio_bytes_written < target_bytes:
                    deficit_bytes = target_bytes - self.audio_bytes_written
                    chunk = silence_chunk if deficit_bytes >= len(silence_chunk) else b"\x00" * deficit_bytes
                    os.write(self.audio_fifo_fd, chunk)
                    self.audio_bytes_written += len(chunk)
                else:
                    # Comfortably ahead -- just sleep a bit.
                    time.sleep(0.01)

            except (BrokenPipeError, OSError) as e:
                queue_log.error(f"❌ Audio FIFO write failed (ffmpeg likely exited): {e}")
                break
            except Exception as e:
                queue_log.error(f"❌ Audio loop error: {e}")
                time.sleep(0.05)

        queue_log.info("🔊 Audio loop ended")

    

    def _enqueue_frame_array(self, frame_array):
        """Put a prepared RGB frame into the stream queue."""
        try:
            self.frame_queue.put_nowait(frame_array)
        except:
            # Queue full - drop oldest frame and add new one
            try:
                self.frame_queue.get_nowait()
                self.frames_dropped += 1
                self.frame_queue.put_nowait(frame_array)
            except Empty:
                pass

    def add_frame(self, pil_frame):
        """Add PIL Image frame to stream queue"""
        if not self.is_streaming:
            return
        
        try:
            # Convert PIL frame to numpy array for FFmpeg
            frame_array = np.array(pil_frame.convert('RGB'))
            if frame_array.shape[:2] != (self.height, self.width):
                frame_array = cv2.resize(frame_array, (self.width, self.height))
            
            self._enqueue_frame_array(frame_array)
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")

    def add_frame_batch(self, pil_frames, playback_seconds: float | None = None):
        """Add multiple frames efficiently using batch processing.

        If playback_seconds is provided and longer than the clip's native
        duration, spread the generated frames across that wall-clock window.
        This avoids the Twitch-visible pattern where a newly generated clip
        plays quickly, then the stream repeats the last frame while waiting for
        the next generation.
        """
        if not self.is_streaming:
            queue_log.warning(f"❌ RTMP not streaming - rejecting {len(pil_frames) if pil_frames else 0} frames")
            return 0
            
        if not pil_frames:
            queue_log.warning("❌ No frames provided to add_frame_batch")
            return 0
        
        clip_seconds = len(pil_frames) / max(1, float(self.fps))
        target_seconds = max(clip_seconds, float(playback_seconds or 0))
        target_frame_count = max(
            len(pil_frames),
            int(np.ceil(target_seconds * float(self.fps))),
        )
        max_hold_seconds = float(os.getenv("RTMP_MAX_GENERATED_FRAME_HOLD_SECONDS", "6"))
        max_repeat_per_frame = max(1, int(round(max_hold_seconds * float(self.fps))))
        max_target_count = len(pil_frames) * max_repeat_per_frame
        if target_frame_count > max_target_count:
            queue_log.info(
                f"📺 DRIP CAP: limiting generated frame hold from "
                f"{target_frame_count / len(pil_frames) / float(self.fps):.1f}s "
                f"to {max_hold_seconds:.1f}s per source frame"
            )
            target_frame_count = max_target_count

        queue_log.info(f"📺 BATCH START: Processing {len(pil_frames)} frames...")
        if target_frame_count > len(pil_frames):
            queue_log.info(
                f"📺 DRIP PLAYBACK: evenly resampling {len(pil_frames)} generated "
                f"frames to {target_frame_count} stream frames "
                f"(~{target_frame_count / float(self.fps):.1f}s)"
            )
        queue_log.info(f"📊 Current queue size: {self.frame_queue.qsize()}/{self.frame_queue.maxsize}")
        
        batch_start_time = time.time()
        prepared_frames = []

        # Convert the complete source batch before enqueueing anything.  This
        # preserves the engine's all-or-nothing handoff transaction.
        for pil_frame in pil_frames:
            try:
                frame_array = np.array(pil_frame.convert('RGB'))
                if frame_array.shape[:2] != (self.height, self.width):
                    frame_array = cv2.resize(frame_array, (self.width, self.height))
                prepared_frames.append(frame_array)
            except Exception as e:
                print(f"❌ Error processing frame in batch: {e}")
                return 0

        # Distribute repeated frames across the clip rather than holding the
        # tail for several seconds.  Nearest-neighbour temporal resampling keeps
        # pixels intact and avoids cross-fade ghost images.
        source_indices = np.rint(
            np.linspace(0, len(prepared_frames) - 1, target_frame_count)
        ).astype(int)
        for source_index in source_indices:
            self._enqueue_frame_array(prepared_frames[int(source_index)])

        batch_duration = time.time() - batch_start_time
        batch_fps = target_frame_count / batch_duration if batch_duration > 0 else 0

        queue_log.info(f"📺 BATCH COMPLETE: {target_frame_count} stream frames from {len(pil_frames)} generated frames in {batch_duration:.2f}s ({batch_fps:.1f} fps)")
        queue_log.info(f"📊 Final queue size: {self.frame_queue.qsize()}/{self.frame_queue.maxsize}")

        return len(pil_frames)

    def _monitor_ffmpeg(self):
        """Read FFmpeg stderr in background so errors are logged."""
        try:
            while self.ffmpeg_process and self.ffmpeg_process.stderr:
                line = self.ffmpeg_process.stderr.readline()
                if not line:
                    break
                decoded = line.decode(errors='replace').strip()
                if decoded:
                    queue_log.warning(f"⚠️ FFmpeg: {decoded}")
        except Exception as e:
            queue_log.error(f"FFmpeg monitor error: {e}")
        # Log exit code
        if self.ffmpeg_process:
            rc = self.ffmpeg_process.poll()
            queue_log.error(f"❌ FFmpeg exited with code {rc}")

    def _stream_loop(self):
        """Send frames to FFmpeg at consistent FPS - REDUCED LOGGING"""
        frame_duration = 1.0 / self.fps
        last_real_frame = None
        frame_repeat_count = 0
        last_queue_size = 0
        
        queue_log.info("📺 Starting continuous frame streaming loop...")
        queue_log.info(f"📺 Target FPS: {self.fps}, Frame duration: {frame_duration:.3f}s")
        
        loop_count = 0
        while self.is_streaming and self.ffmpeg_process:
            loop_count += 1
            
            # Fix: Check if ffmpeg already exited
            if self.ffmpeg_process.poll() is not None:
                queue_log.error("❌ FFmpeg process ended; stopping stream loop.")
                self.is_streaming = False
                break
                
            # Debug logging every 30 seconds
            if loop_count % (self.fps * 30) == 0:
                queue_log.info(f"🔄 Stream loop alive: {loop_count} iterations, queue: {self.frame_queue.qsize()}")

            loop_start = time.time()
            
            try:
                # Try to get a real frame from the queue
                current_queue_size = self.frame_queue.qsize()
                
              
                
                try:
                    # Use shorter timeout for better responsiveness
                    timeout = 0.1 if current_queue_size == 0 else 0.001
                    frame = self.frame_queue.get(timeout=timeout)
                    last_real_frame = frame
                    frame_repeat_count = 0
                    
                    # Only log significant queue changes
                    if last_queue_size == 0 and current_queue_size > 5:
                        print(f"📺 Queue building up: {current_queue_size} frames")
                        
                except Empty:
                    # Use last frame with subtle variation
                    if last_real_frame is not None:
                        frame = self._create_varied_frame(last_real_frame, frame_repeat_count)
                        frame_repeat_count += 1
                        
                        # Only log queue empty occasionally
                        if frame_repeat_count % (self.fps * 5) == 0:  # Every 5 seconds
                            print(f"⚠️ Queue empty for {frame_repeat_count/self.fps:.1f}s - repeating frames")
                    else:
                        frame = self._create_placeholder_frame(self.frames_sent)
                        if self.frames_sent % (self.fps * 5) == 0:  # Every 5 seconds
                            print(f"⚠️ No frames available - using placeholder")

                last_queue_size = current_queue_size

                # Fix: Check if stdin is still available
                if not self.ffmpeg_process or not self.ffmpeg_process.stdin:
                    print("❌ FFmpeg stdin unavailable.")
                    self.is_streaming = False
                    break

                # Send frame to FFmpeg
                self.ffmpeg_process.stdin.write(frame.tobytes())
                self.ffmpeg_process.stdin.flush()
                self.frames_sent += 1

            except (BrokenPipeError, ValueError) as e:
                print(f"❌ Streaming error (pipe closed): {e}")
                self.is_streaming = False
                break
            except Exception as e:
                print(f"❌ Streaming error: {e}")
                time.sleep(0.1)
                continue
            
            # Maintain precise FPS timing
            elapsed = time.time() - loop_start
            sleep_time = max(0, frame_duration - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Only log performance issues occasionally
                if elapsed > frame_duration * 2 and self.frames_sent % (self.fps * 2) == 0:  # Every 2 seconds, only if 2x slower
                    print(f"⚠️ Frame processing slow: {elapsed:.3f}s (target: {frame_duration:.3f}s)")
        
        print("📺 Frame streaming loop ended")

    def _load_placeholder_frame(self):
        """Load a visible fallback frame so an empty queue never turns Twitch black."""
        candidates = [
            os.getenv("RTMP_PLACEHOLDER_IMAGE"),
            os.path.join(os.getcwd(), "outputs", "twitch-rubberhose", "twitch-rubberhose-01.png"),
            os.path.join(os.getcwd(), "outputs", "twitch-rubberhose", "twitch-rubberhose-01-day.png"),
        ]
        for path in candidates:
            if not path or not os.path.exists(path):
                continue
            try:
                image = Image.open(path).convert("RGB")
                frame_array = np.array(image)
                if frame_array.shape[:2] != (self.height, self.width):
                    frame_array = cv2.resize(frame_array, (self.width, self.height))
                queue_log.info(f"🖼️ Loaded RTMP placeholder image: {path}")
                return frame_array
            except Exception as e:
                queue_log.warning(f"⚠️ Could not load RTMP placeholder image {path}: {e}")
        return None

    def _create_placeholder_frame(self, frame_count):
        """Create a visible placeholder frame when no content is available."""
        if self.placeholder_frame is not None:
            return self._create_varied_frame(self.placeholder_frame, frame_count)

        frame = np.full((self.height, self.width, 3), 24, dtype=np.uint8)
        
        # Optional: Add subtle visual indicator
        if frame_count % (self.fps * 4) < (self.fps * 2):  # Blink every 4 seconds
            frame[10:20, 10:20] = [80, 80, 80]  # Small gray square
        
        return frame

    def _create_varied_frame(self, base_frame, variation_count):
        """Create subtle variation of the last frame to avoid static appearance"""
        # Make a copy to avoid modifying the original
        frame = base_frame.copy()
        
        # Add very subtle brightness variation (±1-2 levels)
        variation = (variation_count % 3) - 1  # -1, 0, or 1
        if variation != 0:
            frame = np.clip(frame.astype(np.int16) + variation, 0, 255).astype(np.uint8)
        
        return frame


    def get_status(self) -> dict:
        """Get current streaming status and performance metrics"""
        queue_size = self.frame_queue.qsize()
        return {
            "is_streaming": self.is_streaming,
            "frames_sent": self.frames_sent,
            "frames_dropped": self.frames_dropped,
            "queue_size": queue_size,
            "queue_seconds": round(queue_size / max(1.0, float(self.fps)), 1),
            "current_fps": round(self.frames_sent / max(1, time.time() - (self.start_time or time.time())), 1),
            "target_fps": self.fps
        }

