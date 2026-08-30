import ffmpeg
import os
import tempfile
import threading
import time
import numpy as np
from queue import Queue, Empty
from PIL import Image, ImageDraw, ImageFont
import cv2
from streaming_pipeline.utils.logger_config import queue_log
from streaming_pipeline.models import Monitorable


# Audio output is fixed at stereo s16le @ 44.1 kHz (Twitch ingest spec).
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 2
AUDIO_BYTES_PER_SAMPLE = 2  # int16
AUDIO_BYTES_PER_SECOND = AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_BYTES_PER_SAMPLE


class FFmpegRTMPStreamer(Monitorable):
    def __init__(self, stream_key: str, fps: int = 24, width: int = 640, height: int = 480, enable_audio: bool = False):
        self.stream_key = stream_key
        self.fps = fps
        self.width = width
        self.height = height
        self.enable_audio = enable_audio
        # Fix: Use correct Twitch RTMP path
        self.rtmp_url = f"rtmp://live.twitch.tv/app/{stream_key}"

        # Stream state
        self.is_streaming = False
        self.ffmpeg_process = None
        self.stream_thread = None
        self.monitor_thread = None

        # Frame management - Optimized buffer size
        self.frame_queue = Queue(maxsize=1000)  # ~9 seconds at 16fps

        # Audio plumbing (only used when enable_audio=True)
        self.audio_queue: "Queue[bytes]" = Queue(maxsize=64)  # ~64 clips of audio
        self.audio_thread: threading.Thread | None = None
        self.audio_fifo_path: str | None = None
        self.audio_fifo_fd: int | None = None
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
            print(f"   RTMP URL: {self.rtmp_url[:50]}...")

            video_in = ffmpeg.input(
                'pipe:',
                format='rawvideo',
                pix_fmt='rgb24',
                s=f'{self.width}x{self.height}',
                framerate=self.fps,  # Use 'framerate' instead of 'r' for raw pipe
            )

            # Build the audio input: native PCM via FIFO if enabled, else silent.
            if self.enable_audio:
                # Create a FIFO that ffmpeg will read PCM from.  We open it for
                # writing in a background thread *after* spawning ffmpeg, so
                # that ffmpeg has opened it for reading first (otherwise the
                # writer-side open() will block forever).
                fifo_dir = tempfile.mkdtemp(prefix="ltx_audio_")
                self.audio_fifo_path = os.path.join(fifo_dir, "audio.pcm")
                os.mkfifo(self.audio_fifo_path)
                queue_log.info(f"🔊 Created audio FIFO at {self.audio_fifo_path}")
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

            # Drain ffmpeg's stderr: with pipe_stderr=True and no reader,
            # ffmpeg blocks once the pipe fills, and the real reason for an
            # exit (e.g. Twitch rejecting the stream key) is never logged.
            self.stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
            self.stderr_thread.start()

            # Start the audio loop if native audio is enabled.
            if self.enable_audio:
                self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
                self.audio_thread.start()
                queue_log.info("🔊 Audio FIFO writer thread started")

            queue_log.info(f"✅ FFmpeg RTMP stream started - NOW LIVE ON TWITCH!")
            queue_log.info(f"🔗 RTMP URL: {self.rtmp_url}")
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
        """Close the audio FIFO writer fd and remove the temp file."""
        if self.audio_fifo_fd is not None:
            try:
                os.close(self.audio_fifo_fd)
            except OSError:
                pass
            self.audio_fifo_fd = None
        if self.audio_fifo_path:
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
            # Blocking open: returns once ffmpeg has opened the FIFO for reading.
            queue_log.info(f"🔊 Opening audio FIFO for writing: {self.audio_fifo_path}")
            self.audio_fifo_fd = os.open(self.audio_fifo_path, os.O_WRONLY)
            queue_log.info(f"🔊 Audio FIFO writer connected; streaming PCM at {AUDIO_SAMPLE_RATE} Hz stereo")
        except Exception as e:
            queue_log.error(f"❌ Could not open audio FIFO writer: {e}")
            return

        # Pace silence padding so we don't write faster than wall-clock.
        # When the queue has data we write as fast as we can (ffmpeg buffers).
        silence_chunk_seconds = 0.1
        silence_chunk = b"\x00" * int(AUDIO_BYTES_PER_SECOND * silence_chunk_seconds)
        self.audio_start_time = time.time()

        # Every write MUST be a whole number of stereo s16 frames (4 bytes).
        # A single misaligned write shifts every later sample across byte
        # boundaries and the rest of the stream decodes as white noise.
        bytes_per_pcm_frame = AUDIO_CHANNELS * 2

        def write_all(data: bytes):
            """os.write can be partial on pipes; write everything, count truth."""
            view = memoryview(data)
            while view:
                written = os.write(self.audio_fifo_fd, view)
                self.audio_bytes_written += written
                view = view[written:]

        while self.is_streaming:
            try:
                try:
                    pcm = self.audio_queue.get(timeout=0.05)
                    write_all(pcm)
                    continue
                except Empty:
                    pass

                # Underrun: pad with silence, but only if we've fallen behind
                # wall clock.  This keeps ffmpeg's audio decoder fed without
                # spamming gigabytes when generation is comfortably ahead.
                wall_seconds = time.time() - self.audio_start_time
                target_bytes = int(wall_seconds * AUDIO_BYTES_PER_SECOND)
                deficit_bytes = target_bytes - self.audio_bytes_written
                # Round the deficit DOWN to whole stereo frames.
                deficit_bytes -= deficit_bytes % bytes_per_pcm_frame
                if deficit_bytes >= bytes_per_pcm_frame:
                    chunk = silence_chunk if deficit_bytes >= len(silence_chunk) else b"\x00" * deficit_bytes
                    write_all(chunk)
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

    

    def add_frame(self, pil_frame):
        """Add PIL Image frame to stream queue"""
        if not self.is_streaming:
            return
        
        try:
            # Convert PIL frame to numpy array for FFmpeg
            frame_array = np.array(pil_frame.convert('RGB'))
            if frame_array.shape[:2] != (self.height, self.width):
                frame_array = cv2.resize(frame_array, (self.width, self.height))
            
            # Add to queue with non-blocking put
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
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")

    def add_frame_batch(self, pil_frames):
        """Add multiple frames efficiently using batch processing"""
        if not self.is_streaming:
            queue_log.warning(f"❌ RTMP not streaming - rejecting {len(pil_frames) if pil_frames else 0} frames")
            return 0
            
        if not pil_frames:
            queue_log.warning("❌ No frames provided to add_frame_batch")
            return 0
        
        queue_log.info(f"📺 BATCH START: Processing {len(pil_frames)} frames...")
        queue_log.info(f"📊 Current queue size: {self.frame_queue.qsize()}/{self.frame_queue.maxsize}")
        
        batch_start_time = time.time()
        processed_count = 0
        
        # Simply loop through frames and reuse existing add_frame logic
        for pil_frame in pil_frames:
            try:
                self.add_frame(pil_frame)  # Reuse existing optimized logic
                processed_count += 1
            except Exception as e:
                print(f"❌ Error processing frame in batch: {e}")
                continue
        
        batch_duration = time.time() - batch_start_time
        batch_fps = processed_count / batch_duration if batch_duration > 0 else 0
        
        queue_log.info(f"📺 BATCH COMPLETE: {processed_count}/{len(pil_frames)} frames in {batch_duration:.2f}s ({batch_fps:.1f} fps)")
        queue_log.info(f"📊 Final queue size: {self.frame_queue.qsize()}/{self.frame_queue.maxsize}")
        
        return processed_count

    def _stderr_loop(self):
        """Log ffmpeg's stderr lines; RTMP failures otherwise vanish silently."""
        proc = self.ffmpeg_process
        if not proc or not proc.stderr:
            return
        try:
            for raw in iter(proc.stderr.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    queue_log.warning(f"🎥 ffmpeg: {line}")
        except Exception:
            pass
        rc = proc.poll()
        if rc is not None and rc != 0:
            queue_log.error(f"❌ ffmpeg exited with code {rc} - see the ffmpeg lines above for the reason")

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

    def _create_placeholder_frame(self, frame_count):
        """Create a black placeholder frame when no content is available"""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Optional: Add subtle visual indicator
        if frame_count % (self.fps * 4) < (self.fps * 2):  # Blink every 4 seconds
            frame[10:20, 10:20] = [30, 30, 30]  # Small dark gray square
        
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
        return {
            "is_streaming": self.is_streaming,
            "frames_sent": self.frames_sent,
            "frames_dropped": self.frames_dropped,
            "queue_size": self.frame_queue.qsize(),
            "current_fps": round(self.frames_sent / max(1, time.time() - (self.start_time or time.time())), 1),
            "target_fps": self.fps
        }

