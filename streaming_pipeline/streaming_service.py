import os
import time
# Removed unused typing imports
from streaming_pipeline.video_generation.video_generator import RealtimeGenerator
from streaming_pipeline.utils.monitoring import ComponentMonitor
from streaming_pipeline.output.rtmp_streamer import FFmpegRTMPStreamer
from streaming_pipeline.output.webrtc_streamer import WebRTCStreamer
from streaming_pipeline.core.streaming_engine import RealtimeVideoStreamer
from streaming_pipeline.input.twitch_listener import TwitchChatListener
from streaming_pipeline.prompt_generation.prompt_generator import PromptGenerator
from streaming_pipeline.postprocessing.text_overlay import TextOverlay
#from dotenv import load_dotenv

#load_dotenv()

class StreamingService:
    """Shared streaming service with core logic (no FAL decorators)"""
    
    def __init__(self):
        self.video_generator = None
        self.video_streamer = None
        self.monitor = None
        self.webrtc_streamer = None  # Created alongside RTMP, used when output_mode="webrtc"
        self._initialized = False
    
    def setup(self):
        """Setup the streaming components"""
        if self._initialized:
            return
            
    
        
        load_local = os.getenv("LOAD_LOCAL_PIPELINE", "false").lower() == "true"
        load_ltx23 = os.getenv("LOAD_LTX23_PIPELINE", "true").lower() == "true"
        self.video_generator = RealtimeGenerator(
            load_local_pipeline=load_local,
            load_ltx23_pipeline=load_ltx23,
        )
        self.video_generator.setup()
        
        # Get environment variables
        twitch_channel = os.getenv("TWITCH_CHANNEL", "shroud")
        openai_key = os.getenv("OPENAI_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        fal_key = os.getenv("FAL_KEY")
        stream_key = os.getenv("TWITCH_STREAM_KEY")

        if not (fal_key or openai_key or groq_key):
            raise ValueError(
                "At least one of FAL_KEY, OPENAI_API_KEY, or GROQ_API_KEY "
                "must be set for prompt generation."
            )

        # Create all dependencies independently (Dependency Injection pattern)
        self.twitch_listener = TwitchChatListener(twitch_channel)
        self.prompt_generator = PromptGenerator(
            openai_api_key=openai_key,
            groq_api_key=groq_key,
            fal_key=fal_key,
        )

        # RTMP streamer (requires TWITCH_STREAM_KEY; optional for webrtc-only)
        if stream_key:
            self.rtmp_streamer = FFmpegRTMPStreamer(
                stream_key=stream_key,
                fps=9,
                width=640,
                height=480,
            )
        else:
            self.rtmp_streamer = None
            print("⚠️ TWITCH_STREAM_KEY not set -- RTMP output disabled, WebRTC only")

        # WebRTC streamer (always available; no external secrets needed)
        self.webrtc_streamer = WebRTCStreamer(fps=14, width=512, height=384)

        self.text_overlay = TextOverlay(width=640, height=480)

        # Default streamer is RTMP (if available), switchable per /start_stream request
        active_streamer = self.rtmp_streamer or self.webrtc_streamer

        # Inject all dependencies into video streamer
        self.video_streamer = RealtimeVideoStreamer(
            twitch_listener=self.twitch_listener,
            prompt_generator=self.prompt_generator,
            realtime_generator=self.video_generator,
            rtmp_streamer=active_streamer,
            text_overlay=self.text_overlay,
        )

        # Create generic component monitor
        monitor_components = {
            "video": self.video_streamer,
            "prompt": self.prompt_generator,
            "generator": self.video_generator,
            "overlay": self.text_overlay,
            "twitch": self.twitch_listener,
        }
        if self.rtmp_streamer:
            monitor_components["rtmp"] = self.rtmp_streamer
        monitor_components["webrtc"] = self.webrtc_streamer
        self.monitor = ComponentMonitor(monitor_components)
        
        # Start monitoring all components
        self.monitor.start_monitoring()
        
        self._initialized = True
        print("✅ Complete streaming pipeline setup complete!")
    
    def start_streaming(self, request):
        """Start the complete Twitch streaming pipeline with full LTX configuration"""
        if not self._initialized:
            self.setup()
            
        try:

            # Update LTX configuration cleanly using the base model
            ltx_updates = {}
            
            # Model selection
            if request.model:
                ltx_updates['model_type'] = request.model
                print(f"   🎯 Model: {request.model}")
            
            # Common parameters
            if request.num_frames:
                ltx_updates['num_frames'] = request.num_frames
            if request.timesteps:
                ltx_updates['timesteps'] = request.timesteps
            if request.guidance_scale is not None:
                ltx_updates['guidance_scale'] = request.guidance_scale
            if request.strength is not None:
                ltx_updates['strength'] = request.strength
            if request.negative_prompt:
                ltx_updates['negative_prompt'] = request.negative_prompt
            if request.width:
                ltx_updates['width'] = request.width
            if request.height:
                ltx_updates['height'] = request.height

            # LTX 2.3-local fixation-control parameters
            if request.stg_scale is not None:
                ltx_updates['stg_scale'] = request.stg_scale
                print(f"   🌀 stg_scale: {request.stg_scale}")
            if request.spatio_temporal_guidance_blocks is not None:
                ltx_updates['spatio_temporal_guidance_blocks'] = request.spatio_temporal_guidance_blocks
                print(f"   🌀 stg_blocks: {request.spatio_temporal_guidance_blocks}")
            if request.noise_scale is not None:
                ltx_updates['noise_scale'] = request.noise_scale
                print(f"   🔊 noise_scale: {request.noise_scale}")
            if request.seed is not None:
                ltx_updates['seed'] = request.seed
                print(f"   🌱 seed (pinned): {request.seed}")
            else:
                # Explicitly clear any previously-pinned seed so generations stay random
                ltx_updates['seed'] = None
                print(f"   🌱 seed: random per generation")

            # LTXv2-specific parameters
            if request.duration is not None:
                ltx_updates['duration'] = request.duration
                print(f"   ⏱️ Duration: {request.duration}s")
            if request.resolution:
                ltx_updates['resolution'] = request.resolution
                print(f"   📐 Resolution: {request.resolution}")
            if request.aspect_ratio:
                ltx_updates['aspect_ratio'] = request.aspect_ratio
                print(f"   📏 Aspect Ratio: {request.aspect_ratio}")
            # Apply style preset (system prompt + generation param overrides).
            # Must happen BEFORE ltx_updates are applied so preset params
            # get merged, and before the generation loop starts.
            PRESET_PARAMS = {
                "cohesive":  {"guidance_scale": 2.0, "noise_scale": 0.03},
                "chaotic":   {"guidance_scale": 3.0, "noise_scale": 0.15},
                "nightmare": {"guidance_scale": 3.5, "noise_scale": 0.20},
            }
            preset = getattr(request, "style_preset", None) or "cohesive"
            if preset != "custom":
                self.prompt_generator.set_style_preset(preset)
                if preset in PRESET_PARAMS:
                    ltx_updates.update(PRESET_PARAMS[preset])
                if preset == "nightmare":
                    self.video_streamer.state.mode = "nightmare"
                print(f"   🎨 Style preset: {preset}")
            else:
                print(f"   🎨 Style preset: custom (using Advanced panel values)")

            # Explicit LLM temperature override (after preset, so it wins)
            if getattr(request, "llm_temperature", None) is not None:
                self.prompt_generator.temperature = float(request.llm_temperature)
                print(f"   🌡️ LLM temperature: {request.llm_temperature}")

            # Apply all updates at once
            if ltx_updates:
                self.video_streamer.update_ltx_config(**ltx_updates)

            # Select output backend and swap the streamer injected into the
            # video_streamer before starting the generation loop.
            output_mode = getattr(request, "output_mode", "rtmp") or "rtmp"
            if output_mode == "webrtc":
                active_streamer = self.webrtc_streamer
                print(f"   📡 Output mode: WebRTC (direct-to-browser)")
            else:
                if self.rtmp_streamer is None:
                    raise ValueError("RTMP output requested but TWITCH_STREAM_KEY is not set")
                active_streamer = self.rtmp_streamer
                print(f"   📡 Output mode: RTMP (Twitch)")

            # Update FPS on whichever streamer is active
            if request.target_fps:
                active_streamer.fps = request.target_fps
                print(f"   🎛️ Set target_fps: {request.target_fps}")

            # Keep generation frame_rate aligned with the stream's FPS so
            # the LTX 2.3 audio path (duration_s = num_frames / frame_rate)
            # produces audio that matches actual playback duration.
            effective_frame_rate = request.frame_rate
            if effective_frame_rate is None and request.target_fps:
                effective_frame_rate = float(request.target_fps)
            if effective_frame_rate is not None:
                ltx_updates['frame_rate'] = float(effective_frame_rate)
                print(f"   🎬 frame_rate: {effective_frame_rate} (audio duration matches stream playback)")

            # Toggle native audio (RTMP only; WebRTC always streams audio)
            if output_mode == "rtmp" and self.rtmp_streamer and request.enable_audio is not None:
                self.rtmp_streamer.enable_audio = bool(request.enable_audio)
                print(f"   🔊 enable_audio: {self.rtmp_streamer.enable_audio}")

            if request.width and request.height:
                active_streamer.width = request.width
                active_streamer.height = request.height
                print(f"   🎛️ Set resolution: {request.width}x{request.height}")

            # Hot-swap the streamer reference on the video_streamer
            self.video_streamer.rtmp_streamer = active_streamer
            
            # Store character references for condition pipeline
            if getattr(request, "character_refs", None):
                self.video_streamer.state.character_refs = request.character_refs[:4]
                self.video_streamer.state.character_names = [
                    ref.get("label", f"Character {i+1}")
                    for i, ref in enumerate(request.character_refs[:4])
                    if ref.get("label")
                ]
                print(f"   🧑 Character refs: {', '.join(self.video_streamer.state.character_names) or 'unnamed'}")
            else:
                self.video_streamer.state.character_refs = []
                self.video_streamer.state.character_names = []

            # Set custom initial state if provided
            if request.initial_prompt or request.initial_image_url:
                print(f"🎨 Using custom initial state:")
                if request.initial_prompt:
                    print(f"   📝 Custom prompt: {request.initial_prompt}")
                    self.video_streamer.initial_prompt = request.initial_prompt
                if request.initial_image_url:
                    print(f"   🖼️ Custom image: {request.initial_image_url}")
                    self.video_streamer.initial_image_url = request.initial_image_url
            else:
                print(f"�� Using default initial state:")
                print(f"   📝 Default prompt: {self.video_streamer.initial_prompt}")
                print(f"   🖼️ Default image: {self.video_streamer.initial_image_url}")
            
            # Set generation mode in streaming state
            if hasattr(request, 'mode') and request.mode:
                self.video_streamer.state.mode = request.mode
                print(f"   🎭 Mode: {request.mode}")
            else:
                self.video_streamer.state.mode = "regular"
                print(f"   🎭 Mode: regular (default)")
            
            # Restart monitoring if it was stopped
            if self.monitor and not self.monitor.monitoring:
                self.monitor.start_monitoring()
            
            # Start video generation and streaming (RTMP is started internally)
            self.video_streamer.start_streaming()
            
            return {
                "status": "started",
                "message": f"Streaming started ({output_mode} mode).",
                "output_mode": output_mode,
                "twitch_channel_input": self.video_streamer.twitch_listener.channel_name,
                "rtmp_url": self.rtmp_streamer.rtmp_url if self.rtmp_streamer else None,
                "initial_prompt": self.video_streamer.initial_prompt,
                "initial_image_url": self.video_streamer.initial_image_url,
                "configuration": {
                    "num_frames": self.video_streamer.ltx_config.num_frames,
                    "timesteps": self.video_streamer.ltx_config.timesteps,
                    "target_fps": active_streamer.fps,
                    "resolution": f"{self.video_streamer.ltx_config.width}x{self.video_streamer.ltx_config.height}",
                    "guidance_scale": self.video_streamer.ltx_config.guidance_scale,
                    "strength": self.video_streamer.ltx_config.strength,
                    "negative_prompt": self.video_streamer.ltx_config.negative_prompt,
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to start streaming: {e}"
            }
    
    def update_config(self, request):
        """Hot-reload generation params on a live stream.

        Only touches params that can change mid-stream without restarting
        (no model swap, no output_mode change, no initial image change).
        The next generation cycle picks up the new values immediately.
        """
        if not self.video_streamer or not self.video_streamer.state.is_running:
            return {"status": "error", "message": "No active stream to update"}

        ltx_updates = {}
        updated_fields = []

        if request.guidance_scale is not None:
            ltx_updates["guidance_scale"] = request.guidance_scale
            updated_fields.append(f"guidance_scale={request.guidance_scale}")
        if request.noise_scale is not None:
            ltx_updates["noise_scale"] = request.noise_scale
            updated_fields.append(f"noise_scale={request.noise_scale}")
        if request.seed is not None:
            ltx_updates["seed"] = request.seed
            updated_fields.append(f"seed={request.seed}")
        if request.negative_prompt is not None:
            ltx_updates["negative_prompt"] = request.negative_prompt
            updated_fields.append("negative_prompt")
        if request.num_frames is not None:
            ltx_updates["num_frames"] = request.num_frames
            updated_fields.append(f"num_frames={request.num_frames}")
        if request.stg_scale is not None:
            ltx_updates["stg_scale"] = request.stg_scale
            updated_fields.append(f"stg_scale={request.stg_scale}")
        if request.spatio_temporal_guidance_blocks is not None:
            ltx_updates["spatio_temporal_guidance_blocks"] = request.spatio_temporal_guidance_blocks
            updated_fields.append(f"stg_blocks={request.spatio_temporal_guidance_blocks}")

        # Explicit LLM temperature override
        if request.llm_temperature is not None:
            self.prompt_generator.temperature = float(request.llm_temperature)
            updated_fields.append(f"llm_temperature={request.llm_temperature}")

        # Apply style preset if provided (swaps system prompt + overrides params)
        PRESET_PARAMS = {
            "cohesive":  {"guidance_scale": 2.0, "noise_scale": 0.03},
            "chaotic":   {"guidance_scale": 3.0, "noise_scale": 0.15},
            "nightmare": {"guidance_scale": 3.5, "noise_scale": 0.20},
        }
        if request.style_preset and request.style_preset != "custom":
            self.prompt_generator.set_style_preset(request.style_preset)
            if request.style_preset in PRESET_PARAMS:
                ltx_updates.update(PRESET_PARAMS[request.style_preset])
            if request.style_preset == "nightmare":
                self.video_streamer.state.mode = "nightmare"
            else:
                self.video_streamer.state.mode = "regular"
            updated_fields.append(f"style_preset={request.style_preset}")

        if ltx_updates:
            self.video_streamer.update_ltx_config(**ltx_updates)

        summary = ", ".join(updated_fields) if updated_fields else "no changes"
        print(f"🔄 Hot-reload config: {summary}")

        return {
            "status": "updated",
            "message": f"Config updated: {summary}",
            "updated_fields": updated_fields,
        }

    def stop_streaming(self):
        """Stop the streaming pipeline"""
        try:
            print("🛑 Stopping streaming pipeline...")
            print(f"   video_streamer: {self.video_streamer}")
            print(f"   monitor: {self.monitor}")
            
            # Stop components in the right order and return immediately
            # Don't wait for threads to finish to avoid blocking the response
            
            # Keep monitor running to show "stopped" state
            # Monitor will continue showing metrics with is_streaming=False
            
            if self.video_streamer:
                print("   Stopping video streamer (includes internal RTMP)...")
                self.video_streamer.stop_streaming()
            
            result = {
                "status": "stopped",
                "message": "Streaming pipeline stopped successfully"
            }
            print(f"   Stop result: {result}")
            return result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "message": f"Error stopping stream: {e}"
            }
            print(f"   Stop error: {error_result}")
            return error_result
    
    def get_metrics(self):
        """Get latest streaming metrics - clean and simple"""
        try:
            # Get latest metrics if monitor exists and is monitoring
            if self.monitor and self.monitor.monitoring:
                latest_metrics = self.monitor.get_latest_metrics()
                
                if latest_metrics is None:
                    return {"error": "No metrics available yet"}
                
                # Return metrics directly (ComponentMonitor returns dict)
                return latest_metrics
            else:
                # Return minimal state when not monitoring
                return {
                    "error": "Monitoring not active",
                    "timestamp": time.time()
                }
            
        except Exception as e:
            return {
                "error": f"Failed to get metrics: {e}",
                "timestamp": time.time()
            }
    
    async def handle_webrtc(self, websocket):
        """Delegate WebRTC signaling to the active WebRTCStreamer.

        If the streamer is not a WebRTCStreamer (e.g. the user started with
        RTMP output mode) this falls through gracefully with an error message.
        """
        streamer = getattr(self, "webrtc_streamer", None)
        if streamer is None or not isinstance(streamer, WebRTCStreamer):
            await websocket.accept()
            await websocket.send_json({
                "type": "error",
                "message": "WebRTC output is not enabled. Start the stream with output_mode='webrtc'.",
            })
            await websocket.close()
            return

        await streamer.handle_signaling(websocket)

    async def handle_metrics_websocket(self, websocket, logger=None):
        """Handle WebSocket connection for real-time metrics streaming
        
        This method can be used by both gpu_server.py and FAL app
        """
        import asyncio
        
        await websocket.accept()
        
        try:
            if logger:
                logger.info("📡 WebSocket client connected for metrics streaming")
            else:
                print("📡 WebSocket client connected for metrics streaming")
            
            while True:
                try:
                    # Get current metrics
                    metrics = self.get_metrics()
                    
                    # Send metrics to client
                    await websocket.send_json({
                        "type": "metrics",
                        "data": metrics,
                        "timestamp": time.time()
                    })
                    
                    # Wait 1 second before next update
                    await asyncio.sleep(1.0)
                    
                except Exception as e:
                    if logger:
                        logger.error(f"❌ Error sending metrics: {e}")
                    else:
                        print(f"❌ Error sending metrics: {e}")
                    
                    # Send error message to client
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                        "timestamp": time.time()
                    })
                    await asyncio.sleep(1.0)
                    
        except Exception as e:
            if logger:
                logger.error(f"❌ WebSocket connection error: {e}")
            else:
                print(f"❌ WebSocket connection error: {e}")
        finally:
            if logger:
                logger.info("📡 WebSocket client disconnected")
            else:
                print("📡 WebSocket client disconnected")
            try:
                await websocket.close()
            except:
                pass