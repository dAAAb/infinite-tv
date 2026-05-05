// Shared types for the FAL Realtime Dashboard

// Component-specific metrics types
export interface RTMPMetrics {
  queue_size: number
  frames_sent: number
  frames_dropped: number
  current_fps: number
  target_fps: number
  is_streaming: boolean
}

export interface VideoMetrics {
  is_running: boolean
  generation_count: number
  current_prompt: string
  generation_params_history: GenerationParams[]
}

export interface PromptMetrics {
  prompts_generated: number
  avg_response_time: number
  last_input_length: number
  last_output_length: number
  last_generation_time: number
}

export interface GeneratorMetrics {
  videos_generated: number
  avg_generation_time: number
  last_generation_time: number
}

export interface OverlayMetrics {
  frames_processed: number
  avg_time_per_frame: number
  has_overlay: boolean
  last_batch_size: number
  last_batch_time: number
  last_batch_avg_per_frame: number
}

export interface TwitchMetrics {
  channel: string
  is_listening: boolean
  queue_size: number
}

// Main metrics interface with nested component metrics
export interface ComponentMetrics {
  timestamp: number
  gpu_memory_allocated: number
  
  // Component-specific metrics
  rtmp: RTMPMetrics
  video: VideoMetrics
  prompt: PromptMetrics
  generator: GeneratorMetrics
  overlay: OverlayMetrics
  twitch: TwitchMetrics
}


export interface GenerationParams {
  timestamp: number
  generation_id: number
  prompt: string
  negative_prompt: string
  width: number
  height: number
  num_frames: number
  strength: number
  guidance_scale: number
  timesteps: number[]
}

export interface RealtimeData {
  metrics: ComponentMetrics | null
  history: Array<ComponentMetrics>
  isConnected: boolean
  error: string | null
}

export interface WebSocketMessage {
  type: 'metrics' | 'error'
  data?: ComponentMetrics
  message?: string
  timestamp: number
}

export interface GenerationHistoryProps {
  generationHistory?: GenerationParams[]
}

// LTX 2.3 configuration types
export interface LTXv2Config {
  model: 'ltx-2.3'
  image_url: string
  prompt: string
  duration?: 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20
  resolution?: '1080p' | '1440p' | '2160p'
  aspect_ratio?: 'auto' | '16:9' | '9:16'
  // Streaming parameters (applied after generation)
  target_fps?: number
  width?: number
  height?: number
}

// LTXv1 configuration types (existing streaming model)
export interface LTXv1Config {
  model: 'ltxv1'
  initial_prompt: string
  initial_image_url: string
  negative_prompt: string
  height: number
  width: number
  num_frames: number
  strength: number
  guidance_scale: number
  timesteps: number[]
  target_fps: number
  mode: 'regular' | 'nightmare'
}

// LTX 2.3 Local configuration types
export interface LTX23LocalConfig {
  model: 'ltx-2.3-local'
  initial_prompt: string
  initial_image_url: string
  negative_prompt: string
  height: number
  width: number
  num_frames: number
  target_fps: number
  mode: 'regular' | 'nightmare'
  // Fixation-control parameters (see streaming_pipeline/models/api.py)
  guidance_scale: number
  stg_scale: number
  // Required when stg_scale > 0; transformer block indices to apply STG at.
  // null/empty = STG disabled even if stg_scale > 0 (server skips it safely).
  spatio_temporal_guidance_blocks: number[] | null
  noise_scale: number
  // null = random seed per generation; number = pinned seed
  seed: number | null
  // LLM temperature for prompt generation
  llm_temperature: number
  // Stream LTX 2.3's natively-generated audio instead of silent anullsrc
  enable_audio: boolean
  // Output backend: 'rtmp' for Twitch, 'webrtc' for direct-to-browser
  output_mode: 'rtmp' | 'webrtc'
  // Named combination of system prompt + generation parameters
  style_preset: 'cohesive' | 'chaotic' | 'nightmare' | 'custom'
}

export interface CharacterRef {
  image: string       // base64 data URI or HTTPS URL
  strength: number    // 0-1, default 0.4
  label: string       // display name, e.g. "Homer Simpson"
}

export interface LTX23ConditionConfig extends LTX23LocalConfig {
  model: 'ltx-2.3-condition'
  character_refs: CharacterRef[]
}

export type ModelType = 'ltxv1' | 'ltx-2.3' | 'ltx-2.3-local' | 'ltx-2.3-condition'
export type TestConfig = LTXv1Config | LTXv2Config | LTX23LocalConfig | LTX23ConditionConfig
