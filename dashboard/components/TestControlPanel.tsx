'use client'

import { useState, useRef } from 'react'
import { Play, Square, Upload, Settings, Sliders, Zap, RefreshCw } from 'lucide-react'
import type { TestConfig, LTXv1Config, LTXv2Config, LTX23LocalConfig, LTX23ConditionConfig, CharacterRef, ModelType } from '../types'

interface TestControlPanelProps {
  onStartTest: (config: TestConfig) => void
  onStopTest: () => void
  onUpdateConfig?: (config: any) => Promise<void> | void
  isStreaming: boolean
}

// Resolution presets for the streaming output.  All dimensions are divisible
// by 32 (LTX requirement).  Ordered roughly by speed (smallest first) within
// each aspect ratio group.
type Preset = { label: string; width: number; height: number }
const RESOLUTION_PRESETS: Preset[] = [
  { label: '384x256  (3:2, fastest)',  width: 384, height: 256 },
  { label: '512x288  (16:9, fast)',    width: 512, height: 288 },
  { label: '512x384  (4:3, fast)',     width: 512, height: 384 },
  { label: '640x384  (5:3)',           width: 640, height: 384 },
  { label: '640x480  (4:3)',           width: 640, height: 480 },
  { label: '768x432  (16:9)',          width: 768, height: 432 },
  { label: '768x512  (3:2)',           width: 768, height: 512 },
  { label: '512x512  (1:1)',           width: 512, height: 512 },
  { label: '288x512  (9:16, vertical)', width: 288, height: 512 },
  { label: '432x768  (9:16, vertical)', width: 432, height: 768 },
]

const PRESET_CUSTOM = 'custom'

function findPresetKey(width: number, height: number): string {
  const match = RESOLUTION_PRESETS.find(p => p.width === width && p.height === height)
  return match ? `${match.width}x${match.height}` : PRESET_CUSTOM
}

export default function TestControlPanel({ onStartTest, onStopTest, onUpdateConfig, isStreaming }: TestControlPanelProps) {
  // Default to ltx-2.3-local because the backend's default LOAD_LTX23_PIPELINE=true
  // only loads that pipeline.  Selecting ltxv1 here without setting
  // LOAD_LOCAL_PIPELINE=true on the backend will hit "Pipeline not loaded".
  const [selectedModel, setSelectedModel] = useState<ModelType>('ltx-2.3-local')
  
  const [ltxv1Config, setLtxv1Config] = useState<LTXv1Config>({
    model: 'ltxv1',
    initial_prompt: "Spongebob leaves the room through the door",
    initial_image_url: "https://storage.googleapis.com/remade-v2/uploads/a185f836a3e9ca84cc75f5c12bb10dd4.jpg",
    negative_prompt: "worst quality, inconsistent motion, blurry, jittery, distorted",
    height: 480,
    width: 640,
    num_frames: 240,
    strength: 1.2,
    guidance_scale: 5.0,
    timesteps: [1000, 981, 909, 725, 0.03],
    target_fps: 14.0,
    mode: 'regular'
  })

  const [ltxv2Config, setLtxv2Config] = useState<LTXv2Config>({
    model: 'ltx-2.3',
    image_url: "https://storage.googleapis.com/remade-v2/uploads/a185f836a3e9ca84cc75f5c12bb10dd4.jpg",
    prompt: "A cinematic video with smooth camera movement and realistic motion",
    duration: 6,
    resolution: '1080p',
    aspect_ratio: '16:9',
    // Streaming parameters
    target_fps: 14.0,
    width: 640,
    height: 480,
  })

  const [ltx23LocalConfig, setLtx23LocalConfig] = useState<LTX23LocalConfig>({
    model: 'ltx-2.3-local',
    initial_prompt: "A cinematic video with smooth camera movement and realistic motion",
    initial_image_url: "https://scriptmag.com/uploads/MTY3Mzc4OTYwMzA5ODg4NjI0/image-placeholder-title.jpg?format=auto&optimize=high&width=1440",
    negative_prompt: "worst quality, inconsistent motion, blurry, jittery, distorted, static scene, frozen frame, no motion, repetitive, looping",
    height: 384,
    width: 512,
    num_frames: 121,
    target_fps: 9.0,
    mode: 'regular',
    // Fixation-control defaults (mirror backend api.py defaults).
    // STG defaults to OFF because it requires spatio_temporal_guidance_blocks
    // which depend on the model architecture; enable from Advanced if you
    // know the right block indices for your checkpoint.
    guidance_scale: 1.0,
    stg_scale: 0.0,
    spatio_temporal_guidance_blocks: null,
    noise_scale: 0.15,
    seed: null,
    llm_temperature: 0.55,
    enable_audio: true,
    output_mode: 'webrtc',
    style_preset: 'cohesive',
  })

  const [ltx23ConditionConfig, setLtx23ConditionConfig] = useState<LTX23ConditionConfig>({
    ...ltx23LocalConfig,
    model: 'ltx-2.3-condition',
    character_refs: [],
  })

  const [showAdvanced, setShowAdvanced] = useState(true)
  const [uploadingImage, setUploadingImage] = useState(false)
  const [applyStatus, setApplyStatus] = useState<'idle' | 'applying' | 'applied'>('idle')
  const [showSystemPrompt, setShowSystemPrompt] = useState(false)

  // System prompt text for each preset (read-only display)
  const SYSTEM_PROMPT_PREVIEWS: Record<string, string> = {
    cohesive: `You are the writer and director of an ongoing animated story. The characters on screen are PROTAGONISTS with personalities, goals, and emotions.\n\nSTORY RHYTHM (based on clip number):\n- Clips 1-3: ESTABLISH the scene\n- Clips 4-6: INTRODUCE a situation (something catches attention)\n- Clips 7-10: DEVELOP tension (investigate, encounter problems)\n- Clips 11-15: ESCALATE (consequences, choices, surprises)\n- Clips 16+: RESOLVE and RESET (new cycle)\n\nCHARACTER DIRECTION:\n- Characters WANT things, TRY things, REACT to things\n- They have emotions: curiosity, surprise, frustration, joy\n- They make DECISIONS that drive the story forward\n\nUses initial prompt as "story premise" and generation_count for rhythm.`,
    chaotic: `You are creating video prompts for continuous video generation with VISUAL AWARENESS.\n\nCRITICAL STORYTELLING RULES:\n1. NEVER REPEAT - If recent prompts are similar, FORCE dramatic change\n2. ALWAYS PROGRESS - Each prompt must ADD something new or CHANGE something significant\n3. BE BOLD - Don't just describe what you see, TRANSFORM it\n4. FIX PROBLEMS - If scene is messy/boring, use dramatic transitions`,
    nightmare: `Same as Chaotic, but with MODE = "nightmare":\nMake ALL prompts nightmarish/bizarre/outlandish. Transform normal actions into surreal/disturbing scenarios.`,
    custom: `Using the current system prompt on the server. Edit the .txt files and redeploy to change, or use the Apply Changes button with custom parameter values.`,
  }

  // When user manually edits an Advanced param, auto-switch to "custom" preset
  const updateLocalParam = (updates: Partial<LTX23LocalConfig>) => {
    setLtx23LocalConfig(prev => ({ ...prev, ...updates, style_preset: 'custom' }))
  }
  const fileInputRef = useRef<HTMLInputElement>(null)
  
  // Get current config based on selected model
  const config = selectedModel === 'ltxv1' ? ltxv1Config 
    : selectedModel === 'ltx-2.3-condition' ? ltx23ConditionConfig
    : selectedModel === 'ltx-2.3-local' ? ltx23LocalConfig 
    : ltxv2Config
  
  // Computed prompt value that includes nightmare prefix when needed (LTXv1 only)
  const displayedPrompt = selectedModel === 'ltxv1' && ltxv1Config.mode === 'nightmare' && !ltxv1Config.initial_prompt.startsWith('(Nightmare Started)')
    ? `(Nightmare Started) ${ltxv1Config.initial_prompt}`
    : selectedModel === 'ltxv1' ? ltxv1Config.initial_prompt 
    : selectedModel === 'ltx-2.3-condition' ? ltx23ConditionConfig.initial_prompt
    : selectedModel === 'ltx-2.3-local' ? ltx23LocalConfig.initial_prompt
    : ltxv2Config.prompt

  const handleImageUpload = async (file: File) => {
    // Simple validation
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file')
      return
    }
    
    if (file.size > 10 * 1024 * 1024) {
      alert('Image must be smaller than 10MB')
      return
    }

    setUploadingImage(true)
    try {
      console.log('📄 Converting image:', file.name)
      
      // Convert to base64 data URL for now
      const reader = new FileReader()
      reader.onload = (e) => {
        const result = e.target?.result as string
        
        if (selectedModel === 'ltxv1') {
          setLtxv1Config(prev => ({ ...prev, initial_image_url: result }))
        } else if (selectedModel === 'ltx-2.3-local') {
          setLtx23LocalConfig(prev => ({ ...prev, initial_image_url: result }))
        } else {
          setLtxv2Config(prev => ({ ...prev, image_url: result }))
        }
        
        console.log('✅ Image converted successfully')
        setUploadingImage(false)
      }
      reader.onerror = () => {
        console.error('Failed to read file')
        alert('Failed to read image file')
        setUploadingImage(false)
      }
      reader.readAsDataURL(file)
      
    } catch (error) {
      console.error('Failed to process image:', error)
      alert('Failed to process image. Please try again.')
      setUploadingImage(false)
    }
  }



  return (
    <div className="fal-card">
      <div className="fal-card-header">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Settings className="w-5 h-5 text-fal-yellow-500" />
            <h3 className="text-lg font-semibold text-fal-gray-900">Video Generation</h3>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="btn-secondary text-sm"
            >
              <Sliders className="w-4 h-4 mr-2" />
              {showAdvanced ? 'Hide' : 'Show'} Advanced
            </button>
          </div>
        </div>
      </div>
      
      <div className="fal-card-content space-y-6">
        
        {/* Model Selector */}
        <div>
          <label className="metric-label mb-3 block">Select Model</label>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <button
              type="button"
              onClick={() => setSelectedModel('ltxv1')}
              className={`px-4 py-3 rounded-lg text-sm font-medium transition-all border-2 ${
                selectedModel === 'ltxv1'
                  ? 'bg-fal-primary-500 text-white border-fal-primary-500 shadow-lg'
                  : 'bg-white text-fal-gray-700 hover:bg-fal-gray-50 border-fal-gray-300'
              }`}
            >
              <div className="flex items-center justify-center space-x-2">
                <Zap className="w-4 h-4" />
                <span>LTX v1</span>
              </div>
              <p className="text-xs mt-1 opacity-75">Local 0.9.8 pipeline</p>
            </button>
            
            <button
              type="button"
              onClick={() => setSelectedModel('ltx-2.3')}
              className={`px-4 py-3 rounded-lg text-sm font-medium transition-all border-2 ${
                selectedModel === 'ltx-2.3'
                  ? 'bg-fal-primary-500 text-white border-fal-primary-500 shadow-lg'
                  : 'bg-white text-fal-gray-700 hover:bg-fal-gray-50 border-fal-gray-300'
              }`}
            >
              <div className="flex items-center justify-center space-x-2">
                <Zap className="w-4 h-4" />
                <span>LTX 2.3 API</span>
              </div>
              <p className="text-xs mt-1 opacity-75">fal.ai hosted</p>
            </button>

            <button
              type="button"
              onClick={() => setSelectedModel('ltx-2.3-local')}
              className={`px-4 py-3 rounded-lg text-sm font-medium transition-all border-2 ${
                selectedModel === 'ltx-2.3-local'
                  ? 'bg-fal-primary-500 text-white border-fal-primary-500 shadow-lg'
                  : 'bg-white text-fal-gray-700 hover:bg-fal-gray-50 border-fal-gray-300'
              }`}
            >
              <div className="flex items-center justify-center space-x-2">
                <Zap className="w-4 h-4" />
                <span>LTX 2.3 Local</span>
              </div>
              <p className="text-xs mt-1 opacity-75">22B distilled FP8</p>
            </button>

            <button
              type="button"
              onClick={() => setSelectedModel('ltx-2.3-condition')}
              className={`px-4 py-3 rounded-lg text-sm font-medium transition-all border-2 ${
                selectedModel === 'ltx-2.3-condition'
                  ? 'bg-fal-primary-500 text-white border-fal-primary-500 shadow-lg'
                  : 'bg-white text-fal-gray-700 hover:bg-fal-gray-50 border-fal-gray-300'
              }`}
            >
              <div className="flex items-center justify-center space-x-2">
                <Zap className="w-4 h-4" />
                <span>LTX 2.3 Condition</span>
              </div>
              <p className="text-xs mt-1 opacity-75">Multi-character refs</p>
            </button>
          </div>
        </div>

        {/* Basic Configuration */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Prompt Configuration */}
          <div className="space-y-4">
            <div>
              <label className="metric-label mb-2 block">
                {selectedModel === 'ltxv1' ? 'Initial Prompt' : 'Prompt'}
              </label>
              <textarea
                value={displayedPrompt}
                onChange={(e) => {
                  let newPrompt = e.target.value
                  
                  if (selectedModel === 'ltxv1') {
                    if (newPrompt.startsWith('(Nightmare Started) ')) {
                      newPrompt = newPrompt.replace('(Nightmare Started) ', '')
                    }
                    setLtxv1Config(prev => ({ ...prev, initial_prompt: newPrompt }))
                  } else if (selectedModel === 'ltx-2.3-local') {
                    setLtx23LocalConfig(prev => ({ ...prev, initial_prompt: newPrompt }))
                  } else {
                    setLtxv2Config(prev => ({ ...prev, prompt: newPrompt }))
                  }
                }}
                className="w-full bg-fal-gray-100 border border-fal-gray-300 rounded-lg p-3 text-fal-gray-900 text-sm"
                rows={3}
                placeholder={selectedModel === 'ltxv1' 
                  ? "Describe the initial video content..." 
                  : "Describe the video you want to generate..."}
              />
            </div>
            
            {/* Mode Selector - LTXv1 only */}
            {selectedModel === 'ltxv1' && (
              <div>
                <label className="metric-label mb-2 block">Generation Mode</label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setLtxv1Config(prev => ({ 
                      ...prev, 
                      mode: 'regular',
                      strength: prev.strength === 1.4 ? 1.0 : prev.strength
                    }))}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                      ltxv1Config.mode === 'regular'
                        ? 'bg-fal-primary-500 text-white'
                        : 'bg-fal-gray-100 text-fal-gray-700 hover:bg-fal-gray-200 border border-fal-gray-300'
                    }`}
                  >
                    🌟 Regular
                  </button>
                  <button
                    type="button"
                    onClick={() => setLtxv1Config(prev => ({ 
                      ...prev, 
                      mode: 'nightmare',
                      strength: 1.4
                    }))}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                      ltxv1Config.mode === 'nightmare'
                        ? 'bg-fal-red-500 text-white'
                        : 'bg-fal-gray-100 text-fal-gray-700 hover:bg-fal-gray-200 border border-fal-gray-300'
                    }`}
                  >
                    😈 Nightmare
                  </button>
                </div>
                {ltxv1Config.mode === 'nightmare' && (
                  <div className="mt-2 p-2 bg-fal-red-50 border border-fal-red-200 rounded text-xs text-fal-red-700">
                    <strong>Nightmare Mode:</strong> Prompts will be enhanced to create nightmarish/outlandish content.
                    <br />Strength automatically set to 1.4 for more dramatic effects.
                  </div>
                )}
              </div>
            )}
            
            {/* LTX 2.3 Options */}
            {selectedModel === 'ltx-2.3' && (
              <>
                <div>
                  <label className="metric-label mb-2 block">Duration</label>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      onClick={() => setLtxv2Config(prev => ({ ...prev, duration: 6 }))}
                      className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                        ltxv2Config.duration === 6
                          ? 'bg-fal-primary-500 text-white'
                          : 'bg-fal-gray-100 text-fal-gray-700 hover:bg-fal-gray-200 border border-fal-gray-300'
                      }`}
                    >
                      6 seconds
                    </button>
                    <button
                      type="button"
                      onClick={() => setLtxv2Config(prev => ({ ...prev, duration: 8 }))}
                      className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                        ltxv2Config.duration === 8
                          ? 'bg-fal-primary-500 text-white'
                          : 'bg-fal-gray-100 text-fal-gray-700 hover:bg-fal-gray-200 border border-fal-gray-300'
                      }`}
                    >
                      8 seconds
                    </button>
                  </div>
                </div>
                
                <div>
                  <label className="metric-label mb-2 block">Resolution</label>
                  <select
                    value={ltxv2Config.resolution}
                    onChange={(e) => setLtxv2Config(prev => ({ 
                      ...prev, 
                      resolution: e.target.value as '1080p' | '1440p' | '2160p' 
                    }))}
                    className="w-full bg-fal-gray-100 border border-fal-gray-300 rounded-lg p-3 text-fal-gray-900 text-sm"
                  >
                    <option value="1080p">1080p</option>
                    <option value="1440p">1440p</option>
                    <option value="2160p">2160p (4K)</option>
                  </select>
                </div>
                
                <div>
                  <label className="metric-label mb-2 block">Aspect Ratio</label>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      onClick={() => setLtxv2Config(prev => ({ ...prev, aspect_ratio: '16:9' }))}
                      className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                        ltxv2Config.aspect_ratio === '16:9'
                          ? 'bg-fal-primary-500 text-white'
                          : 'bg-fal-gray-100 text-fal-gray-700 hover:bg-fal-gray-200 border border-fal-gray-300'
                      }`}
                    >
                      16:9 Landscape
                    </button>
                    <button
                      type="button"
                      onClick={() => setLtxv2Config(prev => ({ ...prev, aspect_ratio: '9:16' }))}
                      className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                        ltxv2Config.aspect_ratio === '9:16'
                          ? 'bg-fal-primary-500 text-white'
                          : 'bg-fal-gray-100 text-fal-gray-700 hover:bg-fal-gray-200 border border-fal-gray-300'
                      }`}
                    >
                      9:16 Portrait
                    </button>
                  </div>
                </div>
              </>
            )}
            
            {/* Negative Prompt - LTXv1 only */}
            {selectedModel === 'ltxv1' && (
              <div>
                <label className="metric-label mb-2 block">Negative Prompt</label>
                <textarea
                  value={ltxv1Config.negative_prompt}
                  onChange={(e) => setLtxv1Config(prev => ({ ...prev, negative_prompt: e.target.value }))}
                  className="w-full bg-fal-gray-100 border border-fal-gray-300 rounded-lg p-3 text-fal-gray-900 text-sm"
                  rows={2}
                  placeholder="What to avoid in the generation..."
                />
              </div>
            )}
          </div>

          {/* Image Upload */}
          <div className="space-y-4">
            <div>
              <label className="metric-label mb-2 block">
                {selectedModel === 'ltxv1' ? 'Initial Image' : 'Source Image'}
              </label>
              <div className="space-y-3">
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={(e) => {
                    const file = e.target.files?.[0]
                    if (file) handleImageUpload(file)
                  }}
                  accept="image/*"
                  className="hidden"
                />
                
                {/* Unified Upload/Preview Area */}
                <div 
                  className={`relative border-2 border-dashed rounded-lg transition-all ${
                    uploadingImage 
                      ? 'border-fal-primary-400 bg-fal-primary-50' 
                      : (selectedModel === 'ltxv1' ? ltxv1Config.initial_image_url : selectedModel === 'ltx-2.3-local' ? ltx23LocalConfig.initial_image_url : ltxv2Config.image_url)
                        ? 'border-fal-gray-300 bg-fal-gray-50' 
                        : 'border-fal-gray-300 bg-fal-gray-100 hover:border-fal-primary-400 hover:bg-fal-primary-50 cursor-pointer'
                  }`}
                  onClick={() => !uploadingImage && fileInputRef.current?.click()}
                >
                  {(selectedModel === 'ltxv1' ? ltxv1Config.initial_image_url : selectedModel === 'ltx-2.3-local' ? ltx23LocalConfig.initial_image_url : ltxv2Config.image_url) ? (
                    // Image Preview State - Click to replace
                    <div className="relative group cursor-pointer">
                      <img
                        src={selectedModel === 'ltxv1' ? ltxv1Config.initial_image_url : selectedModel === 'ltx-2.3-local' ? ltx23LocalConfig.initial_image_url : ltxv2Config.image_url}
                        alt="Initial frame"
                        className="w-full max-h-48 object-contain rounded"
                        onError={(e) => {
                          e.currentTarget.style.display = 'none'
                        }}
                      />
                      {/* Simple hover overlay */}
                      <div className="absolute inset-0 bg-fal-primary-600 bg-opacity-0 group-hover:bg-opacity-20 transition-all rounded flex items-center justify-center">
                        <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                          <Upload className="w-8 h-8 text-white drop-shadow-lg" />
                        </div>
                      </div>
                      {/* Clear button */}
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          if (selectedModel === 'ltxv1') {
                            setLtxv1Config(prev => ({ ...prev, initial_image_url: '' }))
                          } else if (selectedModel === 'ltx-2.3-local') {
                            setLtx23LocalConfig(prev => ({ ...prev, initial_image_url: '' }))
                          } else {
                            setLtxv2Config(prev => ({ ...prev, image_url: '' }))
                          }
                        }}
                        className="absolute top-2 right-2 w-6 h-6 bg-fal-red-500 hover:bg-fal-red-600 text-white rounded-full flex items-center justify-center text-sm font-bold transition-colors opacity-70 hover:opacity-100"
                        title="Remove image"
                      >
                        ×
                      </button>
                    </div>
                  ) : uploadingImage ? (
                    // Uploading State
                    <div className="flex flex-col items-center justify-center py-12">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-fal-primary-500 mb-3"></div>
                      <p className="text-fal-primary-600 font-medium">Uploading image...</p>
                    </div>
                  ) : (
                    // Empty State
                    <div className="flex flex-col items-center justify-center py-12 text-center">
                      <Upload className="w-12 h-12 text-fal-gray-400 mb-3" />
                      <p className="text-fal-gray-600 font-medium mb-1">Click to upload an image</p>
                      <p className="text-fal-gray-500 text-sm">PNG, JPG, GIF up to 10MB</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Streaming Parameters - Both Models */}
        <div>
          <label className="metric-label mb-3 block">Streaming Parameters</label>
          {(() => {
            const currentWidth = selectedModel === 'ltxv1' ? ltxv1Config.width : selectedModel === 'ltx-2.3-local' ? ltx23LocalConfig.width : ltxv2Config.width
            const currentHeight = selectedModel === 'ltxv1' ? ltxv1Config.height : selectedModel === 'ltx-2.3-local' ? ltx23LocalConfig.height : ltxv2Config.height
            const currentPresetKey = findPresetKey(currentWidth, currentHeight)
            const isCustom = currentPresetKey === PRESET_CUSTOM

            const setDimensions = (w: number, h: number) => {
              if (selectedModel === 'ltxv1') {
                setLtxv1Config(prev => ({ ...prev, width: w, height: h }))
              } else if (selectedModel === 'ltx-2.3-local') {
                setLtx23LocalConfig(prev => ({ ...prev, width: w, height: h }))
              } else {
                setLtxv2Config(prev => ({ ...prev, width: w, height: h }))
              }
            }

            return (
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="metric-label mb-2 block text-xs">Target FPS</label>
                    <input
                      type="number"
                      value={selectedModel === 'ltxv1' ? ltxv1Config.target_fps : selectedModel === 'ltx-2.3-local' ? ltx23LocalConfig.target_fps : ltxv2Config.target_fps}
                      onChange={(e) => {
                        const value = parseFloat(e.target.value)
                        if (selectedModel === 'ltxv1') {
                          setLtxv1Config(prev => ({ ...prev, target_fps: value }))
                        } else if (selectedModel === 'ltx-2.3-local') {
                          setLtx23LocalConfig(prev => ({ ...prev, target_fps: value }))
                        } else {
                          setLtxv2Config(prev => ({ ...prev, target_fps: value }))
                        }
                      }}
                      className="w-full bg-fal-gray-100 border border-fal-gray-300 rounded-lg p-3 text-fal-gray-900 font-mono"
                      min={1}
                      max={30}
                      step={0.5}
                    />
                  </div>

                  <div>
                    <label className="metric-label mb-2 block text-xs">Stream Resolution</label>
                    <select
                      value={currentPresetKey}
                      onChange={(e) => {
                        const key = e.target.value
                        if (key === PRESET_CUSTOM) return
                        const preset = RESOLUTION_PRESETS.find(p => `${p.width}x${p.height}` === key)
                        if (preset) setDimensions(preset.width, preset.height)
                      }}
                      className="w-full bg-fal-gray-100 border border-fal-gray-300 rounded-lg p-3 text-fal-gray-900 font-mono"
                    >
                      {RESOLUTION_PRESETS.map(p => (
                        <option key={`${p.width}x${p.height}`} value={`${p.width}x${p.height}`}>
                          {p.label}
                        </option>
                      ))}
                      <option value={PRESET_CUSTOM}>Custom ({currentWidth}x{currentHeight})</option>
                    </select>
                  </div>
                </div>

                {isCustom && (
                  <div className="grid grid-cols-2 gap-4 mt-3">
                    <div>
                      <label className="metric-label mb-2 block text-xs">Custom Width</label>
                      <input
                        type="number"
                        value={currentWidth}
                        onChange={(e) => setDimensions(parseInt(e.target.value), currentHeight)}
                        className="w-full bg-fal-gray-100 border border-fal-gray-300 rounded-lg p-3 text-fal-gray-900 font-mono"
                        min={256}
                        max={1920}
                        step={32}
                      />
                    </div>
                    <div>
                      <label className="metric-label mb-2 block text-xs">Custom Height</label>
                      <input
                        type="number"
                        value={currentHeight}
                        onChange={(e) => setDimensions(currentWidth, parseInt(e.target.value))}
                        className="w-full bg-fal-gray-100 border border-fal-gray-300 rounded-lg p-3 text-fal-gray-900 font-mono"
                        min={256}
                        max={1920}
                        step={32}
                      />
                    </div>
                  </div>
                )}

                <p className="text-xs text-fal-gray-600 mt-2">
                  {selectedModel === 'ltx-2.3'
                    ? 'Stream resolution: videos from the remote LTX 2.3 API are resized to these dimensions before streaming.'
                    : 'Generation and streaming resolution. Must be divisible by 32 (all presets are).'}
                </p>
              </>
            )
          })()}
        </div>

        {/* LTXv1-specific Parameters */}
        {selectedModel === 'ltxv1' && (
          <div className="grid grid-cols-2 md:grid-cols-2 gap-4">
            <div>
              <label className="metric-label mb-2 block">Frames</label>
              <input
                type="number"
                value={ltxv1Config.num_frames}
                onChange={(e) => setLtxv1Config(prev => ({ ...prev, num_frames: parseInt(e.target.value) }))}
                className="w-full bg-fal-gray-100 border border-fal-gray-300 rounded-lg p-3 text-fal-gray-900 font-mono"
                min={60}
                max={500}
                step={10}
              />
            </div>
            
            <div>
              <label className="metric-label mb-2 block">Guidance Scale</label>
              <input
                type="number"
                value={ltxv1Config.guidance_scale}
                onChange={(e) => setLtxv1Config(prev => ({ ...prev, guidance_scale: parseFloat(e.target.value) }))}
                className="w-full bg-fal-gray-100 border border-fal-gray-300 rounded-lg p-3 text-fal-gray-900 font-mono"
                min={1}
                max={10}
                step={0.1}
              />
            </div>
            
            <div>
              <label className="metric-label mb-2 block">Strength</label>
              <input
                type="number"
                value={ltxv1Config.strength}
                onChange={(e) => setLtxv1Config(prev => ({ ...prev, strength: parseFloat(e.target.value) }))}
                className="w-full bg-fal-gray-100 border border-fal-gray-300 rounded-lg p-3 text-fal-gray-900 font-mono"
                min={0.1}
                max={2.0}
                step={0.1}
              />
            </div>
          </div>
        )}

        {/* Advanced Parameters - LTXv1 only */}
        {showAdvanced && selectedModel === 'ltxv1' && (
          <div className="space-y-4 border-t border-fal-gray-700 pt-6">
            <h4 className="text-fal-gray-900 font-medium">Advanced LTXv1 Parameters</h4>
            
            <div>
              <label className="metric-label mb-2 block">Timesteps (comma-separated)</label>
              <input
                type="text"
                value={ltxv1Config.timesteps.join(', ')}
                onChange={(e) => {
                  try {
                    const timesteps = e.target.value.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n))
                    setLtxv1Config(prev => ({ ...prev, timesteps }))
                  } catch (error) {
                    // Invalid input, ignore
                  }
                }}
                className="w-full bg-fal-gray-100 border border-fal-gray-300 rounded-lg p-3 text-fal-gray-900 font-mono text-sm"
                placeholder="1000, 981, 909, 725, 0.03"
              />
            </div>


          </div>
        )}

        {/* Advanced Parameters - LTX 2.3 Local + Condition (fixation-control knobs) */}
        {showAdvanced && (selectedModel === 'ltx-2.3-local' || selectedModel === 'ltx-2.3-condition') && (
          <div className="space-y-4 border-t border-fal-gray-700 pt-6">
            <h4 className="text-fal-gray-900 font-medium">
              Advanced LTX 2.3 Local Parameters
              {ltx23LocalConfig.style_preset !== 'custom' && (
                <span className="text-xs font-normal text-fal-gray-500 ml-2">(preset: {ltx23LocalConfig.style_preset} -- edit to customize)</span>
              )}
            </h4>
            <p className="text-xs text-fal-gray-600">
              Tune these to fight scene fixation. Higher <code>guidance_scale</code> makes
              the prompt matter more; <code>noise_scale</code> injects entropy into the
              latents; leave seed empty for a fresh random seed every generation
              (recommended). <code>stg_scale</code> adds motion variety but also requires
              non-empty <code>STG blocks</code> (transformer indices) -- if you do not know
              the right indices for the checkpoint, leave both at 0 and the request will
              skip STG safely.
            </p>

            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div>
                <label className="metric-label mb-2 block text-xs">Frames (8k+1)</label>
                <input
                  type="number"
                  value={ltx23LocalConfig.num_frames}
                  onChange={(e) => updateLocalParam({ num_frames: parseInt(e.target.value) })}
                  className="w-full bg-fal-gray-100 border border-fal-gray-300 rounded-lg p-3 text-fal-gray-900 font-mono"
                  min={9}
                  max={241}
                  step={8}
                />
              </div>

              <div>
                <label className="metric-label mb-2 block text-xs">guidance_scale</label>
                <input
                  type="number"
                  value={ltx23LocalConfig.guidance_scale}
                  onChange={(e) => updateLocalParam({ guidance_scale: parseFloat(e.target.value) })}
                  className="w-full bg-fal-gray-100 border border-fal-gray-300 rounded-lg p-3 text-fal-gray-900 font-mono"
                  min={1}
                  max={7}
                  step={0.1}
                />
              </div>

              <div>
                <label className="metric-label mb-2 block text-xs">stg_scale</label>
                <input
                  type="number"
                  value={ltx23LocalConfig.stg_scale}
                  onChange={(e) => updateLocalParam({ stg_scale: parseFloat(e.target.value) })}
                  className="w-full bg-fal-gray-100 border border-fal-gray-300 rounded-lg p-3 text-fal-gray-900 font-mono"
                  min={0}
                  max={3}
                  step={0.1}
                />
              </div>

              <div>
                <label className="metric-label mb-2 block text-xs">STG blocks (csv)</label>
                <input
                  type="text"
                  placeholder="e.g. 19 or 14,18,22"
                  value={(ltx23LocalConfig.spatio_temporal_guidance_blocks ?? []).join(',')}
                  onChange={(e) => {
                    const raw = e.target.value.trim()
                    if (!raw) {
                      updateLocalParam({ spatio_temporal_guidance_blocks: null })
                      return
                    }
                    const blocks = raw.split(',')
                      .map(s => parseInt(s.trim()))
                      .filter(n => !isNaN(n) && n >= 0)
                    updateLocalParam({ spatio_temporal_guidance_blocks: blocks.length ? blocks : null })
                  }}
                  className="w-full bg-fal-gray-100 border border-fal-gray-300 rounded-lg p-3 text-fal-gray-900 font-mono text-sm"
                />
              </div>

              <div>
                <label className="metric-label mb-2 block text-xs">noise_scale</label>
                <input
                  type="number"
                  value={ltx23LocalConfig.noise_scale}
                  onChange={(e) => updateLocalParam({ noise_scale: parseFloat(e.target.value) })}
                  className="w-full bg-fal-gray-100 border border-fal-gray-300 rounded-lg p-3 text-fal-gray-900 font-mono"
                  min={0}
                  max={0.3}
                  step={0.01}
                />
              </div>

              <div>
                <label className="metric-label mb-2 block text-xs">LLM temperature</label>
                <input
                  type="number"
                  value={ltx23LocalConfig.llm_temperature}
                  onChange={(e) => updateLocalParam({ llm_temperature: parseFloat(e.target.value) })}
                  className="w-full bg-fal-gray-100 border border-fal-gray-300 rounded-lg p-3 text-fal-gray-900 font-mono"
                  min={0.1}
                  max={1.5}
                  step={0.1}
                />
              </div>

              <div className="md:col-span-2">
                <label className="metric-label mb-2 block text-xs">Seed</label>
                <div className="flex items-center space-x-3">
                  <input
                    type="number"
                    value={ltx23LocalConfig.seed ?? ''}
                    placeholder="random"
                    disabled={ltx23LocalConfig.seed === null}
                    onChange={(e) => {
                      const v = e.target.value
                      updateLocalParam({ seed: v === '' ? null : parseInt(v) })
                    }}
                    className="flex-1 bg-fal-gray-100 border border-fal-gray-300 rounded-lg p-3 text-fal-gray-900 font-mono disabled:opacity-50"
                    min={0}
                    step={1}
                  />
                  <label className="flex items-center space-x-2 text-xs text-fal-gray-700 select-none">
                    <input
                      type="checkbox"
                      checked={ltx23LocalConfig.seed === null}
                      onChange={(e) => {
                        updateLocalParam({ seed: e.target.checked ? null : 0 })
                      }}
                    />
                    <span>Random per generation</span>
                  </label>
                </div>
              </div>
            </div>

            <div>
              <label className="metric-label mb-2 block text-xs">Negative Prompt</label>
              <textarea
                value={ltx23LocalConfig.negative_prompt}
                onChange={(e) => updateLocalParam({ negative_prompt: e.target.value })}
                className="w-full bg-fal-gray-100 border border-fal-gray-300 rounded-lg p-3 text-fal-gray-900 font-mono text-sm"
                rows={2}
              />
            </div>

            <div>
              <label className="flex items-center space-x-2 select-none cursor-pointer">
                <input
                  type="checkbox"
                  checked={ltx23LocalConfig.enable_audio}
                  onChange={(e) => setLtx23LocalConfig(prev => ({ ...prev, enable_audio: e.target.checked }))}
                />
                <span className="text-sm text-fal-gray-900 font-medium">Stream native audio (experimental)</span>
              </label>
              <p className="text-xs text-fal-gray-600 mt-1 ml-6">
                LTX 2.3 jointly generates audio with video. When off, the stream uses
                silent <code>anullsrc</code> like before. When on, the model's PCM
                output is fed into ffmpeg via a FIFO. The stream must be restarted
                for this toggle to take effect.
              </p>
            </div>

            <div>
              <button
                type="button"
                onClick={() => setShowSystemPrompt(!showSystemPrompt)}
                className="text-sm text-fal-primary-500 hover:text-fal-primary-600 font-medium"
              >
                {showSystemPrompt ? 'Hide' : 'Show'} System Prompt
              </button>
              {showSystemPrompt && (
                <div className="mt-2 bg-fal-gray-50 border border-fal-gray-200 rounded-lg p-3">
                  <p className="text-xs text-fal-gray-500 mb-2 font-medium">
                    System prompt for preset: <strong>{ltx23LocalConfig.style_preset}</strong>
                  </p>
                  <pre className="text-xs text-fal-gray-700 whitespace-pre-wrap font-mono leading-relaxed max-h-64 overflow-y-auto">
                    {SYSTEM_PROMPT_PREVIEWS[ltx23LocalConfig.style_preset] || 'Unknown preset'}
                  </pre>
                </div>
              )}
            </div>
          </div>
        )}



        {/* Character References (LTX 2.3 Condition only) */}
        {selectedModel === 'ltx-2.3-condition' && (
          <div>
            <label className="metric-label mb-3 block">Character References</label>
            <p className="text-xs text-fal-gray-600 mb-3">
              Add up to 4 character reference images. The model uses these as visual anchors
              for character consistency. Characters appear in the story when the narrative calls for them.
            </p>
            <div className="space-y-3">
              {ltx23ConditionConfig.character_refs.map((ref, idx) => (
                <div key={idx} className="flex items-start space-x-3 bg-fal-gray-50 border border-fal-gray-200 rounded-lg p-3">
                  {ref.image && (
                    <img
                      src={ref.image.startsWith('data:') ? ref.image : ref.image}
                      alt={ref.label || `Character ${idx + 1}`}
                      className="w-16 h-16 rounded object-cover flex-shrink-0"
                    />
                  )}
                  <div className="flex-1 space-y-2">
                    <input
                      type="text"
                      placeholder="Character name (e.g. Homer Simpson)"
                      value={ref.label}
                      onChange={(e) => {
                        const updated = [...ltx23ConditionConfig.character_refs]
                        updated[idx] = { ...ref, label: e.target.value }
                        setLtx23ConditionConfig(prev => ({ ...prev, character_refs: updated }))
                      }}
                      className="w-full bg-white border border-fal-gray-300 rounded p-2 text-sm text-fal-gray-900"
                    />
                    <div className="flex items-center space-x-3">
                      <input
                        type="text"
                        placeholder="Image URL or upload below"
                        value={ref.image.startsWith('data:') ? '(uploaded)' : ref.image}
                        onChange={(e) => {
                          const updated = [...ltx23ConditionConfig.character_refs]
                          updated[idx] = { ...ref, image: e.target.value }
                          setLtx23ConditionConfig(prev => ({ ...prev, character_refs: updated }))
                        }}
                        className="flex-1 bg-white border border-fal-gray-300 rounded p-2 text-xs text-fal-gray-900 font-mono"
                      />
                      <label className="btn-secondary text-xs cursor-pointer px-2 py-1">
                        Upload
                        <input
                          type="file"
                          accept="image/*"
                          className="hidden"
                          onChange={(e) => {
                            const file = e.target.files?.[0]
                            if (!file) return
                            const reader = new FileReader()
                            reader.onload = (ev) => {
                              const updated = [...ltx23ConditionConfig.character_refs]
                              updated[idx] = { ...ref, image: ev.target?.result as string }
                              setLtx23ConditionConfig(prev => ({ ...prev, character_refs: updated }))
                            }
                            reader.readAsDataURL(file)
                          }}
                        />
                      </label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <label className="text-xs text-fal-gray-600 w-20">Strength:</label>
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={ref.strength}
                        onChange={(e) => {
                          const updated = [...ltx23ConditionConfig.character_refs]
                          updated[idx] = { ...ref, strength: parseFloat(e.target.value) }
                          setLtx23ConditionConfig(prev => ({ ...prev, character_refs: updated }))
                        }}
                        className="flex-1"
                      />
                      <span className="text-xs text-fal-gray-700 font-mono w-8">{ref.strength.toFixed(2)}</span>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      const updated = ltx23ConditionConfig.character_refs.filter((_, i) => i !== idx)
                      setLtx23ConditionConfig(prev => ({ ...prev, character_refs: updated }))
                    }}
                    className="text-fal-gray-400 hover:text-fal-red-500 text-lg font-bold"
                    title="Remove character"
                  >
                    ×
                  </button>
                </div>
              ))}
              {ltx23ConditionConfig.character_refs.length < 4 && (
                <button
                  type="button"
                  onClick={() => {
                    setLtx23ConditionConfig(prev => ({
                      ...prev,
                      character_refs: [
                        ...prev.character_refs,
                        { image: '', strength: 0.4, label: '' },
                      ],
                    }))
                  }}
                  className="btn-secondary text-sm w-full"
                >
                  + Add Character Reference
                </button>
              )}
            </div>
          </div>
        )}

        {/* Style Preset Selector (LTX 2.3 Local + Condition) */}
        {(selectedModel === 'ltx-2.3-local' || selectedModel === 'ltx-2.3-condition') && (
          <div>
            <label className="metric-label mb-3 block">Style Preset</label>
            <div className="grid grid-cols-4 gap-2">
              {([
                { id: 'cohesive', label: 'Cohesive', desc: 'Smooth story evolution' },
                { id: 'chaotic', label: 'Chaotic', desc: 'Dramatic scene changes' },
                { id: 'nightmare', label: 'Nightmare', desc: 'Surreal fever dream' },
                { id: 'custom', label: 'Custom', desc: 'Manual Advanced values' },
              ] as const).map(preset => (
                <button
                  key={preset.id}
                  type="button"
                  onClick={() => {
                    const PRESET_PARAMS: Record<string, Partial<LTX23LocalConfig>> = {
                      cohesive:  { guidance_scale: 1.0, noise_scale: 0.03, llm_temperature: 0.55, mode: 'regular' },
                      chaotic:   { guidance_scale: 3.0, noise_scale: 0.15, llm_temperature: 0.7, mode: 'regular' },
                      nightmare: { guidance_scale: 3.5, noise_scale: 0.20, llm_temperature: 0.9, mode: 'nightmare' },
                    }
                    const overrides = PRESET_PARAMS[preset.id] || {}
                    setLtx23LocalConfig(prev => ({ ...prev, ...overrides, style_preset: preset.id }))
                  }}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-all border-2 ${
                    ltx23LocalConfig.style_preset === preset.id
                      ? 'bg-fal-primary-500 text-white border-fal-primary-500 shadow-lg'
                      : 'bg-white text-fal-gray-700 hover:bg-fal-gray-50 border-fal-gray-300'
                  }`}
                >
                  {preset.label}
                  <p className="text-xs mt-0.5 opacity-75">{preset.desc}</p>
                </button>
              ))}
            </div>
            {ltx23LocalConfig.style_preset !== 'custom' && (
              <p className="text-xs text-fal-gray-600 mt-2">
                Preset overrides guidance_scale, noise_scale, and LLM temperature.
                Switch to <strong>Custom</strong> for full manual control via Advanced.
              </p>
            )}
          </div>
        )}

        {/* Output Mode Selector (LTX 2.3 Local + Condition) */}
        {(selectedModel === 'ltx-2.3-local' || selectedModel === 'ltx-2.3-condition') && (
          <div>
            <label className="metric-label mb-3 block">Output Mode</label>
            <div className="grid grid-cols-2 gap-3">
              <button
                type="button"
                onClick={() => setLtx23LocalConfig(prev => ({ ...prev, output_mode: 'rtmp' }))}
                className={`px-4 py-3 rounded-lg text-sm font-medium transition-all border-2 ${
                  ltx23LocalConfig.output_mode === 'rtmp'
                    ? 'bg-fal-primary-500 text-white border-fal-primary-500 shadow-lg'
                    : 'bg-white text-fal-gray-700 hover:bg-fal-gray-50 border-fal-gray-300'
                }`}
              >
                RTMP / Twitch
                <p className="text-xs mt-1 opacity-75">Push to Twitch via FFmpeg</p>
              </button>
              <button
                type="button"
                onClick={() => setLtx23LocalConfig(prev => ({ ...prev, output_mode: 'webrtc' }))}
                className={`px-4 py-3 rounded-lg text-sm font-medium transition-all border-2 ${
                  ltx23LocalConfig.output_mode === 'webrtc'
                    ? 'bg-fal-primary-500 text-white border-fal-primary-500 shadow-lg'
                    : 'bg-white text-fal-gray-700 hover:bg-fal-gray-50 border-fal-gray-300'
                }`}
              >
                WebRTC Direct
                <p className="text-xs mt-1 opacity-75">Stream to browser, sub-1s latency</p>
              </button>
            </div>
          </div>
        )}

        {/* Control Buttons */}
        <div className="flex items-center justify-between pt-4 border-t border-fal-gray-700">
          <div className="flex items-center space-x-4">
            {!isStreaming ? (
              <button
                onClick={() => {
                  if (selectedModel === 'ltxv1') {
                    onStartTest({ ...ltxv1Config, initial_prompt: displayedPrompt })
                  } else if (selectedModel === 'ltx-2.3-condition') {
                    onStartTest({ ...ltx23ConditionConfig, initial_prompt: displayedPrompt })
                  } else if (selectedModel === 'ltx-2.3-local') {
                    onStartTest({ ...ltx23LocalConfig, initial_prompt: displayedPrompt })
                  } else {
                    const processedConfig: any = {
                      ...ltxv2Config,
                      initial_image_url: ltxv2Config.image_url,
                      initial_prompt: ltxv2Config.prompt
                    }
                    onStartTest(processedConfig)
                  }
                }}
                className="btn-primary flex items-center space-x-2"
              >
                <Play className="w-4 h-4" />
                <span>Start Stream</span>
              </button>
            ) : (
              <div className="flex items-center space-x-3">
                <button
                  onClick={onStopTest}
                  className="bg-fal-red-500 hover:bg-fal-red-600 text-white font-medium px-6 py-2 rounded-lg transition-colors flex items-center space-x-2"
                >
                  <Square className="w-4 h-4" />
                  <span>Stop Stream</span>
                </button>
                {onUpdateConfig && (selectedModel === 'ltx-2.3-local' || selectedModel === 'ltx-2.3-condition') && (
                  <button
                    onClick={async () => {
                      setApplyStatus('applying')
                      try {
                        await onUpdateConfig({
                          guidance_scale: ltx23LocalConfig.guidance_scale,
                          noise_scale: ltx23LocalConfig.noise_scale,
                          seed: ltx23LocalConfig.seed,
                          negative_prompt: ltx23LocalConfig.negative_prompt,
                          num_frames: ltx23LocalConfig.num_frames,
                          stg_scale: ltx23LocalConfig.stg_scale,
                          spatio_temporal_guidance_blocks: ltx23LocalConfig.spatio_temporal_guidance_blocks,
                          llm_temperature: ltx23LocalConfig.llm_temperature,
                          style_preset: ltx23LocalConfig.style_preset,
                        })
                        setApplyStatus('applied')
                        setTimeout(() => setApplyStatus('idle'), 3000)
                      } catch {
                        setApplyStatus('idle')
                      }
                    }}
                    disabled={applyStatus === 'applying'}
                    className={`font-medium px-6 py-2 rounded-lg transition-all flex items-center space-x-2 ${
                      applyStatus === 'applied'
                        ? 'bg-green-500 text-white'
                        : applyStatus === 'applying'
                        ? 'bg-fal-yellow-300 text-black opacity-75 cursor-wait'
                        : 'bg-fal-yellow-500 hover:bg-fal-yellow-600 text-black'
                    }`}
                  >
                    <RefreshCw className={`w-4 h-4 ${applyStatus === 'applying' ? 'animate-spin' : ''}`} />
                    <span>
                      {applyStatus === 'applied' ? 'Applied! (next clip)' : applyStatus === 'applying' ? 'Applying...' : 'Apply Changes'}
                    </span>
                  </button>
                )}
              </div>
            )}
          </div>
          

        </div>
      </div>
    </div>
  )
}
