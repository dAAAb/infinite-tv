'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { Volume2, VolumeX, Wifi, WifiOff } from 'lucide-react'

interface WebRTCPlayerProps {
  apiUrl: string
}

const TOKEN_EXPIRATION_MS = 300

const fetchTemporaryToken = async (appName: string): Promise<string> => {
  const response = await fetch('/api/fal/proxy', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Fal-Target-Url': 'https://rest.alpha.fal.ai/tokens/',
    },
    body: JSON.stringify({
      allowed_apps: [appName],
      token_expiration: TOKEN_EXPIRATION_MS,
    }),
  })
  if (!response.ok) throw new Error('Failed to fetch token')
  return await response.json()
}

export default function WebRTCPlayer({ apiUrl }: WebRTCPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const pcRef = useRef<RTCPeerConnection | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  const [connected, setConnected] = useState(false)
  const [muted, setMuted] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const cleanup = useCallback(() => {
    if (pcRef.current) {
      pcRef.current.close()
      pcRef.current = null
    }
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setConnected(false)
  }, [])

  const connect = useCallback(async () => {
    cleanup()
    setError(null)

    try {
      const appName = apiUrl.includes('fal.run')
        ? apiUrl.split('/').pop() || 'realtime-streaming'
        : 'realtime-streaming'

      const token = await fetchTemporaryToken(appName)

      const wsUrl = apiUrl
        .replace(/^http:\/\//, 'ws://')
        .replace(/^https:\/\//, 'wss://')
        + `/webrtc?fal_jwt_token=${encodeURIComponent(token)}`

      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        console.log('WebRTC signaling: connected')
      }

      ws.onclose = () => {
        console.log('WebRTC signaling: disconnected')
        setConnected(false)
      }

      ws.onerror = () => {
        setError('WebSocket connection failed')
        setConnected(false)
      }

      ws.onmessage = async (event) => {
        const msg = JSON.parse(event.data)

        if (msg.type === 'ready') {
          const pc = new RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
          })
          pcRef.current = pc

          pc.onicecandidate = (e) => {
            if (ws.readyState !== WebSocket.OPEN) return
            if (e.candidate) {
              ws.send(JSON.stringify({
                type: 'icecandidate',
                candidate: {
                  candidate: e.candidate.candidate,
                  sdpMid: e.candidate.sdpMid,
                  sdpMLineIndex: e.candidate.sdpMLineIndex,
                },
              }))
            } else {
              ws.send(JSON.stringify({ type: 'icecandidate', candidate: null }))
            }
          }

          pc.onconnectionstatechange = () => {
            const state = pc.connectionState
            console.log(`WebRTC: connection state -> ${state}`)
            if (state === 'connected') {
              setConnected(true)
              setError(null)
            } else if (state === 'failed' || state === 'disconnected' || state === 'closed') {
              setConnected(false)
            }
          }

          // Collect all incoming tracks into a single MediaStream so both
          // video and audio are attached to the same <video> element.
          const remoteStream = new MediaStream()

          pc.ontrack = (event) => {
            console.log(`WebRTC: received track kind=${event.track.kind} id=${event.track.id}`)
            remoteStream.addTrack(event.track)
            if (videoRef.current) {
              videoRef.current.srcObject = remoteStream
              // Re-trigger play in case the element paused waiting for tracks
              videoRef.current.play().catch(() => {})
            }
          }

          pc.addTransceiver('video', { direction: 'recvonly' })
          pc.addTransceiver('audio', { direction: 'recvonly' })

          const offer = await pc.createOffer()
          await pc.setLocalDescription(offer)

          ws.send(JSON.stringify({ type: 'offer', sdp: offer.sdp }))

        } else if (msg.type === 'answer') {
          if (pcRef.current) {
            await pcRef.current.setRemoteDescription(
              new RTCSessionDescription({ type: 'answer', sdp: msg.sdp })
            )
          }

        } else if (msg.type === 'icecandidate') {
          if (pcRef.current && msg.candidate) {
            try {
              await pcRef.current.addIceCandidate(new RTCIceCandidate(msg.candidate))
            } catch (e) {
              console.warn('WebRTC: failed to add ICE candidate', e)
            }
          }

        } else if (msg.type === 'error') {
          setError(msg.message || 'WebRTC error')
        }
      }

    } catch (e: any) {
      setError(e.message || 'Connection failed')
    }
  }, [apiUrl, cleanup])

  useEffect(() => {
    connect()
    return cleanup
  }, [connect, cleanup])

  const toggleMute = () => {
    if (videoRef.current) {
      const shouldMute = !videoRef.current.muted
      videoRef.current.muted = shouldMute
      if (!shouldMute) {
        videoRef.current.volume = 1.0
        videoRef.current.play().catch(() => {})
      }
      setMuted(shouldMute)
    }
  }

  return (
    <div className="fal-card">
      <div className="fal-card-header">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {connected ? (
              <Wifi className="w-5 h-5 text-green-500" />
            ) : (
              <WifiOff className="w-5 h-5 text-fal-gray-400" />
            )}
            <h3 className="text-lg font-semibold text-fal-gray-900">
              WebRTC Stream
            </h3>
            {connected && (
              <span className="text-xs bg-green-100 text-green-800 px-2 py-0.5 rounded-full font-medium">
                LIVE
              </span>
            )}
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={toggleMute}
              className="btn-secondary text-sm p-2"
              title={muted ? 'Unmute' : 'Mute'}
            >
              {muted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
            </button>
            <button
              onClick={connect}
              className="btn-secondary text-sm"
            >
              Reconnect
            </button>
          </div>
        </div>
      </div>

      <div className="fal-card-content">
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4 text-sm text-red-700">
            {error}
          </div>
        )}
        <div className="relative bg-black rounded-lg overflow-hidden" style={{ aspectRatio: '4/3' }}>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted={muted}
            className="w-full h-full object-contain"
          />
          {!connected && (
            <div className="absolute inset-0 flex items-center justify-center text-fal-gray-400">
              <div className="text-center">
                <WifiOff className="w-12 h-12 mx-auto mb-2" />
                <p className="text-sm">Waiting for WebRTC connection...</p>
                <p className="text-xs mt-1">Start a stream with output_mode = "webrtc"</p>
              </div>
            </div>
          )}
        </div>
        {muted && connected && (
          <p className="text-xs text-fal-gray-600 mt-2 text-center">
            Click the speaker icon to unmute audio
          </p>
        )}
      </div>
    </div>
  )
}
