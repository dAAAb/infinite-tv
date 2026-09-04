$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# A forcibly stopped backend can leave its FFmpeg child alive. Twitch accepts
# only one publisher per stream key, so that orphan makes the replacement
# FFmpeg fail after its first frame while the channel appears superficially
# live. Stop only Infinite TV's Twitch RTMP publishers before restarting.
$stalePublishers = Get-CimInstance Win32_Process -Filter "Name = 'ffmpeg.exe'" |
  Where-Object { $_.CommandLine -like "*rtmp://live.twitch.tv/app/*" }
foreach ($publisher in $stalePublishers) {
  Stop-Process -Id $publisher.ProcessId -Force -ErrorAction SilentlyContinue
  Write-Host "stopped stale Twitch FFmpeg PID $($publisher.ProcessId)"
}
if ($stalePublishers) {
  # Twitch ingest can retain the old publishing session briefly after a hard
  # disconnect. Give it time to release the stream key before FFmpeg reconnects.
  Start-Sleep -Seconds 8
}

$conn = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($conn) {
  $pids = $conn | Select-Object -ExpandProperty OwningProcess -Unique
  foreach ($processId in $pids) {
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    Write-Host "stopped backend PID $processId"
  }
  Start-Sleep -Seconds 2
} else {
  Write-Host "backend not running"
}

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
# Continuity-first LTX-2.5 mode: do not repeatedly inject a visual style prompt.
Remove-Item Env:H3_MAX_STYLE_PREFIX -ErrorAction SilentlyContinue
# FAL_KEY may remain available for explicitly selected fal video models, but
# prompt/vision routing for the local LTX stream must use OpenAI directly.
$env:USE_FAL_OPENROUTER = "false"
$env:ENABLE_FAL_VIDEO = "false"
# Never let the visual quality gate deadlock the channel. After one retry,
# commit a seamless local push-in and continue ComfyUI from its clean tail.
$env:LTX25_MAX_CORRUPT_RETRIES = "2"
$env:LTX25_RECOVERY_INSET_RATIO = "0.18"
# Bound interaction latency so Twitch comments reach viewers within roughly one
# buffered clip instead of sitting behind several minutes of generated frames.
$env:RTMP_TARGET_QUEUE_SECONDS = "18"
# torch 2.9 Windows: disable inductor static CUDA launcher so torch.compile
# (reduce-overhead) doesn't hit OverflowError. Required for LTX23_TORCH_COMPILE=true.
$env:TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER = "0"

Start-Process `
  -FilePath ".\.venv\Scripts\python.exe" `
  -ArgumentList @("-m", "uvicorn", "streaming_pipeline.local_app:app", "--host", "127.0.0.1", "--port", "8000") `
  -WorkingDirectory $root `
  -RedirectStandardOutput (Join-Path $root "logs\backend.log") `
  -RedirectStandardError (Join-Path $root "logs\backend.err.log") `
  -WindowStyle Hidden

Write-Host "Backend restarting on http://127.0.0.1:8000"
