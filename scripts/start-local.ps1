$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

New-Item -ItemType Directory -Force -Path "logs" | Out-Null

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Start-Process `
  -FilePath ".\.venv\Scripts\python.exe" `
  -ArgumentList @("-m", "uvicorn", "streaming_pipeline.local_app:app", "--host", "127.0.0.1", "--port", "8000") `
  -WorkingDirectory $root `
  -RedirectStandardOutput (Join-Path $root "logs\backend.log") `
  -RedirectStandardError (Join-Path $root "logs\backend.err.log") `
  -WindowStyle Hidden

Start-Process `
  -FilePath "npm.cmd" `
  -ArgumentList @("run", "dev", "--", "--hostname", "127.0.0.1", "--port", "3000") `
  -WorkingDirectory (Join-Path $root "dashboard") `
  -RedirectStandardOutput (Join-Path $root "logs\dashboard.log") `
  -RedirectStandardError (Join-Path $root "logs\dashboard.err.log") `
  -WindowStyle Hidden

Write-Host "Backend:   http://127.0.0.1:8000"
Write-Host "Dashboard: http://127.0.0.1:3000"
