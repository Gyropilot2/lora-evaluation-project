@echo off

set "SCRIPT_DIR=%~dp0"
set "CONFIG_PATH=%SCRIPT_DIR%runtime_config.json"

for /f "usebackq tokens=1,* delims==" %%A in (`powershell -NoProfile -Command ^
  "$cfg = Get-Content -Raw '%CONFIG_PATH%' | ConvertFrom-Json; " ^
  "Write-Output ('BACKEND_HOST=' + $cfg.backend.host); " ^
  "Write-Output ('BACKEND_PORT=' + $cfg.backend.port); " ^
  "Write-Output ('BACKEND_HEALTHZ_PATH=' + $cfg.backend.healthz_path); " ^
  "Write-Output ('FRONTEND_HOST=' + $cfg.frontend.host); " ^
  "Write-Output ('FRONTEND_PORT=' + $cfg.frontend.port)"`) do set "%%A=%%B"
