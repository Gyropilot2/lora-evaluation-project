@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
call "%ROOT%load_runtime_env.bat"
set "BACKEND_HEALTH_URL=http://%BACKEND_HOST%:%BACKEND_PORT%%BACKEND_HEALTHZ_PATH%"
set "FRONTEND_URL=http://%FRONTEND_HOST%:%FRONTEND_PORT%"

echo Starting Operator App backend...
start "LEP Operator Backend" cmd /k "cd /d "%ROOT%backend" && call run_backend.bat"

echo Starting Operator App frontend...
start "LEP Operator Frontend" cmd /k "cd /d "%ROOT%frontend" && call run_frontend.bat"

echo Waiting for backend health...
call :wait_for_url "%BACKEND_HEALTH_URL%" "backend"

echo Waiting for frontend...
call :wait_for_url "%FRONTEND_URL%" "frontend"

echo Opening browser...
start "" "%FRONTEND_URL%"

exit /b 0

:wait_for_url
set "TARGET_URL=%~1"
set "TARGET_NAME=%~2"
for /l %%I in (1,1,45) do (
  powershell -NoProfile -Command "try { $resp = Invoke-WebRequest -UseBasicParsing '%TARGET_URL%' -TimeoutSec 1; if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 500) { exit 0 } else { exit 1 } } catch { exit 1 }"
  if not errorlevel 1 goto :wait_ok
  timeout /t 1 /nobreak >nul
)
echo Warning: %TARGET_NAME% did not become ready in time. Opening anyway.
goto :eof

:wait_ok
echo %TARGET_NAME% ready.
goto :eof
