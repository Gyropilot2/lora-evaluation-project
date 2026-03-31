@echo off
setlocal EnableExtensions

set "BACKEND_DIR=%~dp0"
set "PROJECT_ROOT=%BACKEND_DIR%..\.."
set "PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%"
call "%PROJECT_ROOT%\operator_app\load_runtime_env.bat"

python -m uvicorn app.main:app --host %BACKEND_HOST% --port %BACKEND_PORT%
