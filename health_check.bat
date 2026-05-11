@echo off
cd /d %~dp0..
call venv\Scripts\activate
python run_system.py --validate-only
if %errorlevel% equ 0 (
    echo System is healthy
    exit /b 0
) else (
    echo System health check failed
    exit /b 1
)
