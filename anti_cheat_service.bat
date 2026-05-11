@echo off
title Anti-Cheat Detection System
cd /d %~dp0..
call venv\Scripts\activate
python run_system.py --config config\testing.json
pause
