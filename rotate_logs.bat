@echo off
forfiles /p logs\testing /s /m *.log /d -7 /c "cmd /c del @path"
echo Log rotation completed
