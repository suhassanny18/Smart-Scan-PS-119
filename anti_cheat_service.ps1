$Host.UI.RawUI.WindowTitle = 'Anti-Cheat Detection System'
Set-Location $PSScriptRoot\..
& .\venv\Scripts\Activate.ps1
python run_system.py --config config\testing.json
Read-Host 'Press Enter to exit'
