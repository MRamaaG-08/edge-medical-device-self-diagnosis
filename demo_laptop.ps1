# Starts the review demo on the laptop: the hub in one window, the dashboard in another.
# Run from the project folder:   .\demo_laptop.ps1            (uses COM6)
#                                .\demo_laptop.ps1 -Port COM5
param([string]$Port = "COM6")
$here = (Get-Location).Path
$session = "DEMO_" + (Get-Date -Format "MMdd_HHmm")
Write-Host "Session name: $session  (port $Port)"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$here'; .venv\Scripts\Activate.ps1; python -m phase2.hardware.hub.live --port $Port --session $session"
Start-Sleep -Seconds 3
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$here'; .venv\Scripts\Activate.ps1; streamlit run phase2\hardware\dashboard\app.py -- --log data\sessions\$session.jsonl"
Write-Host "Two windows opened. The dashboard opens in the browser at http://localhost:8501"
