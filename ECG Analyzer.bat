@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-win\Scripts\pythonw.exe" (
  echo Setting up ECG Analyzer, first run only...
  where py >nul 2>nul && (py -3 -m venv .venv-win) || (python -m venv .venv-win)
  if not exist ".venv-win\Scripts\python.exe" (
    echo Python 3.11 or newer is required: https://www.python.org/downloads/
    pause
    exit /b 1
  )
  ".venv-win\Scripts\python.exe" -m pip install -r requirements-app.txt
  ".venv-win\Scripts\python.exe" -m pip install -e . --no-deps
)
start "" ".venv-win\Scripts\pythonw.exe" -m ecgpit.desktop
