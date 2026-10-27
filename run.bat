@echo off
cd /d "%~dp0"
"..\.venv_face\Scripts\python.exe" "source\main.py"
if errorlevel 1 echo [ERROR] Please read README.md.
pause
