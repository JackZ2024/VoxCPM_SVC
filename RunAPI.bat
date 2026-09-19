@echo off
set PATH=%CD%\runtime\Scripts\;%CD%\runtime\;%PATH%

runtime\python.exe api.py --host 0.0.0.0 --port 7862

pause
