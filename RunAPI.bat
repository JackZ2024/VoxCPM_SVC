@echo off
set PATH=%CD%\runtime\Scripts\;%CD%\runtime\;%PATH%

runtime\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8080

pause
