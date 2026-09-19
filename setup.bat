@echo off

set PATH=%CD%\runtime\Scripts\;%CD%\runtime\;%PATH%

runtime\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
runtime\python.exe -m pip install -r requirements.txt
runtime\python.exe -m pip install -e ./VoxCPM
runtime\python.exe -m pip install --no-cache-dir --force-reinstall "triton-windows==3.4.0.post21"


echo 安装包完成。
pause
