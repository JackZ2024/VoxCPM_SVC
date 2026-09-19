#!/bin/bash

# 设置虚拟环境名称，默认是 "myenv"
ENV_NAME=${1:-myenv}

echo ">>> 检查 Python3 是否已安装..."
if ! command -v python3 &> /dev/null; then
    echo "Python3 未安装，请先安装 Python3。"
    exit 1
fi

echo ">>> 检查 python3-venv 模块..."
if ! dpkg -s python3-venv &> /dev/null; then
    echo "python3-venv 未安装，正在安装..."
    sudo apt update
    sudo apt install -y python3-venv
fi

if [ -d "$ENV_NAME" ]; then
    echo ">>> 虚拟环境 '$ENV_NAME' 已存在，跳过创建。"
else
    echo ">>> 创建虚拟环境：$ENV_NAME"
    python3 -m venv "$ENV_NAME"
    echo ">>> 虚拟环境已创建在 ./$ENV_NAME"
fi

# 激活虚拟环境
echo ">>> 激活虚拟环境：$ENV_NAME"
# shellcheck disable=SC1090
source "$ENV_NAME/bin/activate"

# 安装依赖
if [ -f "requirements.txt" ]; then
    echo ">>> 检测到 requirements.txt，正在安装依赖..."
    pip install --upgrade pip
    python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu129
    pip install -r requirements.txt
    python -m pip install -e ./VoxCPM
    echo ">>> 依赖安装完成。"
else
    echo ">>> 未找到 requirements.txt，跳过依赖安装。"
fi

echo ">>> 虚拟环境已准备就绪。你现在可以使用 Python 环境。"
