#!/bin/bash

# 虚拟环境目录，默认是 myenv
ENV_NAME=myenv

if [ ! -d "$ENV_NAME" ]; then
    echo "虚拟环境不存在，请先运行setup.sh脚本创建虚拟环境并安装依赖。"
    exit 1
fi

# 激活虚拟环境
# shellcheck disable=SC1090
source "$ENV_NAME/bin/activate"

# 运行 Python 脚本
python3 api.py --host 0.0.0.0 --port 7862
