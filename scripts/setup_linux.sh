#!/usr/bin/env bash
set -euo pipefail

# 快速在 Linux 上安装依赖并创建虚拟环境。
# 可通过环境变量自定义：
#   PYTHON      - Python 可执行文件名，默认 python3
#   ENV_DIR     - 虚拟环境目录，默认 .venv
#   TORCH_VARIANT - 依赖版本：cpu / gpu / amd，默认 cpu

PYTHON_BIN="${PYTHON:-python3}"
ENV_DIR="${ENV_DIR:-.venv}"
TORCH_VARIANT="${TORCH_VARIANT:-cpu}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "未找到 $PYTHON_BIN，请先安装 Python 3.12+" >&2
  exit 1
fi

# 创建虚拟环境
"$PYTHON_BIN" -m venv "$ENV_DIR"
# shellcheck disable=SC1090
source "$ENV_DIR/bin/activate"

# 升级 pip
python -m pip install --upgrade pip

case "$TORCH_VARIANT" in
  cpu)
    REQ_FILE="requirements_cpu.txt"
    ;;
  gpu)
    REQ_FILE="requirements_gpu.txt"
    ;;
  amd)
    REQ_FILE="requirements_amd.txt"
    ;;
  *)
    echo "未知的 TORCH_VARIANT: $TORCH_VARIANT，可选值 cpu/gpu/amd" >&2
    exit 1
    ;;
esac

echo "使用依赖文件: $REQ_FILE"
pip install --no-cache-dir -r "$REQ_FILE"

echo "安装完成。激活虚拟环境后可运行："
echo "  python -m manga_translator local -i your_image.jpg"
