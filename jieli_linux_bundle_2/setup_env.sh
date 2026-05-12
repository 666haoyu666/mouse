#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f .env ]]; then
  set -a
  source ./.env
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERR] 找不到 Python: $PYTHON_BIN"
  echo "[HINT] 先安装 python3 / python3-venv 再重试。"
  exit 1
fi

chmod +x ./*.sh ./*.py || true

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[INFO] 创建虚拟环境: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "[INFO] 已生成 .env，请按需修改。"
fi

echo "[OK] 环境准备完成。"
echo "[NEXT] 前台启动 UDP: ./start_udp.sh"
echo "[NEXT] 前台启动 CTP: ./start_ctp.sh"
echo "[NEXT] 一键启动两者: ./start_all.sh"
