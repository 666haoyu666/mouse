#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f .env ]]; then
  set -a
  source ./.env
  set +a
fi

VENV_DIR="${VENV_DIR:-.venv}"
LOG_DIR="${LOG_DIR:-logs}"
DEVICE_IP="${DEVICE_IP:-192.168.1.1}"
CTP_PORT="${CTP_PORT:-3333}"
HEARTBEAT="${HEARTBEAT:-10}"
CTP_MODE="${CTP_MODE:-client}"
LISTEN_IP="${LISTEN_IP:-0.0.0.0}"

mkdir -p "$LOG_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[ERR] 虚拟环境不存在，请先运行 ./setup_env.sh"
  exit 1
fi

source "$VENV_DIR/bin/activate"

ARGS=(
  --mode "$CTP_MODE"
  --port "$CTP_PORT"
  --heartbeat "$HEARTBEAT"
  --log-file "$LOG_DIR/ctp_client.log"
)

if [[ "$CTP_MODE" == "client" ]]; then
  ARGS+=(--host "$DEVICE_IP")
else
  ARGS+=(--listen "$LISTEN_IP")
fi

echo "[INFO] 启动 CTP 客户端/服务端..."
exec python jieli_min_ctp_client.py "${ARGS[@]}"
