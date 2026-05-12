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
UDP_PORT="${UDP_PORT:-2224}"
SAVE_DIR="${SAVE_DIR:-udp_frames}"
SAVE_EVERY="${SAVE_EVERY:-60}"
SHOW_WINDOW="${SHOW_WINDOW:-1}"
VERBOSE_UDP="${VERBOSE_UDP:-0}"
NO_FILTER="${NO_FILTER:-0}"
CLEANUP_TIMEOUT="${CLEANUP_TIMEOUT:-3.0}"

mkdir -p "$LOG_DIR" "$SAVE_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[ERR] 虚拟环境不存在，请先运行 ./setup_env.sh"
  exit 1
fi

source "$VENV_DIR/bin/activate"

ARGS=(
  --port "$UDP_PORT"
  --save-dir "$SAVE_DIR"
  --save-every "$SAVE_EVERY"
  --cleanup-timeout "$CLEANUP_TIMEOUT"
)

if [[ "$NO_FILTER" == "1" ]]; then
  ARGS+=(--no-filter)
else
  ARGS+=(--device-ip "$DEVICE_IP")
fi

if [[ "$SHOW_WINDOW" != "1" ]] || [[ -z "${DISPLAY:-}" ]]; then
  ARGS+=(--no-window)
fi

if [[ "$VERBOSE_UDP" == "1" ]]; then
  ARGS+=(--verbose)
fi

echo "[INFO] 启动 UDP 接收器..."
exec python jieli_min_udp_client.py "${ARGS[@]}"
