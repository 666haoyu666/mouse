#!/usr/bin/env bash
set -euo pipefail

# AC79 ROI UI 最终版启动脚本
# 作用：可选启动 CTP 开流，然后以前台方式启动 PySide6 ROI UI。
# 注意：UI 自己接收 UDP + 调 RKNN，不要再同时启动 jieli_rknn_udp_infer.py，否则会抢占 2224 端口。

cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
START_CTP="${START_CTP:-1}"

if [ "$START_CTP" = "1" ] && [ -f ./start_ctp.sh ]; then
  echo "[INFO] 启动 CTP 控制链路，打开 AC79 视频流..."
  bash ./start_ctp.sh || echo "[WARN] start_ctp.sh 返回非 0，请确认 AC79 网络和 CTP 配置"
else
  echo "[INFO] 跳过 CTP 启动，假设 AC79 已经在发送 UDP JPEG"
fi

echo "[INFO] 启动 ROI UI..."
exec "$PYTHON_BIN" ./run_roi_ui.py
