#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# 如果是 start_roi_ui_all.sh 前台运行，Ctrl+C 或关闭窗口即可退出。
# 本脚本主要用于清理已有 CTP 后台进程。

if [ -f ctp.pid ]; then
  pid="$(cat ctp.pid || true)"
  if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
    echo "[INFO] kill CTP pid=$pid"
    kill "$pid" || true
  fi
  rm -f ctp.pid
fi

echo "[OK] stop_roi_ui 完成"
