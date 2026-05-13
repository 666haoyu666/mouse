#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv_nano}"
PYTHON_BIN="$VENV_DIR/bin/python"
RUNTIME_DEPS_DIR="${RUNTIME_DEPS_DIR:-$SCRIPT_DIR/.python_deps/runtime}"
FALLBACK_MARKER="$RUNTIME_DEPS_DIR/.use_system_python"

ensure_target_package() {
    local module_name="$1"
    local package_spec="$2"
    if ! PYTHONPATH="$RUNTIME_DEPS_DIR:${PYTHONPATH:-}" python3 - <<PY
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("$module_name") else 1)
PY
    then
        mkdir -p "$RUNTIME_DEPS_DIR"
        python3 -m pip install --target "$RUNTIME_DEPS_DIR" "$package_spec"
    fi
}

if [[ -f "$FALLBACK_MARKER" ]]; then
    PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
    export PYTHONPATH="$RUNTIME_DEPS_DIR:${PYTHONPATH:-}"
elif [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[INFO] Creating project venv: $VENV_DIR"
    if python3 -m venv --system-site-packages "$VENV_DIR"; then
        "$PYTHON_BIN" -m pip install --upgrade pip
        "$PYTHON_BIN" -m pip install -r "$SCRIPT_DIR/requirements_ui.txt"
    else
        echo "[WARN] python3 venv is unavailable; using project-local dependency target instead."
        rm -rf "$VENV_DIR"
        PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
        ensure_target_package "PySide6" "PySide6>=6.5"
        ensure_target_package "cuda" "cuda-python==12.6.2.post1"
        touch "$FALLBACK_MARKER"
        export PYTHONPATH="$RUNTIME_DEPS_DIR:${PYTHONPATH:-}"
    fi
fi

export ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.nano}"
export DETECTOR_BACKEND="${DETECTOR_BACKEND:-tensorrt}"
export BGR_INPUT="${BGR_INPUT:-0}"
export CLASS_FILTER="${CLASS_FILTER:-0}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/run_roi_ui.py" "$@"
