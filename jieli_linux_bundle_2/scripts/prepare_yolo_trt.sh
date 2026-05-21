#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/Desktop/yolo/.yolo/bin/python}"
PIP_PYTHON="${PIP_PYTHON:-python3}"
DEPS_DIR="${DEPS_DIR:-$BUNDLE_DIR/.python_deps/export}"
PT=""
ONNX=""
ENGINE=""
IMGSZ=640
OPSET=17
FP16=1
SIMPLIFY=0

usage() {
    cat <<'EOF'
Usage:
  prepare_yolo_trt.sh --pt model.pt --engine model.engine [--onnx model.onnx] [--imgsz 640]

Notes:
  The source .pt is copied into this project before export, so the source model directory is not modified.
  Set PYTHON_BIN to choose the Python environment that has ultralytics and torch.
  Set PIP_PYTHON if that environment has no pip; dependencies are installed into --deps-dir only.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pt)
            PT="$2"
            shift 2
            ;;
        --onnx)
            ONNX="$2"
            shift 2
            ;;
        --engine)
            ENGINE="$2"
            shift 2
            ;;
        --imgsz)
            IMGSZ="$2"
            shift 2
            ;;
        --opset)
            OPSET="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --pip-python)
            PIP_PYTHON="$2"
            shift 2
            ;;
        --deps-dir)
            DEPS_DIR="$2"
            shift 2
            ;;
        --fp16)
            FP16=1
            shift
            ;;
        --no-fp16)
            FP16=0
            shift
            ;;
        --simplify)
            SIMPLIFY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERR] Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$PT" || -z "$ENGINE" ]]; then
    usage >&2
    exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERR] Python not executable: $PYTHON_BIN" >&2
    exit 1
fi

mkdir -p "$(dirname "$ENGINE")" "$DEPS_DIR"
mkdir -p "$BUNDLE_DIR/.ultralytics"
if [[ -z "$ONNX" ]]; then
    ENGINE_BASE="$(basename "$ENGINE")"
    ONNX="$(dirname "$ENGINE")/${ENGINE_BASE%.*}.onnx"
fi

echo "[INFO] Export Python: $PYTHON_BIN"
echo "[INFO] Pip Python: $PIP_PYTHON"
echo "[INFO] Dependency target: $DEPS_DIR"

if ! PYTHONPATH="$DEPS_DIR:${PYTHONPATH:-}" "$PYTHON_BIN" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("onnx") else 1)
PY
then
    echo "[INFO] Installing ONNX export dependency into project-local target..."
    "$PIP_PYTHON" -m pip install --target "$DEPS_DIR" "onnx>=1.12"
fi

EXPORT_ARGS=(
    "--pt" "$PT"
    "--out" "$ONNX"
    "--imgsz" "$IMGSZ"
    "--opset" "$OPSET"
    "--vendor-dir" "$DEPS_DIR"
)
if [[ "$SIMPLIFY" == "1" ]]; then
    if ! PYTHONPATH="$DEPS_DIR:${PYTHONPATH:-}" "$PYTHON_BIN" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("onnxslim") else 1)
PY
    then
        echo "[INFO] Installing ONNX simplifier dependency into project-local target..."
        "$PIP_PYTHON" -m pip install --target "$DEPS_DIR" "onnxslim>=0.1.59"
    fi
    EXPORT_ARGS+=("--simplify")
fi

PYTHONPATH="$DEPS_DIR:${PYTHONPATH:-}" \
YOLO_CONFIG_DIR="$BUNDLE_DIR/.ultralytics" \
"$PYTHON_BIN" "$SCRIPT_DIR/export_yolo_onnx.py" "${EXPORT_ARGS[@]}"

BUILD_ARGS=("--onnx" "$ONNX" "--engine" "$ENGINE")
if [[ "$FP16" == "1" ]]; then
    BUILD_ARGS+=("--fp16")
else
    BUILD_ARGS+=("--no-fp16")
fi
"$SCRIPT_DIR/build_trt_engine.sh" "${BUILD_ARGS[@]}"
