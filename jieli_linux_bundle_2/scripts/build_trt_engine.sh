#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRTEXEC="${TRTEXEC:-}"
ONNX=""
ENGINE=""
FP16=1
WORKSPACE_MB="${WORKSPACE_MB:-2048}"

usage() {
    cat <<'EOF'
Usage:
  build_trt_engine.sh --onnx model.onnx --engine model.engine [--fp16|--no-fp16]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --onnx)
            ONNX="$2"
            shift 2
            ;;
        --engine)
            ENGINE="$2"
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

if [[ -z "$ONNX" || -z "$ENGINE" ]]; then
    usage >&2
    exit 2
fi

if [[ -z "$TRTEXEC" ]]; then
    if command -v trtexec >/dev/null 2>&1; then
        TRTEXEC="$(command -v trtexec)"
    elif [[ -x /usr/src/tensorrt/bin/trtexec ]]; then
        TRTEXEC="/usr/src/tensorrt/bin/trtexec"
    else
        echo "[ERR] trtexec not found. Set TRTEXEC=/path/to/trtexec." >&2
        exit 1
    fi
fi

mkdir -p "$(dirname "$ENGINE")"
ARGS=(
    "--onnx=$ONNX"
    "--saveEngine=$ENGINE"
    "--memPoolSize=workspace:${WORKSPACE_MB}"
)
if [[ "$FP16" == "1" ]]; then
    ARGS+=("--fp16")
fi

echo "[INFO] Building TensorRT engine:"
echo "       $TRTEXEC ${ARGS[*]}"
"$TRTEXEC" "${ARGS[@]}"
echo "[OK] Engine written: $ENGINE"
