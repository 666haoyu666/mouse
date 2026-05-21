#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AC79 UDP + Hamster Scene UI Stage2: detection + grid + fixed ROI + text + tracking + stats"
    )
    parser.add_argument("--helper-script", default=None, help="jieli_rknn_udp_infer.py 的路径；默认使用当前目录下的同名文件")
    parser.add_argument("--model", default=None, help="覆盖 MODEL_PATH")
    parser.add_argument("--labels", default=None, help="覆盖 LABELS_PATH")
    parser.add_argument("--detector-backend", choices=["rknn", "tensorrt"], default=None, help="覆盖 DETECTOR_BACKEND")
    parser.add_argument("--class-filter", default=None, help="覆盖 CLASS_FILTER，例如 0 或 0,1；empty/all 表示不过滤")
    parser.add_argument("--input-size", nargs=2, type=int, default=None, metavar=("W", "H"), help="覆盖 INPUT_WIDTH/INPUT_HEIGHT")
    parser.add_argument("--device-ip", default=None, help="覆盖 DEVICE_IP；传 empty 可关闭过滤")
    parser.add_argument("--port", type=int, default=None, help="覆盖 UDP_PORT")
    parser.add_argument("--roi-json", default=None, help="覆盖 ROI_JSON")
    parser.add_argument("--env-file", default=None, help="指定 .env 文件，默认读取当前目录 .env")
    parser.add_argument("--heatmap", action="store_true", help="启动时直接显示活动热区叠加")
    parser.add_argument("--no-tracker", action="store_true", help="关闭第二阶段跟踪逻辑，只保留第一阶段显示")
    parser.add_argument("--no-frame-preprocess", action="store_true", help="关闭 YOLO 前 LAB/CLAHE/锐化帧处理")
    parser.add_argument("--frame-preprocess-display", action="store_true", help="界面显示处理后的帧；默认只把处理后的帧送入 YOLO")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    bundle_dir = Path(__file__).resolve().parent
    os.chdir(bundle_dir)

    if args.env_file:
        os.environ["ENV_FILE"] = args.env_file
    if args.model:
        os.environ["MODEL_PATH"] = args.model
    if args.labels:
        os.environ["LABELS_PATH"] = args.labels
    if args.detector_backend:
        os.environ["DETECTOR_BACKEND"] = args.detector_backend
    if args.class_filter is not None:
        os.environ["CLASS_FILTER"] = "" if args.class_filter.lower() in {"empty", "all"} else args.class_filter
    if args.input_size:
        os.environ["INPUT_WIDTH"] = str(args.input_size[0])
        os.environ["INPUT_HEIGHT"] = str(args.input_size[1])
    if args.device_ip is not None:
        os.environ["DEVICE_IP"] = "" if args.device_ip.lower() == "empty" else args.device_ip
    if args.port is not None:
        os.environ["UDP_PORT"] = str(args.port)
    if args.roi_json:
        os.environ["ROI_JSON"] = args.roi_json
    if args.heatmap:
        os.environ["HEATMAP_OVERLAY_ENABLED"] = "1"
    if args.no_tracker:
        os.environ["TRACKER_ENABLED"] = "0"
    if args.no_frame_preprocess:
        os.environ["FRAME_PREPROCESS_ENABLED"] = "0"
    if args.frame_preprocess_display:
        os.environ["FRAME_PREPROCESS_DISPLAY"] = "1"

    from roi_ui.config import AppConfig
    from roi_ui.main_window import MainWindow

    app = QApplication(sys.argv)
    cfg = AppConfig()
    helper_script = args.helper_script or str(bundle_dir / "jieli_rknn_udp_infer.py")
    window = MainWindow(cfg=cfg, helper_script=helper_script)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
