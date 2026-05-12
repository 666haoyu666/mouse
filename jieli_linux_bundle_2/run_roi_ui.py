#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from roi_ui.config import AppConfig
from roi_ui.main_window import MainWindow


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AC79 UDP + RKNN ROI UI")
    p.add_argument("--helper-script", default=None, help="jieli_rknn_udp_infer.py 的路径；默认使用当前目录下的同名文件")
    p.add_argument("--model", default=None, help="覆盖 MODEL_PATH")
    p.add_argument("--labels", default=None, help="覆盖 LABELS_PATH")
    p.add_argument("--device-ip", default=None, help="覆盖 DEVICE_IP；传 empty 可关闭过滤")
    p.add_argument("--port", type=int, default=None, help="覆盖 UDP_PORT")
    p.add_argument("--roi-json", default=None, help="覆盖 ROI_JSON")
    p.add_argument("--env-file", default=None, help="指定 .env 文件，默认读取当前目录 .env")
    return p


def main() -> int:
    args = build_parser().parse_args()

    # 保证从 jieli_linux_bundle 目录运行时，相对路径与原项目一致。
    bundle_dir = Path(__file__).resolve().parent
    os.chdir(bundle_dir)

    if args.env_file:
        os.environ["ENV_FILE"] = args.env_file
    if args.model:
        os.environ["MODEL_PATH"] = args.model
    if args.labels:
        os.environ["LABELS_PATH"] = args.labels
    if args.device_ip is not None:
        os.environ["DEVICE_IP"] = "" if args.device_ip.lower() == "empty" else args.device_ip
    if args.port is not None:
        os.environ["UDP_PORT"] = str(args.port)
    if args.roi_json:
        os.environ["ROI_JSON"] = args.roi_json

    helper_script = args.helper_script or str(bundle_dir / "jieli_rknn_udp_infer.py")
    app = QApplication(sys.argv)
    cfg = AppConfig()
    win = MainWindow(cfg, helper_script=helper_script)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
