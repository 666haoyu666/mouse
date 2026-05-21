#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export an Ultralytics YOLO .pt model to ONNX without writing beside the source .pt.")
    parser.add_argument("--pt", required=True, help="Source .pt model path")
    parser.add_argument("--out", required=True, help="Destination .onnx path")
    parser.add_argument("--imgsz", type=int, default=640, help="Square input size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset")
    parser.add_argument("--vendor-dir", default=None, help="Extra dependency directory added to sys.path")
    parser.add_argument("--simplify", action="store_true", help="Ask Ultralytics to simplify the exported ONNX")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pt_path = Path(args.pt).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if not pt_path.exists():
        raise FileNotFoundError(f"PT model not found: {pt_path}")

    if args.vendor_dir:
        vendor = Path(args.vendor_dir).expanduser().resolve()
        if vendor.exists():
            sys.path.insert(0, str(vendor))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = out_path.parent / ".export_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(work_dir / ".ultralytics"))

    local_pt = work_dir / pt_path.name
    if local_pt.resolve() != pt_path:
        shutil.copy2(pt_path, local_pt)

    from ultralytics import YOLO

    model = YOLO(str(local_pt))
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=bool(args.simplify),
        dynamic=False,
    )
    exported_path = Path(exported).expanduser().resolve()
    if not exported_path.exists():
        raise FileNotFoundError(f"Ultralytics did not produce ONNX: {exported_path}")

    if exported_path != out_path:
        if out_path.exists():
            out_path.unlink()
        shutil.move(str(exported_path), str(out_path))

    if local_pt.exists() and local_pt.resolve() != pt_path:
        local_pt.unlink()

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
