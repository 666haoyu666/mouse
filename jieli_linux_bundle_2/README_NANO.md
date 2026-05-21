# Jetson Nano / Orin Nano TensorRT notes

This adapted launcher keeps the AC79 UDP, CTP, ROI and alarm logic, and swaps the detector backend to TensorRT.

Prepare the default model:

```bash
cd /home/ubuntu/Desktop/AC791-RK3588_withUI_nano/jieli_linux_bundle_2
./scripts/prepare_yolo_trt.sh \
  --pt /home/ubuntu/Desktop/yolo/yolo11n.pt \
  --engine ./model/yolo11n.engine \
  --imgsz 640
```

Run the UI:

```bash
./start_roi_ui_nano.sh --model ./model/yolo11n.engine --class-filter 0
```

Use another model by preparing another engine and passing its path with `--model`.
