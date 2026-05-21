# Hamster Scene UI Patch

这是一个基于 `666haoyu666/mouse` 仓库改造的 **第一阶段 UI 版本**：

- 检测
- 九宫格
- 固定 ROI
- 中文描述

## 适用目录

把本补丁中的文件覆盖到原仓库的：

```text
jieli_linux_bundle_2/
├── run_roi_ui.py
├── .env.hamster.example
└── roi_ui/
    ├── __init__.py
    ├── config.py
    ├── main_window.py
    ├── region_analyzer.py
    ├── roi_model.py
    ├── text_generator.py
    ├── video_widget.py
    └── worker.py
```

## 运行方式

先在原仓库 `jieli_linux_bundle_2` 目录下准备：

- `jieli_rknn_udp_infer.py`
- 原仓库的检测后端实现（例如 `tensorrt_detector.py`）
- 模型文件，例如 `./model/hamster.engine`
- 标签文件，例如 `./model/hamster_labels.txt`

然后：

```bash
cp .env.hamster.example .env
python3 run_roi_ui.py --env-file .env
```

## 这个版本和原仓库的区别

原仓库更偏向：

- 人员占用检测
- 底边中点进入 ROI
- 驻留告警/会议室模式

这个补丁改成了：

- 仓鼠检测框中心点
- 九宫格位置判断
- 固定场景 ROI（木屋/跑轮/食盆/饮水器）
- 中文描述生成

## 默认 ROI

首次接收到视频帧时，如果没有已有 ROI 配置，会自动生成一组固定 ROI：

- 木屋
- 跑轮
- 食盆
- 饮水器

你也可以进入 ROI 编辑模式后，直接拖拽调整位置。

## 输出

- ROI 配置：`roi_ui_output/hamster_rois.json`
- 事件日志：`roi_ui_output/hamster_events.jsonl`
