# AC79 → RK3588 空间占用检测系统（UDP + CTP + RKNN + ROI UI）

## 1. 项目简介

本项目基于 **杰理 AC79 摄像头开发板 + RK3588**，构建一套面向低清晰度视频流的空间占用检测与事件触发系统。系统保留现有 AC79 端 UDP JPEG 视频流与 RK3588 端 RKNN 推理链路，在 RK 端新增 PySide6 图形界面，用于实时显示视频、框选 ROI 区域、保存 ROI 配置、判断目标是否进入区域、统计驻留时间，并在超时后触发报警联动。

当前最终版重点是：

- AC79 通过 UDP 发送 JPEG 视频流；
- RK3588 监听 UDP 端口并解码图像；
- RK3588 使用 RKNN 模型进行人体检测；
- PySide6 UI 实时显示视频、检测框和 ROI；
- 鼠标框选矩形 ROI，并保存为 JSON；
- 使用检测框 **底边中点** 判断目标是否进入 ROI；
- 对每个 ROI 分别统计连续驻留时间；
- 超过阈值后进行蜂鸣、截图保存、事件日志记录和可选外部报警命令执行。

---

## 2. 系统架构

```text
AC79 摄像头开发板
    │
    │  CTP 控制：open / close / date / app
    │
    ├─────────────── TCP 3333 ────────────────┐
    │                                         │
    │  UDP JPEG 视频流                         │
    │                                         │
    └─────────────── UDP 2224 ────────────────▶ RK3588
                                                  │
                                                  ├─ UDP 分片重组
                                                  ├─ JPEG 解码
                                                  ├─ RKNN YOLO 推理
                                                  ├─ PySide6 视频显示
                                                  ├─ ROI 框选与 JSON 保存
                                                  ├─ 底边中点进入 ROI 判定
                                                  ├─ 驻留计时
                                                  └─ 报警联动
```

---

## 3. 最终目录结构

建议保持如下结构，新增 UI 文件都放在 `jieli_linux_bundle` 下，这样可以直接复用已有的 `jieli_rknn_udp_infer.py`、`start_ctp.sh`、`.env` 和 `model/` 目录。

```text
ac79-camera-UDP/
├── README.md
├── model/
│   ├── person.rknn
│   └── labels.txt
└── jieli_linux_bundle/
    ├── .env
    ├── .env.example
    ├── .env.roi.example
    ├── jieli_min_ctp_client.py
    ├── jieli_min_udp_client.py
    ├── jieli_rknn_udp_infer.py
    ├── run_roi_ui.py
    ├── start_ctp.sh
    ├── start_infer_all.sh
    ├── start_roi_ui_all.sh
    ├── stop_roi_ui.sh
    ├── requirements.txt
    ├── requirements_ui.txt
    ├── model/
    │   ├── person.rknn
    │   └── labels.txt
    ├── roi_ui/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── dwell.py
    │   ├── main_window.py
    │   ├── roi_model.py
    │   ├── video_widget.py
    │   └── worker.py
    └── roi_ui_output/
        ├── rois.json
        ├── events.jsonl
        └── screenshots/
```

---

## 4. 快速运行

### 4.1 进入工程目录

```bash
cd ac79-camera-UDP/jieli_linux_bundle
```

### 4.2 准备配置文件

```bash
cp .env.roi.example .env
nano .env
```

重点检查：

```bash
DEVICE_IP=192.168.1.1
UDP_PORT=2224
MODEL_PATH=./model/person.rknn
LABELS_PATH=./model/labels.txt
ROI_JSON=./roi_ui_output/rois.json
SCREENSHOT_DIR=./roi_ui_output/screenshots
ALARM_CMD=
```

如果你当前 AC79 不是 `192.168.1.1`，需要修改 `DEVICE_IP`。如果不想过滤设备来源，可以写成：

```bash
DEVICE_IP=
```

### 4.3 安装 UI 依赖

建议使用你现在已经能跑 RKNN 推理的 Python 环境，不要重新破坏原环境。

```bash
python3 -m pip install -r requirements_ui.txt
```

注意：`rknnlite2` 或 `rknnlite` 请按 RK3588 当前系统环境单独安装。本项目的 `requirements_ui.txt` 不强行安装 RKNN 运行时，避免破坏你已经打通的推理环境。

### 4.4 启动 ROI UI

推荐：

```bash
chmod +x start_roi_ui_all.sh stop_roi_ui.sh run_roi_ui.py
./start_roi_ui_all.sh
```

这个脚本会先尝试调用已有的 `start_ctp.sh` 打开 AC79 视频流，然后启动 PySide6 ROI UI。

如果你已经手动打开了 AC79 视频流，不希望脚本启动 CTP，可以在 `.env` 中设置：

```bash
START_CTP=0
```

然后执行：

```bash
./start_roi_ui_all.sh
```

也可以直接启动 UI：

```bash
python3 run_roi_ui.py
```

---

## 5. UI 使用方法

### 5.1 启动视频显示

打开 UI 后，点击右侧 **启动** 按钮。正常情况下，界面中央会显示 AC79 发来的实时视频流，并叠加 RKNN 检测框。

如果中间一直显示“等待 UDP 视频流”，请检查：

- RK3588 是否已经连接 AC79 热点或处于同一局域网；
- AC79 是否已经开始发送 UDP JPEG；
- `UDP_PORT` 是否为 `2224`；
- `DEVICE_IP` 是否设置正确；
- 是否已经有另一个进程占用了 `2224` 端口。

### 5.2 框选 ROI

操作步骤：

1. 点击 **进入 ROI 编辑**；
2. 在视频画面中按住鼠标左键拖拽；
3. 松开鼠标后生成矩形 ROI；
4. 在右侧 ROI 列表中选择 ROI；
5. 修改名称和驻留阈值；
6. 点击 **应用到选中 ROI**；
7. 点击 **保存 ROI**。

保存后的 ROI 默认路径：

```text
jieli_linux_bundle/roi_ui_output/rois.json
```

ROI JSON 示例：

```json
{
  "frame_size": {
    "width": 640,
    "height": 480
  },
  "rois": [
    {
      "roi_id": 1,
      "name": "desk_area",
      "x1": 120,
      "y1": 80,
      "x2": 420,
      "y2": 300,
      "dwell_sec": 10.0,
      "enabled": true,
      "color": [0, 255, 255],
      "alarm_enabled": true
    }
  ]
}
```

---

## 6. ROI 命中判定逻辑

本版本采用“检测框底边中点”进行 ROI 命中判断。

对于一个检测框：

```text
bbox = (x1, y1, x2, y2)
```

计算底边中点：

```text
bottom_x = (x1 + x2) / 2
bottom_y = y2
```

如果 `(bottom_x, bottom_y)` 落在某个 ROI 矩形内部，则认为该目标进入该 ROI。

这样比直接用检测框中心点更适合人体检测，因为底边中点更接近人的站立位置，适合后续做工位、门口、区域占用等逻辑。

---

## 7. 驻留计时与报警联动

每个 ROI 都有自己的 `dwell_sec` 阈值。只要检测目标的底边中点持续位于 ROI 内，就开始计时。

当连续驻留时间超过阈值时，系统会触发报警：

1. UI 蜂鸣；
2. 保存报警截图；
3. 写入事件日志；
4. 如果配置了 `ALARM_CMD`，执行外部报警命令。

报警截图默认目录：

```text
jieli_linux_bundle/roi_ui_output/screenshots/
```

事件日志默认路径：

```text
jieli_linux_bundle/roi_ui_output/events.jsonl
```

事件日志示例：

```json
{"time":"2026-04-24T10:30:12","event":"roi_dwell_alarm","roi_id":1,"roi_name":"desk_area","dwell_time":10.235,"threshold":10.0,"det_count":1}
```

---

## 8. 音频报警联动

如果后续你已经在 AC79 端实现了 CTP 播放本地音频，可以把命令写入 `.env` 的 `ALARM_CMD`。

示例：

```bash
ALARM_CMD=python3 jieli_min_ctp_client.py --host 192.168.1.1 --cmd alarm
```

如果你的 CTP 播放脚本是单独文件，也可以写成：

```bash
ALARM_CMD=python3 send_audio_alarm.py --host 192.168.1.1 --file alarm.mp3
```

`ALARM_CMD` 每次 ROI 超时报警时执行一次。为了防止同一次驻留重复报警，本版本在目标离开 ROI 前只报警一次；目标离开后再次进入，会重新开始计时。

---

## 9. 环境变量说明

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `DEVICE_IP` | `192.168.1.1` | AC79 设备 IP；为空表示不过滤来源 IP |
| `BIND_IP` | `0.0.0.0` | RK3588 UDP 监听地址 |
| `UDP_PORT` | `2224` | AC79 JPEG 视频流 UDP 端口 |
| `MODEL_PATH` | `./model/person.rknn` | RKNN 模型路径 |
| `LABELS_PATH` | `./model/labels.txt` | 标签文件路径 |
| `INPUT_WIDTH` | `640` | 模型输入宽度 |
| `INPUT_HEIGHT` | `640` | 模型输入高度 |
| `OBJ_THRESH` | `0.25` | 目标置信度阈值 |
| `NMS_THRESH` | `0.45` | NMS 阈值 |
| `MAX_DET` | `10` | 单帧最大检测数量 |
| `BGR_INPUT` | `1` | 与原推理脚本保持一致，1 表示按 BGR 输入配置 |
| `SINGLE_CORE` | `1` | 是否使用单 NPU 核心 |
| `ROI_JSON` | `./roi_ui_output/rois.json` | ROI 配置保存路径 |
| `SCREENSHOT_DIR` | `./roi_ui_output/screenshots` | 报警截图保存目录 |
| `ROI_EVENT_LOG` | `./roi_ui_output/events.jsonl` | ROI 报警事件日志 |
| `ALARM_CMD` | 空 | 报警时执行的外部命令 |
| `START_CTP` | `1` | 启动 UI 前是否调用 `start_ctp.sh` |

---

## 10. 常见问题

### 10.1 UI 打开了，但是没有画面

排查顺序：

```bash
cd jieli_linux_bundle
python3 jieli_min_udp_client.py --port 2224 --device-ip 192.168.1.1
```

如果这个脚本也没有画面，说明问题在 AC79 发流或网络链路，不在 UI。

### 10.2 提示 2224 端口被占用

不要同时运行：

```bash
python3 jieli_rknn_udp_infer.py
python3 run_roi_ui.py
```

因为二者都会监听 UDP 2224。ROI UI 已经内置 UDP 接收和 RKNN 推理，不需要再额外启动原推理脚本。

查看占用：

```bash
sudo lsof -i:2224
```

杀掉旧进程：

```bash
sudo kill -9 <PID>
```

### 10.3 RKNN 模型加载失败

检查：

```bash
ls -lh ./model/person.rknn
ls -lh ./model/labels.txt
```

同时确认当前 Python 环境中能导入 RKNNLite：

```bash
python3 -c "from rknnlite.api import RKNNLite; print('rknn ok')"
```

### 10.4 PySide6 安装失败

RK3588 上如果 pip 安装慢，可以先换源：

```bash
python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple PySide6 numpy opencv-python
```

如果系统已有 OpenCV，且 `opencv-python` 安装失败，可以先只装：

```bash
python3 -m pip install PySide6 numpy
```

### 10.5 ROI 框选位置和实际判断不一致

本版本内部保存的是原始视频帧坐标，不是 UI 缩放后的显示坐标。只要视频帧分辨率稳定，ROI 判断就会稳定。如果 AC79 输出分辨率发生变化，建议重新框选 ROI 并保存。

---

## 11. 开发说明

### 11.1 `roi_ui/worker.py`

负责：

- 加载原有 `jieli_rknn_udp_infer.py`；
- 复用其中的 `YoloRknnDetector` 和 `FrameState`；
- 监听 UDP 端口；
- 重组 JPEG 分片；
- 解码图像；
- 调用 RKNN 推理；
- 把原图和检测结果发送给 UI。

### 11.2 `roi_ui/video_widget.py`

负责：

- 显示视频画面；
- 绘制检测框；
- 绘制检测框底边中点；
- 绘制 ROI；
- 处理鼠标拖拽框选；
- 做 UI 显示坐标与原始帧坐标之间的换算。

### 11.3 `roi_ui/dwell.py`

负责：

- 计算检测框底边中点；
- 判断底边中点是否进入 ROI；
- 统计 ROI 驻留时间；
- 管理报警状态，防止同一次驻留重复报警。

### 11.4 `roi_ui/main_window.py`

负责：

- 主窗口布局；
- ROI 列表；
- ROI 名称和阈值编辑；
- ROI 保存和加载；
- 报警截图；
- 事件日志；
- 外部报警命令执行。

---

## 12. 推荐测试顺序

第一步，确认原始 UDP 视频流：

```bash
python3 jieli_min_udp_client.py --port 2224
```

第二步，确认原始 RKNN 推理脚本：

```bash
python3 jieli_rknn_udp_infer.py --model ./model/person.rknn --labels ./model/labels.txt
```

第三步，关闭上面的推理脚本，启动 ROI UI：

```bash
./start_roi_ui_all.sh
```

第四步，框选 ROI 并保存：

```text
进入 ROI 编辑 → 鼠标拖框 → 修改名称和阈值 → 应用 → 保存 ROI
```

第五步，让目标进入 ROI，观察：

```text
ROI 颜色变化 → 驻留时间增加 → 超阈值报警 → 截图保存 → events.jsonl 写入
```

---

## 13. 后续升级方向

本版本先实现矩形 ROI 和单目标/多目标混合的区域占用判断。后续可以继续升级：

1. 接入 ByteTrack，为不同人分配稳定 ID；
2. 多边形 ROI；
3. ROI 拖拽缩放编辑；
4. 按时间段切换规则；
5. 白天统计空间使用率，夜间触发异常驻留；
6. 把 `ALARM_CMD` 正式接到 AC79 本地音频播放；
7. 增加事件回放页面；
8. 增加 Web UI 或远程配置接口。

---

## 14. 当前版本定位

这是面向你当前工程状态的 **ROI UI 最终集成版**。它不是另起炉灶，而是直接贴合当前仓库目录，保留已有：

- CTP 控制链路；
- UDP JPEG 收流链路；
- RKNN 推理脚本；
- `model/person.rknn` 与 `labels.txt` 配置；
- 原有 Shell 启动方式。

新增部分集中在：

```text
jieli_linux_bundle/run_roi_ui.py
jieli_linux_bundle/start_roi_ui_all.sh
jieli_linux_bundle/stop_roi_ui.sh
jieli_linux_bundle/requirements_ui.txt
jieli_linux_bundle/.env.roi.example
jieli_linux_bundle/roi_ui/
jieli_linux_bundle/roi_ui_output/
```
