# AC79 → Jetson Nano 空间占用检测系统

> UDP JPEG 视频流 + CTP 控制 + TensorRT YOLO 推理 + PySide6 ROI 图形界面  
> 本 README 是面向 Jetson Nano / Orin Nano 的中文版说明，用于替代旧版 RK3588 / RKNN 工作流描述。

---

## 1. 项目简介

本项目用于将 **杰理 AC79 摄像头开发板** 与 **Jetson Nano / Orin Nano** 连接起来，构建一个轻量化的空间占用检测与事件触发系统。

AC79 负责摄像头采集和 UDP JPEG 视频流发送；Jetson Nano 负责接收 UDP 数据、重组 JPEG 帧、解码图像、运行 TensorRT YOLO 推理、显示检测结果、绘制 ROI 区域，并根据区域驻留规则触发报警或音频播放。

当前 Nano 版本主要实现：

- 接收 AC79 发送的 UDP JPEG 视频流。
- 通过 CTP TCP 控制链路控制 AC79。
- 使用 TensorRT `.engine` 模型进行 YOLO 目标检测。
- 使用 PySide6 显示实时画面、检测框和 ROI 区域。
- 支持鼠标拖拽绘制矩形 ROI。
- 支持 ROI 分组保存和加载。
- 使用检测框“底边中点”判断目标是否进入 ROI。
- 支持 ROI 内连续驻留时间统计。
- 支持普通 ROI 报警模式。
- 支持会议室占用检测模式。
- 支持通过 CTP 向 AC79 发送 `sd:1` ~ `sd:6` 音频播放命令。

---

## 2. 系统架构

```text
AC79 摄像头开发板
    │
    │  CTP 控制链路
    │  TCP 3333
    │  命令：app / date / open / sd:x / quit
    │
    ├────────────────────────────────────────────┐
    │                                            │
    │  UDP JPEG 视频流                           │
    │  UDP 2224                                  │
    │                                            ▼
Jetson Nano / Orin Nano
    │
    ├─ UDP 数据包接收
    ├─ JPEG 分片重组
    ├─ JPEG 解码为 OpenCV BGR 图像
    ├─ TensorRT YOLO 推理
    ├─ PySide6 实时视频显示
    ├─ 检测框与 ROI 绘制
    ├─ 底边中点进入 ROI 判断
    ├─ 驻留时间统计
    ├─ 普通 ROI 报警模式
    ├─ 会议室占用检测模式
    └─ 通过 CTP 发送 sd:x 命令，控制 AC79 本地音频播放
```

---

## 3. 仓库目录结构

当前仓库主要围绕 `jieli_linux_bundle_2` 目录展开：

```text
jetson-nano-jieli-ac79/
├── README.md
├── model/
└── jieli_linux_bundle_2/
    ├── .env
    ├── .env.example
    ├── .env.nano
    ├── .env.roi.example
    ├── README.md
    ├── README_LINUX.md
    ├── README_NANO.md
    ├── jieli_min_ctp_client.py
    ├── jieli_min_udp_client.py
    ├── jieli_rknn_udp_infer.py
    ├── requirements.txt
    ├── requirements_ui.txt
    ├── run_roi_ui.py
    ├── setup_env.sh
    ├── start_all.sh
    ├── start_ctp.sh
    ├── start_infer_all.sh
    ├── start_roi_ui_all.sh
    ├── start_roi_ui_nano.sh
    ├── start_udp.sh
    ├── stop_all.sh
    ├── stop_roi_ui.sh
    ├── scripts/
    │   └── prepare_yolo_trt.sh
    ├── model/
    │   ├── yolo11n.engine
    │   └── coco_labels.txt
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

> 注意：部分文件名仍保留历史命名，例如 `jieli_rknn_udp_infer.py`。在 Nano 工作流中，关键是使用 `DETECTOR_BACKEND=tensorrt` 和 TensorRT `.engine` 模型。

---

## 4. 硬件与网络准备

### 4.1 硬件准备

推荐硬件：

- 杰理 AC79 摄像头开发板。
- Jetson Nano 或 Orin Nano。
- 如果使用本地图形界面，需要给 Jetson 连接显示器。
- 如果需要播放 `sd:1` ~ `sd:6` 音频，需要 AC79 端 SD 卡中存在对应音频文件。

### 4.2 网络准备

需要确保 Jetson 能够和 AC79 正常通信。

常见连接方式：

```text
AC79 作为 WiFi AP
Jetson 连接 AC79 WiFi
AC79 默认 IP：192.168.1.1
CTP TCP 端口：3333
UDP 视频端口：2224
```

基础检查命令：

```bash
ping 192.168.1.1
sudo ss -lunp | grep ':2224'
```

如果 UI 能启动但没有画面，可以临时关闭来源 IP 过滤：

```bash
DEVICE_IP=
```

---

## 5. 准备 TensorRT YOLO 模型

Nano 版本使用 TensorRT engine，而不是 RKNN 模型。

### 5.1 进入项目目录

```bash
cd /home/ubuntu/Desktop/AC791-RK3588_withUI_nano/jieli_linux_bundle_2
```

如果你的仓库路径不同，请替换成自己的实际路径。

### 5.2 转换或准备 TensorRT engine

使用仓库提供的辅助脚本：

```bash
./scripts/prepare_yolo_trt.sh \
  --pt /home/ubuntu/Desktop/yolo/yolo11n.pt \
  --engine ./model/yolo11n.engine \
  --imgsz 640
```

转换完成后检查 engine 是否存在：

```bash
ls -lh ./model/yolo11n.engine
```

如果使用自己的 YOLO 模型，可以生成新的 `.engine` 文件，然后在启动 UI 时通过 `--model` 参数指定模型路径。

---

## 6. Nano 配置文件说明

Nano 专用配置文件是：

```bash
jieli_linux_bundle_2/.env.nano
```

推荐配置如下：

```env
DETECTOR_BACKEND=tensorrt
MODEL_PATH=./model/yolo11n.engine
LABELS_PATH=./model/coco_labels.txt

INPUT_WIDTH=640
INPUT_HEIGHT=640
BGR_INPUT=0
CLASS_FILTER=0

OBJ_THRESH=0.25
NMS_THRESH=0.45
MAX_DET=10
AGNOSTIC_NMS=0

BIND_IP=0.0.0.0
UDP_PORT=2224
DEVICE_IP=192.168.1.1
CLEANUP_TIMEOUT=3.0

SCREENSHOT_DIR=./roi_ui_output/screenshots
ROI_JSON=./roi_ui_output/rois.json
ROI_EVENT_LOG=./roi_ui_output/events.jsonl
```

关键变量说明：

| 变量 | 作用 |
|---|---|
| `DETECTOR_BACKEND=tensorrt` | 在 Jetson Nano / Orin Nano 上使用 TensorRT 后端。 |
| `MODEL_PATH` | TensorRT `.engine` 模型路径。 |
| `LABELS_PATH` | 类别标签文件路径。 |
| `CLASS_FILTER=0` | 只检测 COCO 类别 0，通常是 person。 |
| `BGR_INPUT=0` | TensorRT YOLO 通常使用 RGB 风格预处理。 |
| `UDP_PORT=2224` | AC79 UDP JPEG 视频流端口。 |
| `DEVICE_IP=192.168.1.1` | AC79 设备 IP。为空时不做来源 IP 过滤。 |
| `ROI_JSON` | ROI 配置保存路径。 |
| `ROI_EVENT_LOG` | 事件日志保存路径。 |

---

## 7. 安装依赖

### 7.1 推荐方式

优先使用 Nano 启动脚本，它会尝试创建项目虚拟环境并安装 UI 依赖：

```bash
cd jieli_linux_bundle_2
chmod +x start_roi_ui_nano.sh
./start_roi_ui_nano.sh --help
```

脚本会使用：

```bash
.venv_nano/
requirements_ui.txt
.python_deps/runtime/
```

如果系统中没有 `python3 -m venv`，脚本会回退到项目本地依赖目录。

### 7.2 手动安装

```bash
python3 -m pip install -r requirements_ui.txt
```

如果 PySide6 安装较慢或失败，可以尝试清华源：

```bash
python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple PySide6 numpy opencv-python
```

TensorRT、CUDA、JetPack 相关 Python 包建议优先使用 Jetson 系统自带版本。不要随意升级系统 CUDA 或 TensorRT，否则可能导致兼容性问题。

---

## 8. 启动系统

### 8.1 推荐 Nano 启动方式

```bash
cd jieli_linux_bundle_2
chmod +x start_roi_ui_nano.sh
./start_roi_ui_nano.sh --model ./model/yolo11n.engine --class-filter 0
```

该启动脚本会设置 Nano 相关默认参数：

```bash
ENV_FILE=.env.nano
DETECTOR_BACKEND=tensorrt
BGR_INPUT=0
CLASS_FILTER=0
```

### 8.2 使用其他 TensorRT engine 启动

```bash
./start_roi_ui_nano.sh \
  --model ./model/best.engine \
  --class-filter 0
```

### 8.3 显示相关设置

如果使用本地显示器运行 UI：

```bash
export DISPLAY=:0
```

如果 UI 界面太大：

```bash
export QT_SCALE_FACTOR=0.75
```

---

## 9. AC79 视频流工作流程

UI 启动后：

1. 点击 UI 中的 **Start**。
2. 程序启动 UDP 接收和推理线程。
3. 程序启动或调用 CTP 控制流程。
4. 向 AC79 发送以下命令：

```text
app
date
open 640 480 20 8000 0
```

5. AC79 通过 UDP 端口 `2224` 发送 JPEG 视频帧。
6. Jetson 解码视频帧并运行 TensorRT YOLO 推理。
7. UI 显示实时画面、检测框、ROI 和当前状态。

推荐使用的 `open` 命令：

```text
open 640 480 20 8000 0
```

最后一个参数建议保持为 `0`，因为当前接收逻辑主要围绕 UDP JPEG 视频流设计。

---

## 10. ROI 判断逻辑

系统使用检测框的 **底边中点** 判断目标是否进入 ROI。

对于检测框：

```python
bbox = (x1, y1, x2, y2)
bottom_x = (x1 + x2) / 2
bottom_y = y2
```

如果 `(bottom_x, bottom_y)` 落在某个 ROI 矩形内部，就认为该目标进入了该 ROI。

相比使用检测框中心点，底边中点更接近人的脚部位置，更适合判断“人是否真正站在某个区域内”。

---

## 11. 普通 ROI 报警模式

普通模式适合一般区域驻留报警。

操作流程：

```text
启动 UI
↓
点击 Start
↓
选择 Default Mode
↓
进入 ROI 编辑
↓
在视频画面上拖拽创建 ROI
↓
选择报警音频 sd:1 ~ sd:6
↓
在 ROI 列表中选择该 ROI
↓
设置 ROI 名称、驻留阈值和报警音频
↓
点击 Apply ROI
↓
保存 ROI Group
```

报警逻辑：

```text
目标进入 ROI
↓
连续驻留时间 >= dwell_sec
↓
保存报警截图
↓
写入事件日志
↓
通过 CTP 向 AC79 发送 sd:x 音频命令
```

每个 ROI 都可以独立配置：

- ROI 名称。
- 驻留时间阈值。
- 报警音频编号。

---

## 12. 会议室模式

会议室模式用于判断会议室是否处于占用、空闲、长时间占用或异常占用状态。

### 12.1 工作时间内

```text
有人进入任意 ROI
↓
连续驻留 >= MEETING_USE_START_SEC
↓
播放 sd:1 “会议室开始使用”
↓
显示“会议室使用中”
```

如果持续占用：

```text
占用时间 >= MEETING_LONG_USE_SEC
↓
播放 sd:3 “长时间占用提醒”
```

如果继续占用：

```text
占用时间 >= MEETING_LONG_USE_SEC + MEETING_ABNORMAL_EXTRA_SEC
↓
播放 sd:6 “异常占用持续”
```

进入异常占用后，`sd:6` 可以按照 `MEETING_ABNORMAL_REPEAT_SEC` 间隔重复播放。

释放逻辑：

```text
会议室开始使用后，ROI 内无人持续 MEETING_RELEASE_EMPTY_SEC
↓
播放 sd:2 “会议室空闲”
↓
等待 MEETING_RELEASE_AUDIO_GAP_SEC
↓
播放 sd:4 “房间已空，请关闭设备”
↓
显示“会议室空闲”
```

### 12.2 非工作时间

```text
有人进入任意 ROI
↓
连续驻留 >= MEETING_USE_START_SEC
↓
播放 sd:5 “非工作时间，请离开”
↓
显示“会议室使用中”
```

如果继续占用：

```text
占用时间 >= MEETING_USE_START_SEC + MEETING_ABNORMAL_EXTRA_SEC
↓
播放 sd:6 “异常占用持续”
```

非工作时间释放逻辑：

```text
ROI 内无人持续 MEETING_RELEASE_EMPTY_SEC
↓
只播放 sd:4
↓
不播放 sd:2
↓
显示“会议室空闲”
```

---

## 13. AC79 音频命令映射

| 编号 | 命令 | 含义 |
|---|---|---|
| 1 | `sd:1` | 会议室开始使用 |
| 2 | `sd:2` | 会议室空闲 |
| 3 | `sd:3` | 长时间占用提醒 |
| 4 | `sd:4` | 房间已空，请关闭设备 |
| 5 | `sd:5` | 非工作时间提醒 |
| 6 | `sd:6` | 异常占用持续提醒 |

普通模式下，可以为每个 ROI 手动选择 `sd:1` ~ `sd:6`。  
会议室模式下，系统会根据当前状态自动选择音频。

---

## 14. ROI 保存与加载

点击 **Save Group** 可以保存当前所有 ROI 到 JSON 文件。

默认路径：

```bash
./roi_ui_output/rois.json
```

保存内容包括：

- ROI 名称。
- ROI 坐标。
- 普通模式下的驻留阈值。
- 普通模式下的音频编号。
- `frame_size`。
- `group_name`。
- `saved_at`。
- 会议室工作时间相关信息。

注意：ROI 坐标保存的是原始视频帧坐标，不是 UI 缩放后的显示坐标。如果 AC79 输出分辨率发生变化，需要重新绘制并保存 ROI。

---

## 15. 视频流看门狗与恢复机制

如果视频卡住，可能原因包括：

- AC79 停止发送 UDP 视频。
- CTP 控制链路断开。
- WiFi 波动。
- UDP 丢包严重。
- AC79 推流状态异常。

推荐看门狗参数：

```env
VIDEO_STALL_TIMEOUT_SEC=5
VIDEO_REOPEN_COOLDOWN_SEC=8
VIDEO_FULL_RESTART_SEC=20
```

恢复策略：

```text
超过 VIDEO_STALL_TIMEOUT_SEC 没有新帧
↓
重新发送 app / date / open 640 480 20 8000 0
↓
如果超过 VIDEO_FULL_RESTART_SEC 仍然没有新帧
↓
重启 UDP/TensorRT worker
↓
再次发送 open 命令
```

---

## 16. 常见问题排查

### 16.1 UI 显示 “waiting for UDP video stream”

检查：

```bash
ping 192.168.1.1
sudo ss -lunp | grep ':2224'
python3 jieli_min_udp_client.py --port 2224 --device-ip 192.168.1.1
```

同时确认：

- Jetson 已连接 AC79 WiFi 或处于同一局域网。
- `DEVICE_IP` 配置正确。
- `UDP_PORT=2224`。
- AC79 已收到 `open 640 480 20 8000 0` 命令。
- 没有其他程序占用 UDP `2224`。

### 16.2 UDP 2224 端口被占用

不要同时运行以下程序：

```bash
python3 jieli_rknn_udp_infer.py
python3 jieli_min_udp_client.py
python3 run_roi_ui.py
```

检查并结束旧进程：

```bash
sudo lsof -i:2224
sudo kill -9 <PID>
```

### 16.3 TensorRT engine 无法加载

检查：

```bash
ls -lh ./model/yolo11n.engine
```

确保 `.engine` 是在当前 Jetson 设备、JetPack、CUDA、TensorRT 版本和模型输入尺寸下生成的。TensorRT engine 通常不能跨设备或跨 TensorRT 版本直接复用。

### 16.4 检测类别不正确

如果使用 COCO 标签做人检测，保持：

```env
CLASS_FILTER=0
```

如果是自己训练的模型，需要根据自己的类别顺序修改 `LABELS_PATH` 和 `CLASS_FILTER`。

### 16.5 ROI 位置不准确

系统保存的是原始视频帧坐标。如果将：

```text
open 640 480 20 8000 0
```

改成其他分辨率，需要重新绘制并保存 ROI。

### 16.6 PySide6 安装失败

可以尝试：

```bash
python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple PySide6 numpy
```

如果 Jetson 上 `opencv-python` 安装失败，优先使用系统自带 OpenCV。

### 16.7 AC79 音频不播放

检查：

```bash
cat roi_ui_output/ctp_auto.log
```

同时确认：

- AC79 SD 卡中存在对应音频文件。
- 手动发送 `sd:1` ~ `sd:6` 能播放音频。
- CTP 进程仍在运行。
- UI 日志中显示已经发送了 `sd:x` 命令。

---

## 17. 推荐测试顺序

### 第一步：测试 AC79 UDP 视频流

```bash
python3 jieli_min_udp_client.py --port 2224 --device-ip 192.168.1.1
```

如果这一步没有图像，问题通常在 AC79 推流或网络链路，不在 UI。

### 第二步：准备 TensorRT 模型

```bash
./scripts/prepare_yolo_trt.sh \
  --pt /home/ubuntu/Desktop/yolo/yolo11n.pt \
  --engine ./model/yolo11n.engine \
  --imgsz 640
```

### 第三步：启动 Nano UI

```bash
./start_roi_ui_nano.sh --model ./model/yolo11n.engine --class-filter 0
```

### 第四步：测试普通 ROI 模式

```text
Default Mode
↓
Enter ROI Edit
↓
Draw ROI
↓
Select audio
↓
Set dwell threshold
↓
Apply ROI
↓
Save Group
↓
Person enters ROI
↓
Reach threshold and trigger alarm audio
```

### 第五步：测试会议室模式

为了快速测试，可以临时缩短阈值：

```env
MEETING_USE_START_SEC=5
MEETING_RELEASE_EMPTY_SEC=5
MEETING_LONG_USE_SEC=10
MEETING_ABNORMAL_EXTRA_SEC=10
MEETING_ABNORMAL_REPEAT_SEC=10
MEETING_RELEASE_AUDIO_GAP_SEC=3
```

测试流程：

```text
选择 Meeting Mode
↓
设置工作时间
↓
绘制一个或多个 ROI
↓
人员进入 ROI
↓
5 秒后：会议室开始使用
↓
10 秒后：长时间占用提醒
↓
再过 10 秒：异常占用提醒
↓
持续占用时 sd:6 重复播放
↓
人员离开 ROI 5 秒后：会议室释放
```

---

## 18. 主要文件说明

### `start_roi_ui_nano.sh`

Nano 专用启动脚本。负责准备 Python 运行环境、加载 `.env.nano`、设置 TensorRT 后端，并启动 `run_roi_ui.py`。

### `.env.nano`

Nano 专用环境配置文件。包含 `DETECTOR_BACKEND=tensorrt`、TensorRT 模型路径、标签路径、输入尺寸、类别过滤、UDP 端口、设备 IP、ROI 输出路径等配置。

### `run_roi_ui.py`

UI 入口文件。读取配置、初始化主窗口并启动 ROI UI 应用。

### `roi_ui/config.py`

读取环境变量，并管理 UDP、模型路径、ROI 输出、会议室阈值和看门狗参数。

### `roi_ui/worker.py`

负责 UDP 接收、JPEG 重组、JPEG 解码、调用检测后端，并将画面和检测结果发送给 UI。

### `roi_ui/video_widget.py`

负责视频显示、检测框绘制、ROI 绘制、会议室状态文字显示、鼠标拖拽和坐标映射。

### `roi_ui/dwell.py`

负责底边中点计算、ROI 命中判断和驻留时间统计。

### `roi_ui/roi_model.py`

负责 ROI 数据结构、ROI 分组保存和加载。

### `jieli_min_ctp_client.py`

用于向 AC79 发送 CTP 命令，例如视频流控制和本地音频播放命令。

### `jieli_min_udp_client.py`

用于单独测试 UDP JPEG 视频流。调试视频问题时建议先运行它。

---

## 19. 与旧 RK3588 版本的区别

旧版 README 主要描述：

```text
RK3588 + RKNNLite + person.rknn
```

当前 Nano 版本应描述为：

```text
Jetson Nano / Orin Nano + TensorRT + yolo11n.engine
```

主要变化：

| 旧 RK 版本 | 当前 Nano 版本 |
|---|---|
| RK3588 | Jetson Nano / Orin Nano |
| RKNNLite / RKNN | TensorRT |
| `person.rknn` | `yolo11n.engine` 或自定义 `.engine` |
| `BGR_INPUT=1` | 通常使用 `BGR_INPUT=0` |
| `start_roi_ui_all.sh` | 优先使用 `start_roi_ui_nano.sh` |
| `.env.roi.example` / `.env` | 优先使用 `.env.nano` |

---

## 20. 后续可优化方向

后续可以继续扩展：

1. 加入 ByteTrack / BoT-SORT，实现稳定多目标 ID 跟踪。
2. 增加多边形 ROI 支持。
3. 增加 ROI 拖动、缩放、删除手柄。
4. 增加 Web 配置页面。
5. 增加事件回放和截图查看页面。
6. 增加 TensorRT engine 兼容性检查。
7. 增加 systemd 服务，实现 Jetson 开机自启。
8. 增加 MQTT / HTTP 事件上传。
9. 使用轻量数据库保存事件历史。
10. 增加工作日、周末、白天、夜间等多时段规则。
