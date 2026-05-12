# AC79 → RK3588 空间占用检测系统（UDP + CTP + RKNN + ROI UI）

本项目基于 **杰理 AC79 摄像头开发板 + RK3588**，实现从摄像头视频采集、UDP JPEG 传输、RK3588 RKNN 人体检测，到 ROI 区域驻留判断、会议室占用判断和本地音频播报的一体化空间占用检测系统。

项目保留原有 AC79 UDP 视频流和 RK3588 RKNN 推理链路，在 RK3588 端新增 PySide6 图形界面，用于实时显示视频、绘制检测框、框选 ROI、保存/加载 ROI 组、设置默认报警逻辑和会议室模式逻辑。

---

## 1. 系统工作流程

```text
AC79 摄像头开发板
    │
    │  CTP 控制链路：app / date / open / sd:x / quit
    │  TCP 3333
    │
    ├───────────────────────────────────────┐
    │                                       │
    │  UDP JPEG 视频流                       │
    │  UDP 2224                              │
    │                                       ▼
RK3588 Linux
    │
    ├─ UDP 收包
    ├─ JPEG 分片重组
    ├─ JPEG 解码为 OpenCV BGR 图像
    ├─ RKNNLite 调用 YOLO 人体检测模型
    ├─ PySide6 UI 实时显示视频、检测框、ROI
    ├─ 使用人体检测框“底边中点”判断目标是否进入 ROI
    ├─ 默认模式：按每个 ROI 的驻留阈值触发报警
    ├─ 会议室模式：按统一会议室规则判断使用、空闲、长时间占用、异常占用
    └─ 通过 CTP 发送 sd:1 ~ sd:6 指令，让 AC79 播放本地音频
```

### 1.1 视频链路

1. RK3588 启动 UI。
2. 点击 UI 的 **启动** 按钮。
3. UI 启动 UDP/RKNN 工作线程，监听 UDP `2224`。
4. UI 自动启动 `jieli_min_ctp_client.py --host 192.168.1.1`。
5. UI 自动发送：

```text
app
date
open 640 480 20 8000 0
```

6. AC79 开始通过 UDP 发送 JPEG 视频流。
7. RK3588 解码视频帧并进行 RKNN 人体检测。
8. UI 显示实时画面、人体框、ROI 框和当前状态。

> 注意：`open 640 480 20 8000 0` 最后一个参数建议保持为 `0`，当前 UI 收流和解码逻辑按 UDP JPEG 视频流处理。

### 1.2 ROI 判断逻辑

系统使用人体检测框的 **底边中点** 判断是否进入 ROI：

```text
bbox = (x1, y1, x2, y2)
bottom_x = (x1 + x2) / 2
bottom_y = y2
```

如果 `(bottom_x, bottom_y)` 落在某个 ROI 矩形区域内，则认为人员进入该 ROI。

这种方式比使用检测框中心点更适合人体站立区域判断，因为底边中点更接近人的脚底位置。

---

## 2. 目录结构

建议 UI 文件放在 `jieli_linux_bundle` 下，这样可以直接复用原有的 CTP、UDP、RKNN 推理脚本和模型目录。

```text
ac79-rk3588/
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

## 3. 环境准备

### 3.1 硬件与网络

确保：

1. RK3588 已连接 AC79 板子的 WiFi，或两者处于同一局域网。
2. RK3588 能访问 AC79 的 CTP 地址，默认通常是：

```text
192.168.1.1
```

3. AC79 UDP 视频流默认端口为：

```text
2224
```

4. RK3588 已安装并能正常使用 RKNNLite/RKNN 运行环境。

### 3.2 安装 UI 依赖

进入 `jieli_linux_bundle`：

```bash
cd ac79-rk3588/jieli_linux_bundle
```

安装 UI 依赖：

```bash
python3 -m pip install -r requirements_ui.txt
```

如果 RK3588 上 `opencv-python` 安装困难，而系统已经有 OpenCV，可以只安装 PySide6 和 numpy：

```bash
python3 -m pip install PySide6 numpy
```

> 注意：`rknnlite2` 或 `rknnlite` 建议使用你当前已经跑通 RKNN 推理的 Python 环境，不建议为了 UI 重新破坏原有 RKNN 环境。

---

## 4. 配置文件 `.env`

可以从示例文件复制：

```bash
cp .env.roi.example .env
nano .env
```

推荐配置：

```bash
# AC79 / UDP
DEVICE_IP=192.168.1.1
BIND_IP=0.0.0.0
UDP_PORT=2224
CLEANUP_TIMEOUT=3.0

# RKNN 模型
MODEL_PATH=./model/person.rknn
LABELS_PATH=./model/labels.txt
INPUT_WIDTH=640
INPUT_HEIGHT=640
OBJ_THRESH=0.25
NMS_THRESH=0.45
MAX_DET=10
BGR_INPUT=1
SINGLE_CORE=1
AGNOSTIC_NMS=0

# UI 输出
ROI_JSON=./roi_ui_output/rois.json
SCREENSHOT_DIR=./roi_ui_output/screenshots
ROI_EVENT_LOG=./roi_ui_output/events.jsonl
ALARM_CMD=

# UI 模式：default 或 meeting
UI_MODE=default

# 会议室模式默认工作时间
MEETING_WORK_START=09:00
MEETING_WORK_END=18:00

# 会议室模式阈值，单位秒
MEETING_USE_START_SEC=60
MEETING_RELEASE_EMPTY_SEC=30
MEETING_LONG_USE_SEC=120
MEETING_ABNORMAL_EXTRA_SEC=180
MEETING_ABNORMAL_REPEAT_SEC=180
MEETING_RELEASE_AUDIO_GAP_SEC=3

# 会议室模式左上角显示文字
MEETING_IDLE_TEXT=会议室空闲
MEETING_BUSY_TEXT=会议室正在使用

# 可选：视频流 watchdog，单位秒
VIDEO_STALL_TIMEOUT_SEC=5
VIDEO_REOPEN_COOLDOWN_SEC=8
VIDEO_FULL_RESTART_SEC=20
```

### 4.1 配置注意事项

#### `DEVICE_IP`

默认 AC79 AP 地址通常是：

```bash
DEVICE_IP=192.168.1.1
```

如果 UI 能启动但一直没有画面，可以临时关闭来源 IP 过滤：

```bash
DEVICE_IP=
```

或者启动时传参：

```bash
python3 run_roi_ui.py --device-ip empty
```

#### `UDP_PORT`

默认端口为：

```bash
UDP_PORT=2224
```

不要同时运行多个监听 `2224` 的程序，例如：

```bash
python3 jieli_rknn_udp_infer.py
python3 jieli_min_udp_client.py
python3 run_roi_ui.py
```

ROI UI 已经内置 UDP 收流和 RKNN 推理，不需要再同时启动 `jieli_rknn_udp_infer.py`。

#### `DISPLAY`

如果在 RK3588 LCD 上运行 UI，通常需要：

```bash
export DISPLAY=:0
```

如果界面显示不全，可以临时设置缩放：

```bash
export QT_SCALE_FACTOR=0.75
```

或者在代码中限制右侧控制区宽度。

#### 模型路径

确认模型文件存在：

```bash
ls -lh ./model/person.rknn
ls -lh ./model/labels.txt
```

确认 RKNNLite 可导入：

```bash
python3 -c "from rknnlite.api import RKNNLite; print('rknn ok')"
```

---

## 5. 启动方式

### 5.1 推荐启动

```bash
cd ac79-rk3588/jieli_linux_bundle
export DISPLAY=:0
python3 run_roi_ui.py --device-ip 192.168.1.1
```

打开 UI 后点击 **启动**。

点击启动后，UI 会自动：

1. 启动 UDP/RKNN 工作线程；
2. 监听 UDP `2224`；
3. 启动 CTP 客户端；
4. 发送 `app`、`date`、`open 640 480 20 8000 0`；
5. 接收 AC79 UDP JPEG 视频流；
6. 调用 RKNN 模型进行人体检测；
7. 在 LCD/UI 上显示视频、人体框、ROI 和状态。

### 5.2 使用脚本启动

```bash
chmod +x run_roi_ui.py start_roi_ui_all.sh stop_roi_ui.sh
./start_roi_ui_all.sh
```

如果 `start_roi_ui_all.sh` 已经会先调用 `start_ctp.sh`，而 UI 本身也会在点击启动时自动开流，建议避免重复开流。可以在 `.env` 中设置：

```bash
START_CTP=0
```

然后再执行：

```bash
./start_roi_ui_all.sh
```

---

## 6. UI 使用流程

### 6.1 默认模式

默认模式用于普通 ROI 驻留报警。

操作流程：

```text
启动 UI
↓
点击“启动”
↓
选择“默认模式”
↓
点击“进入编辑”
↓
在视频画面中拖拽创建 ROI
↓
弹出音频选择框，选择 sd:1 ~ sd:6
↓
在 ROI 列表中选择 ROI
↓
设置名称、驻留阈值、报警音频
↓
点击“应用ROI”
↓
点击“保存组”
```

默认模式逻辑：

```text
人员进入某个 ROI
↓
持续驻留时间 >= 该 ROI 的 dwell_sec
↓
保存报警截图
↓
写入事件日志
↓
通过 CTP 发送对应 sd:x 音频指令
```

默认模式下，每个 ROI 可以有独立的：

- 名称；
- 驻留阈值；
- 报警音频编号。

### 6.2 会议室模式

会议室模式用于判断会议室是否正在使用、是否长时间占用、是否异常占用。

进入会议室模式时，会弹出工作时间设置框，使用 24 小时制，只设置一个时间段。

会议室模式下：

- 创建 ROI 时不选择音频；
- 不显示单个 ROI 的驻留阈值；
- 不允许设置单个 ROI 的阈值；
- ROI 可以创建多个；
- 会议室正在使用时，不允许删除或清空已有 ROI，但允许继续新增 ROI；
- 画面左上角显示会议室状态：
  - 绿色：`会议室空闲`
  - 红色：`会议室正在使用`

会议室模式逻辑：

#### 工作时间内

```text
人员进入任意 ROI
↓
连续驻留 >= MEETING_USE_START_SEC
↓
播放 sd:1 “会议室已开始使用”
↓
左上角显示红色“会议室正在使用”
```

如果持续占用：

```text
占用时间 >= MEETING_LONG_USE_SEC
↓
播放 sd:3 “当前会议室已被长时间占用，请注意使用时长”
```

如果继续占用：

```text
占用时间 >= MEETING_LONG_USE_SEC + MEETING_ABNORMAL_EXTRA_SEC
↓
播放 sd:6 “异常占用持续，请管理员介入”
```

异常占用后，如果仍然持续占用，每隔：

```bash
MEETING_ABNORMAL_REPEAT_SEC
```

再次播放 `sd:6`。

释放会议室：

```text
已开始使用后，ROI 内连续 MEETING_RELEASE_EMPTY_SEC 秒无人
↓
播放 sd:2 “会议室已空闲”
↓
等待 MEETING_RELEASE_AUDIO_GAP_SEC 秒
↓
播放 sd:4 “会议室无人，请关闭设备”
↓
左上角显示绿色“会议室空闲”
```

#### 非工作时间内

```text
人员进入任意 ROI
↓
连续驻留 >= MEETING_USE_START_SEC
↓
播放 sd:5 “当前为非工作时间，请尽快离开会议室”
↓
左上角显示红色“会议室正在使用”
```

如果继续占用：

```text
占用时间 >= MEETING_USE_START_SEC + MEETING_ABNORMAL_EXTRA_SEC
↓
播放 sd:6 “异常占用持续，请管理员介入”
```

异常占用后，如果仍然持续占用，每隔：

```bash
MEETING_ABNORMAL_REPEAT_SEC
```

再次播放 `sd:6`。

非工作时间释放会议室：

```text
ROI 内连续 MEETING_RELEASE_EMPTY_SEC 秒无人
↓
只播放 sd:4 “会议室无人，请关闭设备”
↓
不播放 sd:2 “会议室已空闲”
↓
左上角显示绿色“会议室空闲”
```

---

## 7. 音频编号说明

AC79 SD 卡音频编号约定：

| 编号 | 命令 | 内容 |
|---:|---|---|
| 1 | `sd:1` | 会议室已开始使用 |
| 2 | `sd:2` | 会议室已空闲 |
| 3 | `sd:3` | 当前会议室已被长时间占用，请注意使用时长 |
| 4 | `sd:4` | 会议室无人，请关闭设备 |
| 5 | `sd:5` | 当前为非工作时间，请尽快离开会议室 |
| 6 | `sd:6` | 异常占用持续，请管理员介入 |

默认模式下，创建 ROI 时可选择 `1 ~ 6` 中任意音频作为该 ROI 的报警音频。

会议室模式下，不需要手动选择音频，系统根据会议室状态自动播放对应音频。

---

## 8. ROI 组保存与加载

点击 **保存组** 会把当前多个 ROI 一次性保存到 JSON 文件。

点击 **加载组** 会一次性加载上次保存的多个 ROI。

ROI 文件默认路径：

```text
./roi_ui_output/rois.json
```

保存内容包括：

- ROI 名称；
- ROI 坐标；
- 默认模式下的驻留阈值；
- 默认模式下的音频编号；
- frame_size；
- group_name；
- saved_at；
- 会议室工作时间元信息。

注意：ROI 坐标保存的是原始视频帧坐标，不是 UI 缩放后的显示坐标。如果 AC79 输出分辨率改变，建议重新框选 ROI 并保存。

---

## 9. 视频流断连与自动恢复

如果视频流断连，UI 画面可能停留在最后一帧，看起来像“画面静止”。常见原因包括：

- AC79 停止发送 UDP 视频；
- CTP 控制链路断开；
- WiFi 短暂波动；
- UDP 分片丢失；
- AC79 端开流状态丢失。

推荐在 UI 中加入 watchdog 机制，定期检查最后一帧更新时间。

建议参数：

```bash
VIDEO_STALL_TIMEOUT_SEC=5
VIDEO_REOPEN_COOLDOWN_SEC=8
VIDEO_FULL_RESTART_SEC=20
```

处理策略：

```text
超过 VIDEO_STALL_TIMEOUT_SEC 没有新帧
↓
重新发送 app / date / open 640 480 20 8000 0
↓
如果仍然超过 VIDEO_FULL_RESTART_SEC 没恢复
↓
重启 UDP/RKNN worker
↓
再次发送 open 开流命令
```

这样可以减少手动点击“停止 → 启动”的次数。

---

## 10. 常见问题

### 10.1 UI 显示“等待 UDP 视频流”

检查：

1. RK3588 是否连接 AC79 WiFi。
2. `DEVICE_IP` 是否正确。
3. `UDP_PORT` 是否为 `2224`。
4. 是否点击了 UI 的 **启动** 按钮。
5. AC79 是否成功收到 `open 640 480 20 8000 0`。
6. 是否有其它程序占用了 UDP `2224`。

可以检查端口：

```bash
sudo ss -lunp | grep ':2224'
```

如果没有 `ss`，可用：

```bash
sudo lsof -i:2224
```

### 10.2 不能同时运行哪些程序

不要同时运行：

```bash
python3 jieli_rknn_udp_infer.py
python3 jieli_min_udp_client.py
python3 run_roi_ui.py
```

因为它们可能都会尝试监听 UDP `2224`。

### 10.3 CTP 音频不播放

检查：

1. AC79 SD 卡中是否存在对应音频。
2. 手动运行 CTP 客户端时，`sd:1` ~ `sd:6` 是否能播放。
3. UI 日志是否显示 `已发送杰理音频命令: sd:x`。
4. CTP 进程是否仍然存活。
5. 是否在发送音频前 CTP 已经断开。

可查看：

```bash
cat roi_ui_output/ctp_auto.log
```

### 10.4 UI 显示不全

可以临时设置：

```bash
export QT_SCALE_FACTOR=0.75
```

也可以在代码中限制右侧面板宽度，例如：

```python
side.setMinimumWidth(220)
side.setMaximumWidth(300)
splitter.setSizes([900, 260])
```

### 10.5 ROI 框选位置不准

本系统内部保存的是原始视频帧坐标。只要 AC79 输出分辨率稳定，ROI 判断就稳定。

如果改了 `open` 分辨率，例如从 `640x480` 改成其它分辨率，需要重新框选 ROI。

### 10.6 会议室模式不显示阈值

这是预期行为。

会议室模式不使用单个 ROI 的 `dwell_sec`，而是统一使用：

```bash
MEETING_USE_START_SEC
MEETING_RELEASE_EMPTY_SEC
MEETING_LONG_USE_SEC
MEETING_ABNORMAL_EXTRA_SEC
MEETING_ABNORMAL_REPEAT_SEC
```

因此会议室模式下只显示 ROI 名称，不显示和不设置单个 ROI 阈值。

---

## 11. 推荐测试顺序

### 第一步：确认 AC79 UDP 视频流

```bash
python3 jieli_min_udp_client.py --port 2224 --device-ip 192.168.1.1
```

如果这个脚本也没有画面，问题通常在 AC79 发流或网络链路，不在 UI。

### 第二步：确认 RKNN 推理脚本

```bash
python3 jieli_rknn_udp_infer.py --model ./model/person.rknn --labels ./model/labels.txt
```

确认能正常加载模型并进行检测。

### 第三步：关闭其它 UDP 监听程序

```bash
pkill -f jieli_rknn_udp_infer.py
pkill -f jieli_min_udp_client.py
pkill -f run_roi_ui.py
```

### 第四步：启动 UI

```bash
export DISPLAY=:0
python3 run_roi_ui.py --device-ip 192.168.1.1
```

点击 **启动**。

### 第五步：测试默认模式

```text
默认模式
↓
进入编辑
↓
画 ROI
↓
选择音频
↓
设置阈值
↓
应用 ROI
↓
保存组
↓
人员进入 ROI
↓
超过阈值后播放对应音频
```

### 第六步：测试会议室模式

测试时可以把 `.env` 阈值改短：

```bash
MEETING_USE_START_SEC=5
MEETING_RELEASE_EMPTY_SEC=5
MEETING_LONG_USE_SEC=10
MEETING_ABNORMAL_EXTRA_SEC=10
MEETING_ABNORMAL_REPEAT_SEC=10
MEETING_RELEASE_AUDIO_GAP_SEC=3
```

测试流程：

```text
选择会议室模式
↓
设置工作时间
↓
画多个 ROI
↓
人员进入 ROI
↓
5 秒后会议室开始使用
↓
10 秒后长时间占用
↓
再 10 秒后异常占用
↓
继续占用每 10 秒重复播放 sd:6
↓
人员离开 ROI 5 秒后释放会议室
```

---

## 12. 主要文件说明

### `run_roi_ui.py`

UI 启动入口。负责解析命令行参数、读取 `.env`、创建 `AppConfig`、启动 `MainWindow`。

### `roi_ui/config.py`

读取环境变量，管理 UDP、模型路径、ROI 输出、会议室模式阈值和 watchdog 参数。

### `roi_ui/worker.py`

负责 UDP 收流、JPEG 分片重组、JPEG 解码、调用 RKNN 推理，并把原图和检测结果发送给 UI。

### `roi_ui/video_widget.py`

负责视频显示、人物检测框绘制、ROI 绘制、会议室状态文字绘制、鼠标拖拽创建 ROI，以及 UI 坐标和原始帧坐标之间的映射。

### `roi_ui/dwell.py`

负责根据人体检测框底边中点判断是否进入 ROI，并统计驻留时间。

### `roi_ui/roi_model.py`

负责 ROI 数据结构、ROI 组保存和加载。

### `roi_ui/main_window.py`

负责主界面、按钮、模式切换、ROI 编辑、默认模式报警、会议室模式状态机、音频播报、事件日志和 CTP 控制。

---

## 13. 版本注意事项

1. UI 运行时不要同时运行旧的 `jieli_rknn_udp_infer.py`，否则可能抢占 UDP `2224`。
2. 启动 UI 前确认 RK3588 已连接 AC79 网络。
3. `open` 命令最后一个参数建议使用 `0`，保持 UDP JPEG 格式。
4. 会议室模式下不设置单个 ROI 阈值，统一使用会议室全局阈值。
5. 非工作时间开始的会议室占用，释放时只播放 `sd:4`，不播放 `sd:2`。
6. 工作时间开始的会议室占用，释放时播放 `sd:2`，再延迟播放 `sd:4`。
7. 如果视频流偶发卡住，建议启用 watchdog 参数自动重开发流。
8. 如果 LCD 分辨率较小，建议限制右侧控制区宽度或设置 `QT_SCALE_FACTOR`。
9. 如果 AC79 IP 变化，需要同步修改 `.env` 中的 `DEVICE_IP`。
10. 如果更换模型输入尺寸或视频分辨率，建议重新验证 ROI 坐标和检测效果。

