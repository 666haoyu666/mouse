# 仓鼠监控系统 UI 第二阶段增强版

本补丁基于第一阶段 UI 继续扩展，完成第二阶段主线：

```text
检测结果增强主线 = 多帧连续跟踪 + 停留时间统计 + 活动热区统计 + 场景行为规则判断
```

它不引入 VLM，也不替代第一阶段 YOLO 检测链路，而是在已有“检测 + 九宫格 + 固定 ROI + 中文描述”的基础上，增加连续状态分析能力。

## 1. 覆盖位置

将压缩包解压后，把其中的 `jieli_linux_bundle_2/` 覆盖到原仓库对应目录：

```text
mouse/
└── jieli_linux_bundle_2/
    ├── run_roi_ui.py
    ├── .env.hamster.stage2.example
    ├── README_hamster_ui_stage2.md
    └── roi_ui/
        ├── config.py
        ├── main_window.py
        ├── region_analyzer.py
        ├── roi_model.py
        ├── scene_stats.py
        ├── text_generator.py
        ├── tracker.py
        ├── video_widget.py
        └── worker.py
```

注意：本补丁只替换 UI 与业务逻辑层，仍然依赖原仓库已有的检测后端文件，例如：

```text
jieli_rknn_udp_infer.py
model/hamster.engine
model/hamster_labels.txt
```

## 2. 配置模型路径

复制示例配置：

```bash
cd jieli_linux_bundle_2
cp .env.hamster.stage2.example .env
```

然后修改 `.env`：

```bash
DETECTOR_BACKEND=tensorrt
MODEL_PATH=./model/hamster.engine
LABELS_PATH=./model/hamster_labels.txt
CLASS_FILTER=
```

标签文件建议：

```text
hamster
```

如果你的模型类别 0 是仓鼠，可以保持 `CLASS_FILTER=` 不过滤，也可以写：

```bash
CLASS_FILTER=0
```

## 3. 启动 UI

```bash
python3 run_roi_ui.py --env-file .env
```

如果要启动时直接显示热区叠加：

```bash
python3 run_roi_ui.py --env-file .env --heatmap
```

如果只想测试第一阶段效果，临时关闭跟踪：

```bash
python3 run_roi_ui.py --env-file .env --no-tracker
```

## 4. 第二阶段新增功能

### 4.1 多帧连续跟踪

新增 `roi_ui/tracker.py`，采用轻量 IoU + 中心距离关联方法，为连续帧中的仓鼠分配稳定 ID。

UI 会显示：

- 当前活动 ID
- 主目标 ID
- 连续跟踪时间
- 目标移动轨迹
- 目标中心点

### 4.2 停留时间统计

新增 `roi_ui/scene_stats.py`，持续统计：

- 每个 ID 的连续出现时间
- 当前九宫格停留时间
- 当前 ROI 停留时间
- ROI 累计停留时间
- 宫格累计停留时间

当达到阈值后，中文描述会出现：

```text
已形成连续停留
连续停留时间较长
在木屋附近停留
在同一画面区域持续停留
```

### 4.3 活动热区统计

系统会累计仓鼠在九宫格和固定 ROI 中的停留时间，并在右侧显示：

```text
宫格热区：右下: 23.5s；中央: 10.2s
ROI 热区：木屋: 18.3s；食盆: 6.7s
```

勾选“显示活动热区”后，视频画面上会以浅色块叠加显示活动更集中的九宫格区域。

### 4.4 场景行为规则判断

规则引擎会根据连续跟踪、ROI 停留、移动幅度输出解释性标签，例如：

```text
已形成连续停留
连续停留时间较长
在木屋附近停留
活动幅度较小
移动较活跃
```

事件会写入：

```text
roi_ui_output/hamster_events.jsonl
```

统计总表会写入：

```text
roi_ui_output/hamster_stats.json
```

## 5. 推荐调参

### 仓鼠跑得快，ID 容易断

增大：

```bash
TRACKER_CENTER_DISTANCE=100
TRACKER_MAX_MISSED=30
```

### 误把靠近 ROI 判断得太宽

减小：

```bash
ROI_DISTANCE_THRESHOLD=15
ROI_IOU_THRESHOLD=0.08
```

### “连续停留”太容易触发

增大：

```bash
DWELL_THRESHOLD_SEC=20
LONG_STAY_THRESHOLD_SEC=60
ROI_STAY_THRESHOLD_SEC=15
```

### 活动幅度较小误判

增大或减小：

```bash
STATIONARY_SPEED_THRESHOLD_PX=6
STATIONARY_THRESHOLD_SEC=12
```

## 6. 第二阶段验收标准

运行后应能看到：

1. 仓鼠被检测框框出；
2. 画面显示九宫格；
3. 木屋、跑轮、食盆、饮水器 ROI 可显示、拖拽、保存；
4. 同一只仓鼠跨帧获得稳定 ID；
5. 右侧显示连续跟踪时间；
6. 右侧显示当前 ROI / 宫格停留时间；
7. 右侧显示活动热区统计；
8. 中文描述包含连续状态与行为判断。

示例输出：

```text
当前检测到 1 只仓鼠，主目标 ID 3，位于画面右下区域，靠近木屋，已连续跟踪 18.2 秒，在木屋附近停留 9.4 秒，行为判断：已形成连续停留、在木屋附近停留。
活动热区：宫格 右下18.2s、中央6.5s；ROI 木屋9.4s、食盆2.1s。
```
