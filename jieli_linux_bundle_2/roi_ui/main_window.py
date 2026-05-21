from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json
import subprocess
import sys
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .config import AppConfig
from .region_analyzer import analyze_detection
from .roi_model import RectROI, default_scene_rois, load_rois, save_rois
from .scene_stats import ActivityStats, BehaviorRuleEngine
from .text_generator import generate_hotspot_text, generate_text
from .tracker import IoUTracker
from .video_widget import VideoWidget
from .worker import UdpInferWorker


class MainWindow(QMainWindow):
    def __init__(self, cfg: AppConfig, helper_script: str | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.helper_script = helper_script
        self.worker: UdpInferWorker | None = None
        self.ctp_proc: subprocess.Popen | None = None
        self.rois: list[RectROI] = []
        self.last_frame_shape: tuple[int, int] | None = None
        self.last_description = ""
        self.backend_label = cfg.detector_backend.upper()
        self.last_frame_ts = 0.0
        self.last_stream_reopen_ts = 0.0
        self.last_worker_restart_ts = 0.0
        self.tracker = IoUTracker(
            iou_threshold=cfg.tracker_iou_threshold,
            center_distance=cfg.tracker_center_distance,
            max_missed=cfg.tracker_max_missed,
            min_hits=cfg.tracker_min_hits,
            trail_length=cfg.trail_length,
            stationary_speed_threshold_px=cfg.stationary_speed_threshold_px,
        )
        self.stats = ActivityStats(cfg)
        self.rules = BehaviorRuleEngine(cfg)
        self._last_event_key_ts: dict[str, float] = {}

        self.video_watchdog_timer = QTimer(self)
        self.video_watchdog_timer.setInterval(1000)
        self.video_watchdog_timer.timeout.connect(self._check_video_watchdog)

        self.setWindowTitle(f"{cfg.ui_title} - UDP + {self.backend_label}")
        screen = QApplication.primaryScreen()
        if screen:
            g = screen.availableGeometry()
            self.resize(min(1480, int(g.width() * 0.95)), min(880, int(g.height() * 0.92)))
        else:
            self.resize(1480, 880)
        self._build_ui()
        self._load_rois_on_startup()
        self._refresh_roi_list()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        self.video = VideoWidget(self)
        self.video.set_grid_enabled(self.cfg.grid_enabled)
        self.video.set_grid_labels_enabled(self.cfg.grid_labels_enabled)
        self.video.set_heatmap_overlay_enabled(self.cfg.heatmap_overlay_enabled)
        self.video.rois_changed.connect(self._on_rois_changed)
        self.video.roi_selected.connect(self._on_roi_selected)
        splitter.addWidget(self.video)

        right = QWidget(self)
        right.setMinimumWidth(460)
        splitter.addWidget(right)
        splitter.setSizes([980, 500])
        panel = QVBoxLayout(right)

        panel.addWidget(self._build_top_controls())
        panel.addWidget(self._build_status_panel())
        panel.addWidget(self._build_tracking_panel())
        panel.addWidget(self._build_stats_panel())
        panel.addWidget(self._build_roi_panel())
        panel.addWidget(self._build_log_panel(), stretch=1)

    def _build_top_controls(self) -> QWidget:
        box = QGroupBox("运行控制")
        layout = QGridLayout(box)
        self.btn_start = QPushButton("启动")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.chk_edit = QCheckBox("ROI 编辑模式")
        self.chk_grid = QCheckBox("显示九宫格")
        self.chk_grid.setChecked(self.cfg.grid_enabled)
        self.chk_grid_labels = QCheckBox("显示宫格标签")
        self.chk_grid_labels.setChecked(self.cfg.grid_labels_enabled)
        self.chk_heatmap = QCheckBox("显示活动热区")
        self.chk_heatmap.setChecked(self.cfg.heatmap_overlay_enabled)

        layout.addWidget(self.btn_start, 0, 0)
        layout.addWidget(self.btn_stop, 0, 1)
        layout.addWidget(self.chk_edit, 1, 0, 1, 2)
        layout.addWidget(self.chk_grid, 2, 0)
        layout.addWidget(self.chk_grid_labels, 2, 1)
        layout.addWidget(self.chk_heatmap, 3, 0, 1, 2)

        self.btn_start.clicked.connect(self.start_worker)
        self.btn_stop.clicked.connect(self.stop_worker)
        self.chk_edit.toggled.connect(self.video.set_edit_mode)
        self.chk_grid.toggled.connect(self.video.set_grid_enabled)
        self.chk_grid_labels.toggled.connect(self.video.set_grid_labels_enabled)
        self.chk_heatmap.toggled.connect(self.video.set_heatmap_overlay_enabled)
        return box

    def _build_status_panel(self) -> QWidget:
        box = QGroupBox("第一阶段结构化结果")
        form = QFormLayout(box)
        self.lbl_detected = QLabel("否")
        self.lbl_count = QLabel("0")
        self.lbl_bbox = QLabel("-")
        self.lbl_center = QLabel("-")
        self.lbl_grid = QLabel("-")
        self.lbl_near = QLabel("-")
        self.lbl_inside = QLabel("-")
        self.lbl_fps = QLabel("0.0")
        self.txt_desc = QPlainTextEdit()
        self.txt_desc.setReadOnly(True)
        self.txt_desc.setMaximumBlockCount(80)
        self.txt_desc.setFixedHeight(82)
        self.txt_desc.setPlaceholderText("中文描述会显示在这里")

        form.addRow("是否检测到仓鼠", self.lbl_detected)
        form.addRow("检测数量", self.lbl_count)
        form.addRow("BBox", self.lbl_bbox)
        form.addRow("中心点", self.lbl_center)
        form.addRow("九宫格位置", self.lbl_grid)
        form.addRow("靠近区域", self.lbl_near)
        form.addRow("进入区域", self.lbl_inside)
        form.addRow("FPS", self.lbl_fps)
        form.addRow("中文描述", self.txt_desc)
        return box

    def _build_tracking_panel(self) -> QWidget:
        box = QGroupBox("第二阶段：多帧跟踪与停留时间")
        form = QFormLayout(box)
        self.lbl_active_ids = QLabel("-")
        self.lbl_main_track = QLabel("-")
        self.lbl_track_age = QLabel("0.0 s")
        self.lbl_roi_dwell = QLabel("-")
        self.lbl_grid_dwell = QLabel("-")
        self.lbl_behaviors = QLabel("-")
        self.lbl_behaviors.setWordWrap(True)
        form.addRow("当前活动 ID", self.lbl_active_ids)
        form.addRow("主目标", self.lbl_main_track)
        form.addRow("连续跟踪时间", self.lbl_track_age)
        form.addRow("当前 ROI 停留", self.lbl_roi_dwell)
        form.addRow("当前宫格停留", self.lbl_grid_dwell)
        form.addRow("规则判断", self.lbl_behaviors)
        return box

    def _build_stats_panel(self) -> QWidget:
        box = QGroupBox("第二阶段：活动热区统计")
        form = QFormLayout(box)
        self.lbl_top_grids = QLabel("-")
        self.lbl_top_rois = QLabel("-")
        self.lbl_running = QLabel("0.0 s")
        self.lbl_top_grids.setWordWrap(True)
        self.lbl_top_rois.setWordWrap(True)
        row = QHBoxLayout()
        self.btn_reset_stats = QPushButton("重置统计")
        self.btn_save_stats = QPushButton("保存统计")
        row.addWidget(self.btn_reset_stats)
        row.addWidget(self.btn_save_stats)
        form.addRow("运行时长", self.lbl_running)
        form.addRow("宫格热区", self.lbl_top_grids)
        form.addRow("ROI 热区", self.lbl_top_rois)
        form.addRow(row)
        self.btn_reset_stats.clicked.connect(self._reset_stats)
        self.btn_save_stats.clicked.connect(self._save_stats)
        return box

    def _build_roi_panel(self) -> QWidget:
        box = QGroupBox("固定 ROI：木屋 / 跑轮 / 食盆 / 饮水器")
        layout = QVBoxLayout(box)
        row1 = QHBoxLayout()
        self.btn_seed = QPushButton("生成默认 ROI")
        self.btn_save = QPushButton("保存 ROI")
        row1.addWidget(self.btn_seed)
        row1.addWidget(self.btn_save)
        layout.addLayout(row1)
        row2 = QHBoxLayout()
        self.btn_load = QPushButton("加载 ROI")
        self.btn_clear = QPushButton("清空 ROI")
        row2.addWidget(self.btn_load)
        row2.addWidget(self.btn_clear)
        layout.addLayout(row2)

        self.roi_list = QListWidget()
        self.roi_list.setMaximumHeight(120)
        self.roi_name = QLineEdit()
        self.btn_apply_name = QPushButton("修改名称")
        self.btn_delete_roi = QPushButton("删除选中 ROI")
        layout.addWidget(self.roi_list)
        layout.addWidget(QLabel("名称"))
        layout.addWidget(self.roi_name)
        row3 = QHBoxLayout()
        row3.addWidget(self.btn_apply_name)
        row3.addWidget(self.btn_delete_roi)
        layout.addLayout(row3)

        self.btn_seed.clicked.connect(self._seed_default_rois)
        self.btn_save.clicked.connect(self._save_rois)
        self.btn_load.clicked.connect(self._load_rois_manually)
        self.btn_clear.clicked.connect(self._clear_rois)
        self.roi_list.currentItemChanged.connect(self._on_roi_item_changed)
        self.btn_apply_name.clicked.connect(self._apply_roi_name)
        self.btn_delete_roi.clicked.connect(self._delete_selected_roi)
        return box

    def _build_log_panel(self) -> QWidget:
        box = QGroupBox("运行日志 / 行为事件")
        layout = QVBoxLayout(box)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(600)
        layout.addWidget(self.log_edit)
        return box

    def append_log(self, text: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_edit.appendPlainText(f"[{ts}] {text}")

    def _send_ctp_line(self, line: str) -> bool:
        try:
            if self.ctp_proc and self.ctp_proc.poll() is None and self.ctp_proc.stdin:
                self.ctp_proc.stdin.write(line if line.endswith("\n") else line + "\n")
                self.ctp_proc.stdin.flush()
                return True
        except Exception as exc:
            self.append_log(f"发送 CTP 命令失败: {exc}")
        return False

    def start_ctp_stream(self) -> None:
        if self.ctp_proc and self.ctp_proc.poll() is None:
            self.append_log("CTP 已在运行，不重复启动")
            return

        ctp_script = Path(__file__).resolve().parent.parent / "jieli_min_ctp_client.py"
        if not ctp_script.exists():
            self.append_log(f"CTP 脚本不存在: {ctp_script}")
            return

        host = self.cfg.device_ip.strip() or "192.168.1.1"
        try:
            log_path = Path("./roi_ui_output/ctp_auto.log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("a", encoding="utf-8")
            self.ctp_proc = subprocess.Popen(
                [sys.executable, str(ctp_script), "--host", host],
                stdin=subprocess.PIPE,
                stdout=log_file,
                stderr=log_file,
                text=True,
                bufsize=1,
            )

            for cmd in ["app", "date", "open 640 480 20 8000 0"]:
                if self.ctp_proc.stdin:
                    self.ctp_proc.stdin.write(cmd + "\n")
                    self.ctp_proc.stdin.flush()
                    time.sleep(0.3)

            self.append_log(f"已自动发送 CTP 开流命令: host={host}, open 640x480 fps=20 format=0")
            self.append_log(f"CTP 日志: {log_path}")
        except Exception as exc:
            self.append_log(f"自动启动 CTP 失败: {exc}")

    def _stop_ctp_stream(self) -> None:
        if not self.ctp_proc or self.ctp_proc.poll() is not None:
            self.ctp_proc = None
            return
        try:
            self._send_ctp_line("quit")
            self.ctp_proc.terminate()
            self.ctp_proc.wait(timeout=2)
        except Exception:
            try:
                self.ctp_proc.kill()
            except Exception:
                pass
        self.ctp_proc = None
        self.append_log("CTP 已停止")

    def _reopen_video_stream(self) -> None:
        if self.ctp_proc and self.ctp_proc.poll() is None:
            ok = True
            for line in ["app", "date", "open 640 480 20 8000 0"]:
                if not self._send_ctp_line(line):
                    ok = False
                    break
            if ok:
                self.append_log("视频流 watchdog：已重新发送 CTP 开流命令")
                return

        self.append_log("视频流 watchdog：CTP 不可用，尝试重新启动 CTP")
        self._restart_ctp_process()

    def _restart_ctp_process(self) -> None:
        self._stop_ctp_stream()
        self.start_ctp_stream()

    def _restart_udp_worker_only(self) -> None:
        self.append_log(f"视频流 watchdog：开始重启 UDP/{self.backend_label} worker")
        if self.worker:
            try:
                self.worker.finished.disconnect(self._on_worker_finished)
            except Exception:
                pass
            try:
                self.worker.stop()
                self.worker.wait(2000)
            except Exception as exc:
                self.append_log(f"停止旧 worker 失败: {exc}")
            self.worker = None

        self.worker = UdpInferWorker(config=self.cfg, helper_script=self.helper_script)
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.log_message.connect(self.append_log)
        self.worker.error_message.connect(self._on_worker_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()
        self.last_frame_ts = time.time()
        self.append_log(f"视频流 watchdog：UDP/{self.backend_label} worker 已重启")
        QTimer.singleShot(1000, self._reopen_video_stream)

    def _check_video_watchdog(self) -> None:
        if not self.worker or not self.worker.isRunning():
            return

        now = time.time()
        if self.last_frame_ts <= 0:
            self.last_frame_ts = now
            return

        stale_sec = now - self.last_frame_ts
        stall_timeout = float(getattr(self.cfg, "video_stall_timeout_sec", 5.0))
        reopen_cooldown = float(getattr(self.cfg, "video_reopen_cooldown_sec", 8.0))
        full_restart_sec = float(getattr(self.cfg, "video_full_restart_sec", 20.0))
        if stale_sec < stall_timeout:
            return

        if now - self.last_stream_reopen_ts >= reopen_cooldown:
            self.last_stream_reopen_ts = now
            self.append_log(f"视频流 watchdog：{stale_sec:.1f}s 未收到新帧，尝试重开发流")
            self._reopen_video_stream()

        if stale_sec >= full_restart_sec and now - self.last_worker_restart_ts >= full_restart_sec:
            self.last_worker_restart_ts = now
            self.append_log(f"视频流 watchdog：{stale_sec:.1f}s 未恢复，重启 UDP/{self.backend_label} worker")
            self._restart_udp_worker_only()

    def _load_rois_on_startup(self) -> None:
        self.rois = load_rois(self.cfg.roi_json_path)
        self.video.set_rois(self.rois)
        if self.rois:
            self.append_log(f"已从 {self.cfg.roi_json_path} 读取 {len(self.rois)} 个 ROI")

    def _load_rois_manually(self) -> None:
        self._load_rois_on_startup()
        self._refresh_roi_list()

    def _save_rois(self) -> None:
        save_rois(self.cfg.roi_json_path, self.rois)
        self.append_log(f"ROI 已保存到 {self.cfg.roi_json_path}")

    def _clear_rois(self) -> None:
        self.rois = []
        self.video.set_rois(self.rois)
        self._refresh_roi_list()
        self.append_log("已清空 ROI")

    def _seed_default_rois(self) -> None:
        if not self.last_frame_shape:
            QMessageBox.information(self, "提示", "请先启动视频，拿到真实画面尺寸后再自动生成固定 ROI。")
            return
        frame_h, frame_w = self.last_frame_shape
        self.rois = default_scene_rois(frame_w, frame_h)
        self.video.set_rois(self.rois)
        self._refresh_roi_list()
        self.append_log("已按当前画面尺寸生成默认固定 ROI")

    def _on_rois_changed(self, rois) -> None:
        self.rois = [roi.normalized() for roi in rois]
        self.video.set_rois(self.rois)
        self._refresh_roi_list(select_id=self.video.selected_roi_id)

    def _on_roi_selected(self, roi_id: int) -> None:
        self.video.selected_roi_id = roi_id
        self._refresh_roi_list(select_id=roi_id)

    def _refresh_roi_list(self, select_id: int | None = None) -> None:
        self.roi_list.blockSignals(True)
        self.roi_list.clear()
        for roi in self.rois:
            item = QListWidgetItem(f"{roi.roi_id}: {roi.name}  ({roi.x1},{roi.y1})-({roi.x2},{roi.y2})")
            item.setData(Qt.ItemDataRole.UserRole, roi.roi_id)
            self.roi_list.addItem(item)
            if select_id is not None and roi.roi_id == select_id:
                self.roi_list.setCurrentItem(item)
        self.roi_list.blockSignals(False)

    def _on_roi_item_changed(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        if current is None:
            self.roi_name.clear()
            return
        roi_id = int(current.data(Qt.ItemDataRole.UserRole))
        self.video.selected_roi_id = roi_id
        roi = next((item for item in self.rois if item.roi_id == roi_id), None)
        self.roi_name.setText(roi.name if roi else "")
        self.video.update()

    def _apply_roi_name(self) -> None:
        item = self.roi_list.currentItem()
        if item is None:
            return
        roi_id = int(item.data(Qt.ItemDataRole.UserRole))
        new_name = self.roi_name.text().strip()
        if not new_name:
            return
        updated: list[RectROI] = []
        for roi in self.rois:
            if roi.roi_id == roi_id:
                updated.append(
                    RectROI(
                        roi_id=roi.roi_id,
                        name=new_name,
                        x1=roi.x1,
                        y1=roi.y1,
                        x2=roi.x2,
                        y2=roi.y2,
                        enabled=roi.enabled,
                        color=roi.color,
                        dwell_sec=roi.dwell_sec,
                        audio_id=roi.audio_id,
                        alarm_enabled=roi.alarm_enabled,
                    )
                )
            else:
                updated.append(roi)
        self.rois = updated
        self.video.set_rois(self.rois)
        self._refresh_roi_list(select_id=roi_id)

    def _delete_selected_roi(self) -> None:
        item = self.roi_list.currentItem()
        if item is None:
            return
        roi_id = int(item.data(Qt.ItemDataRole.UserRole))
        self.rois = [roi for roi in self.rois if roi.roi_id != roi_id]
        self.video.selected_roi_id = None
        self.video.set_rois(self.rois)
        self._refresh_roi_list()

    def start_worker(self) -> None:
        if self.worker and self.worker.isRunning():
            self.append_log("检测线程已在运行")
            return
        self.worker = UdpInferWorker(config=self.cfg, helper_script=self.helper_script)
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.log_message.connect(self.append_log)
        self.worker.error_message.connect(self._on_worker_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.last_frame_ts = time.time()
        self.last_stream_reopen_ts = 0.0
        self.last_worker_restart_ts = 0.0
        if not self.video_watchdog_timer.isActive():
            self.video_watchdog_timer.start()
        self.append_log(f"第二阶段增强 UI 已启动：UDP/{self.backend_label} + 跟踪 + 停留统计 + 热区 + 行为规则")
        QTimer.singleShot(1500, self.start_ctp_stream)

    def stop_worker(self) -> None:
        if self.video_watchdog_timer.isActive():
            self.video_watchdog_timer.stop()
        self._stop_ctp_stream()
        if self.worker:
            self.worker.stop()
            self.worker.wait(2500)
            self.worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._save_stats()
        self.append_log("已停止")

    def _on_worker_finished(self) -> None:
        self.worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.append_log("检测线程已停止")

    def _on_worker_error(self, msg: str) -> None:
        self.append_log("[ERROR] " + msg)
        QMessageBox.critical(self, "工作线程错误", msg)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_frame_ready(self, frame_bgr, detections: list[dict[str, Any]], fps: float) -> None:
        self.last_frame_ts = time.time()
        frame_h, frame_w = frame_bgr.shape[:2]
        self.last_frame_shape = (frame_h, frame_w)
        if not self.rois and self.cfg.auto_seed_default_rois:
            self.rois = default_scene_rois(frame_w, frame_h)
            self.video.set_rois(self.rois)
            self._refresh_roi_list()
            self.append_log("首次收到画面，已自动生成默认固定 ROI；请按真实笼子拖拽校准并保存。")

        now = time.time()
        primary_analysis = analyze_detection(
            detections,
            self.rois,
            frame_w,
            frame_h,
            distance_threshold=self.cfg.roi_distance_threshold,
            iou_threshold=self.cfg.roi_iou_threshold,
            max_regions=self.cfg.max_description_regions,
        )

        track_dicts = self.tracker.update(detections, now=now) if self.cfg.tracker_enabled else []
        analyses_by_track: dict[int, Any] = {}
        for tr in track_dicts:
            tid = int(tr.get("track_id", -1))
            tdet = {
                "label": tr.get("label", "hamster"),
                "class_id": tr.get("class_id", 0),
                "score": tr.get("score", 0.0),
                "bbox": tr.get("bbox", [0, 0, 0, 0]),
            }
            analyses_by_track[tid] = analyze_detection(
                [tdet],
                self.rois,
                frame_w,
                frame_h,
                distance_threshold=self.cfg.roi_distance_threshold,
                iou_threshold=self.cfg.roi_iou_threshold,
                max_regions=self.cfg.max_description_regions,
            )

        stats_summary = self.stats.update(track_dicts, analyses_by_track, now=now)
        main_track = track_dicts[0] if track_dicts else None
        main_runtime = None
        behaviors: list[str] = []
        if main_track:
            main_tid = int(main_track["track_id"])
            main_runtime = stats_summary.get("tracks", {}).get(main_tid)
            behaviors = self.rules.evaluate(main_runtime)

        description = generate_text(primary_analysis, active_track=main_runtime or main_track, behaviors=behaviors)
        hotspot = generate_hotspot_text(stats_summary.get("top_grids", []), stats_summary.get("top_rois", []))
        if hotspot:
            description = f"{description}\n{hotspot}"

        self._update_first_stage_panel(primary_analysis, fps)
        self._update_second_stage_panel(track_dicts, main_runtime, behaviors, stats_summary)
        self._maybe_log_behavior_events(main_runtime, behaviors, description)

        tags = list(primary_analysis.description_tags)
        tags.extend(behaviors)
        self.video.set_frame(
            frame_bgr,
            detections,
            tracks=track_dicts,
            analysis_text=description.splitlines()[0],
            analysis_tags=tags,
            heatmap=stats_summary.get("grid_dwell", {}),
        )
        self.txt_desc.setPlainText(description)
        self.last_description = description

    def _update_first_stage_panel(self, analysis, fps: float) -> None:
        self.lbl_detected.setText("是" if analysis.detected else "否")
        self.lbl_count.setText(str(analysis.count))
        self.lbl_bbox.setText(str(analysis.bbox) if analysis.bbox else "-")
        if analysis.center:
            self.lbl_center.setText(f"({analysis.center[0]:.1f}, {analysis.center[1]:.1f})")
        else:
            self.lbl_center.setText("-")
        self.lbl_grid.setText(analysis.grid_position)
        self.lbl_near.setText("、".join(analysis.near_regions) if analysis.near_regions else "-")
        self.lbl_inside.setText("、".join(analysis.inside_regions) if analysis.inside_regions else "-")
        self.lbl_fps.setText(f"{fps:.1f}")

    def _update_second_stage_panel(self, track_dicts, main_runtime, behaviors, stats_summary) -> None:
        ids = [str(t.get("track_id")) for t in track_dicts]
        self.lbl_active_ids.setText("、".join(ids) if ids else "-")
        if main_runtime:
            tid = main_runtime.get("track_id")
            self.lbl_main_track.setText(f"ID {tid}")
            self.lbl_track_age.setText(f"{float(main_runtime.get('age_sec', 0.0)):.1f} s")
            roi = main_runtime.get("current_roi") or "-"
            roi_dwell = float(main_runtime.get("current_roi_dwell_sec", 0.0))
            self.lbl_roi_dwell.setText(f"{roi} / {roi_dwell:.1f} s")
            grid = main_runtime.get("current_grid") or "-"
            grid_dwell = float(main_runtime.get("current_grid_dwell_sec", 0.0))
            self.lbl_grid_dwell.setText(f"{grid} / {grid_dwell:.1f} s")
        else:
            self.lbl_main_track.setText("-")
            self.lbl_track_age.setText("0.0 s")
            self.lbl_roi_dwell.setText("-")
            self.lbl_grid_dwell.setText("-")
        self.lbl_behaviors.setText("、".join(behaviors) if behaviors else "-")
        self.lbl_running.setText(f"{float(stats_summary.get('running_sec', 0.0)):.1f} s")
        self.lbl_top_grids.setText(self._format_top(stats_summary.get("top_grids", [])))
        self.lbl_top_rois.setText(self._format_top(stats_summary.get("top_rois", [])))

    @staticmethod
    def _format_top(items: list[tuple[str, float]]) -> str:
        if not items:
            return "-"
        return "；".join([f"{name}: {sec:.1f}s" for name, sec in items])

    def _maybe_log_behavior_events(self, runtime: dict[str, Any] | None, behaviors: list[str], description: str) -> None:
        if not runtime or not behaviors:
            return
        tid = runtime.get("track_id")
        now = time.time()
        for behavior in behaviors:
            key = f"{tid}:{behavior}"
            last = self._last_event_key_ts.get(key, 0.0)
            if now - last < 5.0:
                continue
            self._last_event_key_ts[key] = now
            event = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "track_id": tid,
                "behavior": behavior,
                "runtime": runtime,
                "description": description,
            }
            self._append_event_jsonl(event)
            self.append_log(f"行为事件：ID {tid} {behavior}")

    def _append_event_jsonl(self, event: dict[str, Any]) -> None:
        p = self.cfg.event_log_path
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _reset_stats(self) -> None:
        self.tracker.reset()
        self.stats.reset()
        self._last_event_key_ts.clear()
        self.append_log("已重置跟踪器和统计数据")

    def _save_stats(self) -> None:
        try:
            self.stats.save_json(self.cfg.stats_json_path)
            self.append_log(f"统计数据已保存到 {self.cfg.stats_json_path}")
        except Exception as exc:
            self.append_log(f"统计保存失败：{exc}")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_worker()
        self._save_rois()
        event.accept()
