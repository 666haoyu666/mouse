from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
from PySide6.QtCore import QTime, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QDoubleSpinBox,
    QSplitter,
    QTextEdit,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from .config import AppConfig
from .dwell import DwellTracker
from .roi_model import RectROI, load_roi_bundle, save_rois
from .video_widget import VideoCanvas
from .worker import UdpInferWorker


@dataclass
class MeetingRoomState:
    in_use: bool = False
    started_at: float | None = None
    started_in_work_time: bool = True
    last_presence_at: float | None = None
    long_warned: bool = False
    abnormal_warned: bool = False

    def reset(self) -> None:
        self.in_use = False
        self.started_at = None
        self.started_in_work_time = True
        self.last_presence_at = None
        self.long_warned = False
        self.abnormal_warned = False


class WorkPeriodDialog(QDialog):
    def __init__(self, parent, start_text: str, end_text: str) -> None:
        super().__init__(parent)
        self.setWindowTitle('设置会议室工作时间')

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.edit_start = QTimeEdit()
        self.edit_start.setDisplayFormat('HH:mm')
        start_time = QTime.fromString(start_text, 'HH:mm')
        self.edit_start.setTime(start_time if start_time.isValid() else QTime(9, 0))

        self.edit_end = QTimeEdit()
        self.edit_end.setDisplayFormat('HH:mm')
        end_time = QTime.fromString(end_text, 'HH:mm')
        self.edit_end.setTime(end_time if end_time.isValid() else QTime(18, 0))

        form.addRow('开始时间', self.edit_start)
        form.addRow('结束时间', self.edit_end)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_period(self) -> tuple[str, str]:
        return self.edit_start.time().toString('HH:mm'), self.edit_end.time().toString('HH:mm')


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig, helper_script: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.helper_script = helper_script
        self.worker: UdpInferWorker | None = None
        self.ctp_proc: subprocess.Popen | None = None
        self.rois: List[RectROI] = []
        self.frame_size = None
        self.last_frame = None
        self.dwell_tracker = DwellTracker(absence_reset_sec=1.0)

        self.mode = config.ui_mode if config.ui_mode in {'default', 'meeting'} else 'default'
        self._changing_mode = False
        self.work_start = config.meeting_default_work_start
        self.work_end = config.meeting_default_work_end
        self.meeting_state = MeetingRoomState()

        self.release_audio_timer = QTimer(self)
        self.release_audio_timer.setSingleShot(True)
        self.release_audio_timer.timeout.connect(lambda: self._play_audio(4, '会议室无人，请关闭设备'))

        self.setWindowTitle('AC79 ROI UI - UDP + RKNN')
        screen = QApplication.primaryScreen()
        if screen:
            g = screen.availableGeometry()
            self.resize(min(1280, int(g.width() * 0.95)), min(720, int(g.height() * 0.92)))
        else:
            self.resize(1280, 720)

        self._build_ui()
        self._load_default_rois()
        self._sync_mode_ui(initial=True)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        splitter = QSplitter()
        root.addWidget(splitter)

        self.canvas = VideoCanvas()
        self.canvas.roi_created.connect(self.on_roi_created)
        splitter.addWidget(self.canvas)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)

        top_buttons = QGridLayout()
        self.btn_start = QPushButton('启动')
        self.btn_stop = QPushButton('停止')
        self.btn_edit = QPushButton('进入 ROI 编辑')
        self.btn_save = QPushButton('保存 ROI 组')
        self.btn_load = QPushButton('加载 ROI 组')
        self.btn_clear = QPushButton('清空 ROI')
        for i, btn in enumerate([self.btn_start, self.btn_stop, self.btn_edit, self.btn_save, self.btn_load, self.btn_clear]):
            top_buttons.addWidget(btn, i // 2, i % 2)
        side_layout.addLayout(top_buttons)

        self.btn_start.clicked.connect(self.start_worker)
        self.btn_stop.clicked.connect(self.stop_worker)
        self.btn_edit.clicked.connect(self.toggle_edit)
        self.btn_save.clicked.connect(self.save_rois_dialog)
        self.btn_load.clicked.connect(self.load_rois_dialog)
        self.btn_clear.clicked.connect(self.clear_rois)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem('默认模式', 'default')
        self.mode_combo.addItem('会议室模式', 'meeting')
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        side_layout.addWidget(QLabel('工作模式'))
        side_layout.addWidget(self.mode_combo)

        self.work_time_label = QLabel('工作时间：—')
        side_layout.addWidget(self.work_time_label)

        self.roi_list = QListWidget()
        side_layout.addWidget(QLabel('ROI 列表'))
        side_layout.addWidget(self.roi_list)
        self.roi_list.currentRowChanged.connect(self.on_roi_selected)

        form = QFormLayout()
        self.edit_name = QLineEdit()
        self.spin_dwell = QDoubleSpinBox()
        self.spin_dwell.setRange(0.5, 3600)
        self.spin_dwell.setDecimals(1)
        self.spin_dwell.setValue(10.0)

        self.label_audio = QLabel('报警音频')
        self.combo_audio = QComboBox()
        for i in range(1, 7):
            self.combo_audio.addItem(f'音频 {i} (sd:{i})', i)

        form.addRow('名称', self.edit_name)
        form.addRow('驻留阈值(秒)', self.spin_dwell)
        form.addRow(self.label_audio, self.combo_audio)
        side_layout.addLayout(form)

        row2 = QHBoxLayout()
        self.btn_apply = QPushButton('应用到选中 ROI')
        self.btn_delete = QPushButton('删除选中 ROI')
        row2.addWidget(self.btn_apply)
        row2.addWidget(self.btn_delete)
        side_layout.addLayout(row2)
        self.btn_apply.clicked.connect(self.apply_roi_edit)
        self.btn_delete.clicked.connect(self.delete_selected_roi)

        self.label_status = QLabel('状态：未启动')
        side_layout.addWidget(self.label_status)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        side_layout.addWidget(self.log_text)

    def log(self, msg: str) -> None:
        stamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f'[{stamp}] {msg}')

    def _send_ctp_line(self, line: str) -> bool:
        try:
            if self.ctp_proc and self.ctp_proc.poll() is None and self.ctp_proc.stdin:
                self.ctp_proc.stdin.write(line if line.endswith('\n') else line + '\n')
                self.ctp_proc.stdin.flush()
                return True
        except Exception as e:
            self.log(f'发送 CTP 命令失败: {e}')
        return False

    def _play_audio(self, audio_id: int, reason: str) -> None:
        ok = self._send_ctp_line(f'sd:{audio_id}')
        if ok:
            self.log(f'已发送杰理音频命令: sd:{audio_id} | {reason}')
        else:
            self.log(f'CTP 未连接，无法发送 sd:{audio_id} | {reason}')

    def _apply_canvas_room_overlay(self) -> None:
        if self.mode != 'meeting':
            self.canvas.set_room_overlay(False, '', (0, 255, 0))
            return
        font_size = self.btn_start.font().pointSize() or 12
        if self.meeting_state.in_use:
            self.canvas.set_room_overlay(True, self.config.meeting_busy_text, (255, 0, 0), font_size=font_size)
        else:
            self.canvas.set_room_overlay(True, self.config.meeting_idle_text, (0, 255, 0), font_size=font_size)

    def _sync_mode_ui(self, initial: bool = False) -> None:
        self._changing_mode = True
        self.mode_combo.setCurrentIndex(1 if self.mode == 'meeting' else 0)
        self._changing_mode = False

        show_audio = self.mode == 'default'
        self.label_audio.setVisible(show_audio)
        self.combo_audio.setVisible(show_audio)
        self.work_time_label.setVisible(self.mode == 'meeting')
        self.work_time_label.setText(f'工作时间：{self.work_start} - {self.work_end}' if self.mode == 'meeting' else '工作时间：—')
        self.refresh_roi_list()
        self._apply_canvas_room_overlay()

    def on_mode_changed(self, index: int) -> None:
        if self._changing_mode:
            return
        new_mode = self.mode_combo.currentData()
        if new_mode == self.mode:
            return
        if new_mode == 'meeting':
            dialog = WorkPeriodDialog(self, self.work_start, self.work_end)
            if dialog.exec() != int(QDialog.DialogCode.Accepted):
                self._changing_mode = True
                self.mode_combo.setCurrentIndex(0 if self.mode == 'default' else 1)
                self._changing_mode = False
                return
            self.work_start, self.work_end = dialog.get_period()
            self.meeting_state.reset()
        else:
            self.release_audio_timer.stop()
            self.meeting_state.reset()
        self.mode = new_mode
        self._sync_mode_ui()

    def _load_default_rois(self) -> None:
        bundle = load_roi_bundle(self.config.roi_json_path)
        self.rois = bundle.get('rois', [])
        self.refresh_roi_list()
        self.canvas.set_rois(self.rois)
        if self.rois:
            self.log(f'已加载默认 ROI 组: {self.config.roi_json_path}')

    def refresh_roi_list(self) -> None:
        self.roi_list.clear()
        for roi in self.rois:
            if self.mode == 'meeting':
                text = f'#{roi.roi_id} {roi.name} | 会议室 ROI | 阈值 {roi.dwell_sec:.1f}s'
            else:
                text = f'#{roi.roi_id} {roi.name} | 阈值 {roi.dwell_sec:.1f}s | 音频 sd:{getattr(roi, "audio_id", 1)}'
            self.roi_list.addItem(QListWidgetItem(text))

    def start_worker(self) -> None:
        if self.worker and self.worker.isRunning():
            self.log('工作线程已在运行')
            return

        self.worker = UdpInferWorker(self.config, helper_script=self.helper_script)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.log_message.connect(self.log)
        self.worker.error_message.connect(self.on_worker_error)
        self.worker.start()

        self.label_status.setText('状态：运行中')
        self.log('UDP/RKNN 工作线程启动中...')
        QTimer.singleShot(1500, self.start_ctp_stream)

    def start_ctp_stream(self) -> None:
        if self.ctp_proc and self.ctp_proc.poll() is None:
            self.log('CTP 已在运行，不重复启动')
            return

        ctp_script = Path(__file__).resolve().parent.parent / 'jieli_min_ctp_client.py'
        if not ctp_script.exists():
            self.log(f'CTP 脚本不存在: {ctp_script}')
            return

        host = self.config.device_ip.strip() or '192.168.1.1'
        try:
            log_path = Path('./roi_ui_output/ctp_auto.log')
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open('a', encoding='utf-8')

            self.ctp_proc = subprocess.Popen(
                [sys.executable, str(ctp_script), '--host', host],
                stdin=subprocess.PIPE,
                stdout=log_file,
                stderr=log_file,
                text=True,
                bufsize=1,
            )

            for cmd in ['app\n', 'date\n', 'open 640 480 20 8000 0\n']:
                if self.ctp_proc.stdin:
                    self.ctp_proc.stdin.write(cmd)
                    self.ctp_proc.stdin.flush()
                    time.sleep(0.3)

            self.log(f'已自动发送 CTP 开流命令: host={host}, open 640x480 fps=20 format=0')
            self.log(f'CTP 日志: {log_path}')
        except Exception as e:
            self.log(f'自动启动 CTP 失败: {e}')

    def stop_worker(self) -> None:
        self.release_audio_timer.stop()

        if self.ctp_proc and self.ctp_proc.poll() is None:
            try:
                self._send_ctp_line('quit')
                self.ctp_proc.terminate()
                self.ctp_proc.wait(timeout=2)
            except Exception:
                try:
                    self.ctp_proc.kill()
                except Exception:
                    pass
            self.ctp_proc = None
            self.log('CTP 已停止')

        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None

        self.label_status.setText('状态：已停止')
        self.log('已停止')

    def toggle_edit(self) -> None:
        enabled = not self.canvas.edit_mode
        self.canvas.set_edit_mode(enabled)
        self.btn_edit.setText('退出 ROI 编辑' if enabled else '进入 ROI 编辑')
        self.log('ROI 编辑模式已开启' if enabled else 'ROI 编辑模式已关闭')

    def _next_roi_id(self) -> int:
        return max([r.roi_id for r in self.rois], default=0) + 1

    def on_roi_created(self, roi: RectROI) -> None:
        roi.roi_id = self._next_roi_id()
        roi.name = f'roi_{roi.roi_id}'

        if self.mode == 'default':
            text, ok = QInputDialog.getItem(
                self,
                '选择 ROI 报警音频',
                f'为 {roi.name} 选择报警音频：',
                [str(i) for i in range(1, 7)],
                0,
                False,
            )
            if not ok:
                self.log('已取消创建 ROI')
                return
            roi.audio_id = int(text)
        else:
            roi.audio_id = 1

        self.rois.append(roi)
        self.refresh_roi_list()
        self.canvas.set_rois(self.rois)
        self.log(f'新增 ROI: {roi.name} ({roi.x1},{roi.y1})-({roi.x2},{roi.y2})')

    def on_roi_selected(self, row: int) -> None:
        if row < 0 or row >= len(self.rois):
            return
        roi = self.rois[row]
        self.edit_name.setText(roi.name)
        self.spin_dwell.setValue(float(roi.dwell_sec))
        idx = self.combo_audio.findData(int(getattr(roi, 'audio_id', 1)))
        if idx >= 0:
            self.combo_audio.setCurrentIndex(idx)

    def apply_roi_edit(self) -> None:
        row = self.roi_list.currentRow()
        if row < 0 or row >= len(self.rois):
            QMessageBox.information(self, '提示', '请先选择一个 ROI')
            return
        roi = self.rois[row]
        roi.name = self.edit_name.text().strip() or roi.name
        roi.dwell_sec = float(self.spin_dwell.value())
        if self.mode == 'default':
            roi.audio_id = int(self.combo_audio.currentData() or 1)
        self.refresh_roi_list()
        self.canvas.set_rois(self.rois)
        self.log(f'已更新 ROI: {roi.name}')

    def _meeting_block_roi_delete(self) -> bool:
        if self.mode == 'meeting' and self.meeting_state.in_use:
            QMessageBox.information(self, '提示', '会议室正在使用中：已禁止删除/清空已有 ROI，但仍可继续新增 ROI。')
            return True
        return False

    def delete_selected_roi(self) -> None:
        if self._meeting_block_roi_delete():
            return
        row = self.roi_list.currentRow()
        if row < 0 or row >= len(self.rois):
            return
        roi = self.rois.pop(row)
        self.refresh_roi_list()
        self.canvas.set_rois(self.rois)
        self.log(f'已删除 ROI: {roi.name}')

    def clear_rois(self) -> None:
        if self._meeting_block_roi_delete():
            return
        self.rois.clear()
        self.refresh_roi_list()
        self.canvas.set_rois(self.rois)
        self.log('已清空 ROI')

    def save_rois_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, '保存 ROI 组', str(self.config.roi_json_path), 'JSON Files (*.json)')
        if not path:
            return
        frame_size = self.frame_size if self.frame_size else None
        extra_meta = {
            'mode': self.mode,
            'meeting_work_start': self.work_start if self.mode == 'meeting' else None,
            'meeting_work_end': self.work_end if self.mode == 'meeting' else None,
        }
        save_rois(Path(path), self.rois, frame_size=frame_size, group_name=Path(path).stem, extra_meta=extra_meta)
        self.log(f'ROI 组已保存到: {path}')

    def load_rois_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, '加载 ROI 组', str(self.config.roi_json_path), 'JSON Files (*.json)')
        if not path:
            return
        bundle = load_roi_bundle(Path(path))
        self.rois = bundle.get('rois', [])
        if self.mode == 'meeting':
            self.work_start = bundle.get('meeting_work_start') or self.work_start
            self.work_end = bundle.get('meeting_work_end') or self.work_end
            self.work_time_label.setText(f'工作时间：{self.work_start} - {self.work_end}')
        self.refresh_roi_list()
        self.canvas.set_rois(self.rois)
        self.log(f'ROI 组已加载: {path}')

    def _save_alarm_snapshot(self, frame_bgr, roi: RectROI, dwell_time: float) -> Path:
        self.config.screenshot_dir_path.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = self.config.screenshot_dir_path / f'alarm_{roi.name}_{stamp}.jpg'
        vis = frame_bgr.copy()
        n = roi.normalized()
        cv2.rectangle(vis, (n.x1, n.y1), (n.x2, n.y2), (0, 0, 255), 3)
        cv2.putText(vis, f'ALARM {roi.name} {dwell_time:.1f}s', (max(10, n.x1), max(30, n.y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imwrite(str(path), vis)
        return path

    def _write_event_log(self, event_name: str, payload: dict) -> None:
        try:
            self.config.event_log_path.parent.mkdir(parents=True, exist_ok=True)
            event = {
                'time': datetime.now().isoformat(timespec='seconds'),
                'event': event_name,
            }
            event.update(payload)
            with self.config.event_log_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
        except Exception as e:
            self.log(f'事件日志写入失败: {e}')

    def _trigger_default_alarm(self, roi: RectROI, dwell_time: float, det_count: int) -> None:
        QApplication.beep()
        if self.last_frame is not None:
            shot = self._save_alarm_snapshot(self.last_frame, roi, dwell_time)
            self.log(f'报警截图已保存: {shot}')
        self._write_event_log('roi_dwell_alarm', {
            'roi_id': roi.roi_id,
            'roi_name': roi.name,
            'dwell_time': round(float(dwell_time), 3),
            'threshold': float(roi.dwell_sec),
            'det_count': int(det_count),
            'audio_id': int(getattr(roi, 'audio_id', 1)),
        })
        self._play_audio(int(getattr(roi, 'audio_id', 1)), f'默认模式 ROI 报警: {roi.name}')
        if self.config.alarm_cmd.strip():
            try:
                subprocess.Popen(self.config.alarm_cmd, shell=True)
                self.log(f'已执行报警命令: {self.config.alarm_cmd}')
            except Exception as e:
                self.log(f'报警命令执行失败: {e}')

    def _time_in_work_period(self, dt: datetime) -> bool:
        start = QTime.fromString(self.work_start, 'HH:mm')
        end = QTime.fromString(self.work_end, 'HH:mm')
        now_t = QTime(dt.hour, dt.minute, dt.second)
        if not start.isValid() or not end.isValid():
            return True
        if start <= end:
            return start <= now_t < end
        return now_t >= start or now_t < end

    def _meeting_start_room(self, now_ts: float, started_at: float, in_work: bool, trigger_roi: RectROI | None) -> None:
        if self.release_audio_timer.isActive():
            self.release_audio_timer.stop()
        self.meeting_state.in_use = True
        self.meeting_state.started_at = started_at
        self.meeting_state.started_in_work_time = in_work
        self.meeting_state.last_presence_at = now_ts
        self.meeting_state.long_warned = False
        self.meeting_state.abnormal_warned = False
        self._apply_canvas_room_overlay()

        if in_work:
            self._write_event_log('meeting_room_started', {
                'started_at_ts': round(float(started_at), 3),
                'trigger': 'work_time',
                'roi_name': trigger_roi.name if trigger_roi else None,
                'work_period': f'{self.work_start}-{self.work_end}',
            })
            self._play_audio(1, '会议室已开始使用')
        else:
            self._write_event_log('meeting_room_nonwork_started', {
                'started_at_ts': round(float(started_at), 3),
                'trigger': 'non_work_time',
                'roi_name': trigger_roi.name if trigger_roi else None,
                'work_period': f'{self.work_start}-{self.work_end}',
            })
            self._play_audio(5, '当前为非工作时间，请尽快离开会议室')

    def _meeting_release_room(self, now_ts: float) -> None:
        if not self.meeting_state.in_use:
            return
        occupied_for = max(0.0, now_ts - (self.meeting_state.started_at or now_ts))
        self._write_event_log('meeting_room_released', {
            'occupied_for_sec': round(occupied_for, 3),
            'work_period': f'{self.work_start}-{self.work_end}',
        })
        self._play_audio(2, '会议室已空闲')
        gap_ms = int(self.config.meeting_release_audio_gap_sec * 1000)
        self.release_audio_timer.start(max(0, gap_ms))
        self.meeting_state.reset()
        self._apply_canvas_room_overlay()

    def _update_meeting_mode(self, status: dict, detections, now_ts: float) -> None:
        active_items = []
        for roi in self.rois:
            st = status.get(roi.roi_id)
            if st and st.active:
                active_items.append((roi, st))

        any_active = bool(active_items)
        if any_active:
            self.meeting_state.last_presence_at = now_ts

        in_work = self._time_in_work_period(datetime.now())

        if not self.meeting_state.in_use:
            for roi, st in active_items:
                if st.dwell_time >= self.config.meeting_use_start_sec:
                    started_at = st.entered_at or (now_ts - st.dwell_time)
                    self._meeting_start_room(now_ts, started_at, in_work, roi)
                    break
        else:
            started_at = self.meeting_state.started_at or now_ts
            occupied_for = max(0.0, now_ts - started_at)

            if any_active:
                self.meeting_state.last_presence_at = now_ts
                if self.meeting_state.started_in_work_time:
                    if (not self.meeting_state.long_warned) and occupied_for >= self.config.meeting_long_use_sec:
                        self.meeting_state.long_warned = True
                        self._write_event_log('meeting_room_long_use', {
                            'occupied_for_sec': round(occupied_for, 3),
                            'threshold_sec': self.config.meeting_long_use_sec,
                        })
                        self._play_audio(3, '当前会议室已被长时间占用，请注意使用时长')
                    abnormal_total = self.config.meeting_long_use_sec + self.config.meeting_abnormal_extra_sec
                else:
                    abnormal_total = self.config.meeting_use_start_sec + self.config.meeting_abnormal_extra_sec

                if (not self.meeting_state.abnormal_warned) and occupied_for >= abnormal_total:
                    self.meeting_state.abnormal_warned = True
                    self._write_event_log('meeting_room_abnormal_use', {
                        'occupied_for_sec': round(occupied_for, 3),
                        'threshold_sec': abnormal_total,
                        'started_in_work_time': self.meeting_state.started_in_work_time,
                    })
                    self._play_audio(6, '异常占用持续，请管理员介入')
            else:
                last_presence_at = self.meeting_state.last_presence_at or now_ts
                if now_ts - last_presence_at >= self.config.meeting_release_empty_sec:
                    self._meeting_release_room(now_ts)

        self._apply_canvas_room_overlay()

    def on_frame_ready(self, frame_bgr, detections, fps: float) -> None:
        self.last_frame = frame_bgr.copy()
        self.frame_size = (frame_bgr.shape[1], frame_bgr.shape[0])
        now_ts = time.time()
        status = self.dwell_tracker.update(self.rois, detections, now=now_ts)

        if self.mode == 'default':
            for roi in self.rois:
                st = status.get(roi.roi_id)
                if not st:
                    continue
                if roi.alarm_enabled and st.active and not st.alarmed and st.dwell_time >= roi.dwell_sec:
                    st.alarmed = True
                    self.log(f'触发报警: {roi.name}, 驻留 {st.dwell_time:.1f}s')
                    self._trigger_default_alarm(roi, st.dwell_time, len(detections))
        else:
            self._update_meeting_mode(status, detections, now_ts)

        self.canvas.set_roi_status(status)
        self.canvas.set_frame(frame_bgr, detections)
        det_in_roi = sum(1 for st in status.values() if st.active)
        self.label_status.setText(f'状态：运行中 | 模式 {self.mode} | FPS {fps:.1f} | DET {len(detections)} | ROI_ACTIVE {det_in_roi}')

    def on_worker_error(self, msg: str) -> None:
        self.log(msg)
        QMessageBox.critical(self, '工作线程错误', msg)

    def closeEvent(self, event) -> None:
        self.stop_worker()
        super().closeEvent(event)
