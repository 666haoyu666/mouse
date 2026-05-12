from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from .dwell import DwellTracker
from .roi_model import RectROI

Detection = Tuple[int, float, Tuple[int, int, int, int]]


@dataclass
class DisplayMapping:
    x: int
    y: int
    w: int
    h: int
    scale_x: float
    scale_y: float


class VideoCanvas(QWidget):
    roi_created = Signal(object)
    log_message = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.frame_bgr = None
        self.qimage: Optional[QImage] = None
        self.detections: List[Detection] = []
        self.rois: List[RectROI] = []
        self.roi_status = {}
        self.show_roi_threshold = True
        self.edit_mode = False
        self.dragging = False
        self.drag_start: Optional[QPoint] = None
        self.drag_end: Optional[QPoint] = None
        self._mapping: Optional[DisplayMapping] = None

        self.room_overlay_visible = False
        self.room_overlay_text = ""
        self.room_overlay_color = QColor(0, 255, 0)
        self.room_overlay_font_size = 12

    def set_show_roi_threshold(self, enabled: bool) -> None:
        self.show_roi_threshold = bool(enabled)
        self.update()

    def set_edit_mode(self, enabled: bool) -> None:
        self.edit_mode = enabled
        self.update()

    def set_rois(self, rois: List[RectROI]) -> None:
        self.rois = list(rois)
        self.update()

    def set_roi_status(self, status: dict) -> None:
        self.roi_status = status
        self.update()

    def set_room_overlay(self, visible: bool, text: str, color: Tuple[int, int, int], font_size: int | None = None) -> None:
        self.room_overlay_visible = bool(visible)
        self.room_overlay_text = text
        self.room_overlay_color = QColor(*color)
        if font_size is not None and font_size > 0:
            self.room_overlay_font_size = int(font_size)
        self.update()

    def set_frame(self, frame_bgr, detections: List[Detection]) -> None:
        self.frame_bgr = frame_bgr
        self.detections = detections
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        self.qimage = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        self.update()

    def _calc_mapping(self) -> Optional[DisplayMapping]:
        if self.qimage is None:
            return None
        iw, ih = self.qimage.width(), self.qimage.height()
        ww, wh = self.width(), self.height()
        if iw <= 0 or ih <= 0 or ww <= 0 or wh <= 0:
            return None
        scale = min(ww / iw, wh / ih)
        draw_w = int(iw * scale)
        draw_h = int(ih * scale)
        off_x = (ww - draw_w) // 2
        off_y = (wh - draw_h) // 2
        return DisplayMapping(off_x, off_y, draw_w, draw_h, iw / draw_w, ih / draw_h)

    def _display_to_frame(self, p: QPoint) -> Optional[Tuple[int, int]]:
        m = self._mapping
        if m is None:
            return None
        if not (m.x <= p.x() <= m.x + m.w and m.y <= p.y() <= m.y + m.h):
            return None
        fx = int((p.x() - m.x) * m.scale_x)
        fy = int((p.y() - m.y) * m.scale_y)
        return fx, fy

    def _frame_to_display(self, x: int, y: int) -> Tuple[int, int]:
        m = self._mapping
        if m is None:
            return x, y
        dx = int(m.x + x / m.scale_x)
        dy = int(m.y + y / m.scale_y)
        return dx, dy

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.edit_mode and event.button() == Qt.MouseButton.LeftButton and self.qimage is not None:
            self.dragging = True
            self.drag_start = event.position().toPoint()
            self.drag_end = self.drag_start
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.edit_mode and self.dragging:
            self.drag_end = event.position().toPoint()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self.edit_mode and self.dragging and self.drag_start and self.drag_end:
            self.dragging = False
            a = self._display_to_frame(self.drag_start)
            b = self._display_to_frame(self.drag_end)
            if a and b:
                x1, y1 = a
                x2, y2 = b
                if abs(x2 - x1) >= 8 and abs(y2 - y1) >= 8:
                    roi = RectROI(roi_id=-1, name='new_roi', x1=x1, y1=y1, x2=x2, y2=y2)
                    self.roi_created.emit(roi.normalized())
            self.drag_start = None
            self.drag_end = None
            self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 20))
        if self.qimage is None:
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, '等待 UDP 视频流...')
            return
        self._mapping = self._calc_mapping()
        m = self._mapping
        if m is None:
            return
        pix = QPixmap.fromImage(self.qimage).scaled(m.w, m.h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
        painter.drawPixmap(m.x, m.y, pix)
        self._draw_overlays(painter)

    def _draw_room_overlay(self, painter: QPainter) -> None:
        if not self.room_overlay_visible or not self.room_overlay_text or self._mapping is None:
            return
        
        m = self._mapping
        font = QFont(painter.font())
        font.setPointSize(self.room_overlay_font_size)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(self.room_overlay_color)
        painter.drawText(m.x + 10, m.y + 26, self.room_overlay_text)

    def _draw_overlays(self, painter: QPainter) -> None:
        self._draw_room_overlay(painter)

        # =========================s
        # 颜色设计
        # =========================
        # 人物检测框：绿色边框 + 淡红色透明填充
        PERSON_BORDER = QColor(0, 255, 0)
        PERSON_FILL = QColor(255, 0, 0, 50)        # 约 20% 透明红色

        # ROI 默认状态：青色边框，不填充
        ROI_IDLE_BORDER = QColor(0, 220, 255)

        # ROI 内有人：橙色边框 + 透明橙色填充
        ROI_ACTIVE_BORDER = QColor(255, 180, 0)
        ROI_ACTIVE_FILL = QColor(255, 180, 0, 64)  # 约 25%

        # ROI 报警：红色边框 + 透明红色填充
        ROI_ALARM_BORDER = QColor(255, 40, 40)
        ROI_ALARM_FILL = QColor(255, 40, 40, 90)   # 约 35%

        # =========================
        # 1. 绘制人物检测框
        # =========================
        for _, score, box in self.detections:
            x1, y1, x2, y2 = box

            dx1, dy1 = self._frame_to_display(x1, y1)
            dx2, dy2 = self._frame_to_display(x2, y2)

            det_rect = QRect(dx1, dy1, dx2 - dx1, dy2 - dy1).normalized()

            # 先画人物框透明填充
            painter.fillRect(det_rect, PERSON_FILL)

            # 再画人物边框。关键：画矩形前必须 NoBrush，否则会被实心填充
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(PERSON_BORDER, 2))
            painter.drawRect(det_rect)

            # 画检测框底边中点
            bm = DwellTracker.bottom_midpoint(box)
            bx, by = self._frame_to_display(*bm)

            painter.setBrush(PERSON_BORDER)
            painter.setPen(QPen(PERSON_BORDER, 1))
            painter.drawEllipse(QPoint(bx, by), 4, 4)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            # 画人物置信度文字
            painter.setPen(PERSON_BORDER)
            painter.drawText(
                det_rect.x() + 4,
                max(16, det_rect.y() - 6),
                f'person {score:.2f}'
            )

        # =========================
        # 2. 绘制 ROI 区域
        # =========================
        for roi in self.rois:
            n = roi.normalized()

            dx1, dy1 = self._frame_to_display(n.x1, n.y1)
            dx2, dy2 = self._frame_to_display(n.x2, n.y2)

            roi_rect = QRect(dx1, dy1, dx2 - dx1, dy2 - dy1).normalized()

            st = self.roi_status.get(n.roi_id)

            # 默认 ROI：青色边框，不填充
            border_color = ROI_IDLE_BORDER
            fill_color = None

            # ROI 报警：红色
            if st and st.alarmed:
                border_color = ROI_ALARM_BORDER
                fill_color = ROI_ALARM_FILL

            # ROI 内有人：橙色
            elif st and st.active:
                border_color = ROI_ACTIVE_BORDER
                fill_color = ROI_ACTIVE_FILL

            # 先画 ROI 透明填充
            if fill_color is not None:
                painter.fillRect(roi_rect, fill_color)

            # 再画 ROI 边框。关键：必须清空 brush
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(
                QPen(
                    border_color,
                    2,
                    Qt.PenStyle.DashLine if self.edit_mode else Qt.PenStyle.SolidLine
                )
            )
            painter.drawRect(roi_rect)

            # 画 ROI 文字
            dwell = f'{st.dwell_time:.1f}s' if st else '0.0s'

            if self.show_roi_threshold:
                text = f'{n.name} | 阈值 {n.dwell_sec:.1f}s | 当前 {dwell}'
            else:
                text = f'{n.name} | 当前 {dwell}'
                
            painter.setPen(border_color)
            painter.drawText(roi_rect.x() + 4, roi_rect.y() + 18, text)

            # 画 ROI 内检测到底边点
            if st and st.last_detection_bottom:
                bx, by = self._frame_to_display(*st.last_detection_bottom)

                painter.setBrush(border_color)
                painter.setPen(QPen(border_color, 1))
                painter.drawEllipse(QPoint(bx, by), 5, 5)
                painter.setBrush(Qt.BrushStyle.NoBrush)

        # =========================
        # 3. 鼠标拖拽创建 ROI 时的临时框
        # =========================
        if self.edit_mode and self.dragging and self.drag_start and self.drag_end:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(0, 200, 255), 2, Qt.PenStyle.DashLine))
            rect = QRect(self.drag_start, self.drag_end).normalized()
            painter.drawRect(rect)