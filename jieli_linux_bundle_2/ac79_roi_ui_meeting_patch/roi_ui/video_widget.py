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
        self.edit_mode = False
        self.dragging = False
        self.drag_start: Optional[QPoint] = None
        self.drag_end: Optional[QPoint] = None
        self._mapping: Optional[DisplayMapping] = None

        self.room_overlay_visible = False
        self.room_overlay_text = ""
        self.room_overlay_color = QColor(0, 255, 0)
        self.room_overlay_font_size = 12

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

        for _, score, box in self.detections:
            x1, y1, x2, y2 = box
            dx1, dy1 = self._frame_to_display(x1, y1)
            dx2, dy2 = self._frame_to_display(x2, y2)
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.drawRect(QRect(dx1, dy1, dx2 - dx1, dy2 - dy1))
            bm = DwellTracker.bottom_midpoint(box)
            bx, by = self._frame_to_display(*bm)
            painter.setBrush(QColor(255, 0, 0))
            painter.drawEllipse(QPoint(bx, by), 4, 4)
            painter.setPen(QColor(0, 255, 0))
            painter.drawText(dx1 + 4, max(16, dy1 - 6), f'person {score:.2f}')

        for roi in self.rois:
            n = roi.normalized()
            dx1, dy1 = self._frame_to_display(n.x1, n.y1)
            dx2, dy2 = self._frame_to_display(n.x2, n.y2)
            st = self.roi_status.get(n.roi_id)
            color = QColor(*n.color)
            if st and st.alarmed:
                color = QColor(255, 60, 60)
            elif st and st.active:
                color = QColor(255, 180, 0)
            painter.setPen(QPen(color, 2, Qt.PenStyle.DashLine if self.edit_mode else Qt.PenStyle.SolidLine))
            painter.drawRect(QRect(dx1, dy1, dx2 - dx1, dy2 - dy1))
            dwell = f'{st.dwell_time:.1f}s' if st else '0.0s'
            text = f'{n.name} | 阈值 {n.dwell_sec:.1f}s | 当前 {dwell}'
            painter.setPen(color)
            painter.drawText(dx1 + 4, dy1 + 18, text)
            if st and st.last_detection_bottom:
                bx, by = self._frame_to_display(*st.last_detection_bottom)
                painter.setBrush(color)
                painter.drawEllipse(QPoint(bx, by), 5, 5)

        if self.edit_mode and self.dragging and self.drag_start and self.drag_end:
            painter.setPen(QPen(QColor(0, 200, 255), 2, Qt.PenStyle.DashLine))
            rect = QRect(self.drag_start, self.drag_end).normalized()
            painter.drawRect(rect)
