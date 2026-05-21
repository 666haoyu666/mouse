from __future__ import annotations

from typing import Any
import cv2

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from .roi_model import RectROI


DET_COLOR = QColor(68, 214, 44)
GRID_COLOR = QColor(255, 255, 255, 110)
TEXT_BG = QColor(8, 18, 28, 185)
HEAT_COLOR = QColor(255, 193, 7, 65)


class VideoWidget(QWidget):
    rois_changed = Signal(object)
    roi_selected = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(720, 480)
        self.setMouseTracking(True)
        self._pixmap = QPixmap()
        self._frame_w = 0
        self._frame_h = 0
        self.detections: list[dict[str, Any]] = []
        self.tracks: list[dict[str, Any]] = []
        self.rois: list[RectROI] = []
        self.analysis_text = "等待视频输入"
        self.analysis_tags: list[str] = []
        self.heatmap: dict[str, float] = {}
        self.heatmap_overlay_enabled = False
        self.edit_mode = False
        self.grid_enabled = True
        self.grid_labels_enabled = True
        self.selected_roi_id: int | None = None
        self._drag_start: QPoint | None = None
        self._drag_mode: str | None = None
        self._drag_anchor_frame: tuple[int, int] | None = None
        self._moving_roi_origin: RectROI | None = None
        self._draft_rect: RectROI | None = None

    def set_grid_enabled(self, enabled: bool) -> None:
        self.grid_enabled = bool(enabled)
        self.update()

    def set_grid_labels_enabled(self, enabled: bool) -> None:
        self.grid_labels_enabled = bool(enabled)
        self.update()

    def set_edit_mode(self, enabled: bool) -> None:
        self.edit_mode = bool(enabled)
        self.update()

    def set_heatmap_overlay_enabled(self, enabled: bool) -> None:
        self.heatmap_overlay_enabled = bool(enabled)
        self.update()

    def set_rois(self, rois: list[RectROI]) -> None:
        self.rois = [r.normalized() for r in rois]
        self.update()

    def set_analysis(self, text: str, tags: list[str]) -> None:
        self.analysis_text = text
        self.analysis_tags = tags
        self.update()

    def set_frame(
        self,
        frame_bgr,
        detections: list[dict[str, Any]],
        tracks: list[dict[str, Any]] | None = None,
        analysis_text: str = "",
        analysis_tags: list[str] | None = None,
        heatmap: dict[str, float] | None = None,
    ) -> None:
        if frame_bgr is None:
            return
        h, w = frame_bgr.shape[:2]
        self._frame_w = int(w)
        self._frame_h = int(h)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self.detections = detections or []
        self.tracks = tracks or []
        if analysis_text:
            self.analysis_text = analysis_text
        self.analysis_tags = analysis_tags or []
        self.heatmap = heatmap or {}
        self.update()

    def _view_rect(self) -> QRect:
        if self._pixmap.isNull() or self._frame_w <= 0 or self._frame_h <= 0:
            return self.rect()
        area = self.rect()
        img_ratio = self._frame_w / max(1, self._frame_h)
        widget_ratio = area.width() / max(1, area.height())
        if widget_ratio > img_ratio:
            h = area.height()
            w = int(h * img_ratio)
            x = area.x() + (area.width() - w) // 2
            y = area.y()
        else:
            w = area.width()
            h = int(w / img_ratio)
            x = area.x()
            y = area.y() + (area.height() - h) // 2
        return QRect(x, y, w, h)

    def _frame_to_display(self, x: float, y: float) -> QPoint:
        r = self._view_rect()
        sx = r.width() / max(1, self._frame_w)
        sy = r.height() / max(1, self._frame_h)
        return QPoint(int(r.x() + x * sx), int(r.y() + y * sy))

    def _frame_rect_to_display(self, x1: float, y1: float, x2: float, y2: float) -> QRect:
        p1 = self._frame_to_display(x1, y1)
        p2 = self._frame_to_display(x2, y2)
        return QRect(p1, p2).normalized()

    def _display_to_frame(self, p: QPoint) -> tuple[int, int]:
        r = self._view_rect()
        if r.width() <= 0 or r.height() <= 0:
            return 0, 0
        x = (p.x() - r.x()) * self._frame_w / r.width()
        y = (p.y() - r.y()) * self._frame_h / r.height()
        return int(max(0, min(self._frame_w - 1, x))), int(max(0, min(self._frame_h - 1, y)))

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(24, 24, 24))
        if self._pixmap.isNull():
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "等待视频流 / Waiting for video")
            return

        view = self._view_rect()
        painter.drawPixmap(view, self._pixmap)

        if self.heatmap_overlay_enabled:
            self._draw_heatmap_overlay(painter)
        if self.grid_enabled:
            self._draw_grid(painter)
        self._draw_rois(painter)
        self._draw_detections(painter)
        self._draw_tracks(painter)
        if self._draft_rect:
            self._draw_single_roi(painter, self._draft_rect, QColor(255, 255, 255), dashed=True)
        self._draw_status_overlay(painter)

    def _draw_grid(self, painter: QPainter) -> None:
        view = self._view_rect()
        painter.setPen(QPen(GRID_COLOR, 1, Qt.PenStyle.DashLine))
        for i in (1, 2):
            x = view.x() + view.width() * i // 3
            y = view.y() + view.height() * i // 3
            painter.drawLine(x, view.y(), x, view.y() + view.height())
            painter.drawLine(view.x(), y, view.x() + view.width(), y)
        if not self.grid_labels_enabled:
            return
        labels = [["左上", "上中", "右上"], ["左中", "中央", "右中"], ["左下", "下中", "右下"]]
        painter.setFont(QFont("Microsoft YaHei", 10))
        painter.setPen(QColor(255, 255, 255, 185))
        for r in range(3):
            for c in range(3):
                cell = QRect(
                    view.x() + view.width() * c // 3,
                    view.y() + view.height() * r // 3,
                    view.width() // 3,
                    view.height() // 3,
                )
                painter.drawText(cell.adjusted(8, 6, -8, -6), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, labels[r][c])

    def _draw_heatmap_overlay(self, painter: QPainter) -> None:
        if not self.heatmap:
            return
        view = self._view_rect()
        max_v = max(float(v) for v in self.heatmap.values()) if self.heatmap else 0.0
        if max_v <= 0:
            return
        labels = [["左上", "上中", "右上"], ["左中", "中央", "右中"], ["左下", "下中", "右下"]]
        for r in range(3):
            for c in range(3):
                name = labels[r][c]
                v = float(self.heatmap.get(name, 0.0))
                if v <= 0:
                    continue
                alpha = int(25 + min(150, 150 * v / max_v))
                painter.fillRect(
                    QRect(
                        view.x() + view.width() * c // 3,
                        view.y() + view.height() * r // 3,
                        view.width() // 3,
                        view.height() // 3,
                    ),
                    QColor(255, 193, 7, alpha),
                )

    def _draw_rois(self, painter: QPainter) -> None:
        for roi in self.rois:
            color = QColor(roi.color if roi.color else "#40C4FF")
            self._draw_single_roi(painter, roi, color, dashed=False)

    def _draw_single_roi(self, painter: QPainter, roi: RectROI, color: QColor, dashed: bool = False) -> None:
        n = roi.normalized()
        rect = self._frame_rect_to_display(n.x1, n.y1, n.x2, n.y2)
        pen = QPen(color, 3 if n.roi_id == self.selected_roi_id else 2)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawRect(rect)
        label_rect = QRect(rect.x(), max(0, rect.y() - 24), max(100, rect.width()), 22)
        painter.fillRect(label_rect, QColor(0, 0, 0, 150))
        painter.setPen(color)
        painter.setFont(QFont("Microsoft YaHei", 10))
        painter.drawText(label_rect.adjusted(5, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter, n.name)

    def _draw_detections(self, painter: QPainter) -> None:
        painter.setFont(QFont("Arial", 10))
        for det in self.detections:
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            rect = self._frame_rect_to_display(x1, y1, x2, y2)
            painter.setPen(QPen(DET_COLOR, 2))
            painter.drawRect(rect)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            center = self._frame_to_display(cx, cy)
            painter.setBrush(DET_COLOR)
            painter.drawEllipse(center, 4, 4)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            text = f"{det.get('label', 'hamster')} {float(det.get('score', 0.0)):.2f}"
            tr_id = det.get("track_id")
            if tr_id is not None:
                text = f"ID {tr_id} {text}"
            text_rect = QRect(rect.x(), max(0, rect.y() - 24), max(150, rect.width()), 22)
            painter.fillRect(text_rect, QColor(0, 0, 0, 165))
            painter.setPen(DET_COLOR)
            painter.drawText(text_rect.adjusted(5, 0, -5, 0), Qt.AlignmentFlag.AlignVCenter, text)

    def _draw_tracks(self, painter: QPainter) -> None:
        painter.setFont(QFont("Arial", 10))
        for tr in self.tracks:
            trail = tr.get("trail") or []
            if len(trail) >= 2:
                painter.setPen(QPen(QColor(0, 255, 255, 180), 2))
                points = [self._frame_to_display(float(x), float(y)) for x, y in trail]
                for p1, p2 in zip(points[:-1], points[1:]):
                    painter.drawLine(p1, p2)
            center = tr.get("center")
            if center:
                p = self._frame_to_display(float(center[0]), float(center[1]))
                painter.setBrush(QColor(0, 255, 255, 180))
                painter.drawEllipse(p, 5, 5)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                info = f"ID {tr.get('track_id')} {float(tr.get('age_sec', 0.0)):.1f}s"
                painter.fillRect(QRect(p.x() + 8, p.y() - 10, 110, 22), QColor(0, 0, 0, 155))
                painter.setPen(QColor(0, 255, 255))
                painter.drawText(QRect(p.x() + 12, p.y() - 10, 105, 22), Qt.AlignmentFlag.AlignVCenter, info)

    def _draw_status_overlay(self, painter: QPainter) -> None:
        panel = QRect(12, 12, min(self.width() - 24, 680), 60)
        painter.fillRect(panel, TEXT_BG)
        painter.setFont(QFont("Microsoft YaHei", 10))
        painter.setPen(QColor(245, 245, 245))
        painter.drawText(panel.adjusted(12, 0, -12, -28), Qt.AlignmentFlag.AlignVCenter, self.analysis_text)
        if self.analysis_tags:
            painter.setPen(QColor(170, 220, 255))
            painter.drawText(panel.adjusted(12, 28, -12, 0), Qt.AlignmentFlag.AlignVCenter, " | ".join(self.analysis_tags[:5]))

    def _roi_at_point(self, p: QPoint) -> RectROI | None:
        fx, fy = self._display_to_frame(p)
        for roi in reversed(self.rois):
            if roi.normalized().contains_point(fx, fy):
                return roi.normalized()
        return None

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if not self.edit_mode or self._pixmap.isNull() or event.button() != Qt.MouseButton.LeftButton:
            return
        self._drag_start = event.position().toPoint()
        hit = self._roi_at_point(self._drag_start)
        if hit is not None:
            self.selected_roi_id = hit.roi_id
            self._drag_mode = "move"
            self._drag_anchor_frame = self._display_to_frame(self._drag_start)
            self._moving_roi_origin = hit
            self.roi_selected.emit(hit.roi_id)
        else:
            self._drag_mode = "draw"
            fx, fy = self._display_to_frame(self._drag_start)
            new_id = max([r.roi_id for r in self.rois], default=0) + 1
            self._draft_rect = RectROI(new_id, f"ROI{new_id}", fx, fy, fx, fy, True, "#40C4FF")
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if not self.edit_mode or self._drag_mode is None:
            return
        p = event.position().toPoint()
        fx, fy = self._display_to_frame(p)
        if self._drag_mode == "draw" and self._draft_rect is not None:
            self._draft_rect.x2 = fx
            self._draft_rect.y2 = fy
        elif self._drag_mode == "move" and self._moving_roi_origin is not None and self._drag_anchor_frame is not None:
            ax, ay = self._drag_anchor_frame
            dx, dy = fx - ax, fy - ay
            origin = self._moving_roi_origin.normalized()
            w, h = origin.width, origin.height
            nx1 = max(0, min(self._frame_w - w, origin.x1 + dx))
            ny1 = max(0, min(self._frame_h - h, origin.y1 + dy))
            updated: list[RectROI] = []
            for r in self.rois:
                if r.roi_id == origin.roi_id:
                    updated.append(RectROI(r.roi_id, r.name, nx1, ny1, nx1 + w, ny1 + h, r.enabled, r.color))
                else:
                    updated.append(r)
            self.rois = updated
            self.rois_changed.emit(self.rois)
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if not self.edit_mode or event.button() != Qt.MouseButton.LeftButton:
            return
        if self._drag_mode == "draw" and self._draft_rect is not None:
            roi = self._draft_rect.normalized()
            if roi.width >= 10 and roi.height >= 10:
                self.rois.append(roi)
                self.selected_roi_id = roi.roi_id
                self.rois_changed.emit(self.rois)
                self.roi_selected.emit(roi.roi_id)
        self._drag_start = None
        self._drag_mode = None
        self._drag_anchor_frame = None
        self._moving_roi_origin = None
        self._draft_rect = None
        self.update()
