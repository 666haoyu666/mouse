from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, Iterable, List, Tuple

from .roi_model import RectROI

Detection = Tuple[int, float, Tuple[int, int, int, int]]


@dataclass
class ROIStatus:
    active: bool = False
    entered_at: float | None = None
    dwell_time: float = 0.0
    alarmed: bool = False
    last_detection_bottom: Tuple[int, int] | None = None


class DwellTracker:
    def __init__(self, absence_reset_sec: float = 1.0) -> None:
        self.absence_reset_sec = absence_reset_sec
        self.status: Dict[int, ROIStatus] = {}
        self.last_seen: Dict[int, float] = {}

    @staticmethod
    def bottom_midpoint(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int(y2)

    def update(self, rois: Iterable[RectROI], detections: List[Detection], now: float | None = None) -> Dict[int, ROIStatus]:
        now = now if now is not None else time.time()
        roi_list = list(rois)
        used = {r.roi_id for r in roi_list}
        for roi in roi_list:
            st = self.status.setdefault(roi.roi_id, ROIStatus())
            inside_points = []
            for class_id, score, box in detections:
                bx, by = self.bottom_midpoint(box)
                if roi.enabled and roi.contains_point(bx, by):
                    inside_points.append((bx, by))
            if inside_points:
                self.last_seen[roi.roi_id] = now
                st.last_detection_bottom = inside_points[0]
                if not st.active:
                    st.active = True
                    st.entered_at = now
                    st.dwell_time = 0.0
                    st.alarmed = False
                else:
                    st.dwell_time = max(0.0, now - (st.entered_at or now))
            else:
                last = self.last_seen.get(roi.roi_id)
                if st.active and last is not None and now - last <= self.absence_reset_sec:
                    st.dwell_time = max(0.0, now - (st.entered_at or now))
                else:
                    st.active = False
                    st.entered_at = None
                    st.dwell_time = 0.0
                    st.alarmed = False
                    st.last_detection_bottom = None
        for roi_id in list(self.status.keys()):
            if roi_id not in used:
                self.status.pop(roi_id, None)
                self.last_seen.pop(roi_id, None)
        return self.status
