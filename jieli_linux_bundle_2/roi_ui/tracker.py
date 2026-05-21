from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any
import time

from .region_analyzer import bbox_center, bbox_iou, rect_distance


@dataclass(slots=True)
class Track:
    track_id: int
    bbox: list[int]
    score: float
    label: str
    class_id: int
    created_ts: float
    last_seen_ts: float
    hits: int = 1
    missed: int = 0
    total_visible_sec: float = 0.0
    last_update_ts: float = 0.0
    trail: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=40))
    grid_history: deque[tuple[float, str]] = field(default_factory=lambda: deque(maxlen=120))
    roi_history: deque[tuple[float, str]] = field(default_factory=lambda: deque(maxlen=120))
    grid_dwell: dict[str, float] = field(default_factory=dict)
    roi_dwell: dict[str, float] = field(default_factory=dict)
    current_grid: str = "-"
    current_roi: str = ""
    current_grid_enter_ts: float = 0.0
    current_roi_enter_ts: float = 0.0
    stationary_sec: float = 0.0
    last_center: tuple[float, float] | None = None
    speed_px: float = 0.0

    def to_dict(self, now: float | None = None) -> dict[str, Any]:
        ts = time.time() if now is None else now
        age = max(0.0, ts - self.created_ts)
        roi_dwell = 0.0
        if self.current_roi:
            roi_dwell = max(0.0, ts - self.current_roi_enter_ts)
        grid_dwell = 0.0
        if self.current_grid and self.current_grid != "-":
            grid_dwell = max(0.0, ts - self.current_grid_enter_ts)
        return {
            "track_id": self.track_id,
            "bbox": list(self.bbox),
            "score": self.score,
            "label": self.label,
            "class_id": self.class_id,
            "age_sec": age,
            "missed": self.missed,
            "hits": self.hits,
            "center": bbox_center(self.bbox),
            "trail": list(self.trail),
            "current_grid": self.current_grid,
            "current_roi": self.current_roi,
            "current_grid_dwell_sec": grid_dwell,
            "current_roi_dwell_sec": roi_dwell,
            "grid_dwell": dict(self.grid_dwell),
            "roi_dwell": dict(self.roi_dwell),
            "stationary_sec": self.stationary_sec,
            "speed_px": self.speed_px,
        }


class IoUTracker:
    """轻量级多目标跟踪器，适合仓鼠单/少目标连续状态分析。

    关联策略：IoU 优先，中心距离兜底。无需额外 ReID 模型，适合 Jetson 边缘端第一版演示。
    """

    def __init__(
        self,
        iou_threshold: float = 0.25,
        center_distance: float = 80.0,
        max_missed: int = 20,
        min_hits: int = 1,
        trail_length: int = 40,
        stationary_speed_threshold_px: float = 6.0,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.center_distance = float(center_distance)
        self.max_missed = int(max_missed)
        self.min_hits = int(min_hits)
        self.trail_length = int(trail_length)
        self.stationary_speed_threshold_px = float(stationary_speed_threshold_px)
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1

    def _score_pair(self, track: Track, det: dict[str, Any]) -> float:
        db = [int(v) for v in det.get("bbox", [0, 0, 0, 0])]
        iou = bbox_iou(track.bbox, db)
        if iou >= self.iou_threshold:
            return 1000.0 + iou
        dist = rect_distance(track.bbox, db)
        if dist <= self.center_distance:
            return 1.0 - dist / max(1.0, self.center_distance)
        tc = bbox_center(track.bbox)
        dc = bbox_center(db)
        center_dist = ((tc[0] - dc[0]) ** 2 + (tc[1] - dc[1]) ** 2) ** 0.5
        if center_dist <= self.center_distance:
            return 0.5 - center_dist / (2.0 * max(1.0, self.center_distance))
        return -1.0

    def update(self, detections: list[dict[str, Any]], now: float | None = None) -> list[dict[str, Any]]:
        ts = time.time() if now is None else float(now)
        detections = [d for d in detections if d.get("bbox") is not None]

        # 所有 track 先认为本帧未命中，匹配后再清零 missed。
        for tr in self.tracks.values():
            tr.missed += 1

        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(self.tracks.keys())
        matches: list[tuple[int, int]] = []

        # 贪心匹配：每次取最高得分 track-det 对。
        while unmatched_dets and unmatched_tracks:
            best: tuple[float, int, int] | None = None
            for tid in unmatched_tracks:
                tr = self.tracks[tid]
                for di in unmatched_dets:
                    s = self._score_pair(tr, detections[di])
                    if s < 0:
                        continue
                    if best is None or s > best[0]:
                        best = (s, tid, di)
            if best is None:
                break
            _, tid, di = best
            matches.append((tid, di))
            unmatched_tracks.discard(tid)
            unmatched_dets.discard(di)

        for tid, di in matches:
            self._update_track(self.tracks[tid], detections[di], ts)

        for di in sorted(unmatched_dets):
            self._create_track(detections[di], ts)

        dead = [tid for tid, tr in self.tracks.items() if tr.missed > self.max_missed]
        for tid in dead:
            self.tracks.pop(tid, None)

        active = [tr.to_dict(ts) for tr in self.tracks.values() if tr.hits >= self.min_hits and tr.missed <= self.max_missed]
        active.sort(key=lambda x: (x["missed"], -x["score"], x["track_id"]))
        return active

    def _create_track(self, det: dict[str, Any], ts: float) -> None:
        bbox = [int(v) for v in det.get("bbox", [0, 0, 0, 0])]
        center = bbox_center(bbox)
        tr = Track(
            track_id=self.next_id,
            bbox=bbox,
            score=float(det.get("score", 0.0)),
            label=str(det.get("label", "hamster")),
            class_id=int(det.get("class_id", 0)),
            created_ts=ts,
            last_seen_ts=ts,
            last_update_ts=ts,
            trail=deque(maxlen=self.trail_length),
        )
        tr.trail.append(center)
        tr.last_center = center
        self.tracks[tr.track_id] = tr
        self.next_id += 1

    def _update_track(self, tr: Track, det: dict[str, Any], ts: float) -> None:
        dt = max(0.0, ts - tr.last_seen_ts)
        if tr.missed <= 1:
            tr.total_visible_sec += dt
        old_center = tr.last_center or bbox_center(tr.bbox)
        tr.bbox = [int(v) for v in det.get("bbox", tr.bbox)]
        tr.score = float(det.get("score", tr.score))
        tr.label = str(det.get("label", tr.label))
        tr.class_id = int(det.get("class_id", tr.class_id))
        tr.last_seen_ts = ts
        tr.last_update_ts = ts
        tr.hits += 1
        tr.missed = 0
        new_center = bbox_center(tr.bbox)
        tr.trail.append(new_center)
        if dt > 0:
            tr.speed_px = ((new_center[0] - old_center[0]) ** 2 + (new_center[1] - old_center[1]) ** 2) ** 0.5 / dt
        else:
            tr.speed_px = 0.0
        if ((new_center[0] - old_center[0]) ** 2 + (new_center[1] - old_center[1]) ** 2) ** 0.5 <= self.stationary_speed_threshold_px:
            tr.stationary_sec += dt
        else:
            tr.stationary_sec = 0.0
        tr.last_center = new_center

    def update_track_regions(self, track_id: int, grid_name: str, roi_names: list[str], now: float | None = None) -> None:
        ts = time.time() if now is None else float(now)
        tr = self.tracks.get(track_id)
        if tr is None:
            return
        dt = max(0.0, ts - tr.last_update_ts)
        # grid dwell
        if grid_name and grid_name != "-":
            if tr.current_grid != grid_name:
                tr.current_grid = grid_name
                tr.current_grid_enter_ts = ts
                tr.grid_history.append((ts, grid_name))
            else:
                tr.grid_dwell[grid_name] = tr.grid_dwell.get(grid_name, 0.0) + dt
        # roi dwell：优先 inside，其次 near，取第一个作为当前主要区域。
        roi_name = roi_names[0] if roi_names else ""
        if roi_name:
            if tr.current_roi != roi_name:
                tr.current_roi = roi_name
                tr.current_roi_enter_ts = ts
                tr.roi_history.append((ts, roi_name))
            else:
                tr.roi_dwell[roi_name] = tr.roi_dwell.get(roi_name, 0.0) + dt
        else:
            tr.current_roi = ""
            tr.current_roi_enter_ts = 0.0
        tr.last_update_ts = ts
