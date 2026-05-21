from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import time

from .config import AppConfig


@dataclass(slots=True)
class TrackRuntime:
    track_id: int
    first_seen_ts: float
    last_seen_ts: float
    total_seen_sec: float = 0.0
    current_grid: str = "-"
    current_grid_enter_ts: float = 0.0
    current_roi: str = ""
    current_roi_enter_ts: float = 0.0
    grid_dwell: dict[str, float] = field(default_factory=dict)
    roi_dwell: dict[str, float] = field(default_factory=dict)
    grid_switch_times: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    roi_switch_times: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    last_center: tuple[float, float] | None = None
    stationary_sec: float = 0.0
    speed_px_per_frame: float = 0.0

    def to_dict(self, now: float) -> dict[str, Any]:
        current_grid_dwell = 0.0
        if self.current_grid and self.current_grid != "-" and self.current_grid_enter_ts > 0:
            current_grid_dwell = max(0.0, now - self.current_grid_enter_ts)
        current_roi_dwell = 0.0
        if self.current_roi and self.current_roi_enter_ts > 0:
            current_roi_dwell = max(0.0, now - self.current_roi_enter_ts)
        return {
            "track_id": self.track_id,
            "age_sec": max(0.0, now - self.first_seen_ts),
            "total_seen_sec": self.total_seen_sec,
            "current_grid": self.current_grid,
            "current_roi": self.current_roi,
            "current_grid_dwell_sec": current_grid_dwell,
            "current_roi_dwell_sec": current_roi_dwell,
            "grid_dwell": dict(self.grid_dwell),
            "roi_dwell": dict(self.roi_dwell),
            "stationary_sec": self.stationary_sec,
            "speed_px_per_frame": self.speed_px_per_frame,
        }


class ActivityStats:
    """第二阶段统计模块：停留时间、热区、历史轨迹概览。"""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.start_ts = time.time()
        self.last_ts: float | None = None
        self.grid_dwell: dict[str, float] = defaultdict(float)
        self.roi_dwell: dict[str, float] = defaultdict(float)
        self.grid_hits: dict[str, int] = defaultdict(int)
        self.roi_hits: dict[str, int] = defaultdict(int)
        self.track_runtime: dict[int, TrackRuntime] = {}
        self.frame_count = 0
        self.event_count = 0

    def reset(self) -> None:
        self.__init__(self.cfg)

    def update(self, track_dicts: list[dict[str, Any]], analyses_by_track: dict[int, Any], now: float | None = None) -> dict[str, Any]:
        ts = time.time() if now is None else float(now)
        dt = 0.0 if self.last_ts is None else max(0.0, min(1.0, ts - self.last_ts))
        self.last_ts = ts
        self.frame_count += 1

        active_ids: set[int] = set()
        for tr in track_dicts:
            tid = int(tr.get("track_id", -1))
            if tid < 0:
                continue
            active_ids.add(tid)
            analysis = analyses_by_track.get(tid)
            grid = getattr(analysis, "grid_position", "-") if analysis else "-"
            roi_names: list[str] = []
            if analysis:
                roi_names = list(getattr(analysis, "inside_regions", []) or getattr(analysis, "near_regions", []) or [])
            roi = roi_names[0] if roi_names else ""
            center = tr.get("center")
            if center is None:
                bbox = tr.get("bbox") or [0, 0, 0, 0]
                center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
            center = (float(center[0]), float(center[1]))

            rt = self.track_runtime.get(tid)
            if rt is None:
                rt = TrackRuntime(track_id=tid, first_seen_ts=ts, last_seen_ts=ts)
                self.track_runtime[tid] = rt
            else:
                rt.total_seen_sec += dt
                rt.last_seen_ts = ts

            if rt.last_center is not None:
                move = ((center[0] - rt.last_center[0]) ** 2 + (center[1] - rt.last_center[1]) ** 2) ** 0.5
                rt.speed_px_per_frame = move
                if move <= self.cfg.stationary_speed_threshold_px:
                    rt.stationary_sec += dt
                else:
                    rt.stationary_sec = 0.0
            rt.last_center = center

            if grid and grid != "-":
                self.grid_dwell[grid] += dt
                self.grid_hits[grid] += 1
                rt.grid_dwell[grid] = rt.grid_dwell.get(grid, 0.0) + dt
                if rt.current_grid != grid:
                    rt.current_grid = grid
                    rt.current_grid_enter_ts = ts
                    rt.grid_switch_times.append(ts)

            if roi:
                self.roi_dwell[roi] += dt
                self.roi_hits[roi] += 1
                rt.roi_dwell[roi] = rt.roi_dwell.get(roi, 0.0) + dt
                if rt.current_roi != roi:
                    rt.current_roi = roi
                    rt.current_roi_enter_ts = ts
                    rt.roi_switch_times.append(ts)
            else:
                rt.current_roi = ""
                rt.current_roi_enter_ts = 0.0

        # 保留历史 runtime，不立刻删除，便于热区与统计展示。
        return self.summary(ts, active_ids)

    def top_grids(self, k: int | None = None) -> list[tuple[str, float]]:
        n = self.cfg.heatmap_top_k if k is None else k
        return sorted(self.grid_dwell.items(), key=lambda item: item[1], reverse=True)[:n]

    def top_rois(self, k: int | None = None) -> list[tuple[str, float]]:
        n = self.cfg.heatmap_top_k if k is None else k
        return sorted(self.roi_dwell.items(), key=lambda item: item[1], reverse=True)[:n]

    def summary(self, now: float | None = None, active_ids: set[int] | None = None) -> dict[str, Any]:
        ts = time.time() if now is None else now
        return {
            "running_sec": max(0.0, ts - self.start_ts),
            "frame_count": self.frame_count,
            "active_track_ids": sorted(active_ids or []),
            "grid_dwell": dict(self.grid_dwell),
            "roi_dwell": dict(self.roi_dwell),
            "grid_hits": dict(self.grid_hits),
            "roi_hits": dict(self.roi_hits),
            "top_grids": self.top_grids(),
            "top_rois": self.top_rois(),
            "tracks": {tid: rt.to_dict(ts) for tid, rt in self.track_runtime.items()},
        }

    def save_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2), encoding="utf-8")


class BehaviorRuleEngine:
    """场景行为规则判断：把连续状态转为可解释的中文标签。"""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg

    def evaluate(self, runtime: dict[str, Any] | None) -> list[str]:
        if not runtime:
            return []
        tags: list[str] = []
        age = float(runtime.get("age_sec", 0.0))
        roi = str(runtime.get("current_roi", ""))
        roi_dwell = float(runtime.get("current_roi_dwell_sec", 0.0))
        grid_dwell = float(runtime.get("current_grid_dwell_sec", 0.0))
        stationary = float(runtime.get("stationary_sec", 0.0))
        speed = float(runtime.get("speed_px_per_frame", 0.0))

        if age >= self.cfg.long_stay_threshold_sec:
            tags.append("连续停留时间较长")
        elif age >= self.cfg.dwell_threshold_sec:
            tags.append("已形成连续停留")

        if roi and roi_dwell >= self.cfg.roi_stay_threshold_sec:
            tags.append(f"在{roi}附近停留")

        if stationary >= self.cfg.stationary_threshold_sec:
            tags.append("活动幅度较小")
        elif speed >= self.cfg.active_move_threshold_px:
            tags.append("移动较活跃")

        if grid_dwell >= self.cfg.roi_stay_threshold_sec and not roi:
            tags.append("在同一画面区域持续停留")

        return tags[:4]
