from __future__ import annotations

from typing import Any


def _join_names(names: list[str]) -> str:
    return "、".join(names) if names else ""


def generate_text(analysis, active_track: dict[str, Any] | None = None, behaviors: list[str] | None = None) -> str:
    """第一阶段 + 第二阶段融合中文描述。"""
    if not analysis.detected:
        return "当前未检测到仓鼠。"

    parts: list[str] = [f"当前检测到 {analysis.count} 只仓鼠"]
    if active_track and active_track.get("track_id") is not None:
        parts.append(f"主目标 ID {active_track['track_id']}")
    parts.append(f"位于画面{analysis.grid_position}区域")

    if analysis.inside_regions:
        parts.append(f"进入{_join_names(analysis.inside_regions)}")
    elif analysis.near_regions:
        parts.append(f"靠近{_join_names(analysis.near_regions)}")

    if active_track:
        age = float(active_track.get("age_sec", 0.0))
        parts.append(f"已连续跟踪 {age:.1f} 秒")
        current_roi = active_track.get("current_roi")
        roi_dwell = float(active_track.get("current_roi_dwell_sec", 0.0))
        if current_roi and roi_dwell > 0:
            parts.append(f"在{current_roi}附近停留 {roi_dwell:.1f} 秒")

    if behaviors:
        parts.append("行为判断：" + "、".join(behaviors[:3]))

    return "，".join(parts) + "。"


def generate_hotspot_text(top_grids: list[tuple[str, float]], top_rois: list[tuple[str, float]]) -> str:
    grid_text = "、".join([f"{name}{sec:.1f}s" for name, sec in top_grids]) if top_grids else "暂无"
    roi_text = "、".join([f"{name}{sec:.1f}s" for name, sec in top_rois]) if top_rois else "暂无"
    return f"活动热区：宫格 {grid_text}；ROI {roi_text}。"
