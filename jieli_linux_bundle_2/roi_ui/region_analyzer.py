from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any

from .roi_model import RectROI


GRID_NAMES = [
    ["左上", "上中", "右上"],
    ["左中", "中央", "右中"],
    ["左下", "下中", "右下"],
]


@dataclass(slots=True)
class RegionAnalysis:
    detected: bool
    count: int
    bbox: list[int] | None
    center: tuple[float, float] | None
    grid_position: str
    grid_row: int
    grid_col: int
    near_regions: list[str]
    inside_regions: list[str]
    overlap_regions: list[str]
    distance_regions: dict[str, float]
    description_tags: list[str]

    @property
    def tags(self) -> list[str]:
        return self.description_tags


def bbox_center(bbox: list[int] | tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_area(bbox: list[int] | tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: list[int] | tuple[int, int, int, int], b: list[int] | tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def rect_distance(a: list[int] | tuple[int, int, int, int], b: list[int] | tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return hypot(dx, dy)


def grid_position(center: tuple[float, float], frame_w: int, frame_h: int) -> tuple[str, int, int]:
    cx, cy = center
    col = min(2, max(0, int(cx / max(1, frame_w / 3.0))))
    row = min(2, max(0, int(cy / max(1, frame_h / 3.0))))
    return GRID_NAMES[row][col], row, col


def analyze_detection(
    detections: list[dict[str, Any]],
    rois: list[RectROI],
    frame_w: int,
    frame_h: int,
    distance_threshold: float = 28.0,
    iou_threshold: float = 0.05,
    max_regions: int = 2,
    max_description_regions: int | None = None,
) -> RegionAnalysis:
    if max_description_regions is not None:
        max_regions = max_description_regions

    if not detections:
        return RegionAnalysis(False, 0, None, None, "-", -1, -1, [], [], [], {}, [])

    # 默认使用置信度最高的目标作为主目标，同时 count 保留全部数量。
    det = max(detections, key=lambda d: float(d.get("score", 0.0)))
    bbox = [int(v) for v in det.get("bbox", [0, 0, 0, 0])]
    center = bbox_center(bbox)
    grid_name, row, col = grid_position(center, frame_w, frame_h)

    inside: list[str] = []
    overlap: list[str] = []
    near_candidates: list[tuple[str, float]] = []
    dist_map: dict[str, float] = {}

    for roi in rois:
        if not roi.enabled:
            continue
        n = roi.normalized()
        rb = [n.x1, n.y1, n.x2, n.y2]
        dist = rect_distance(bbox, rb)
        iou = bbox_iou(bbox, rb)
        dist_map[n.name] = dist
        if n.contains_point(*center):
            inside.append(n.name)
        if iou >= iou_threshold:
            overlap.append(n.name)
        if dist <= distance_threshold or iou >= iou_threshold:
            near_candidates.append((n.name, dist))

    near = [name for name, _ in sorted(near_candidates, key=lambda item: item[1])]
    # inside 优先，不重复输出。
    near = [name for name in near if name not in inside][:max_regions]
    inside = inside[:max_regions]
    overlap = overlap[:max_regions]

    tags = [f"{grid_name}区域"]
    if inside:
        tags.extend([f"进入{name}" for name in inside])
    if near:
        tags.extend([f"靠近{name}" for name in near])

    return RegionAnalysis(True, len(detections), bbox, center, grid_name, row, col, near, inside, overlap, dist_map, tags)
