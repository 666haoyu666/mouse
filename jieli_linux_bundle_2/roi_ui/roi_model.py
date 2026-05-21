from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import json


DEFAULT_SCENE_NAMES = ["木屋", "跑轮", "食盆", "饮水器"]


def _coerce_color(value: Any, default: str = "#40C4FF") -> str:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            r, g, b = [max(0, min(255, int(v))) for v in value[:3]]
            return f"#{r:02X}{g:02X}{b:02X}"
        except Exception:
            return default
    return default


@dataclass(slots=True)
class RectROI:
    roi_id: int
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    enabled: bool = True
    color: str = "#40C4FF"
    dwell_sec: float = 10.0
    audio_id: int = 1
    alarm_enabled: bool = True

    def normalized(self) -> "RectROI":
        x1, x2 = sorted((int(self.x1), int(self.x2)))
        y1, y2 = sorted((int(self.y1), int(self.y2)))
        return RectROI(
            roi_id=int(self.roi_id),
            name=str(self.name),
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            enabled=bool(self.enabled),
            color=_coerce_color(self.color),
            dwell_sec=float(self.dwell_sec),
            audio_id=int(self.audio_id),
            alarm_enabled=bool(self.alarm_enabled),
        )

    @property
    def width(self) -> int:
        n = self.normalized()
        return max(0, n.x2 - n.x1)

    @property
    def height(self) -> int:
        n = self.normalized()
        return max(0, n.y2 - n.y1)

    def contains_point(self, x: float, y: float) -> bool:
        n = self.normalized()
        return n.x1 <= x <= n.x2 and n.y1 <= y <= n.y2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.normalized())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RectROI":
        return cls(
            roi_id=int(data.get("roi_id", data.get("id", 0))),
            name=str(data.get("name", "ROI")),
            x1=int(data.get("x1", 0)),
            y1=int(data.get("y1", 0)),
            x2=int(data.get("x2", 0)),
            y2=int(data.get("y2", 0)),
            enabled=bool(data.get("enabled", True)),
            color=_coerce_color(data.get("color", "#40C4FF")),
            dwell_sec=float(data.get("dwell_sec", 10.0)),
            audio_id=int(data.get("audio_id", 1)),
            alarm_enabled=bool(data.get("alarm_enabled", True)),
        ).normalized()


def default_scene_rois(frame_w: int, frame_h: int) -> list[RectROI]:
    """按常见仓鼠笼布局给出初始 ROI，运行后应在 UI 中拖拽校准并保存。"""
    w = max(1, int(frame_w))
    h = max(1, int(frame_h))
    return [
        RectROI(1, "跑轮", int(w * 0.06), int(h * 0.10), int(w * 0.30), int(h * 0.43), True, "#46A3FF"),
        RectROI(2, "木屋", int(w * 0.64), int(h * 0.45), int(w * 0.95), int(h * 0.90), True, "#FFB300"),
        RectROI(3, "食盆", int(w * 0.38), int(h * 0.70), int(w * 0.56), int(h * 0.92), True, "#7CB342"),
        RectROI(4, "饮水器", int(w * 0.84), int(h * 0.08), int(w * 0.96), int(h * 0.38), True, "#AB47BC"),
    ]


def load_roi_bundle(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"group_name": p.stem, "saved_at": None, "frame_size": None, "rois": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {"group_name": p.stem, "saved_at": None, "frame_size": None, "rois": [RectROI.from_dict(item) for item in data]}
        if isinstance(data, dict):
            if isinstance(data.get("rois"), list):
                bundle = dict(data)
                bundle.setdefault("group_name", p.stem)
                bundle.setdefault("saved_at", data.get("updated_at"))
                bundle.setdefault("frame_size", None)
                bundle["rois"] = [RectROI.from_dict(item) for item in data["rois"] if isinstance(item, dict)]
                return bundle
            if isinstance(data.get("groups"), list) and data["groups"]:
                first_group = data["groups"][0]
                if isinstance(first_group, dict) and isinstance(first_group.get("rois"), list):
                    bundle = dict(first_group)
                    bundle.setdefault("group_name", p.stem)
                    bundle.setdefault("saved_at", data.get("updated_at"))
                    bundle.setdefault("frame_size", None)
                    bundle["rois"] = [RectROI.from_dict(item) for item in first_group["rois"] if isinstance(item, dict)]
                    return bundle
    except Exception:
        pass
    return {"group_name": p.stem, "saved_at": None, "frame_size": None, "rois": []}


def load_rois(path: Path | str) -> list[RectROI]:
    return list(load_roi_bundle(path).get("rois", []))


def save_rois(
    path: Path | str,
    rois: list[RectROI],
    extra: dict[str, Any] | None = None,
    *,
    frame_size: tuple[int, int] | None = None,
    group_name: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "type": "hamster_scene_rois",
        "group_name": group_name or p.stem,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "frame_size": {"width": frame_size[0], "height": frame_size[1]} if frame_size else None,
        "description": "固定 ROI：木屋、跑轮、食盆、饮水器，可按真实笼子位置拖拽调整。",
        "rois": [roi.normalized().to_dict() for roi in rois],
    }
    if extra:
        payload["extra"] = extra
    if extra_meta:
        payload.update(extra_meta)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
