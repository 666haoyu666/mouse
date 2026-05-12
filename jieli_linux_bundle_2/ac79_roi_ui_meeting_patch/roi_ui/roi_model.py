from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
from typing import Any, Iterable, List, Tuple


@dataclass
class RectROI:
    roi_id: int
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    dwell_sec: float = 10.0
    audio_id: int = 1
    enabled: bool = True
    color: Tuple[int, int, int] = (0, 255, 255)
    alarm_enabled: bool = True

    def normalized(self) -> "RectROI":
        x1, x2 = sorted((int(self.x1), int(self.x2)))
        y1, y2 = sorted((int(self.y1), int(self.y2)))
        return RectROI(
            roi_id=self.roi_id,
            name=self.name,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            dwell_sec=float(self.dwell_sec),
            audio_id=int(self.audio_id),
            enabled=bool(self.enabled),
            color=tuple(self.color),
            alarm_enabled=bool(self.alarm_enabled),
        )

    def contains_point(self, x: float, y: float) -> bool:
        r = self.normalized()
        return r.x1 <= x <= r.x2 and r.y1 <= y <= r.y2

    def to_dict(self) -> dict:
        d = asdict(self.normalized())
        d["color"] = list(self.color)
        return d

    @staticmethod
    def from_dict(d: dict) -> "RectROI":
        return RectROI(
            roi_id=int(d["roi_id"]),
            name=str(d.get("name", f"roi_{d['roi_id']}")),
            x1=int(d["x1"]),
            y1=int(d["y1"]),
            x2=int(d["x2"]),
            y2=int(d["y2"]),
            dwell_sec=float(d.get("dwell_sec", 10.0)),
            audio_id=int(d.get("audio_id", 1)),
            enabled=bool(d.get("enabled", True)),
            color=tuple(d.get("color", [0, 255, 255])),
            alarm_enabled=bool(d.get("alarm_enabled", True)),
        )


def save_rois(
    path: Path,
    rois: Iterable[RectROI],
    frame_size: Tuple[int, int] | None = None,
    *,
    group_name: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "group_name": group_name or path.stem,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "frame_size": {"width": frame_size[0], "height": frame_size[1]} if frame_size else None,
        "rois": [r.to_dict() for r in rois],
    }
    if extra_meta:
        payload.update(extra_meta)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_roi_bundle(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"group_name": path.stem, "saved_at": None, "frame_size": None, "rois": []}

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"rois": data}
    if not isinstance(data, dict):
        raise ValueError(f"ROI JSON 格式错误: {path}")

    data.setdefault("group_name", path.stem)
    data.setdefault("saved_at", None)
    data.setdefault("frame_size", None)
    data["rois"] = [RectROI.from_dict(item) for item in data.get("rois", [])]
    return data


def load_rois(path: Path) -> List[RectROI]:
    return list(load_roi_bundle(path).get("rois", []))
