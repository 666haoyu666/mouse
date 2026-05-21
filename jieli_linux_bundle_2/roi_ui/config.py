from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _load_env_file(path: str = ".env") -> None:
    """轻量 .env 加载器，避免额外依赖 python-dotenv。"""
    file_path = Path(path)
    if not file_path.exists():
        return
    for raw in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(os.getenv("ENV_FILE", ".env"))


def _get_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _get_int_tuple(name: str, default: str = "") -> tuple[int, ...]:
    value = os.getenv(name, default).strip()
    if not value or value.lower() in {"all", "none", "off", "empty"}:
        return ()
    return tuple(int(part) for part in value.replace(",", " ").split())


def _get_float_list(name: str, default: str = "") -> tuple[float, ...]:
    value = os.getenv(name, default).strip()
    if not value:
        return ()
    return tuple(float(part) for part in value.replace(",", " ").split())


@dataclass(slots=True)
class AppConfig:
    # ===== 网络与视频 =====
    bind_ip: str = os.getenv("BIND_IP", "0.0.0.0")
    port: int = int(os.getenv("UDP_PORT", "2224"))
    device_ip: str = os.getenv("DEVICE_IP", "192.168.1.1")
    cleanup_timeout: float = float(os.getenv("CLEANUP_TIMEOUT", "3.0"))

    # ===== 检测后端 =====
    detector_backend: str = os.getenv("DETECTOR_BACKEND", "tensorrt").strip().lower()
    model_path: str = os.getenv("MODEL_PATH", "./model/hamster.engine")
    labels_path: str = os.getenv("LABELS_PATH", "./model/hamster_labels.txt")
    obj_thresh: float = float(os.getenv("OBJ_THRESH", "0.25"))
    nms_thresh: float = float(os.getenv("NMS_THRESH", "0.45"))
    input_width: int = int(os.getenv("INPUT_WIDTH", "640"))
    input_height: int = int(os.getenv("INPUT_HEIGHT", "640"))
    bgr_input: bool = _get_bool("BGR_INPUT", "1")
    single_core: bool = _get_bool("SINGLE_CORE", "1")
    max_det: int = int(os.getenv("MAX_DET", "10"))
    agnostic_nms: bool = _get_bool("AGNOSTIC_NMS", "0")
    class_filter: tuple[int, ...] = _get_int_tuple("CLASS_FILTER", "")

    # ===== 第一阶段：检测 + 九宫格 + 固定 ROI + 中文描述 =====
    ui_title: str = os.getenv("UI_TITLE", "仓鼠监控系统 UI - 第二阶段增强版")
    auto_seed_default_rois: bool = _get_bool("AUTO_SEED_DEFAULT_ROIS", "1")
    grid_enabled: bool = _get_bool("GRID_ENABLED", "1")
    grid_labels_enabled: bool = _get_bool("GRID_LABELS_ENABLED", "1")
    roi_distance_threshold: float = float(os.getenv("ROI_DISTANCE_THRESHOLD", "28"))
    roi_iou_threshold: float = float(os.getenv("ROI_IOU_THRESHOLD", "0.05"))
    near_priority: str = os.getenv("NEAR_PRIORITY", "distance").strip().lower()
    max_description_regions: int = int(os.getenv("MAX_DESCRIPTION_REGIONS", "2"))

    # ===== 第二阶段：连续跟踪与统计 =====
    tracker_enabled: bool = _get_bool("TRACKER_ENABLED", "1")
    tracker_iou_threshold: float = float(os.getenv("TRACKER_IOU_THRESHOLD", "0.25"))
    tracker_center_distance: float = float(os.getenv("TRACKER_CENTER_DISTANCE", "80"))
    tracker_max_missed: int = int(os.getenv("TRACKER_MAX_MISSED", "20"))
    tracker_min_hits: int = int(os.getenv("TRACKER_MIN_HITS", "1"))
    trail_length: int = int(os.getenv("TRAIL_LENGTH", "40"))

    dwell_enabled: bool = _get_bool("DWELL_ENABLED", "1")
    dwell_threshold_sec: float = float(os.getenv("DWELL_THRESHOLD_SEC", "10"))
    long_stay_threshold_sec: float = float(os.getenv("LONG_STAY_THRESHOLD_SEC", "30"))
    roi_stay_threshold_sec: float = float(os.getenv("ROI_STAY_THRESHOLD_SEC", "8"))
    stationary_speed_threshold_px: float = float(os.getenv("STATIONARY_SPEED_THRESHOLD_PX", "6"))
    stationary_threshold_sec: float = float(os.getenv("STATIONARY_THRESHOLD_SEC", "12"))
    active_move_threshold_px: float = float(os.getenv("ACTIVE_MOVE_THRESHOLD_PX", "35"))
    high_activity_window_sec: float = float(os.getenv("HIGH_ACTIVITY_WINDOW_SEC", "20"))
    high_activity_min_switches: int = int(os.getenv("HIGH_ACTIVITY_MIN_SWITCHES", "5"))

    # ===== 活动热区统计 =====
    heatmap_enabled: bool = _get_bool("HEATMAP_ENABLED", "1")
    heatmap_overlay_enabled: bool = _get_bool("HEATMAP_OVERLAY_ENABLED", "0")
    heatmap_decay: float = float(os.getenv("HEATMAP_DECAY", "0.995"))
    heatmap_grid_rows: int = int(os.getenv("HEATMAP_GRID_ROWS", "3"))
    heatmap_grid_cols: int = int(os.getenv("HEATMAP_GRID_COLS", "3"))
    heatmap_top_k: int = int(os.getenv("HEATMAP_TOP_K", "3"))

    # ===== Nano 现场运行：视频流断连自动恢复 =====
    video_stall_timeout_sec: float = float(os.getenv("VIDEO_STALL_TIMEOUT_SEC", "5"))
    video_reopen_cooldown_sec: float = float(os.getenv("VIDEO_REOPEN_COOLDOWN_SEC", "8"))
    video_full_restart_sec: float = float(os.getenv("VIDEO_FULL_RESTART_SEC", "20"))

    # ===== 输出路径 =====
    screenshot_dir: str = os.getenv("SCREENSHOT_DIR", "./roi_ui_output/screenshots")
    roi_json: str = os.getenv("ROI_JSON", "./roi_ui_output/hamster_rois.json")
    event_log: str = os.getenv("ROI_EVENT_LOG", "./roi_ui_output/hamster_events.jsonl")
    stats_json: str = os.getenv("STATS_JSON", "./roi_ui_output/hamster_stats.json")

    @property
    def screenshot_dir_path(self) -> Path:
        return Path(self.screenshot_dir)

    @property
    def roi_json_path(self) -> Path:
        return Path(self.roi_json)

    @property
    def event_log_path(self) -> Path:
        return Path(self.event_log)

    @property
    def stats_json_path(self) -> Path:
        return Path(self.stats_json)
