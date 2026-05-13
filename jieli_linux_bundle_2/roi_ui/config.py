from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _load_env_file(path: str = ".env") -> None:
    """轻量 .env 加载器，避免额外依赖 python-dotenv。已有环境变量优先级更高。"""
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# UI 一般从 jieli_linux_bundle 目录启动，默认读取当前目录下的 .env。
_load_env_file(os.getenv("ENV_FILE", ".env"))


def _get_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _get_int_tuple(name: str, default: str = "") -> tuple[int, ...]:
    value = os.getenv(name, default).strip()
    if not value or value.lower() in {"all", "none", "off"}:
        return ()
    parts = value.replace(",", " ").split()
    return tuple(int(part) for part in parts)


@dataclass
class AppConfig:
    bind_ip: str = os.getenv("BIND_IP", "0.0.0.0")
    port: int = int(os.getenv("UDP_PORT", "2224"))
    # 为空字符串表示不过滤设备 IP；默认 AC79 AP 常见地址为 192.168.1.1
    device_ip: str = os.getenv("DEVICE_IP", "192.168.1.1")
    cleanup_timeout: float = float(os.getenv("CLEANUP_TIMEOUT", "3.0"))

    detector_backend: str = os.getenv("DETECTOR_BACKEND", "rknn").strip().lower()

    # 与原 jieli_rknn_udp_infer.py 保持一致：默认使用 jieli_linux_bundle/model
    model_path: str = os.getenv("MODEL_PATH", "./model/person.rknn")
    labels_path: str = os.getenv("LABELS_PATH", "./model/labels.txt")
    obj_thresh: float = float(os.getenv("OBJ_THRESH", "0.25"))
    nms_thresh: float = float(os.getenv("NMS_THRESH", "0.45"))
    input_width: int = int(os.getenv("INPUT_WIDTH", "640"))
    input_height: int = int(os.getenv("INPUT_HEIGHT", "640"))
    bgr_input: bool = _get_bool("BGR_INPUT", "1")
    single_core: bool = _get_bool("SINGLE_CORE", "1")
    max_det: int = int(os.getenv("MAX_DET", "10"))
    agnostic_nms: bool = _get_bool("AGNOSTIC_NMS", "0")
    class_filter: tuple[int, ...] = _get_int_tuple("CLASS_FILTER", "")

    # ROI 与报警输出
    screenshot_dir: str = os.getenv("SCREENSHOT_DIR", "./roi_ui_output/screenshots")
    roi_json: str = os.getenv("ROI_JSON", "./roi_ui_output/rois.json")
    alarm_cmd: str = os.getenv("ALARM_CMD", "")
    event_log: str = os.getenv("ROI_EVENT_LOG", "./roi_ui_output/events.jsonl")

    # UI 模式
    ui_mode: str = os.getenv("UI_MODE", "default")

    # 会议室模式参数
    meeting_use_start_sec: float = float(os.getenv("MEETING_USE_START_SEC", "60"))
    meeting_release_empty_sec: float = float(os.getenv("MEETING_RELEASE_EMPTY_SEC", "30"))
    meeting_long_use_sec: float = float(os.getenv("MEETING_LONG_USE_SEC", "120"))
    meeting_abnormal_extra_sec: float = float(os.getenv("MEETING_ABNORMAL_EXTRA_SEC", "180"))
    meeting_abnormal_repeat_sec: float = float(os.getenv("MEETING_ABNORMAL_REPEAT_SEC", "180"))
    meeting_release_audio_gap_sec: float = float(os.getenv("MEETING_RELEASE_AUDIO_GAP_SEC", "3"))
    meeting_default_work_start: str = os.getenv("MEETING_WORK_START", "09:00")
    meeting_default_work_end: str = os.getenv("MEETING_WORK_END", "18:00")
    meeting_idle_text: str = os.getenv("MEETING_IDLE_TEXT", "会议室空闲")
    meeting_busy_text: str = os.getenv("MEETING_BUSY_TEXT", "会议室正在使用")

    # 视频流断连自动恢复
    video_stall_timeout_sec: float = float(os.getenv("VIDEO_STALL_TIMEOUT_SEC", "5"))
    video_reopen_cooldown_sec: float = float(os.getenv("VIDEO_REOPEN_COOLDOWN_SEC", "8"))
    video_full_restart_sec: float = float(os.getenv("VIDEO_FULL_RESTART_SEC", "20"))

    @property
    def screenshot_dir_path(self) -> Path:
        return Path(self.screenshot_dir)

    @property
    def roi_json_path(self) -> Path:
        return Path(self.roi_json)

    @property
    def event_log_path(self) -> Path:
        return Path(self.event_log)
