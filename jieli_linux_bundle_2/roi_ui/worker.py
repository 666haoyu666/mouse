from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
import importlib.util
import socket
import struct
import sys
import time
import traceback

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from . import udp_protocol
from .config import AppConfig
from .image_preprocess import FramePreprocessor


@dataclass(slots=True)
class _FallbackFrameState:
    seq: int
    frame_size: int
    timestamp: int
    media_type: int
    chunks: dict[int, bytes] = field(default_factory=dict)
    last_update_ts: float = field(default_factory=time.time)
    seen_last: bool = False

    def add_chunk(self, offset: int, payload: bytes, is_last: bool) -> None:
        self.chunks[int(offset)] = bytes(payload)
        self.last_update_ts = time.time()
        if is_last:
            self.seen_last = True

    def is_complete(self) -> bool:
        if not self.seen_last or not self.chunks:
            return False
        cursor = 0
        for offset in sorted(self.chunks.keys()):
            if int(offset) != cursor:
                return False
            cursor += len(self.chunks[offset])
        return cursor >= self.frame_size

    def to_bytes(self) -> bytes:
        return b"".join(self.chunks[offset] for offset in sorted(self.chunks.keys()))[: self.frame_size]

    def age(self) -> float:
        return time.time() - self.last_update_ts


class UdpInferWorker(QThread):
    frame_ready = Signal(object, object, float)
    log_message = Signal(str)
    error_message = Signal(str)

    def __init__(self, config: AppConfig, helper_script: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self.helper_script = helper_script
        self._stopping = False
        self.sock: socket.socket | None = None
        self.detector = None
        self.frames: Dict[int, object] = {}
        self.last_fps_ts: float | None = None
        self.fps = 0.0
        self._helper = None
        self._rx_count = 0
        self._complete_count = 0
        self._bad_jpeg_count = 0
        self._last_rx_log = 0.0
        self.preprocessor = FramePreprocessor.from_env()

    def stop(self) -> None:
        self._stopping = True
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass

    def _log(self, msg: str) -> None:
        self.log_message.emit(msg)

    def _load_helper(self):
        candidates: list[Path] = []
        if self.helper_script:
            candidates.append(Path(self.helper_script))
        candidates.extend(
            [
                Path(__file__).resolve().parent.parent / "jieli_rknn_udp_infer.py",
                Path.cwd() / "jieli_rknn_udp_infer.py",
            ]
        )
        for path in candidates:
            if not path.exists():
                continue
            spec = importlib.util.spec_from_file_location("jieli_infer_helper", path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                self._helper = module
                self._log(f"[OK] 已加载推理脚本: {path}")
                return module
        raise FileNotFoundError(
            "未找到 jieli_rknn_udp_infer.py，请将本 UI 放到 jieli_linux_bundle_2 目录旁边，或用 --helper-script 指定路径。"
        )

    def _create_detector(self):
        backend = self.config.detector_backend
        if backend == "tensorrt":
            detector_cls = None
            helper = None
            try:
                from .tensorrt_detector import YoloTensorRTDetector  # type: ignore

                detector_cls = YoloTensorRTDetector
                helper = udp_protocol
            except Exception:
                helper = self._load_helper()
                detector_cls = getattr(helper, "YoloTensorRTDetector", None)
            if detector_cls is None:
                raise ImportError("未找到 YoloTensorRTDetector，请保留原仓库中的 tensorrt_detector.py 或对应实现。")
            detector = detector_cls(
                model_path=self.config.model_path,
                labels_path=self.config.labels_path,
                input_size=(self.config.input_height, self.config.input_width),
                obj_thresh=self.config.obj_thresh,
                nms_thresh=self.config.nms_thresh,
                max_det=self.config.max_det,
                agnostic_nms=self.config.agnostic_nms,
                use_rgb=not self.config.bgr_input,
                class_filter=self.config.class_filter,
                verbose=False,
            )
            self._log(f"[OK] 已加载 TensorRT 推理后端: {self.config.model_path}")
            return helper, detector

        if backend != "rknn":
            raise ValueError(f"未知 DETECTOR_BACKEND: {backend}")

        helper = self._load_helper()
        detector_cls = getattr(helper, "YoloRknnDetector", None)
        if detector_cls is None:
            raise ImportError("helper 脚本中未找到 YoloRknnDetector。")
        detector = detector_cls(
            model_path=self.config.model_path,
            labels_path=self.config.labels_path,
            input_size=(self.config.input_height, self.config.input_width),
            obj_thresh=self.config.obj_thresh,
            nms_thresh=self.config.nms_thresh,
            max_det=self.config.max_det,
            agnostic_nms=self.config.agnostic_nms,
            use_rgb=not self.config.bgr_input,
            use_all_cores=not self.config.single_core,
            verbose=False,
        )
        self._log(f"[OK] 已加载 RKNN 推理后端: {self.config.model_path}")
        return helper, detector

    def _update_fps(self) -> float:
        now = time.time()
        if self.last_fps_ts is not None:
            dt = now - self.last_fps_ts
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0.0 else (0.9 * self.fps + 0.1 * inst)
        self.last_fps_ts = now
        return self.fps

    def _cleanup_stale(self) -> None:
        stale = [seq for seq, st in self.frames.items() if hasattr(st, "age") and st.age() > self.config.cleanup_timeout]
        for seq in stale:
            self.frames.pop(seq, None)

    def _normalize_detections(self, raw_detections: Any) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        if raw_detections is None:
            return normalized
        for item in raw_detections:
            if isinstance(item, dict):
                bbox = item.get("bbox") or item.get("box")
                if bbox is None:
                    continue
                normalized.append(
                    {
                        "label": item.get("label", "hamster"),
                        "class_id": int(item.get("class_id", item.get("cls", 0))),
                        "score": float(item.get("score", item.get("conf", 0.0))),
                        "bbox": [int(v) for v in bbox[:4]],
                    }
                )
                continue
            if isinstance(item, (tuple, list)):
                if len(item) == 3:
                    class_id, score, box = item
                    normalized.append({"label": "hamster", "class_id": int(class_id), "score": float(score), "bbox": [int(v) for v in box[:4]]})
                elif len(item) == 4:
                    label, class_id, score, box = item
                    normalized.append({"label": str(label), "class_id": int(class_id), "score": float(score), "bbox": [int(v) for v in box[:4]]})
        return normalized

    def _decode_and_infer(self, payload: bytes) -> None:
        if not payload.startswith(b"\xFF\xD8") or b"\xFF\xD9" not in payload:
            self._bad_jpeg_count += 1
            if self._bad_jpeg_count <= 10:
                self._log(
                    f"[DBG] 非JPEG帧 #{self._bad_jpeg_count}, len={len(payload)}, "
                    f"head={payload[:8].hex()}, tail={payload[-8:].hex()}"
                )
            return
        arr = np.frombuffer(payload, dtype=np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            self._log(f"[DBG] JPEG解码失败, len={len(payload)}")
            return
        infer_frame_bgr = self.preprocessor.apply(frame_bgr)
        raw_detections = self.detector.infer(infer_frame_bgr)
        detections = self._normalize_detections(raw_detections)
        fps = self._update_fps()
        display_frame_bgr = infer_frame_bgr if self.preprocessor.config.display_processed else frame_bgr
        self.frame_ready.emit(display_frame_bgr, detections, fps)

    def _make_state(self, frame_state_cls, seq: int, frame_size: int, timestamp: int, media_type: int):
        try:
            return frame_state_cls(seq=seq, frame_size=frame_size, timestamp=timestamp, media_type=media_type)
        except TypeError:
            return _FallbackFrameState(seq=seq, frame_size=frame_size, timestamp=timestamp, media_type=media_type)

    def _parse_udp_packet(self, packet: bytes, addr) -> None:
        self._rx_count += 1
        now = time.time()
        if now - self._last_rx_log >= 2.0:
            self._last_rx_log = now
            self._log(f"[DBG] UDP收到包 count={self._rx_count}, from={addr}, len={len(packet)}")
        if self.config.device_ip and addr[0] != self.config.device_ip:
            return
        helper = self._helper
        if helper is None:
            return

        header_len = int(getattr(helper, "UDP_HEADER_LEN", 20))
        jpeg_type = int(getattr(helper, "JPEG_TYPE_VIDEO", 1))
        last_marker = int(getattr(helper, "LAST_VIDEO_MARKER", 0x80))
        frame_state_cls = getattr(helper, "FrameState", _FallbackFrameState)
        unpack_fmt = None
        for fmt in ("<BBHIIII", "<BBHIIIII", "<BBHIIIIQ"):
            if struct.calcsize(fmt) == header_len:
                unpack_fmt = fmt
                break
        if unpack_fmt is None:
            unpack_fmt = "<BBHIIIII"
            header_len = struct.calcsize(unpack_fmt)

        pos = 0
        plen = len(packet)
        while pos + header_len <= plen:
            try:
                fields = struct.unpack_from(unpack_fmt, packet, pos)
            except struct.error:
                break
            media_type = int(fields[0])
            payload_len = int(fields[2])
            seq = int(fields[3])
            frame_size = int(fields[4])
            offset = int(fields[5])
            timestamp = int(fields[6]) if len(fields) >= 7 else 0
            pos += header_len
            if payload_len < 0 or pos + payload_len > plen:
                break
            payload = packet[pos : pos + payload_len]
            pos += payload_len
            if (media_type & 0x7F) != jpeg_type:
                continue
            state = self.frames.get(seq)
            if state is None:
                state = self._make_state(frame_state_cls, seq, frame_size, timestamp, media_type)
                self.frames[seq] = state
            state.add_chunk(offset=offset, payload=payload, is_last=bool(media_type & last_marker))
            if state.is_complete():
                self.frames.pop(seq, None)
                frame_bytes = state.to_bytes()
                self._complete_count += 1
                if self._complete_count <= 10:
                    self._log(f"[DBG] 拼完整帧 #{self._complete_count}, seq={seq}, bytes={len(frame_bytes)}")
                self._decode_and_infer(frame_bytes)

    def run(self) -> None:
        try:
            self._helper, self.detector = self._create_detector()
            self._log(f"[INFO] YOLO 前帧处理: {self.preprocessor.summary()}")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((self.config.bind_ip, self.config.port))
            self.sock.settimeout(1.0)
            self._log(f"[INFO] UDP 监听: {self.config.bind_ip}:{self.config.port}")
            self._log(f"[INFO] 过滤设备 IP: {self.config.device_ip or '已关闭'}")
            while not self._stopping:
                try:
                    packet, addr = self.sock.recvfrom(65535)
                except socket.timeout:
                    self._cleanup_stale()
                    continue
                except OSError:
                    break
                self._parse_udp_packet(packet, addr)
                self._cleanup_stale()
        except Exception as exc:  # pragma: no cover
            self.error_message.emit(f"{exc}\n{traceback.format_exc()}")
        finally:
            if self.detector is not None:
                try:
                    self.detector.close()
                except Exception:
                    pass
            if self.sock is not None:
                try:
                    self.sock.close()
                except OSError:
                    pass
            self._log("[INFO] 工作线程结束")
