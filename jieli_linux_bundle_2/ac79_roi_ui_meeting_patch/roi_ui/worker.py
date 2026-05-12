from __future__ import annotations

import importlib.util
import socket
import struct
import time
import traceback
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from .config import AppConfig


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
        candidates = []
        if self.helper_script:
            candidates.append(Path(self.helper_script))
        candidates.extend([
            Path(__file__).resolve().parent.parent / 'jieli_rknn_udp_infer.py',
            Path.cwd() / 'jieli_rknn_udp_infer.py',
        ])
        for path in candidates:
            if path.exists():
                spec = importlib.util.spec_from_file_location('jieli_infer_helper', path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self._helper = module
                    self._log(f'[OK] 已加载推理脚本: {path}')
                    return module
        raise FileNotFoundError('未找到 jieli_rknn_udp_infer.py，请将本 UI 放到 jieli_linux_bundle 旁边，或用 --helper-script 指定路径。')

    def _create_detector(self):
        helper = self._load_helper()
        detector = helper.YoloRknnDetector(
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

    def _cleanup_stale(self):
        if not self._helper:
            return
        stale = [seq for seq, st in self.frames.items() if st.age() > self.config.cleanup_timeout]
        for seq in stale:
            self.frames.pop(seq, None)

    def _decode_and_infer(self, payload: bytes):
        if not payload.startswith(b'\xFF\xD8') or b'\xFF\xD9' not in payload:
            return
        arr = np.frombuffer(payload, dtype=np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return
        detections = self.detector.infer(frame_bgr)
        fps = self._update_fps()
        self.frame_ready.emit(frame_bgr, detections, fps)

    def _parse_udp_packet(self, packet: bytes, addr: Tuple[str, int]):
        if self.config.device_ip and addr[0] != self.config.device_ip:
            return
        pos = 0
        plen = len(packet)
        while pos + self._helper.UDP_HEADER_LEN <= plen:
            try:
                media_type, reserved, payload_len, seq, frame_size, offset, timestamp = struct.unpack_from('<BBHIIII', packet, pos)
            except struct.error:
                break
            pos += self._helper.UDP_HEADER_LEN
            if payload_len == 0 or pos + payload_len > plen:
                break
            payload = packet[pos: pos + payload_len]
            pos += payload_len
            base_type = media_type & 0x7F
            if base_type != self._helper.JPEG_TYPE_VIDEO:
                continue
            st = self.frames.get(seq)
            if st is None:
                st = self._helper.FrameState(seq=seq, frame_size=frame_size, timestamp=timestamp, media_type=media_type)
                self.frames[seq] = st
            st.add_chunk(offset=offset, payload=payload, is_last=bool(media_type & self._helper.LAST_VIDEO_MARKER))
            if st.is_complete():
                self.frames.pop(seq, None)
                self._decode_and_infer(st.to_bytes())

    def run(self) -> None:
        try:
            self._helper, self.detector = self._create_detector()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((self.config.bind_ip, self.config.port))
            self.sock.settimeout(1.0)
            self._log(f'[INFO] UDP 监听: {self.config.bind_ip}:{self.config.port}')
            self._log(f'[INFO] 过滤设备 IP: {self.config.device_ip}')
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
        except Exception as e:
            self.error_message.emit(f'{e}\n{traceback.format_exc()}')
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
            self._log('[INFO] 工作线程结束')
