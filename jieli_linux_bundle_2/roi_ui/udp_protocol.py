from __future__ import annotations

import time
from typing import Set


PCM_TYPE_AUDIO = 0x01
JPEG_TYPE_VIDEO = 0x02
LAST_VIDEO_MARKER = 0x80
UDP_HEADER_LEN = 20


class FrameState:
    def __init__(self, seq: int, frame_size: int, timestamp: int, media_type: int) -> None:
        self.seq = seq
        self.frame_size = frame_size
        self.timestamp = timestamp
        self.media_type = media_type
        self.buf = bytearray(frame_size)
        self.received = 0
        self.offsets: Set[int] = set()
        self.updated_at = time.time()

    def add_chunk(self, offset: int, payload: bytes, is_last: bool) -> None:
        end = offset + len(payload)
        if offset in self.offsets:
            return
        if offset < 0 or end > self.frame_size:
            return
        self.buf[offset:end] = payload
        self.offsets.add(offset)
        self.received += len(payload)
        self.updated_at = time.time()

    def is_complete(self) -> bool:
        return self.received >= self.frame_size

    def age(self) -> float:
        return time.time() - self.updated_at

    def to_bytes(self) -> bytes:
        return bytes(self.buf[: self.frame_size])
