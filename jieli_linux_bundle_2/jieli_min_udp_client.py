#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Set

import cv2
import numpy as np

PCM_TYPE_AUDIO = 0x01
JPEG_TYPE_VIDEO = 0x02
H264_TYPE_VIDEO = 0x03
PREVIEW_TYPE = 0x04
DATE_TIME_TYPE = 0x05
MEDIA_INFO_TYPE = 0x06
PLAY_OVER_TYPE = 0x07
LAST_VIDEO_MARKER = 0x80

UDP_HEADER_LEN = 20


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}")


def media_type_name(media_type: int) -> str:
    base = media_type & 0x7F
    last = bool(media_type & LAST_VIDEO_MARKER)
    mapping = {
        PCM_TYPE_AUDIO: "PCM_AUDIO",
        JPEG_TYPE_VIDEO: "JPEG_VIDEO",
        H264_TYPE_VIDEO: "H264_VIDEO",
        PREVIEW_TYPE: "PREVIEW",
        DATE_TIME_TYPE: "DATE_TIME",
        MEDIA_INFO_TYPE: "MEDIA_INFO",
        PLAY_OVER_TYPE: "PLAY_OVER",
    }
    name = mapping.get(base, f"UNKNOWN(0x{base:02X})")
    return f"{name}{'|LAST' if last else ''}"


class FrameState:
    def __init__(self, seq: int, frame_size: int, timestamp: int, media_type: int) -> None:
        self.seq = seq
        self.frame_size = frame_size
        self.timestamp = timestamp
        self.media_type = media_type
        self.buf = bytearray(frame_size)
        self.received = 0
        self.offsets: Set[int] = set()
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.last_seen = False

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
        if is_last:
            self.last_seen = True

    def is_complete(self) -> bool:
        return self.received >= self.frame_size

    def age(self) -> float:
        return time.time() - self.updated_at

    def to_bytes(self) -> bytes:
        return bytes(self.buf[: self.frame_size])


class JieliUdpClient:
    def __init__(
        self,
        bind_ip: str,
        bind_port: int,
        device_ip: Optional[str],
        save_dir: Optional[str],
        save_every: int,
        show_window: bool,
        cleanup_timeout: float,
        verbose: bool,
    ) -> None:
        self.bind_ip = bind_ip
        self.bind_port = bind_port
        self.device_ip = device_ip
        self.save_dir = Path(save_dir) if save_dir else None
        self.save_every = max(1, save_every)
        self.show_window = show_window
        self.cleanup_timeout = cleanup_timeout
        self.verbose = verbose

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.bind_ip, self.bind_port))
        self.sock.settimeout(1.0)

        self.frames: Dict[int, FrameState] = {}
        self.packet_count = 0
        self.frame_count = 0
        self.saved_count = 0
        self.fps = 0.0
        self.last_frame_wall_time: Optional[float] = None

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        log(f"[INFO] listening UDP {self.bind_ip}:{self.bind_port}")
        if self.device_ip:
            log(f"[INFO] only accept packets from device_ip={self.device_ip}")
        else:
            log("[INFO] device_ip filter disabled")
        if self.save_dir:
            log(f"[INFO] save_dir={self.save_dir.resolve()} save_every={self.save_every}")
        else:
            log("[INFO] saving disabled")
        log(f"[INFO] show_window={'on' if self.show_window else 'off'}")

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass
        if self.show_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        log("[OK] udp client closed")

    def save_jpeg(self, seq: int, jpeg: bytes, timestamp: int) -> None:
        if not self.save_dir:
            return
        now_ms = int(time.time() * 1000)
        path = self.save_dir / f"frame_{seq}_{timestamp}_{now_ms}.jpg"
        with open(path, "wb") as f:
            f.write(jpeg)
        self.saved_count += 1
        log(f"[SAVE] {path}")

    def decode_and_show(self, seq: int, jpeg: bytes, timestamp: int) -> bool:
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            log(f"[DROP] seq={seq} jpeg decode failed")
            return False

        now = time.time()
        if self.last_frame_wall_time is not None:
            dt = now - self.last_frame_wall_time
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0.0 else (0.9 * self.fps + 0.1 * inst)
        self.last_frame_wall_time = now

        text1 = f"FPS: {self.fps:.1f}  SEQ: {seq}"
        text2 = f"TS: {timestamp}  SIZE: {len(jpeg)}"
        cv2.putText(img, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(img, text2, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Jieli JPEG Stream", img)
        key = cv2.waitKey(1) & 0xFF
        return key in (27, ord("q"))

    def handle_complete_frame(self, state: FrameState) -> bool:
        self.frame_count += 1
        payload = state.to_bytes()
        if not payload.startswith(b"\xFF\xD8"):
            log(f"[DROP] seq={state.seq} not jpeg SOI, size={len(payload)}")
            return False
        if b"\xFF\xD9" not in payload:
            log(f"[DROP] seq={state.seq} no jpeg EOI, size={len(payload)}")
            return False

        if self.verbose:
            log(
                f"[FRAME] seq={state.seq} type={media_type_name(state.media_type)} "
                f"size={state.frame_size} recv={state.received} ts={state.timestamp}"
            )

        should_quit = False
        if self.show_window:
            should_quit = self.decode_and_show(state.seq, payload, state.timestamp)

        if self.save_dir and (self.frame_count % self.save_every == 0):
            self.save_jpeg(state.seq, payload, state.timestamp)
        return should_quit

    def cleanup_stale_frames(self) -> None:
        stale_keys = [seq for seq, st in self.frames.items() if st.age() > self.cleanup_timeout]
        for seq in stale_keys:
            st = self.frames.pop(seq)
            if self.verbose:
                log(
                    f"[CLEAN] drop stale frame seq={seq} recv={st.received}/{st.frame_size} age={st.age():.2f}s"
                )

    def parse_udp_packet(self, packet: bytes, addr: tuple[str, int]) -> bool:
        self.packet_count += 1
        if self.device_ip and addr[0] != self.device_ip:
            return False

        pos = 0
        plen = len(packet)
        should_quit = False
        while pos + UDP_HEADER_LEN <= plen:
            try:
                media_type, reserved, payload_len, seq, frame_size, offset, timestamp = struct.unpack_from(
                    "<BBHIIII", packet, pos
                )
            except struct.error:
                break
            pos += UDP_HEADER_LEN

            if payload_len == 0:
                continue
            if pos + payload_len > plen:
                if self.verbose:
                    log(f"[WARN] malformed packet from {addr[0]}:{addr[1]} payload_len={payload_len} remain={plen - pos}")
                break

            payload = packet[pos: pos + payload_len]
            pos += payload_len

            base_type = media_type & 0x7F
            is_last = bool(media_type & LAST_VIDEO_MARKER)

            if self.verbose:
                log(
                    f"[CHUNK] from={addr[0]}:{addr[1]} type={media_type_name(media_type)} "
                    f"seq={seq} frame_size={frame_size} offset={offset} payload_len={payload_len}"
                )

            if base_type != JPEG_TYPE_VIDEO:
                continue

            state = self.frames.get(seq)
            if state is None:
                state = FrameState(seq=seq, frame_size=frame_size, timestamp=timestamp, media_type=media_type)
                self.frames[seq] = state

            state.add_chunk(offset=offset, payload=payload, is_last=is_last)
            if state.is_complete():
                self.frames.pop(seq, None)
                should_quit = self.handle_complete_frame(state)
                if should_quit:
                    return True
        return should_quit

    def run(self) -> int:
        last_stat_time = time.time()
        try:
            while True:
                try:
                    packet, addr = self.sock.recvfrom(65535)
                except socket.timeout:
                    self.cleanup_stale_frames()
                    now = time.time()
                    if self.verbose and now - last_stat_time >= 5.0:
                        log(
                            f"[STAT] packets={self.packet_count} frames={self.frame_count} "
                            f"saved={self.saved_count} inflight={len(self.frames)} fps={self.fps:.1f}"
                        )
                        last_stat_time = now
                    continue

                should_quit = self.parse_udp_packet(packet, addr)
                self.cleanup_stale_frames()
                now = time.time()
                if self.verbose and now - last_stat_time >= 5.0:
                    log(
                        f"[STAT] packets={self.packet_count} frames={self.frame_count} "
                        f"saved={self.saved_count} inflight={len(self.frames)} fps={self.fps:.1f}"
                    )
                    last_stat_time = now
                if should_quit:
                    log("[INFO] quit by keyboard in window")
                    return 0
        except KeyboardInterrupt:
            print()
            log("[INFO] stopped by Ctrl+C")
            return 0
        finally:
            self.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="杰理 UDP JPEG 实时流接收器")
    p.add_argument("--bind-ip", default="0.0.0.0", help="本地绑定 IP，默认 0.0.0.0")
    p.add_argument("--port", type=int, default=2224, help="前视实时流 UDP 端口，默认 2224")
    p.add_argument("--device-ip", default="192.168.1.1", help="设备 IP 过滤，默认 192.168.1.1")
    p.add_argument("--no-filter", action="store_true", help="关闭 device_ip 过滤")
    p.add_argument("--save-dir", default=None, help="保存 JPEG 的目录；留空则不保存")
    p.add_argument("--save-every", type=int, default=30, help="每隔多少帧保存 1 张，默认 30")
    p.add_argument("--no-window", action="store_true", help="不显示窗口，只接收/可选保存")
    p.add_argument("--cleanup-timeout", type=float, default=3.0, help="分片帧清理超时，默认 3 秒")
    p.add_argument("--verbose", action="store_true", help="打印每个分片和统计信息")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    client = JieliUdpClient(
        bind_ip=args.bind_ip,
        bind_port=args.port,
        device_ip=None if args.no_filter else args.device_ip,
        save_dir=args.save_dir,
        save_every=args.save_every,
        show_window=not args.no_window,
        cleanup_timeout=args.cleanup_timeout,
        verbose=args.verbose,
    )
    return client.run()


if __name__ == "__main__":
    sys.exit(main())
