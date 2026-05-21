from __future__ import annotations

from dataclasses import dataclass
import os

import cv2
import numpy as np


def _env_value(suffix: str, default: str) -> str:
    for prefix in ("FRAME_PREPROCESS", "IMAGE_PREPROCESS"):
        value = os.getenv(f"{prefix}_{suffix}")
        if value is not None:
            return value
    return default


def _env_bool(suffix: str, default: str) -> bool:
    return _env_value(suffix, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class FramePreprocessConfig:
    enabled: bool = True
    a_shift: int = 16
    contrast_alpha: float = 1.0
    brightness_beta: int = 0
    clahe_clip: float = 1.5
    clahe_grid: int = 8
    sharpen_amount: float = 0.7
    sharpen_sigma: float = 1.0
    display_processed: bool = False

    @classmethod
    def from_env(cls) -> "FramePreprocessConfig":
        return cls(
            enabled=_env_bool("ENABLED", "1"),
            a_shift=int(_env_value("A_SHIFT", "16")),
            contrast_alpha=float(_env_value("CONTRAST_ALPHA", "1.0")),
            brightness_beta=int(float(_env_value("BRIGHTNESS_BETA", "0"))),
            clahe_clip=float(_env_value("CLAHE_CLIP", "1.5")),
            clahe_grid=max(1, int(_env_value("CLAHE_GRID", "8"))),
            sharpen_amount=float(_env_value("SHARPEN_AMOUNT", "0.7")),
            sharpen_sigma=max(0.0, float(_env_value("SHARPEN_SIGMA", "1.0"))),
            display_processed=_env_bool("DISPLAY", "0"),
        )


class FramePreprocessor:
    def __init__(self, config: FramePreprocessConfig | None = None) -> None:
        self.config = config or FramePreprocessConfig.from_env()

    @classmethod
    def from_env(cls) -> "FramePreprocessor":
        return cls(FramePreprocessConfig.from_env())

    def summary(self) -> str:
        cfg = self.config
        if not cfg.enabled:
            return "disabled"
        parts = []
        if cfg.a_shift:
            parts.append(f"LAB A{cfg.a_shift:+d}")
        if abs(cfg.contrast_alpha - 1.0) > 1e-6 or cfg.brightness_beta:
            parts.append(f"linear alpha={cfg.contrast_alpha:g} beta={cfg.brightness_beta}")
        if cfg.clahe_clip > 0:
            parts.append(f"CLAHE L clip={cfg.clahe_clip:g} grid={cfg.clahe_grid}")
        if cfg.sharpen_amount > 0 and cfg.sharpen_sigma > 0:
            parts.append(f"unsharp amount={cfg.sharpen_amount:g} sigma={cfg.sharpen_sigma:g}")
        return " -> ".join(parts) if parts else "enabled(no-op)"

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        cfg = self.config
        if not cfg.enabled:
            return frame_bgr
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            return frame_bgr

        adjusted = frame_bgr
        if cfg.a_shift:
            adjusted = self._lab_shift_a(adjusted, cfg.a_shift)
        adjusted = self._linear_contrast(adjusted, cfg.contrast_alpha, cfg.brightness_beta)
        adjusted = self._clahe_l(adjusted, cfg.clahe_clip, cfg.clahe_grid)
        adjusted = self._unsharp_mask(adjusted, cfg.sharpen_amount, cfg.sharpen_sigma)
        return np.ascontiguousarray(adjusted)

    @staticmethod
    def _lab_shift_a(frame_bgr: np.ndarray, a_shift: int) -> np.ndarray:
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        shifted_a = np.clip(a_chan.astype(np.int16) + int(a_shift), 0, 255).astype(np.uint8)
        shifted_lab = cv2.merge([l_chan, shifted_a, b_chan])
        return cv2.cvtColor(shifted_lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _linear_contrast(frame_bgr: np.ndarray, alpha: float, beta: int) -> np.ndarray:
        if abs(alpha - 1.0) < 1e-6 and beta == 0:
            return frame_bgr
        adjusted = frame_bgr.astype(np.float32) * float(alpha) + int(beta)
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    @staticmethod
    def _clahe_l(frame_bgr: np.ndarray, clip_limit: float, grid_size: int) -> np.ndarray:
        if clip_limit <= 0:
            return frame_bgr
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=float(clip_limit),
            tileGridSize=(int(grid_size), int(grid_size)),
        )
        adjusted_l = clahe.apply(l_chan)
        adjusted_lab = cv2.merge([adjusted_l, a_chan, b_chan])
        return cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _unsharp_mask(frame_bgr: np.ndarray, amount: float, sigma: float) -> np.ndarray:
        if amount <= 0 or sigma <= 0:
            return frame_bgr
        blurred = cv2.GaussianBlur(frame_bgr, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
        sharpened = cv2.addWeighted(frame_bgr, 1.0 + float(amount), blurred, -float(amount), 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
