from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import tensorrt as trt
except Exception as exc:  # pragma: no cover - environment dependent
    raise RuntimeError("TensorRT Python package is required for DETECTOR_BACKEND=tensorrt") from exc

try:
    from cuda import cudart
except Exception:
    try:
        from cuda.bindings import runtime as cudart
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("cuda-python is required for TensorRT inference") from exc


Detection = Tuple[int, float, Tuple[int, int, int, int]]


@dataclass
class LetterBoxInfo:
    ratio: float
    pad_w: float
    pad_h: float
    new_w: int
    new_h: int


def _cuda_code(err) -> int:
    if hasattr(err, "value"):
        return int(err.value)
    return int(err)


def _check_cuda(result, op: str):
    if isinstance(result, tuple):
        err = result[0]
        values = result[1:]
    else:
        err = result
        values = ()
    if _cuda_code(err) != 0:
        raise RuntimeError(f"{op} failed: {err}")
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values


class YoloTensorRTDetector:
    def __init__(
        self,
        model_path: str,
        labels_path: Optional[str],
        input_size: Tuple[int, int],
        obj_thresh: float,
        nms_thresh: float,
        max_det: int,
        agnostic_nms: bool,
        use_rgb: bool,
        class_filter: Sequence[int] | None = None,
        verbose: bool = False,
    ) -> None:
        self.model_path = str(model_path)
        self.labels = self._load_labels(labels_path)
        self.input_h, self.input_w = input_size
        self.obj_thresh = obj_thresh
        self.nms_thresh = nms_thresh
        self.max_det = max(1, max_det)
        self.agnostic_nms = agnostic_nms
        self.use_rgb = use_rgb
        self.class_filter = set(class_filter or [])
        self.verbose = verbose

        engine_path = Path(self.model_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        self.logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.stream = _check_cuda(cudart.cudaStreamCreate(), "cudaStreamCreate")
        self.input_name = ""
        self.output_names: List[str] = []
        self.tensor_shapes: dict[str, Tuple[int, ...]] = {}
        self.tensor_dtypes: dict[str, np.dtype] = {}
        self.host_buffers: dict[str, np.ndarray] = {}
        self.device_buffers: dict[str, int] = {}
        self.input_layout = "nchw"

        self._configure_tensors()
        print(f"[OK] TensorRT engine loaded: {engine_path}")
        print(f"[INFO] TensorRT input: {self.input_name} {self.tensor_shapes[self.input_name]}")
        print(f"[INFO] TensorRT outputs: {[(name, self.tensor_shapes[name]) for name in self.output_names]}")
        if self.class_filter:
            print(f"[INFO] Class filter: {sorted(self.class_filter)}")

    @staticmethod
    def _load_labels(labels_path: Optional[str]) -> List[str]:
        if labels_path is None:
            return []
        path = Path(labels_path)
        if not path.exists():
            return []
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def close(self) -> None:
        for ptr in self.device_buffers.values():
            try:
                _check_cuda(cudart.cudaFree(ptr), "cudaFree")
            except Exception:
                pass
        self.device_buffers.clear()
        if getattr(self, "stream", None):
            try:
                _check_cuda(cudart.cudaStreamDestroy(self.stream), "cudaStreamDestroy")
            except Exception:
                pass
            self.stream = None

    def _configure_tensors(self) -> None:
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)

        if not self.input_name:
            raise RuntimeError("TensorRT engine has no input tensor")
        if not self.output_names:
            raise RuntimeError("TensorRT engine has no output tensors")

        input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        if len(input_shape) != 4:
            raise RuntimeError(f"Expected a 4D YOLO input tensor, got {input_shape}")

        if input_shape[1] == 3:
            self.input_layout = "nchw"
            n, c, h, w = input_shape
            desired_shape = (1, 3, self.input_h, self.input_w)
        elif input_shape[-1] == 3:
            self.input_layout = "nhwc"
            n, h, w, c = input_shape
            desired_shape = (1, self.input_h, self.input_w, 3)
        else:
            raise RuntimeError(f"Cannot determine input tensor layout from shape {input_shape}")

        if any(dim < 0 for dim in input_shape):
            if not self.context.set_input_shape(self.input_name, desired_shape):
                raise RuntimeError(f"Failed to set dynamic input shape: {desired_shape}")
            actual_shape = desired_shape
        else:
            actual_shape = input_shape
            if self.input_layout == "nchw":
                self.input_h, self.input_w = int(actual_shape[2]), int(actual_shape[3])
            else:
                self.input_h, self.input_w = int(actual_shape[1]), int(actual_shape[2])

        self.tensor_shapes[self.input_name] = tuple(int(x) for x in actual_shape)

        for name in [self.input_name] + self.output_names:
            shape = tuple(int(x) for x in self.context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(f"Tensor {name} still has dynamic shape after configuration: {shape}")
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            self.tensor_shapes[name] = shape
            self.tensor_dtypes[name] = dtype
            host = np.empty(int(trt.volume(shape)), dtype=dtype)
            device = _check_cuda(cudart.cudaMalloc(host.nbytes), f"cudaMalloc({name})")
            self.host_buffers[name] = host
            self.device_buffers[name] = int(device)
            self.context.set_tensor_address(name, int(device))

    def _letterbox(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, LetterBoxInfo]:
        h, w = image_bgr.shape[:2]
        ratio = min(self.input_w / w, self.input_h / h)
        new_w = int(round(w * ratio))
        new_h = int(round(h * ratio))

        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)

        pad_w = (self.input_w - new_w) / 2.0
        pad_h = (self.input_h - new_h) / 2.0
        left = int(round(pad_w - 0.1))
        top = int(round(pad_h - 0.1))
        canvas[top: top + new_h, left: left + new_w] = resized

        if self.use_rgb:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        return canvas, LetterBoxInfo(ratio, pad_w, pad_h, new_w, new_h)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]

        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter + 1e-6
            iou = inter / union

            inds = np.where(iou <= iou_thr)[0]
            order = order[inds + 1]

        return keep

    def _preprocess(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, LetterBoxInfo]:
        canvas, lb = self._letterbox(image_bgr)
        blob = canvas.astype(np.float32) / 255.0
        if self.input_layout == "nchw":
            blob = np.transpose(blob, (2, 0, 1))[None, ...]
        else:
            blob = blob[None, ...]
        input_dtype = self.tensor_dtypes[self.input_name]
        return np.ascontiguousarray(blob.astype(input_dtype, copy=False)), lb

    def infer(self, image_bgr: np.ndarray) -> List[Detection]:
        blob, lb = self._preprocess(image_bgr)
        host_in = self.host_buffers[self.input_name]
        np.copyto(host_in, blob.ravel())
        _check_cuda(
            cudart.cudaMemcpyAsync(
                self.device_buffers[self.input_name],
                host_in.ctypes.data,
                host_in.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            ),
            "cudaMemcpyAsync(H2D)",
        )

        if not self.context.execute_async_v3(stream_handle=self.stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")

        outputs: List[np.ndarray] = []
        for name in self.output_names:
            host = self.host_buffers[name]
            _check_cuda(
                cudart.cudaMemcpyAsync(
                    host.ctypes.data,
                    self.device_buffers[name],
                    host.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.stream,
                ),
                "cudaMemcpyAsync(D2H)",
            )
            outputs.append(host.reshape(self.tensor_shapes[name]).copy())

        _check_cuda(cudart.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")
        return self._postprocess(outputs, lb, image_bgr.shape[:2])

    def _postprocess(
        self,
        outputs: Sequence[np.ndarray],
        lb: LetterBoxInfo,
        orig_shape: Tuple[int, int],
    ) -> List[Detection]:
        if len(outputs) != 1:
            for output in outputs:
                arr = np.squeeze(np.asarray(output))
                if arr.ndim == 2 and arr.shape[1] in {6, 7}:
                    return self._postprocess_nms_output(arr, lb, orig_shape)
            raise ValueError(f"Only single-output YOLO engines are supported, got {len(outputs)} outputs")
        return self._postprocess_single_output(outputs[0], lb, orig_shape)

    def _postprocess_single_output(
        self,
        output: np.ndarray,
        lb: LetterBoxInfo,
        orig_shape: Tuple[int, int],
    ) -> List[Detection]:
        arr = np.squeeze(np.asarray(output))
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D YOLO output tensor, got {output.shape}")

        if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 512:
            arr = arr.T

        if arr.shape[1] in {6, 7} and np.allclose(arr[:, 5], np.round(arr[:, 5]), atol=1e-3):
            return self._postprocess_nms_output(arr, lb, orig_shape)

        if arr.shape[1] < 5:
            raise ValueError(f"YOLO output must have at least 5 columns, got {arr.shape}")

        boxes_xywh = arr[:, :4].astype(np.float32)
        cls_scores = arr[:, 4:].astype(np.float32)

        if cls_scores.size == 0:
            return []
        if cls_scores.max() > 1.0 or cls_scores.min() < 0.0:
            cls_scores = 1.0 / (1.0 + np.exp(-cls_scores))

        if cls_scores.shape[1] == 1:
            scores = cls_scores[:, 0]
            class_ids = np.zeros((arr.shape[0],), dtype=np.int32)
        else:
            class_ids = np.argmax(cls_scores, axis=1).astype(np.int32)
            scores = np.max(cls_scores, axis=1).astype(np.float32)

        mask = scores > self.obj_thresh
        if self.class_filter:
            mask &= np.isin(class_ids, list(self.class_filter))
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if boxes_xywh.max() <= 2.0:
            boxes_xywh[:, [0, 2]] *= self.input_w
            boxes_xywh[:, [1, 3]] *= self.input_h

        cx = boxes_xywh[:, 0]
        cy = boxes_xywh[:, 1]
        w = boxes_xywh[:, 2]
        h = boxes_xywh[:, 3]
        boxes = np.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1)
        return self._finish_boxes(boxes, scores, class_ids, lb, orig_shape)

    def _postprocess_nms_output(
        self,
        arr: np.ndarray,
        lb: LetterBoxInfo,
        orig_shape: Tuple[int, int],
    ) -> List[Detection]:
        arr = arr.astype(np.float32)
        scores = arr[:, 4]
        class_ids = arr[:, 5].astype(np.int32)
        boxes = arr[:, :4]

        mask = scores > self.obj_thresh
        if self.class_filter:
            mask &= np.isin(class_ids, list(self.class_filter))
        if not np.any(mask):
            return []
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        if boxes.max() <= 2.0:
            boxes[:, [0, 2]] *= self.input_w
            boxes[:, [1, 3]] *= self.input_h
        return self._finish_boxes(boxes, scores, class_ids, lb, orig_shape)

    def _finish_boxes(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        lb: LetterBoxInfo,
        orig_shape: Tuple[int, int],
    ) -> List[Detection]:
        orig_h, orig_w = orig_shape
        boxes[:, [0, 2]] -= lb.pad_w
        boxes[:, [1, 3]] -= lb.pad_h
        boxes[:, :4] /= lb.ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)

        wh = boxes[:, 2:4] - boxes[:, 0:2]
        valid = (wh[:, 0] > 1) & (wh[:, 1] > 1)
        boxes = boxes[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]
        if len(boxes) == 0:
            return []

        results: List[Detection] = []
        if self.agnostic_nms:
            keep = self._nms(boxes, scores, self.nms_thresh)[: self.max_det]
            for k in keep:
                x1, y1, x2, y2 = boxes[k]
                results.append((int(class_ids[k]), float(scores[k]), (int(x1), int(y1), int(x2), int(y2))))
            return results

        for cls_id in np.unique(class_ids):
            inds = np.where(class_ids == cls_id)[0]
            cls_boxes = boxes[inds]
            cls_scores = scores[inds]
            keep = self._nms(cls_boxes, cls_scores, self.nms_thresh)
            for k in keep:
                x1, y1, x2, y2 = cls_boxes[k]
                results.append((int(cls_id), float(cls_scores[k]), (int(x1), int(y1), int(x2), int(y2))))

        results.sort(key=lambda item: item[1], reverse=True)
        return results[: self.max_det]
