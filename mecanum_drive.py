"""Low-latency camera sharing and background crop inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any

import cv2


@dataclass(frozen=True)
class CameraConfig:
    source: int | str = 0
    width: int = 640
    height: int = 360
    fps: int = 30
    rotate_180: bool = True
    use_mjpg: bool = True   


@dataclass(frozen=True)
class FramePacket:
    sequence: int
    captured_at: float
    frame: Any


class LatestFrameCamera:
    """Capture continuously while retaining only the newest frame.

    A one-frame mailbox avoids latency growth when crop inference is slower
    than the camera. Consumers must treat ``FramePacket.frame`` as read-only.
    """

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._capture: cv2.VideoCapture | None = None
        self._latest: FramePacket | None = None
        self._condition = threading.Condition()
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: str | None = None

    @property
    def error(self) -> str | None:
        with self._condition:
            return self._error

    def start(self, first_frame_timeout: float = 3.0) -> "LatestFrameCamera":
        if self._running.is_set():
            return self

        self._capture = self._open_capture()
        self._running.set()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="latest-frame-camera",
            daemon=True,
        )
        self._thread.start()

        packet = self.wait_for_frame(timeout=first_frame_timeout)
        if packet is None:
            error = self.error or "camera produced no frames"
            self.stop()
            raise RuntimeError(error)
        return self

    def _open_capture(self) -> cv2.VideoCapture:
        source = self.config.source
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"could not open camera source {source!r}")

        if self.config.use_mjpg:
            capture.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*"MJPG"),
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        # Not every backend supports this, but V4L2 does on many USB cameras.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _capture_loop(self) -> None:
        assert self._capture is not None
        sequence = 0
        consecutive_failures = 0

        while self._running.is_set():
            ok, frame = self._capture.read()
            captured_at = time.monotonic()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    with self._condition:
                        self._error = "camera read failed five times in a row"
                        self._condition.notify_all()
                    self._running.clear()
                    break
                time.sleep(0.01)
                continue

            consecutive_failures = 0
            if self.config.rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            sequence += 1
            packet = FramePacket(sequence, captured_at, frame)
            with self._condition:
                self._latest = packet
                self._condition.notify_all()

    def latest(self) -> FramePacket | None:
        with self._condition:
            return self._latest

    def wait_for_frame(
        self,
        after_sequence: int = -1,
        timeout: float = 0.5,
    ) -> FramePacket | None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._running.is_set() or self._latest is not None:
                if self._latest is not None and self._latest.sequence > after_sequence:
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return None

    def stop(self) -> None:
        self._running.clear()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        if self._capture is not None:
            self._capture.release()
        self._thread = None
        self._capture = None

    def __enter__(self) -> "LatestFrameCamera":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()


@dataclass(frozen=True)
class CropDetection:
    label: str
    confidence: float
    xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class CropSnapshot:
    frame_sequence: int
    completed_at: float
    inference_ms: float
    detections: tuple[CropDetection, ...]
    annotated_frame: Any | None = None


class CropDetectionWorker:
    """Run the repository's Ultralytics ONNX crop model off the control loop."""

    def __init__(
        self,
        camera: LatestFrameCamera,
        model_path: str | Path,
        *,
        inference_fps: float = 2.0,
        image_size: int = 320,
        confidence: float = 0.40,
        device: str | None = None,
        make_annotated_frame: bool = False,
    ) -> None:
        if inference_fps <= 0:
            raise ValueError("inference_fps must be greater than zero")

        from ultralytics import YOLO

        self.camera = camera
        self.model_path = Path(model_path)
        self.inference_interval = 1.0 / inference_fps
        self.image_size = image_size
        self.confidence = confidence
        self.device = device
        self.make_annotated_frame = make_annotated_frame
        self.model = YOLO(str(self.model_path), task="detect")
        self._running = threading.Event()
        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: CropSnapshot | None = None
        self._error: str | None = None

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def latest(self) -> CropSnapshot | None:
        with self._lock:
            return self._latest

    def start(self) -> "CropDetectionWorker":
        if not self._running.is_set():
            self._stop_requested.clear()
            self._running.set()
            self._thread = threading.Thread(
                target=self._inference_loop,
                name="crop-detection",
                daemon=True,
            )
            self._thread.start()
        return self

    def _inference_loop(self) -> None:
        next_run = 0.0
        last_sequence = -1

        while self._running.is_set():
            delay = next_run - time.monotonic()
            if delay > 0 and self._stop_requested.wait(delay):
                break

            packet = self.camera.latest()
            if packet is None or packet.sequence == last_sequence:
                if self._stop_requested.wait(0.01):
                    break
                continue

            started = time.monotonic()
            try:
                predict_args: dict[str, Any] = {
                    "source": packet.frame,
                    "imgsz": self.image_size,
                    "conf": self.confidence,
                    "verbose": False,
                }
                if self.device:
                    predict_args["device"] = self.device
                result = self.model.predict(**predict_args)[0]
                completed = time.monotonic()
                detections = self._extract_detections(result)
                annotated = result.plot() if self.make_annotated_frame else None
                snapshot = CropSnapshot(
                    frame_sequence=packet.sequence,
                    completed_at=completed,
                    inference_ms=(completed - started) * 1000.0,
                    detections=detections,
                    annotated_frame=annotated,
                )
                with self._lock:
                    self._latest = snapshot
                    self._error = None
            except Exception as exc:  # Keep line following alive if inference fails.
                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"
                if self._stop_requested.wait(0.5):
                    break

            last_sequence = packet.sequence
            next_run = time.monotonic() + self.inference_interval

    @staticmethod
    def _extract_detections(result: Any) -> tuple[CropDetection, ...]:
        detections: list[CropDetection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return ()

        names = getattr(result, "names", {})
        for box in boxes:
            class_id = int(box.cls[0].item())
            if isinstance(names, dict):
                label = str(names.get(class_id, class_id))
            else:
                label = str(names[class_id])
            confidence = float(box.conf[0].item())
            coords = box.xyxy[0].tolist()
            detections.append(
                CropDetection(
                    label=label,
                    confidence=confidence,
                    xyxy=tuple(int(round(value)) for value in coords),
                )
            )
        return tuple(detections)

    def stop(self) -> None:
        self._running.clear()
        self._stop_requested.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
