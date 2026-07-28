#!/usr/bin/env python3
"""Mr Sparky: binary camera, crop detection, and mecanum line following.

This is the self-contained version of the robot program. It only needs the
external ``best.onnx`` model and the Python packages listed in the README.

Safe camera test:
    python3 sparky_all_in_one.py --binary-output

Powered run:
    python3 sparky_all_in_one.py --arm --port /dev/ttyUSB0
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import struct
import threading
import time
from typing import Any

# Leave CPU time for camera capture and motor control on a Raspberry Pi 5.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

try:
    import cv2
    import numpy as np
    import serial
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing package: {exc.name}. Activate .venv and install the requirements."
    ) from exc


# ---------------------------------------------------------------------------
# Minimal Yahboom/Sparky serial motor controller
# ---------------------------------------------------------------------------


class SparkyMotorController:
    HEAD = 0xFF
    DEVICE_ID = 0xFC
    COMPLEMENT = 257 - DEVICE_ID
    MOTOR_FUNCTION = 0x10

    def __init__(self, port: str, baudrate: int = 115200) -> None:
        self.port = port
        self.baudrate = baudrate
        self.serial_port: serial.Serial | None = None
        self._write_lock = threading.Lock()
        self._drain_running = threading.Event()
        self._drain_thread: threading.Thread | None = None

    def connect(self) -> bool:
        try:
            self.serial_port = serial.Serial(
                self.port,
                self.baudrate,
                timeout=0.1,
                write_timeout=0.25,
            )
            self._drain_running.set()
            self._drain_thread = threading.Thread(
                target=self._drain_input,
                name="robot-serial-input",
                daemon=True,
            )
            self._drain_thread.start()
            return self.serial_port.is_open
        except (OSError, serial.SerialException) as exc:
            print(f"Robot connection failed: {exc}")
            return False

    def _drain_input(self) -> None:
        while self._drain_running.is_set():
            port = self.serial_port
            if port is None or not port.is_open:
                return
            try:
                port.read(max(1, port.in_waiting))
            except (OSError, serial.SerialException):
                return

    @staticmethod
    def _limit(value: float) -> int:
        return max(-100, min(100, int(round(value))))

    def set_motor(self, m1: int, m2: int, m3: int, m4: int) -> None:
        port = self.serial_port
        if port is None or not port.is_open:
            raise RuntimeError("robot serial port is not open")
        speeds = [self._limit(value) for value in (m1, m2, m3, m4)]
        packed = [struct.pack("b", value)[0] for value in speeds]
        command = [self.HEAD, self.DEVICE_ID, 0, self.MOTOR_FUNCTION, *packed]
        command[2] = len(command) - 1
        command.append((sum(command) + self.COMPLEMENT) & 0xFF)
        with self._write_lock:
            port.write(bytes(command))

    def disconnect(self) -> None:
        port = self.serial_port
        if port is not None and port.is_open:
            try:
                self.set_motor(0, 0, 0, 0)
            except Exception:
                pass
        self._drain_running.clear()
        if self._drain_thread is not None:
            self._drain_thread.join(timeout=0.5)
        if port is not None:
            port.close()
        self.serial_port = None


class DryRunRobot:
    def set_motor(self, _m1: int, _m2: int, _m3: int, _m4: int) -> None:
        pass


# ---------------------------------------------------------------------------
# Mecanum drive
# ---------------------------------------------------------------------------


def mix_mecanum(
    forward: float,
    strafe_right: float,
    turn_right: float,
    max_power: float,
) -> tuple[int, int, int, int]:
    # Signs match the original working mr_sparky.py motor directions.
    wheels = [
        forward + strafe_right + turn_right,
        -forward + strafe_right + turn_right,
        -forward + strafe_right - turn_right,
        forward + strafe_right - turn_right,
    ]
    peak = max(abs(value) for value in wheels)
    if peak > max_power:
        scale = max_power / peak
        wheels = [value * scale for value in wheels]
    return tuple(int(round(value)) for value in wheels)  # type: ignore[return-value]


class MecanumDrive:
    def __init__(
        self,
        robot: Any,
        max_power: float,
        slew_rate: float = 180.0,
    ) -> None:
        self.robot = robot
        self.max_power = max_power
        self.slew_rate = slew_rate
        self.applied = [0.0, 0.0, 0.0, 0.0]
        self.last_update: float | None = None
        self.last_sent: tuple[int, int, int, int] | None = None
        self.last_send_time = 0.0

    def command(
        self,
        forward: float,
        strafe_right: float,
        turn_right: float,
        now: float,
    ) -> tuple[int, int, int, int]:
        target = mix_mecanum(
            forward,
            strafe_right,
            turn_right,
            self.max_power,
        )
        dt = 1.0 / 30.0 if self.last_update is None else now - self.last_update
        dt = max(0.0, min(0.25, dt))
        change_limit = self.slew_rate * dt
        for index, requested in enumerate(target):
            change = requested - self.applied[index]
            change = max(-change_limit, min(change_limit, change))
            self.applied[index] += change
        self.last_update = now

        output = tuple(int(round(value)) for value in self.applied)
        if output != self.last_sent or now - self.last_send_time >= 0.25:
            self.robot.set_motor(*output)
            self.last_sent = output
            self.last_send_time = now
        return output  # type: ignore[return-value]

    def stop(self) -> None:
        self.robot.set_motor(0, 0, 0, 0)
        self.applied = [0.0, 0.0, 0.0, 0.0]
        self.last_update = None
        self.last_sent = (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Latest-frame camera and binary output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FramePacket:
    sequence: int
    captured_at: float
    frame: Any


class LatestFrameCamera:
    def __init__(
        self,
        source: int,
        width: int,
        height: int,
        fps: int,
        rotate_180: bool,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.rotate_180 = rotate_180
        self.capture: cv2.VideoCapture | None = None
        self.latest_packet: FramePacket | None = None
        self.condition = threading.Condition()
        self.running = threading.Event()
        self.thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self) -> None:
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise RuntimeError(f"could not open camera {self.source}")
        self.capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.running.set()
        self.thread = threading.Thread(
            target=self._capture_loop,
            name="latest-camera-frame",
            daemon=True,
        )
        self.thread.start()
        if self.wait_for_frame(-1, 3.0) is None:
            self.stop()
            raise RuntimeError(self.error or "camera produced no frames")

    def _capture_loop(self) -> None:
        assert self.capture is not None
        sequence = 0
        failures = 0
        while self.running.is_set():
            ok, frame = self.capture.read()
            captured_at = time.monotonic()
            if not ok or frame is None:
                failures += 1
                if failures >= 5:
                    self.error = "camera read failed five times"
                    self.running.clear()
                    with self.condition:
                        self.condition.notify_all()
                    return
                time.sleep(0.01)
                continue
            failures = 0
            if self.rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            sequence += 1
            with self.condition:
                self.latest_packet = FramePacket(sequence, captured_at, frame)
                self.condition.notify_all()

    def latest(self) -> FramePacket | None:
        with self.condition:
            return self.latest_packet

    def wait_for_frame(
        self,
        after_sequence: int,
        timeout: float,
    ) -> FramePacket | None:
        deadline = time.monotonic() + timeout
        with self.condition:
            while time.monotonic() < deadline:
                if (
                    self.latest_packet is not None
                    and self.latest_packet.sequence > after_sequence
                ):
                    return self.latest_packet
                self.condition.wait(max(0.0, deadline - time.monotonic()))
        return None

    def stop(self) -> None:
        self.running.clear()
        with self.condition:
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.capture is not None:
            self.capture.release()


def binary_frame(frame: Any, line_color: str) -> Any:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold_type = (
        cv2.THRESH_BINARY_INV
        if line_color == "dark"
        else cv2.THRESH_BINARY
    )
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        threshold_type | cv2.THRESH_OTSU,
    )
    return binary


# ---------------------------------------------------------------------------
# Line detection and following
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineObservation:
    found: bool
    lateral: float = 0.0
    heading: float = 0.0
    confidence: float = 0.0
    roi_top: int = 0
    near: tuple[int, int] | None = None
    far: tuple[int, int] | None = None


class LineDetector:
    def __init__(self, line_color: str, roi_top_fraction: float = 0.48) -> None:
        self.line_color = line_color
        self.roi_top_fraction = roi_top_fraction
        self.previous_x: float | None = None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: Any) -> LineObservation:
        frame_height, frame_width = frame.shape[:2]
        roi_top = int(frame_height * self.roi_top_fraction)
        roi = frame[roi_top:, :]
        roi_height = roi.shape[0]
        mask = binary_frame(roi, self.line_color)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        image_area = float(roi_height * frame_width)
        reference_x = frame_width / 2 if self.previous_x is None else self.previous_x
        best_index = None
        best_score = -1.0
        for index in range(1, count):
            area = float(stats[index, cv2.CC_STAT_AREA])
            if not image_area * 0.004 <= area <= image_area * 0.55:
                continue
            bottom = (
                stats[index, cv2.CC_STAT_TOP]
                + stats[index, cv2.CC_STAT_HEIGHT]
            )
            bottom_score = bottom / max(1.0, roi_height)
            continuity = 1.0 - min(
                1.0,
                abs(float(centroids[index, 0]) - reference_x) / frame_width,
            )
            score = area * (0.55 + 0.30 * bottom_score + 0.15 * continuity)
            if score > best_score:
                best_score = score
                best_index = index

        if best_index is None:
            return LineObservation(False, roi_top=roi_top)

        selected = np.where(labels == best_index, 255, 0).astype(np.uint8)
        points = cv2.findNonZero(selected)
        if points is None or len(points) < 5:
            return LineObservation(False, roi_top=roi_top)

        vx, vy, x0, y0 = (
            float(value)
            for value in cv2.fitLine(
                points,
                cv2.DIST_L2,
                0,
                0.01,
                0.01,
            ).reshape(-1)
        )
        if abs(vy) < 1e-5:
            near_x = far_x = float(centroids[best_index, 0])
        else:
            near_y = roi_height - 1
            far_y = roi_height * 0.25
            near_x = x0 + (near_y - y0) * vx / vy
            far_x = x0 + (far_y - y0) * vx / vy
        near_x = float(np.clip(near_x, -0.25 * frame_width, 1.25 * frame_width))
        far_x = float(np.clip(far_x, -0.25 * frame_width, 1.25 * frame_width))

        lateral = float(np.clip(
            (near_x - frame_width / 2) / (frame_width / 2),
            -1.5,
            1.5,
        ))
        heading = float(np.clip(
            np.arctan2(far_x - near_x, roi_height * 0.75) / (np.pi / 4),
            -1.0,
            1.0,
        ))
        area = float(stats[best_index, cv2.CC_STAT_AREA])
        confidence = min(1.0, area / max(1.0, image_area * 0.024))
        self.previous_x = near_x
        return LineObservation(
            True,
            lateral,
            heading,
            confidence,
            roi_top,
            (int(round(near_x)), frame_height - 1),
            (int(round(far_x)), roi_top + int(roi_height * 0.25)),
        )


@dataclass(frozen=True)
class Motion:
    forward: float
    strafe: float
    turn: float
    state: str


class LineFollower:
    def __init__(self, speed: float) -> None:
        self.speed = speed
        self.filtered_error = 0.0
        self.previous_error = 0.0
        self.last_seen: float | None = None
        self.last_update: float | None = None
        self.last_direction = 1.0
        self.detections = 0
        self.acquired = False
        self.last_motion = Motion(0, 0, 0, "waiting")

    def update(self, line: LineObservation, now: float) -> Motion:
        if line.found:
            self.detections += 1
            if not self.acquired and self.detections < 2:
                return Motion(0, 0, 0, "waiting")
            self.acquired = True
            return self._track(line, now)

        self.detections = 0
        if self.last_seen is None:
            return Motion(0, 0, 0, "waiting")
        lost_for = now - self.last_seen
        if lost_for <= 0.18:
            return Motion(
                self.speed * 0.3,
                self.last_motion.strafe * 0.5,
                self.last_motion.turn * 0.5,
                "grace",
            )
        if lost_for <= 1.68:
            return Motion(
                0,
                self.last_direction * 8,
                self.last_direction * 14,
                "searching",
            )
        self.acquired = False
        self.last_seen = None
        return Motion(0, 0, 0, "stopped")

    def _track(self, line: LineObservation, now: float) -> Motion:
        self.filtered_error = 0.35 * line.lateral + 0.65 * self.filtered_error
        dt = 1.0 / 30.0 if self.last_update is None else now - self.last_update
        dt = max(0.01, min(0.2, dt))
        derivative = np.clip(
            (self.filtered_error - self.previous_error) / dt,
            -3.0,
            3.0,
        )
        strafe = 20.0 * self.filtered_error + 1.8 * derivative
        turn = 13.0 * self.filtered_error + 26.0 * line.heading
        severity = min(
            1.0,
            0.65 * abs(self.filtered_error) + 0.75 * abs(line.heading),
        )
        minimum_speed = min(14.0, self.speed * 0.55)
        forward = max(
            minimum_speed,
            self.speed
            * (1.0 - 0.55 * severity)
            * (0.65 + 0.35 * line.confidence),
        )
        direction_error = (
            self.filtered_error
            if abs(self.filtered_error) > 0.03
            else line.heading
        )
        if abs(direction_error) > 0.03:
            self.last_direction = 1.0 if direction_error > 0 else -1.0
        self.previous_error = self.filtered_error
        self.last_update = now
        self.last_seen = now
        self.last_motion = Motion(forward, strafe, turn, "tracking")
        return self.last_motion


# ---------------------------------------------------------------------------
# Background crop detector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CropResult:
    inference_ms: float
    labels: tuple[str, ...]
    annotated: Any | None


class CropWorker:
    def __init__(
        self,
        camera: LatestFrameCamera,
        model_path: Path,
        fps: float,
        display: bool,
    ) -> None:
        from ultralytics import YOLO

        self.camera = camera
        self.model = YOLO(str(model_path), task="detect")
        self.interval = 1.0 / fps
        self.display = display
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.result: CropResult | None = None
        self.error: str | None = None
        self.thread = threading.Thread(
            target=self._loop,
            name="crop-detection",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def _loop(self) -> None:
        last_sequence = -1
        while not self.stop_event.is_set():
            packet = self.camera.latest()
            if packet is None or packet.sequence == last_sequence:
                if self.stop_event.wait(0.02):
                    return
                continue
            started = time.monotonic()
            try:
                prediction = self.model.predict(
                    source=packet.frame,
                    imgsz=320,
                    conf=0.40,
                    verbose=False,
                )[0]
                labels = []
                for box in prediction.boxes:
                    class_id = int(box.cls[0].item())
                    labels.append(str(prediction.names[class_id]))
                result = CropResult(
                    (time.monotonic() - started) * 1000,
                    tuple(labels),
                    prediction.plot() if self.display else None,
                )
                with self.lock:
                    self.result = result
                    self.error = None
            except Exception as exc:
                with self.lock:
                    self.error = f"{type(exc).__name__}: {exc}"
            last_sequence = packet.sequence
            if self.stop_event.wait(self.interval):
                return

    def latest(self) -> tuple[CropResult | None, str | None]:
        with self.lock:
            return self.result, self.error

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    arguments = argparse.ArgumentParser(
        description="All-in-one Mr Sparky line follower and crop detector."
    )
    arguments.add_argument("--camera", type=int, default=0)
    arguments.add_argument("--port", default="/dev/ttyUSB0")
    arguments.add_argument("--model", type=Path, default=Path(__file__).with_name("best.onnx"))
    arguments.add_argument("--width", type=int, default=640)
    arguments.add_argument("--height", type=int, default=360)
    arguments.add_argument("--camera-fps", type=int, default=30)
    arguments.add_argument("--crop-fps", type=float, default=2.0)
    arguments.add_argument("--speed", type=float, default=32.0)
    arguments.add_argument("--max-power", type=float, default=55.0)
    arguments.add_argument("--line-color", choices=("dark", "light"), default="dark")
    arguments.add_argument("--no-rotate", action="store_true")
    arguments.add_argument("--disable-crop-detection", action="store_true")
    arguments.add_argument("--binary-output", action="store_true")
    arguments.add_argument("--display", action="store_true")
    arguments.add_argument(
        "--arm",
        action="store_true",
        help="Allow real motor output. Without this flag, motors cannot move.",
    )
    return arguments


def crop_status(worker: CropWorker | None) -> str:
    if worker is None:
        return "crop=off"
    result, error = worker.latest()
    if result is None:
        return f"crop=starting error={error}" if error else "crop=starting"
    counts = Counter(result.labels)
    labels = ",".join(f"{name}:{count}" for name, count in counts.items()) or "none"
    error_text = f" error={error}" if error else ""
    return f"crop={labels} inference={result.inference_ms:.0f}ms{error_text}"


def main() -> int:
    args = parser().parse_args()
    if not 0 < args.max_power <= 100 or not 0 < args.speed <= 100:
        raise SystemExit("--speed and --max-power must be between 1 and 100")
    if not args.disable_crop_detection and not args.model.is_file():
        raise SystemExit(f"Crop model not found: {args.model}")

    cv2.setUseOptimized(True)
    cv2.setNumThreads(2)
    stop_requested = threading.Event()

    def request_stop(_signal: int, _frame: object) -> None:
        stop_requested.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    camera = LatestFrameCamera(
        args.camera,
        args.width,
        args.height,
        args.camera_fps,
        not args.no_rotate,
    )
    crop_worker: CropWorker | None = None
    controller: SparkyMotorController | None = None
    drive: MecanumDrive | None = None

    try:
        print(f"Opening camera {args.camera}...")
        camera.start()

        if not args.disable_crop_detection:
            print(f"Loading crop model {args.model}...")
            crop_worker = CropWorker(
                camera,
                args.model,
                args.crop_fps,
                args.display,
            )
            crop_worker.start()

        if args.arm:
            controller = SparkyMotorController(args.port)
            if not controller.connect():
                raise RuntimeError(f"could not connect on {args.port}")
            robot = controller
            print("MOTORS ARMED. Press Ctrl+C or q to stop.")
        else:
            robot = DryRunRobot()
            print("DRY RUN. Motors cannot move without --arm.")

        drive = MecanumDrive(robot, args.max_power)
        line_detector = LineDetector(args.line_color)
        line_follower = LineFollower(args.speed)
        last_sequence = -1
        last_status = 0.0
        stale_stopped = False

        while not stop_requested.is_set():
            packet = camera.wait_for_frame(last_sequence, 0.35)
            now = time.monotonic()
            if packet is None or now - packet.captured_at > 0.25:
                if not stale_stopped:
                    drive.stop()
                    stale_stopped = True
                    print("Camera frame stale; motors stopped.")
                if camera.error:
                    raise RuntimeError(camera.error)
                continue

            stale_stopped = False
            last_sequence = packet.sequence
            line = line_detector.detect(packet.frame)
            motion = line_follower.update(line, now)
            motors = drive.command(
                motion.forward,
                motion.strafe,
                motion.turn,
                now,
            )

            if now - last_status >= 1.0:
                print(
                    f"line={motion.state} error={line.lateral:+.2f} "
                    f"motors={motors} {crop_status(crop_worker)}"
                )
                last_status = now

            if args.display:
                view = packet.frame.copy()
                cv2.rectangle(
                    view,
                    (0, line.roi_top),
                    (view.shape[1] - 1, view.shape[0] - 1),
                    (255, 180, 0),
                    1,
                )
                if line.near and line.far:
                    cv2.line(view, line.near, line.far, (0, 255, 0), 3)
                cv2.putText(
                    view,
                    f"{motion.state} error={line.lateral:+.2f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Sparky line follower", view)
                if crop_worker is not None:
                    crop_result, _ = crop_worker.latest()
                    if crop_result is not None and crop_result.annotated is not None:
                        cv2.imshow("Crop detections", crop_result.annotated)

            if args.binary_output:
                cv2.imshow(
                    "Binary webcam",
                    binary_frame(packet.frame, args.line_color),
                )

            if args.display or args.binary_output:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_requested.set()

        return 0
    except (RuntimeError, serial.SerialException) as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        if drive is not None:
            try:
                drive.stop()
            except Exception:
                pass
        if crop_worker is not None:
            crop_worker.stop()
        camera.stop()
        if controller is not None:
            controller.disconnect()
        if args.display or args.binary_output:
            cv2.destroyAllWindows()
        print("Stopped safely.")


if __name__ == "__main__":
    raise SystemExit(main())
