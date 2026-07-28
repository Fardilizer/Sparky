#!/usr/bin/env python3
"""Run line following and crop detection from one USB camera.

Examples:
    python3 robot_line_follow.py --display
    python3 robot_line_follow.py --arm --port /dev/ttyUSB0
"""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import signal
import threading
import time
from typing import TYPE_CHECKING, Sequence

# Limit libraries that otherwise try to occupy every Pi CPU core. The crop
# worker needs room while the line-control loop is active.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from mecanum_drive import MecanumDrive, MotorCommand

if TYPE_CHECKING:
    from vision_multitask import CropDetectionWorker


class DryRunRobot:
    """Motor API stand-in used until ``--arm`` is explicitly supplied."""

    def set_motor(self, _m1: int, _m2: int, _m3: int, _m4: int) -> None:
        pass


def camera_source(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def four_floats(value: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected four comma-separated numbers") from exc
    if len(values) != 4:
        raise argparse.ArgumentTypeError("expected four comma-separated numbers")
    return values  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    project_model = Path(__file__).with_name("best.onnx")
    parser = argparse.ArgumentParser(
        description="Low-latency mecanum line following plus background crop detection.",
    )
    camera = parser.add_argument_group("camera")
    camera.add_argument("--camera", type=camera_source, default=0, help="Camera index or video path")
    camera.add_argument("--width", type=int, default=640)
    camera.add_argument("--height", type=int, default=360)
    camera.add_argument("--camera-fps", type=int, default=30)
    camera.add_argument(
        "--no-rotate",
        action="store_true",
        help="Do not rotate camera frames 180 degrees (the old program rotated them)",
    )
    camera.add_argument(
        "--max-frame-age",
        type=float,
        default=0.25,
        help="Stop motors when the newest camera frame is older than this many seconds",
    )
    camera.add_argument("--opencv-threads", type=int, default=2)

    line = parser.add_argument_group("line following")
    line.add_argument("--line-color", choices=("dark", "light"), default="dark")
    line.add_argument("--roi-top", type=float, default=0.48, help="Top of line ROI as frame fraction")
    line.add_argument("--speed", type=float, default=32.0, help="Straight-line motor power")
    line.add_argument("--max-power", type=float, default=55.0)
    line.add_argument("--slew-rate", type=float, default=180.0, help="Maximum motor power change/second")
    line.add_argument("--lateral-kp", type=float, default=20.0)
    line.add_argument("--lateral-kd", type=float, default=1.8)
    line.add_argument("--turn-lateral-gain", type=float, default=13.0)
    line.add_argument("--turn-heading-gain", type=float, default=26.0)
    line.add_argument(
        "--motor-trim",
        type=four_floats,
        default=(1.0, 1.0, 1.0, 1.0),
        metavar="M1,M2,M3,M4",
        help="Per-wheel calibration multipliers",
    )

    crops = parser.add_argument_group("crop detector")
    crops.add_argument("--model", type=Path, default=project_model)
    crops.add_argument("--crop-fps", type=float, default=2.0)
    crops.add_argument("--crop-imgsz", type=int, default=320)
    crops.add_argument("--crop-confidence", type=float, default=0.40)
    crops.add_argument("--device", default=None, help="Optional Ultralytics device, e.g. cpu")
    crops.add_argument("--disable-crop-detection", action="store_true")

    robot = parser.add_argument_group("robot and diagnostics")
    robot.add_argument("--port", default="/dev/ttyUSB0")
    robot.add_argument(
        "--arm",
        action="store_true",
        help="Actually connect and move motors; without this flag the program is a dry run",
    )
    robot.add_argument("--display", action="store_true", help="Show line and crop windows (slower)")
    robot.add_argument("--show-mask", action="store_true", help="Show threshold mask with --display")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.width < 160 or args.height < 90:
        parser.error("camera resolution is too small")
    if not 0.1 <= args.roi_top <= 0.9:
        parser.error("--roi-top must be between 0.1 and 0.9")
    if not 0 < args.speed <= 100 or not 0 < args.max_power <= 100:
        parser.error("--speed and --max-power must be in the range (0, 100]")
    if args.crop_fps <= 0:
        parser.error("--crop-fps must be greater than zero")
    if args.max_frame_age <= 0:
        parser.error("--max-frame-age must be greater than zero")
    if not args.disable_crop_detection and not args.model.is_file():
        parser.error(f"crop model not found: {args.model}")


def format_crop_status(worker: CropDetectionWorker | None) -> str:
    if worker is None:
        return "crop=off"
    snapshot = worker.latest()
    if snapshot is None:
        return f"crop=starting ({worker.error})" if worker.error else "crop=starting"
    counts = Counter(detection.label for detection in snapshot.detections)
    labels = ",".join(f"{label}:{count}" for label, count in sorted(counts.items()))
    labels = labels or "none"
    error = f" error={worker.error}" if worker.error else ""
    return f"crop={labels} inference={snapshot.inference_ms:.0f}ms{error}"


def run(args: argparse.Namespace) -> int:
    import cv2

    from line_following import (
        LineDetector,
        LineDetectorConfig,
        LineFollower,
        LineFollowerConfig,
        draw_line_debug,
    )
    from vision_multitask import (
        CameraConfig,
        CropDetectionWorker,
        LatestFrameCamera,
    )

    cv2.setUseOptimized(True)
    cv2.setNumThreads(max(1, args.opencv_threads))

    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_requested.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    camera = LatestFrameCamera(
        CameraConfig(
            source=args.camera,
            width=args.width,
            height=args.height,
            fps=args.camera_fps,
            rotate_180=not args.no_rotate,
        )
    )
    crop_worker: CropDetectionWorker | None = None
    connected_robot = None
    drive: MecanumDrive | None = None

    try:
        print(f"Opening camera {args.camera!r} at {args.width}x{args.height}@{args.camera_fps}...")
        camera.start()

        if not args.disable_crop_detection:
            print(f"Loading crop model: {args.model}")
            crop_worker = CropDetectionWorker(
                camera,
                args.model,
                inference_fps=args.crop_fps,
                image_size=args.crop_imgsz,
                confidence=args.crop_confidence,
                device=args.device,
                make_annotated_frame=args.display,
            ).start()

        if args.arm:
            from sparkybotmini import SparkyBotMini

            connected_robot = SparkyBotMini(port=args.port)
            if not connected_robot.connect():
                raise RuntimeError(f"could not connect to robot on {args.port}")
            robot_output = connected_robot
            print("MOTORS ARMED. Press Ctrl+C (or q in the display) to stop.")
        else:
            robot_output = DryRunRobot()
            print("DRY RUN: motor values are calculated but not sent. Add --arm to move.")

        drive = MecanumDrive(
            robot_output,
            max_power=args.max_power,
            slew_rate=args.slew_rate,
            motor_trim=args.motor_trim,
        )
        detector = LineDetector(
            LineDetectorConfig(
                roi_top_fraction=args.roi_top,
                line_color=args.line_color,
            )
        )
        follower = LineFollower(
            LineFollowerConfig(
                forward_power=args.speed,
                minimum_forward_power=min(14.0, args.speed * 0.55),
                lateral_kp=args.lateral_kp,
                lateral_kd=args.lateral_kd,
                turn_lateral_gain=args.turn_lateral_gain,
                turn_heading_gain=args.turn_heading_gain,
            )
        )

        last_sequence = -1
        last_status_at = 0.0
        loop_fps = 0.0
        previous_loop_at = time.monotonic()
        last_motor_command = MotorCommand(0, 0, 0, 0)
        stale_stop_sent = False

        while not stop_requested.is_set():
            packet = camera.wait_for_frame(last_sequence, timeout=0.35)
            now = time.monotonic()
            if packet is None or now - packet.captured_at > args.max_frame_age:
                if not stale_stop_sent:
                    drive.stop()
                    stale_stop_sent = True
                    print("Camera frame stale; motors stopped.")
                if camera.error:
                    raise RuntimeError(camera.error)
                continue

            stale_stop_sent = False
            last_sequence = packet.sequence
            observation = detector.detect(
                packet.frame,
                include_mask=args.display and args.show_mask,
            )
            motion = follower.update(observation, now=now)
            last_motor_command = drive.command(
                motion.forward,
                motion.strafe_right,
                motion.turn_right,
                now=now,
            )

            loop_dt = max(1e-4, now - previous_loop_at)
            instantaneous_fps = 1.0 / loop_dt
            loop_fps = instantaneous_fps if loop_fps == 0 else 0.9 * loop_fps + 0.1 * instantaneous_fps
            previous_loop_at = now

            if now - last_status_at >= 1.0:
                print(
                    f"line={motion.state} error={observation.lateral_error:+.2f} "
                    f"motors={last_motor_command.as_tuple()} fps={loop_fps:.1f} "
                    f"{format_crop_status(crop_worker)}"
                )
                last_status_at = now

            if args.display:
                line_view = draw_line_debug(
                    packet.frame,
                    observation,
                    motion,
                    loop_fps=loop_fps,
                )
                cv2.imshow("Sparky line follower", line_view)
                if args.show_mask and observation.mask is not None:
                    cv2.imshow("Line mask", observation.mask)
                if crop_worker is not None:
                    crop_snapshot = crop_worker.latest()
                    if crop_snapshot is not None and crop_snapshot.annotated_frame is not None:
                        cv2.imshow("Crop detections", crop_snapshot.annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_requested.set()

        return 0
    finally:
        # Stop motion before slower thread/camera cleanup.
        if drive is not None:
            try:
                drive.stop()
            except Exception as exc:
                print(f"Warning: motor stop failed: {exc}")
        if crop_worker is not None:
            crop_worker.stop()
        camera.stop()
        if connected_robot is not None:
            connected_robot.disconnect()
        if args.display:
            cv2.destroyAllWindows()
        print("Stopped safely.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    try:
        return run(args)
    except (RuntimeError, ModuleNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
