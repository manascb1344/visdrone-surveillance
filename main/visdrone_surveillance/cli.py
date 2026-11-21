"""Command line interface for the VisDrone surveillance package.

This is the single supported entry point. It replaces the ad-hoc root
scripts (``run.py`` and ``run_police_surveillance.py``) that previously
lived outside the package.

Heavy dependencies (torch, opencv) are imported lazily inside each
subcommand handler so that ``--help`` works without them installed.
"""

import argparse
import os
import sys
import time

__all__ = ["main", "build_parser"]


def build_parser():
    """Build the top level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="visdrone-surveil",
        description="Surveillance ML Project with RTMP support",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- train -------------------------------------------------------
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    train_parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    train_parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of training epochs"
    )

    # --- train-yolo --------------------------------------------------
    train_yolo_parser = subparsers.add_parser(
        "train-yolo", help="Train a YOLO model on the VisDrone dataset"
    )
    train_yolo_parser.add_argument(
        "--config",
        type=str,
        default="configs/visdrone.yaml",
        help="Path to YOLO config",
    )
    train_yolo_parser.add_argument(
        "--model", type=str, default="yolov8m.pt", help="Base YOLO model weights"
    )
    train_yolo_parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    train_yolo_parser.add_argument("--batch", type=int, default=16, help="Batch size")
    train_yolo_parser.add_argument(
        "--device", type=str, default="0", help="Device (0 for GPU, cpu for CPU)"
    )

    # --- rtmp-demo ---------------------------------------------------
    rtmp_parser = subparsers.add_parser(
        "rtmp-demo", help="Run RTMP demo with a video file"
    )
    rtmp_parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model"
    )
    rtmp_parser.add_argument(
        "--input_video", type=str, required=True, help="Path to input video file"
    )
    rtmp_parser.add_argument(
        "--output_video", type=str, help="Path to save output video (optional)"
    )
    rtmp_parser.add_argument(
        "--confidence", type=float, default=0.5, help="Confidence threshold"
    )

    # --- follow ------------------------------------------------------
    follow_parser = subparsers.add_parser(
        "follow", help="Run the autonomous vehicle-following demo in simulation"
    )
    follow_parser.add_argument(
        "--steps", type=int, default=150, help="Number of simulation steps"
    )
    follow_parser.add_argument(
        "--seed", type=int, default=7, help="RNG seed for the simulated world"
    )
    follow_parser.add_argument(
        "--vehicles", type=int, default=5, help="Number of simulated vehicles"
    )
    follow_parser.add_argument(
        "--width", type=int, default=640, help="Image width in pixels"
    )
    follow_parser.add_argument(
        "--height", type=int, default=480, help="Image height in pixels"
    )
    follow_parser.add_argument(
        "--fps", type=float, default=30.0, help="Simulation frame rate"
    )
    follow_parser.add_argument(
        "--altitude", type=float, default=30.0, help="Drone altitude in meters"
    )
    follow_parser.add_argument(
        "--yaw", type=float, default=0.0, help="Initial drone yaw in radians"
    )
    follow_parser.add_argument(
        "--vfov", type=float, default=60.0, help="Vertical field of view in degrees"
    )
    follow_parser.add_argument(
        "--output",
        type=str,
        default="outputs/follow_demo.mp4",
        help="Path to the output video file",
    )
    follow_parser.add_argument(
        "--kp", type=float, default=1.0, help="Proportional gain on yaw error"
    )
    follow_parser.add_argument(
        "--ki", type=float, default=0.2, help="Integral gain on yaw error"
    )
    follow_parser.add_argument(
        "--kd", type=float, default=0.0, help="Derivative gain on yaw error"
    )
    follow_parser.add_argument(
        "--max-forward", type=float, default=3.0, help="Max forward speed (m/s)"
    )
    follow_parser.add_argument(
        "--max-yaw-rate",
        type=float,
        default=1.0,
        help="Max yaw rate (rad/s)",
    )
    follow_parser.add_argument(
        "--max-climb", type=float, default=1.0, help="Max climb rate (m/s)"
    )
    follow_parser.add_argument(
        "--lock-frames", type=int, default=2, help="Hover frames before following"
    )
    follow_parser.add_argument(
        "--max-missed", type=int, default=8, help="Missed frames before reacquire"
    )
    follow_parser.add_argument(
        "--reacquire-timeout",
        type=float,
        default=2.0,
        help="Seconds of search before the mission goes lost",
    )

    # --- surveil -----------------------------------------------------
    surveil_parser = subparsers.add_parser(
        "surveil", help="Run the police surveillance system"
    )
    surveil_parser.add_argument("--video", type=str, help="Path to input video file")
    surveil_parser.add_argument(
        "--camera", type=int, default=0, help="Camera index for live feed"
    )
    surveil_parser.add_argument(
        "--config",
        type=str,
        default="surveillance_config.json",
        help="Path to configuration file",
    )
    surveil_parser.add_argument("--output", type=str, help="Path to output video file")
    surveil_parser.add_argument(
        "--rtmp", action="store_true", help="Enable RTMP streaming"
    )
    surveil_parser.add_argument(
        "--rtmp-url", type=str, default="rtmp://localhost/live/stream", help="RTMP URL"
    )

    # Feature flags
    surveil_parser.add_argument(
        "--disable-alpr", action="store_true", help="Disable ALPR"
    )
    surveil_parser.add_argument(
        "--disable-classification", action="store_true", help="Disable classification"
    )
    surveil_parser.add_argument(
        "--disable-tracking", action="store_true", help="Disable tracking"
    )
    surveil_parser.add_argument(
        "--disable-speed", action="store_true", help="Disable speed estimation"
    )

    # Output control
    surveil_parser.add_argument(
        "--resolution", type=str, default="1280x720", help="Output resolution (WxH)"
    )
    surveil_parser.add_argument("--fps", type=float, default=30.0, help="Output FPS")
    surveil_parser.add_argument(
        "--headless", action="store_true", help="Run in headless mode (no GUI)"
    )

    # Advanced options
    surveil_parser.add_argument(
        "--model-path", type=str, help="Path to detection model"
    )
    surveil_parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device to run on (cuda/cpu)",
    )
    surveil_parser.add_argument(
        "--confidence", type=float, default=0.5, help="Detection confidence threshold"
    )
    surveil_parser.add_argument(
        "--speed-limit", type=float, default=70.0, help="Speed limit in mph"
    )

    return parser


def _run_train(args):
    """Train the detection model."""
    from .training.train_frcnn import train_model

    train_model(args)


def _run_train_yolo(args):
    """Train a YOLO model on the VisDrone dataset."""
    from .training.train_yolo import train_yolo_model

    train_yolo_model(args.config, epochs=args.epochs)


def _run_rtmp_demo(args):
    """Run the RTMP detection demo over a video file."""
    from .streaming.handler import RTMPHandler

    rtmp_handler = RTMPHandler(model_path=args.model_path)
    rtmp_handler.run_demo(
        input_video_path=args.input_video,
        output_video_path=args.output_video,
        confidence_threshold=args.confidence,
    )


def _run_surveil(args):
    """Run the police surveillance system (video file or camera)."""
    import cv2
    import torch

    from .surveillance.surveillance_system import (
        RTMPSurveillanceServer,
        SurveillanceSystem,
    )

    # Parse resolution
    if args.resolution:
        try:
            width, height = map(int, args.resolution.split("x"))
            resolution = (width, height)
        except ValueError:
            print(
                f"Invalid resolution format: {args.resolution}. Using default 1280x720."
            )
            resolution = (1280, 720)
    else:
        resolution = (1280, 720)

    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Police Surveillance System")
    print("==========================")
    print(f"Using device: {device}")
    print(f"Output resolution: {resolution[0]}x{resolution[1]}")

    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)

    # Initialize surveillance system
    system = SurveillanceSystem(
        config_path=args.config if os.path.exists(args.config) else None,
        detection_model_path=args.model_path,
        device=device,
    )

    # Apply feature flags
    if args.disable_alpr:
        system.toggle_feature("alpr", False)
    if args.disable_classification:
        system.toggle_feature("classification", False)
    if args.disable_tracking:
        system.toggle_feature("tracking", False)
    if args.disable_speed:
        system.toggle_feature("speed", False)

    # Update speed limit if specified
    if args.speed_limit:
        system.speed_estimator.speed_limit_mph = args.speed_limit
        print(f"Speed limit set to {args.speed_limit} mph")

    # Initialize RTMP server if requested
    rtmp_server = None
    if args.rtmp:
        rtmp_server = RTMPSurveillanceServer(
            surveillance_system=system,
            rtmp_url=args.rtmp_url,
            output_resolution=resolution,
            fps=args.fps,
        )
        rtmp_server.start_streaming()

    # Start recording if output specified
    if args.output:
        system.start_recording(args.output, args.fps, resolution)

    # Open video source
    if args.video and os.path.exists(args.video):
        cap = cv2.VideoCapture(args.video)
        source_name = os.path.basename(args.video)
        print(f"Processing video: {args.video}")
    else:
        cap = cv2.VideoCapture(args.camera)
        source_name = f"Camera {args.camera}"
        print(f"Processing camera feed: {args.camera}")

    if not cap.isOpened():
        print(f"Error: Could not open video source: {source_name}")
        sys.exit(1)

    # Get video properties
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {video_width}x{video_height} @ {video_fps} FPS")
    if total_frames > 0:
        print(f"Total frames: {total_frames}")

    # Initialize performance metrics
    frame_count = 0
    start_time = time.time()
    last_update_time = start_time
    fps_display = 0

    # Process video frames
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process frame
            processed_frame, results = system.process_frame(
                frame, camera_id=source_name
            )

            # Add to RTMP server if enabled
            if rtmp_server and rtmp_server.is_streaming():
                rtmp_server.add_frame(processed_frame, source_name)

            # Resize for display
            if (
                processed_frame.shape[1] != resolution[0]
                or processed_frame.shape[0] != resolution[1]
            ):
                display_frame = cv2.resize(processed_frame, resolution)
            else:
                display_frame = processed_frame

            # Display result if not in headless mode
            if not args.headless:
                cv2.imshow("Police Surveillance System", display_frame)

                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                # Toggle features with keyboard shortcuts
                if key == ord("a"):
                    system.toggle_feature("alpr")
                elif key == ord("c"):
                    system.toggle_feature("classification")
                elif key == ord("t"):
                    system.toggle_feature("tracking")
                elif key == ord("s"):
                    system.toggle_feature("speed")
                elif key == ord("r"):
                    # Toggle recording
                    if system.recording_enabled:
                        system.stop_recording()
                    else:
                        system.start_recording(
                            f"outputs/recording_{int(time.time())}.mp4",
                            args.fps,
                            resolution,
                        )

            # Update performance metrics every second
            current_time = time.time()
            if current_time - last_update_time >= 1.0:
                elapsed = current_time - start_time
                fps_display = frame_count / elapsed if elapsed > 0 else 0

                # Display progress for video files
                if args.video and total_frames > 0:
                    percent = 100 * frame_count / total_frames
                    sys.stdout.write(
                        f"\rProcessing: {percent:.1f}% ({frame_count}/{total_frames}) - {fps_display:.1f} FPS"
                    )
                else:
                    sys.stdout.write(
                        f"\rProcessed {frame_count} frames - {fps_display:.1f} FPS"
                    )

                sys.stdout.flush()
                last_update_time = current_time

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Clean up
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

        if system.recording_enabled:
            system.stop_recording()

        if rtmp_server and rtmp_server.is_streaming():
            rtmp_server.stop_streaming()

        # Save configuration
        system.save_config(args.config)

        # Print final stats
        elapsed = time.time() - start_time
        print("\n\nSurveillance System Summary")
        print("==========================")
        print(f"Processed {frame_count} frames in {elapsed:.2f} seconds")
        print(f"Average FPS: {frame_count / elapsed if elapsed > 0 else 0:.2f}")

        # Get statistics from components
        if system.enable_alpr:
            alpr_stats = system.alpr.get_plate_statistics()
            print(f"License plates detected: {alpr_stats.get('total_plates', 0)}")
            print(f"Unique plates: {alpr_stats.get('unique_plates', 0)}")

        if system.enable_classification and system.enable_tracking:
            class_stats = system.classification_system.get_statistics()
            print(f"Vehicles classified: {class_stats.get('total_vehicles', 0)}")

        if system.enable_tracking:
            track_stats = system.tracker.get_camera_statistics()
            print(f"Active tracks: {track_stats.get('active_tracks', 0)}")
            print(f"Inactive tracks: {track_stats.get('inactive_tracks', 0)}")

        print("\nSystem shutdown complete")


def _run_follow(args):
    """Run the autonomous vehicle-following demo in the drone simulator."""
    import cv2
    import numpy as np
    import os

    from .control import FollowPilot, GuidanceController
    from .follow import select_by_point
    from .simulation import DroneCamera, SimScenario, World
    from .types import VelocityCommand

    dt = 1.0 / args.fps
    world = World(seed=args.seed, num_vehicles=args.vehicles)
    camera = DroneCamera(
        position=(0.0, 0.0, args.altitude),
        yaw=args.yaw,
        image_width=args.width,
        image_height=args.height,
        vfov_deg=args.vfov,
    )
    sim = SimScenario(world, camera, dt=dt)
    sim.reset(seed=args.seed)

    # Pick the vehicle closest to the image center in the first frame.
    frame, observations, _ = sim.step(VelocityCommand(), dt)
    if not observations:
        print("No vehicles in the initial frame; aborting.")
        return

    center_x, center_y = args.width / 2.0, args.height / 2.0
    target = select_by_point(observations, center_x, center_y)
    if target is None:
        target = min(
            observations,
            key=lambda d: (d.bbox.center[0] - center_x) ** 2
            + (d.bbox.center[1] - center_y) ** 2,
        )

    controller = GuidanceController(
        image_size=(args.width, args.height),
        desired_scale=target.bbox.scale,
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        max_forward=args.max_forward,
        max_yaw_rate=args.max_yaw_rate,
        max_climb=args.max_climb,
    )
    pilot = FollowPilot(
        controller,
        lock_frames=args.lock_frames,
        max_missed=args.max_missed,
        reacquire_timeout=args.reacquire_timeout,
    )
    pilot.set_target(target, 0.0)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )

    center_errors = []
    frames_written = 0
    command = VelocityCommand()
    for step_index in range(args.steps):
        frame, observations, _ = sim.step(command, dt)
        timestamp = (step_index + 1) * dt
        command = pilot.step(observations, timestamp, (args.width, args.height), dt=dt)

        vis = frame.copy()
        target_state = pilot.target_state
        target_det = None
        if target_state is not None and observations:
            target_det = max(observations, key=lambda d: d.bbox.iou(target_state.bbox))

        for det in observations:
            x1, y1, x2, y2 = det.bbox.as_int_tuple()
            color = (0, 200, 0) if det is target_det else (140, 140, 140)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        if target_det is not None:
            cx, cy = target_det.bbox.center
            cv2.drawMarker(
                vis, (int(cx), int(cy)), (0, 200, 0), cv2.MARKER_CROSS, 14, 2
            )
            cv2.drawMarker(
                vis,
                (args.width // 2, args.height // 2),
                (0, 0, 255),
                cv2.MARKER_TILTED_CROSS,
                14,
                1,
            )
            center_errors.append(
                ((cx - args.width / 2.0) ** 2 + (cy - args.height / 2.0) ** 2) ** 0.5
            )

        cv2.putText(
            vis,
            f"state={pilot.state.value} fwd={command.forward:+.2f} "
            f"yaw={command.yaw_rate:+.2f}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        writer.write(vis)
        frames_written += 1

    writer.release()
    print(f"Wrote {frames_written} frames to {args.output}")
    if center_errors:
        mean_error = sum(center_errors) / len(center_errors)
        print(f"Mean target center error: {mean_error:.1f} px")
    print(f"Final state: {pilot.state.value}")


_COMMANDS = {
    "train": _run_train,
    "train-yolo": _run_train_yolo,
    "rtmp-demo": _run_rtmp_demo,
    "follow": _run_follow,
    "surveil": _run_surveil,
}


def main(argv=None):
    """Entry point for the ``visdrone-surveil`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        return 0

    handler(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
