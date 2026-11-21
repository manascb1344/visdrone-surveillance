# Project status

Facts about what works today in `main/visdrone_surveillance/` and what does not.
This page is the maturity record for the 2026-11 restructure. Update it when a
row's status changes.

The test suite under `tests/` pins the behavior of the rows marked **Implemented**
and **Promoted**. Three tests are marked `xfail(strict=True)` and name real bugs: the camera transition crash, the reconnections stat, and `get_latest_frame` consuming the buffer. All three are documented below.

## Status by area

| Area | Module | Status | Notes |
| --- | --- | --- | --- |
| Dataset loading | `data/visdrone_dataset.py` | Implemented | VisDrone VID sequence and annotation parsing, train/val split, Albumentations transforms. Covered by tests. |
| Detection model | `models/detection_model.py` | Implemented | Faster R-CNN ResNet-50 FPN from torchvision, COCO pretrained, VisDrone fine-tuned. |
| Training | `training/train.py` | Implemented | SGD with step decay, per-epoch validation, checkpointing, prediction and metric plots. |
| Evaluation | `utils/evaluation.py` | Implemented | IoU, per-class AP, mAP. Covered by tests. |
| Visualization | `utils/visualization.py` | Implemented | Bounding box drawing and metric plots. |
| RTMP ingestion | `streaming/rtmp_client.py`, `streaming/stream_ingestion.py` | Promoted | Promoted from `research/ml-proj-man/`. Buffered client with reconnect, stats, and events. Covered by smoke tests. |
| RTMP detection handler | `streaming/handler.py` | Partial | `run_demo` writes a demo overlay and does not run inference. `connect_rtmp` and `process_stream` are TODO stubs. |
| Surveillance orchestration | `surveillance/surveillance_system.py` | Implemented | Per-frame pipeline over ALPR, classification, tracking, and speed with feature toggles and recording. |
| License plate recognition | `surveillance/alpr.py` | Partial | Vehicle detection is real (YOLOv5s). Plate boxes are estimated inside vehicle boxes. Plate text is simulated with random strings. |
| Vehicle classification | `surveillance/vehicle_classification.py` | Partial | Type, color, and make predictions return simulated values from `random.choice`. Feature extraction for identification is a placeholder. |
| Multi-camera tracking | `surveillance/vehicle_tracking.py` | Implemented | One known bug, see below. |
| Speed estimation | `surveillance/speed_estimation.py` | Implemented | Depends on camera calibration. Without calibration it uses a default pixels-per-meter value. |
| Autonomous follow, simulation | `simulation/` | Implemented | Deterministic kinematic world, pinhole down-camera, synthetic frames, ground-truth observations. Covered by tests. |
| Autonomous follow, tracking | `follow/` | Implemented | Click selection and single-target tracker with constant-velocity prediction and coasting. Covered by tests. |
| Autonomous follow, guidance | `control/` | Implemented | PID guidance on center offset and target scale, plus a lock, follow, reacquire, lost state machine. Covered by tests. |
| Autonomous follow, hardware bridge | `control/` (planned) | Not started | DJI Tello adapter not yet written. The command interface is ready for it. |
| Command line | `cli.py` | Implemented | `visdrone-surveil train`, `train-yolo`, `rtmp-demo`, `follow`, and `surveil`. Replaces the old `run.py` and `run_police_surveillance.py`. Covered by smoke tests. |

## Known bugs

**Camera transition crash in `MultiCameraTracker`.**
`camera_transitions` is created as `defaultdict(set)` in `__init__` but is
incremented as a counter in `_register_camera_transition`. Any vehicle seen by
a second camera raises `TypeError: unsupported operand type(s) for +=: 'set'
and 'int'`. `load_tracks` rebuilds the same key as `defaultdict(int)`, so the
container type also differs before and after a save/load cycle. The test
`tests/test_surveillance_logic.py::test_multicamera_tracker_records_camera_transition`
is marked `xfail(strict=True)` and fails until this is fixed.

**Reconnection stat never counts.**
`VisDroneRTMPClient.stats["reconnections"]` starts at 0 and nothing increments
it. `RTMPClient.handle_reconnection` reconnects without touching the counter,
so the stat stays 0 no matter how often a stream drops. Pinned in
`tests/test_streaming.py::test_reconnections_counter_is_incremented` as
`xfail(strict=True)`.

**`get_latest_frame` consumes the buffer.**
The docstring of `VisDroneRTMPClient.get_latest_frame` says it returns the
newest frame without removing it. The method drains the queue and re-puts all
frames except the last. With one frame buffered the buffer ends up empty, so
the frame is consumed. Pinned in
`tests/test_streaming.py::test_get_latest_frame_does_not_consume_the_buffer`
as `xfail(strict=True)`.

**Stale `config.NUM_CLASSES`.**
`config.NUM_CLASSES` is 10 while `config.CLASSES` holds 12 entries. Every call
site builds the model with `len(config.CLASSES)`, so the constant is dead code.
The dataset maps VisDrone category `N` to label `N-1`, so labels run 0 to 10
and the 12-way model head has one output that training never uses. Saved
checkpoints were trained with this head shape. Changing `NUM_CLASSES` or
trimming `CLASSES` breaks loading those checkpoints.

**Single-sequence dataset split.**
With `validation_split=0.2`, a dataset of one sequence yields an empty train
split because `int(1 * 0.8) == 0`. The loader returns an empty dataset rather
than raising. Pinned in
`tests/test_visdrone_dataset.py::test_single_sequence_with_validation_split_yields_empty_train_set`.

## Out of scope for this restructure

The 2026-11 work changes structure only. These items are unchanged and open:

- Replace the simulated ALPR, classification, and identification backends with
  trained models.
- Wire real inference into `RTMPHandler.run_demo` and implement
  `connect_rtmp` and `process_stream`.
- Fix the camera transition bug under its own change, with the xfail test
  flipped to a normal test.
- Rotate the API keys that were removed from the working tree. They remain in
  the history of commit `9938600`.

## Added since the restructure

The autonomous vehicle-following pipeline was added on top of the package:
simulation, single-target tracking, PID guidance, and the follow state
machine. All of it runs offline and is covered by tests. The DJI Tello
hardware bridge is the remaining piece.
