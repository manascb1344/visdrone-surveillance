# Architecture

This page explains how the package fits together and why the repository is
split the way it is. For per-area maturity, see [Project status](project-status.md).

## Repository layout

Everything the project supports lives in `main/visdrone_surveillance/`. The
`research/` directory holds prototypes, notebooks, and experiment artifacts.
That split exists so a reader can tell maintained code from experiments at a
glance, and so the test suite and the packaging describe exactly one codebase.

`research/` is not part of the package. Nothing under it is imported by
`visdrone_surveillance` and nothing in it is tested.

## Package map

```text
main/visdrone_surveillance/
├── config.py        # Dataset paths, class list, training defaults, RTMP settings
├── cli.py           # visdrone-surveil entry point
├── data/            # VisDrone dataset loader and transforms
├── models/          # Faster R-CNN build, load, and save
├── training/        # Training loop and validation
├── streaming/       # RTMP clients, stream ingestion, detection handler
├── surveillance/    # ALPR, classification, tracking, speed, orchestration
├── control/         # Guidance, pilot state machine, drone bridge
├── follow/          # Target selection and single-target tracking
├── simulation/      # Deterministic drone and world simulator
├── types.py         # Shared domain types
└── utils/           # Metrics and visualization
```

## Data flow

Training reads VisDrone sequences and annotations through
`data/visdrone_dataset.py`, which converts each frame's annotation rows into
bounding boxes and class labels. `models/detection_model.py` builds a
torchvision Faster R-CNN with a ResNet-50 FPN backbone, pretrained on COCO and
re-headed for the VisDrone classes. `training/train.py` runs the epochs,
validates with the mAP code in `utils/evaluation.py`, and writes checkpoints
and plots to `outputs/`.

Inference paths load a checkpoint through the same model builder. Live video
enters through `streaming/`, where `rtmp_client.py` and `stream_ingestion.py`
handle connections, buffering, reconnection, and stream statistics. The
`handler.py` detection path is still partial, see
[Project status](project-status.md).

`surveillance/surveillance_system.py` runs the per-frame pipeline. It feeds
each frame to ALPR, then vehicle classification and identification, then
multi-camera tracking, then speed estimation. Each stage can be disabled
through feature flags. Tracking maintains per-vehicle trajectories across
cameras, and speed estimation converts pixel movement to speed through per
camera calibration.

## Configuration

`config.py` is the single source of truth for dataset paths, the class list,
training defaults, and RTMP endpoints. Environment variables override the
dataset root, RTMP server and stream key, and the detection thresholds. The
surveillance system reads its own runtime settings from an optional JSON
config file passed on the command line.

The class list keeps all 12 VisDrone entries, including "ignored regions" at
index 0. The dataset drops ignored regions and remaps the remaining categories
to labels 0 through 10. Models are built with `len(CLASSES)` outputs to match
the saved checkpoints. `config.NUM_CLASSES` is stale dead code and is not
used.

## Entry points

`visdrone-surveil` is the only supported interface. Its three subcommands map
to `training.train.train_model`, `streaming.handler.RTMPHandler.run_demo`, and
the `surveillance.surveillance_system` pipeline. The old root scripts
`run.py` and `run_police_surveillance.py` were removed once their argument
sets were ported to the CLI.
