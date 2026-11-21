# VisDrone surveillance

[![CI](https://github.com/manascb1344/visdrone-surveillance/actions/workflows/ci.yml/badge.svg)](https://github.com/manascb1344/visdrone-surveillance/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Aerial object detection and surveillance built on the VisDrone dataset, with
RTMP streaming, a police surveillance pipeline, and an autonomous
vehicle-following demo. This repository is the canonical home of the project.

## What it does

- **Detection.** Two families: a Faster R-CNN (ResNet-50 FPN) detector trained
  on VisDrone VID, and a YOLOv8 detector trained on VisDrone YOLO. Both are
  selectable at runtime.
- **Streaming.** RTMP ingestion with reconnect, buffering, and statistics.
- **Surveillance.** ALPR, vehicle classification and identification,
  multi-camera tracking, and speed estimation, wired into one pipeline.
- **Autonomous follow.** Pick a vehicle in the feed and a simulated drone
  tracks and follows it using a single-target tracker and PID guidance.
  The loop runs offline and deterministically.

## Team members

1. Aarya Bodas
2. Manas Bavaskar

## Repository layout

| Path | What lives there |
| --- | --- |
| `main/visdrone_surveillance/` | The supported package. All tested, maintained code. |
| `tests/` | The test suite that pins package behavior. |
| `docs/` | Architecture and project status. |
| `research/` | Prototypes, notebooks, and experiment artifacts. Not part of the package. |
| `configs/` | Dataset and training configuration files. |
| `scripts/` | Operational helpers. |
| `data/`, `models/`, `outputs/` | Runtime data, weights, and outputs. Gitignored. |

## Install

You need Python 3.8 or newer. FFmpeg is required for streaming features.

```bash
pip install -e ".[test]"
```

Or without editable mode:

```bash
pip install ".[test]"
```

Or with uv:

```bash
uv pip install --python .venv/bin/python -e ".[test]"
```

Copy `.env.example` to `.env` if you want to override settings, or set
environment variables directly. The package reads `config.py`, with overrides
for `VISDRONE_DATASET_ROOT`, `VISDRONE_YOLO_ROOT`, `RTMP_SERVER`,
`RTMP_STREAM_KEY`, `CONFIDENCE_THRESHOLD`, and `NMS_THRESHOLD`.

## Usage

```bash
# Train the Faster R-CNN detector
visdrone-surveil train --batch_size 8 --learning_rate 0.001 --num_epochs 20

# Train a YOLO model
visdrone-surveil train-yolo --config configs/visdrone.yaml --epochs 100

# Run detection over a video file
visdrone-surveil rtmp-demo --model_path outputs/<run>/model_epoch_9.pth --input_video video.mp4

# Run the surveillance pipeline
visdrone-surveil surveil --video video.mp4 --output outputs/tracked.mp4

# Run the autonomous follow demo in simulation
visdrone-surveil follow --steps 150 --seed 7 --output outputs/follow_demo.mp4
```

Run `visdrone-surveil <command> --help` for every flag.

## Dataset

Training expects either the VisDrone2019-VID layout (Faster R-CNN) or the
VisDrone-YOLO layout (YOLO):

```text
VisDrone2019-VID-train/          # VID format
├── sequences/
└── annotations/

datasets/VisDrone-YOLO/           # YOLO format
├── train/images, train/labels
└── val/images, val/labels
```

## Testing

```bash
python -m pytest tests/ -q
```

The suite is offline. It needs no GPU, no dataset, and no network. A few
tests are marked `xfail` and name known bugs.

## Documentation

- [Architecture](docs/architecture.md)
- [Project status](docs/project-status.md)
- [VisDrone monitoring PRD](docs/visdrone-monitoring-prd.md)
- [Research index](research/README.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and pull request guidelines.

## License

MIT. See `LICENSE` for details.
