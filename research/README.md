# research/

Working material for this project: prototype scripts, experiment and notebook
scaffolding, and notes. None of the contents here are part of the supported
package; the shipped library is `main/visdrone_surveillance/` and is the only
code covered by the test suite and packaging.

## Moved items

| Item | Destination | Provenance |
| --- | --- | --- |
| `cl.py` | `research/prototypes/cl.py` | Simple RTMP + YOLOv8n detection demo; predates the restructure. |
| `cl_train.py` | `research/prototypes/cl_train.py` | VisDrone-to-YOLO conversion and YOLOv8 training script; predates the restructure. |
| `rtmp_cli.py` | `research/prototypes/rtmp_cli.py` | CLI for displaying and listing RTMP streams; predates the restructure. |
| `src/rtmp/test_stream.py` | `research/prototypes/test_stream.py` | Manual RTMP stream display and multi-stream test script. |
| `nia194-object-detection.txt` | `research/notes/nia194-object-detection.txt` | Large text artifact related to object detection. |
| `visdrone_monitoring_prd.md` | `docs/visdrone-monitoring-prd.md` | Product requirements document (moved to docs). |

`research/experiments/` holds run outputs, metrics, and manifests.
`research/notebooks/` holds Jupyter notebooks and the `final-notebooks/` system.
`research/notes/` holds planning notes and the pre-rewrite commit SHA.
`research/prototypes/` holds unmaintained demo scripts.

## Maintenance status

The prototype scripts in `research/prototypes/` reference the pre-restructure
`src.` import layout. They are preserved verbatim as historical reference and
are not maintained or covered by tests; they will not run against the current
package layout without edits.
