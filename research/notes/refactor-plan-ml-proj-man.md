# Refactor plan: make ml-proj-man the canonical project

Status: draft for review. Nothing in this plan has been executed yet.
Audience: the two contributors (Aarya Bodas, Manas Bavaskar) and whoever
reviews the pull request.

## 0. Decision points (resolve before any code moves)

1. **Canonical repository.** `ml-proj-man` (remote `manascb1344/ml-proj-man`)
   becomes the single project. `aarya-16/ML-project` becomes a redirect
   (README points at `ml-proj-man`) and is archived after a transition
   window. All future work happens in `ml-proj-man`.
2. **Ownership.** The `ml-proj-man` remote belongs to `manascb1344`. The
   operator of this work (`mcb76`) must have push access on that remote
   before phase 6. Without it, the cutover stops at a local branch.
3. **Package name.** Keep `visdrone_surveillance` (matches the tests and
   the CLI already written) rather than rename to `visdrone_monitor`. The
   PRD title is "VisDrone-RTMP"; the package name is a code identifier and
   renaming it buys nothing.
4. **Binary policy.** No model weights, datasets, zips, or run artifacts
   stay in git. They move to external storage (Hugging Face Hub or a shared
   drive) behind a download script. `ml-proj-man` already tried Git LFS
   (`3508b34`, `f9f0022`) and still carries 2.3 GB, so LFS is not the
   answer; external storage plus a manifest is.
5. **History rewrite.** `ml-proj-man`'s history contains large files added
   and re-added (`d1a6bd3`, `f9f0022`, `f9f0022`). Phase 1 rewrites that
   history before the new content lands, so the rewrite touches only the
   existing history and is done once.
6. **What "final" means.** The end state is one repo, one package, one CLI,
   one test suite, one docs set, one config. The ML-project repo is not
   deleted; it is archived with a pointer.

## 1. Inventory summary

### ml-proj-man (destination)

- Remote: `https://github.com/manascb1344/ml-proj-man`
- Branch: `master` (no `main`)
- 30 commits, 327 tracked files, 2.3 GB tracked
- 26 Git LFS objects, all `.pt` weights under `final-notebooks/.../visdrone_yolo_training/`
- No tests (only `src/rtmp/test_stream.py`, a manual display script)
- Python 3.8+, torch with CUDA, FFmpeg
- Two overlapping implementations of the same domain: `src/` and
  `final-notebooks/drone_detection_system/`

### ML-project (source of the ported additions)

- Branch `refactor/project-structure`, 9 commits on top of the original
- Package `main/visdrone_surveillance/` with 12 subpackages/modules
- 142 tests, 3 xfailed
- Unified CLI `visdrone-surveil` (train, rtmp-demo, follow, surveil)
- Surveillance pipeline, follow mission, simulation, types
- Docs: README, architecture, project-status
- 2.3 GB of ignored binaries (outputs, video, weights)

## 2. Target structure for ml-proj-man

```text
ml-proj-man/
├── README.md
├── pyproject.toml
├── requirements.txt          # kept as a thin pointer for pip users
├── .gitignore
├── .env.example
├── .gitattributes            # LFS rules only if LFS is retained
├── configs/
│   ├── visdrone.yaml         # promoted from configs/
│   ├── visdrone_custom.yaml  # promoted from configs/
│   ├── visdrone_merged.yaml  # promoted from configs/
│   └── surveillance.json     # new, from ML-project
├── docs/
│   ├── index.md
│   ├── getting-started.md
│   ├── architecture.md
│   ├── data-visdrone.md
│   ├── training.md
│   ├── inference.md
│   ├── streaming-rtmp.md
│   ├── surveillance.md
│   └── project-status.md
├── main/
│   └── visdrone_surveillance/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── types.py
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── inference/
│       ├── streaming/
│       ├── surveillance/
│       ├── control/
│       ├── follow/
│       ├── simulation/
│       └── utils/
├── research/
│   ├── README.md
│   ├── prototypes/           # cl.py, cl_train.py, basic_detection.py, ...
│   ├── notebooks/            # ml_proj.ipynb, ml-proj-kaggle.ipynb, final-notebooks/
│   └── experiments/          # metrics, plots, manifests (no weights)
├── tests/
│   └── (all test files from ML-project, imports updated)
├── scripts/
│   └── download_weights.py   # new, pulls weights from external storage
├── data/                     # gitignored
├── models/                   # gitignored
└── outputs/                  # gitignored
```

## 3. File-by-file migration mapping

### 3.1 Already in ml-proj-man, keep and reorganize

| Current path in ml-proj-man | Target path | Action |
|---|---|---|
| `src/data/dataset_utils.py` | `main/visdrone_surveillance/data/dataset_utils.py` | Move, fix relative imports, split if over ~300 lines |
| `src/data/prepare_visdrone.py` | `main/visdrone_surveillance/data/prepare_visdrone.py` | Move, fix imports, keep download logic behind `--no-download` default |
| `src/data/__init__.py` | `main/visdrone_surveillance/data/__init__.py` | Merge with ML-project version |
| `src/models/config.py` | `main/visdrone_surveillance/models/yolo_config.py` | Rename to avoid clash with package config.py |
| `src/models/model_factory.py` | `main/visdrone_surveillance/models/model_factory.py` | Move |
| `src/models/yolo_setup.py` | `main/visdrone_surveillance/models/yolo_setup.py` | Move |
| `src/models/extract_frames.py` | `main/visdrone_surveillance/models/extract_frames.py` | Move |
| `src/models/process_video.py` | `main/visdrone_surveillance/models/process_video.py` | Move |
| `src/models/realtime_stream.py` | `main/visdrone_surveillance/models/realtime_stream.py` | Move |
| `src/models/train.py` | `main/visdrone_surveillance/training/train_yolo.py` | Rename, keep YOLO training |
| `src/models/evaluate.py` | `main/visdrone_surveillance/utils/evaluate_yolo.py` | Move, fix imports |
| `src/models/train_yolo.py` | (delete) | Broken stub, superseded by train_yolo.py |
| `src/rtmp/rtmp_client.py` | `main/visdrone_surveillance/streaming/rtmp_client.py` | Already canonical; keep |
| `src/rtmp/stream_ingestion.py` | `main/visdrone_surveillance/streaming/stream_ingestion.py` | Already canonical; keep |
| `src/rtmp/test_stream.py` | `research/prototypes/test_stream.py` | Move, it is a manual script |
| `src/rtmp/__init__.py` | `main/visdrone_surveillance/streaming/__init__.py` | Merge |
| `src/rtmp_cli.py` | `research/prototypes/rtmp_cli.py` | Move |
| `src/detection/__init__.py` | (delete) | Empty placeholder |
| `src/stream/__init__.py` | (delete) | Empty placeholder |
| `src/visualization/__init__.py` | (delete) | Empty placeholder |
| `src/__init__.py` | (delete) | Replaced by package __init__ |
| `configs/visdrone.yaml` | `configs/visdrone.yaml` | Keep |
| `configs/visdrone_custom.yaml` | `configs/visdrone_custom.yaml` | Keep, fix absolute path |
| `configs/visdrone_merged.yaml` | `configs/visdrone_merged.yaml` | Keep |
| `visdrone_monitoring_prd.md` | `docs/visdrone-monitoring-prd.md` | Move, rename to lowercase |
| `cl.py` | `research/prototypes/cl.py` | Move |
| `cl_train.py` | `research/prototypes/cl_train.py` | Move |
| `final-notebooks/` | `research/notebooks/final-notebooks/` | Move whole tree, keep .gitignored weights out |
| `ml_proj.ipynb`, `ml-proj-kaggle.ipynb` | `research/notebooks/` | Move |
| `runs/` | `research/experiments/runs/` | Move, gitignore |
| `output/` | `research/experiments/output/` | Move, gitignore |
| `filtered/` | `research/experiments/filtered/` | Move, gitignore |
| `visdrone_yolo_training/` (top) | `research/experiments/visdrone_yolo_training_top/` | Move, gitignore |
| `final-notebooks/.../visdrone_yolo_training/` | `research/experiments/visdrone_yolo_training/` | Move, gitignore |
| `.cursor/` | (keep at root) | Keep as editor config |
| `scripts/` | `scripts/` | Keep example_prd.txt, README.md |
| `README.md` | `README.md` | Rewrite |
| `requirements.txt` | `requirements.txt` | Thin pointer to pyproject |
| `LICENSE` | `LICENSE` | Keep |
| `nia194-object-detection.txt` | `research/notes/nia194-object-detection.txt` | Move |

### 3.2 Ported from ML-project (the additions that ml-proj-man lacks)

| Source in ML-project | Target in ml-proj-man | Action |
|---|---|---|
| `main/visdrone_surveillance/config.py` | `main/visdrone_surveillance/config.py` | Port, merge ML-project env overrides with ml-proj-man defaults |
| `main/visdrone_surveillance/types.py` | same | Port as-is |
| `main/visdrone_surveillance/data/visdrone_dataset.py` | `main/visdrone_surveillance/data/visdrone_dataset.py` | Port as-is (ML-project has the cleaner version) |
| `main/visdrone_surveillance/models/detection_model.py` | `main/visdrone_surveillance/models/faster_rcnn.py` | Port as-is |
| `main/visdrone_surveillance/training/train.py` | `main/visdrone_surveillance/training/train_frcnn.py` | Port, rename to avoid clash with train_yolo |
| `main/visdrone_surveillance/utils/evaluation.py` | `main/visdrone_surveillance/utils/metrics.py` | Port as-is |
| `main/visdrone_surveillance/utils/visualization.py` | `main/visdrone_surveillance/utils/visualization.py` | Port, merge with ml-proj-man visualization if needed |
| `main/visdrone_surveillance/streaming/*` | same | Already identical, keep |
| `main/visdrone_surveillance/surveillance/*` | same | Port all five modules |
| `main/visdrone_surveillance/control/*` | same | Port |
| `main/visdrone_surveillance/follow/*` | same | Port |
| `main/visdrone_surveillance/simulation/*` | same | Port |
| `main/visdrone_surveillance/cli.py` | same | Port, add subcommands for ml-proj-man scripts (prepare, train-yolo, eval) |
| `tests/*` | `tests/*` | Port, update imports to new paths |
| `docs/*` | `docs/*` | Port, extend with ml-proj-man content |
| `pyproject.toml` | same | Port, add ml-proj-man deps (ultralytics, av, ffmpeg-python, pycocotools, detectron2 optional) |
| `README.md` | same | Rewrite to cover both detection families |
| `research/README.md` | same | Port, extend |

### 3.3 Dropped (dead or superseded)

| Item | Reason |
|---|---|
| `src/models/train_yolo.py` | Broken stub; `train_yolo.py` replaces it |
| `src/detection/__init__.py`, `src/stream/__init__.py`, `src/visualization/__init__.py` | Empty placeholders |
| `final-notebooks/drone_detection_system/faster_rcnn_impl/faster_rcnn_processor.py` | Empty file |
| `nia194-object-detection.txt` | Large text artifact, not code; move to research/notes |
| `outputs/`, `runs/`, `filtered/`, `visdrone_yolo_training/` weights | Move to research/experiments, keep gitignored |

## 4. Consolidation decisions (behavior-preserving)

1. **Config split.** ml-proj-man has `src/models/config.py` (YOLOConfig class, YAML-based). ML-project has `visdrone_surveillance/config.py` (module constants, env overrides). Keep both: the YOLO config becomes `models/yolo_config.py`, the package config stays `config.py`. Training scripts take a config path argument, so no merge of config semantics is needed.
2. **Hardcoded absolute path.** `configs/visdrone_custom.yaml` line 3 has `/home/user/ml-proj-man/datasets/VisDrone-YOLO`. Replace with the relative `datasets/VisDrone-YOLO` and add a `VISDRONE_DATASET_ROOT` env override in `yolo_setup.py`.
3. **Class count inconsistency.** `visdrone.yaml` and `visdrone_custom.yaml` use 11 classes; `visdrone_merged.yaml` uses 10. Standardize on the 11-class list from `visdrone.yaml` and document the difference in docs/project-status.md.
4. **Broken stub.** `src/models/train_yolo.py` imports `train, evaluate_model` which do not exist. Delete it; `train_yolo.py` is the replacement.
5. **Import style.** ml-proj-man uses `sys.path.append` hacks and `from src.models...`. Convert everything to relative imports inside the package, matching ML-project.
6. **RTMP handler.** ML-project's `streaming/handler.py` has TODO stubs. Keep ml-proj-man's `RTMPProcessor` (in `final-notebooks/drone_detection_system/`) as the reference implementation and wire it into `handler.py` behind the same interface. The promoted `rtmp_client.py` and `stream_ingestion.py` stay as the canonical ingestion layer.
7. **Faster R-CNN.** Two implementations exist: ML-project's `detection_model.py` (torchvision, clean) and `final-notebooks/faster_rcnn_impl/` (notebook export). Keep the ML-project version as canonical; move the notebook version to `research/notebooks/`.

## 5. Binary and artifact policy

- Remove all `.pt`, `.pth`, `.onnx`, `.zip`, `.tar.gz`, `.jpg`, `.png`, `.csv` artifacts from git history using `git filter-repo` with `--path` globs, exactly as done for `.env`.
- Move the artifacts to `research/experiments/` locally, gitignored, with a `manifest.md` recording their original commit and SHA.
- Add `scripts/download_weights.py` that fetches weights from Hugging Face Hub (or a documented shared drive) into `models/`.
- Retain `.gitattributes` LFS rules only if LFS is chosen; the recommendation is external storage, not LFS.
- `Git LFS` currently tracks 26 `.pt` files. After the rewrite, none are tracked.

## 6. Phases (each is a verifiable unit)

### Phase 0: Safety and decisions

- Confirm push access to `manascb1344/ml-proj-man` with `git push --dry-run`.
- Create branch `refactor/canonical` from `master`.
- Record the pre-rewrite commit SHA of `master` in `research/notes/pre-rewrite-sha.txt`.
- Verify the working tree is clean.

Gate: push dry-run succeeds; tree clean.

### Phase 1: History rewrite (binaries out)

- Run `git filter-repo --path-glob '*.pt' --path-glob '*.pth' --path-glob '*.onnx' --path-glob '*.zip' --path-glob '*.tar.gz' --path-glob '*.jpg' --path-glob '*.png' --path-glob '*.csv' --invert-paths --force`.
- Re-add the remote (filter-repo removes it).
- Verify: `git log --all -- .pt` is empty; `git rev-list --objects --all | grep -c` drops by the artifact count; working tree unchanged.
- Purge local objects: `git reflog expire --expire=now --all && git gc --prune=now`.
- Force-push `master` to `origin/master` (this is the only way to remove the blobs from the remote; the user must confirm).

Gate: no tracked weight/image/csv/zip files; remote still reachable.

### Phase 2: Scaffold the new layout

- Create `main/`, `research/`, `docs/`, `tests/`, `configs/`, `scripts/`.
- Create `main/visdrone_surveillance/` with all `__init__.py` files.
- Write `pyproject.toml` (adapted from ML-project, plus ml-proj-man deps).
- Write `.env.example`.
- Write `main/visdrone_surveillance/config.py` (env-overridable, project-root relative).
- Write `configs/surveillance.json`.

Gate: `pip install -e .` succeeds in a clean venv; `visdrone-surveil --help` prints.

### Phase 3: Move existing ml-proj-man code into the package

- `git mv` every file in section 3.1 into its target path.
- Fix relative imports in every moved module (remove `sys.path.append`, convert `from src.x import y` to `from ..x import y`).
- Fix `configs/visdrone_custom.yaml` absolute path.
- Delete `src/models/train_yolo.py` and the empty `__init__.py` placeholders.
- Standardize the class list on the 11-class version.

Gate: `python -c "import visdrone_surveillance"` succeeds; no `sys.path.append` left in `main/`.

### Phase 4: Port the ML-project additions

- Copy the files in section 3.2 from ML-project into ml-proj-man.
- Resolve import paths (already relative in ML-project, so they transfer cleanly).
- Merge `src/models/config.py` (YOLOConfig) into `models/yolo_config.py`.
- Merge visualization modules (keep the richer one, delete the other).
- Port the test suite and update any paths that changed.
- Port docs and README.

Gate: `pytest tests/ -q` passes with 0 failures (xfails allowed).

### Phase 5: Consolidate overlapping implementations

- One training entry point per family: `train_yolo.py` (YOLO) and
  `train_frcnn.py` (Faster R-CNN). Both callable from the CLI.
- One config loader per family: `yolo_config.py` and `config.py`.
- One video inference entry point: `models/process_video.py` (YOLO) plus
  a Faster R-CNN path in `inference/`.
- Delete `src/models/train_yolo.py`.
- Document the class-count inconsistency in docs/project-status.md.

Gate: `grep -rn "sys.path.append" main/` is empty; no duplicate training
entry points remain.

### Phase 6: Cutover

- Update `README.md` at the top of `aarya-16/ML-project` to point at
  `manascb1344/ml-proj-man` (one paragraph, one link).
- Archive the `aarya-16/ML-project` repo on GitHub (Settings -> Archive).
- Update the `research/ml-proj-man` reference in ML-project to note the
  cutover and the final commit SHA.
- Push `refactor/canonical` as a PR against `master` in ml-proj-man.
- Merge only after review.

Gate: PR open, tests green on the PR, archive done.

### Phase 7: Post-cutover cleanup (optional, separate PR)

- Move `final-notebooks/` content into `research/notebooks/` and delete the
  `final-notebooks/` directory.
- Move `cl.py`, `cl_train.py` into `research/prototypes/`.
- Add `scripts/download_weights.py`.
- Add a GitHub Actions workflow running `pytest` on push.

## 7. Verification (run at every phase boundary)

- `pip install -e ".[test]"` in a fresh venv
- `visdrone-surveil --help` and every subcommand's `--help`
- `pytest tests/ -q` (must be green, xfails allowed only for the three
  pinned bugs)
- `python -c "import visdrone_surveillance"` (no torch required for import)
- `grep -rn "sys.path.append" main/` (must be empty)
- `git ls-files | grep -E '\.(pt|pth|onnx|zip|tar.gz|jpg|png|csv)$'` (must
  be empty after phase 1)
- `git status --porcelain` clean before each commit

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Push access to `ml-proj-man` is missing | Phase 0 verifies dry-run; if it fails, stop and ask the owner |
| 2.3 GB of weights make clone slow | Phase 1 removes them from history; download script replaces them |
| Git LFS history is tangled (3 LFS commits) | filter-repo handles LFS pointers as normal files; verify with `git lfs ls-files` after rewrite |
| Overlapping `src/` and `final-notebooks/` code diverges | Phase 5 picks one canonical implementation per capability and deletes the other |
| Hardcoded `/home/user/...` path breaks on other machines | Phase 3 replaces it with a relative path plus env override |
| `src/models/train_yolo.py` is broken | Delete it; `train_yolo.py` replaces it |
| Tests break during the move | Pin tests were written against `src/` first; update imports only, keep assertions |
| Secret leak (none found in ml-proj-man by grep, but verify) | Run `git grep` for `API_KEY`, `password`, `secret`, `token` before phase 1 |
| Two contributors, two branches | Rebase `refactor/canonical` on `master` before merge; resolve conflicts in phase 6 |

## 9. Rollback

- Before phase 1, record `master`'s SHA in `research/notes/pre-rewrite-sha.txt`.
- If anything goes wrong before phase 6, reset `master` to that SHA with
  `git reset --hard <sha>` and force-push (only the operator's own branch
  is affected at that point).
- After phase 6, rollback means un-archiving ML-project and pointing users
  back at it; the ml-proj-man changes remain as a branch.

## 10. Effort estimate

- Phase 0-1: 1-2 hours (mostly filter-repo runtime and verification)
- Phase 2: 1 hour
- Phase 3: 3-4 hours (import rewriting across ~20 modules)
- Phase 4: 2-3 hours (porting + merge conflicts)
- Phase 5: 1-2 hours
- Phase 6: 1 hour plus review wait
- Total: roughly 2-3 focused days, plus review.

## 11. What I will not do

- I will not rewrite `ml-proj-man`'s history without a confirmed push
  dry-run.
- I will not delete the `ml-proj-man` remote or its branches.
- I will not touch the 2.3 GB of weights without moving them to external
  storage first.
- I will not merge the PR; that is the operator's call.
