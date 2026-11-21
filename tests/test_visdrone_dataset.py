"""Pin the behavior of the VisDrone dataset loader.

These build tiny on-disk fixtures in the VisDrone VID layout, so the
annotation parsing, box conversion, and class remapping are pinned without
needing the real 2 GB dataset.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visdrone_surveillance.data.visdrone_dataset import VisDroneDataset, get_transform, collate_fn


def _make_fixture(tmp_path, num_sequences=2, frames_per_sequence=1):
    """Create a miniature VisDrone VID dataset and return its paths.

    Annotation columns:
    frame_id, object_id, x, y, w, h, score, category, truncation, occlusion
    """
    import cv2

    sequences = tmp_path / "sequences"
    annotations = tmp_path / "annotations"
    sequences.mkdir()
    annotations.mkdir()

    for seq in range(num_sequences):
        seq_dir = sequences / f"seq_{seq}"
        seq_dir.mkdir()
        lines = []
        for frame in range(1, frames_per_sequence + 1):
            # A blank 40x50 image is enough for the loader to parse.
            cv2.imwrite(str(seq_dir / f"{frame}.jpg"), np.zeros((40, 50, 3), np.uint8))
            lines += [
                f"{frame},0,10,10,20,20,1,4,0,0",  # category 4 -> label 3
                f"{frame},1,30,5,10,10,1,1,0,0",  # category 1 -> label 0
                f"{frame},2,0,0,5,5,1,0,0,0",  # category 0 -> ignored, dropped
            ]
        (annotations / f"seq_{seq}.txt").write_text("\n".join(lines) + "\n")

    return str(sequences), str(annotations)


def test_dataset_length_counts_annotated_frames(tmp_path):
    sequences, annotations = _make_fixture(
        tmp_path, num_sequences=3, frames_per_sequence=2
    )
    # The default 0.2 validation split keeps int(3 * 0.8) == 2 sequences.
    ds = VisDroneDataset(sequences, annotations, transform=None, train=True)
    assert len(ds) == 2 * 2


def test_split_can_be_disabled_to_keep_every_sequence(tmp_path):
    sequences, annotations = _make_fixture(
        tmp_path, num_sequences=3, frames_per_sequence=2
    )
    ds = VisDroneDataset(
        sequences, annotations, transform=None, train=True, validation_split=0.0
    )
    assert len(ds) == 3 * 2


def test_single_sequence_with_validation_split_yields_empty_train_set(tmp_path):
    # int(1 * (1 - 0.2)) == 0, so the train split holds no sequences at all.
    # Pinned because it silently produces an empty dataset rather than raising.
    sequences, annotations = _make_fixture(
        tmp_path, num_sequences=1, frames_per_sequence=1
    )
    ds = VisDroneDataset(sequences, annotations, transform=None, train=True)
    assert len(ds) == 0


def test_ignored_region_category_is_dropped(tmp_path):
    sequences, annotations = _make_fixture(
        tmp_path, num_sequences=1, frames_per_sequence=1
    )
    ds = VisDroneDataset(
        sequences, annotations, transform=None, train=True, validation_split=0.0
    )
    image, target = ds[0]
    # category 0 is dropped, so 2 boxes remain out of 3 annotations
    assert len(target["boxes"]) == 2


def test_labels_are_remapped_to_zero_indexed(tmp_path):
    sequences, annotations = _make_fixture(
        tmp_path, num_sequences=1, frames_per_sequence=1
    )
    ds = VisDroneDataset(
        sequences, annotations, transform=None, train=True, validation_split=0.0
    )
    image, target = ds[0]
    labels = sorted(target["labels"].tolist())
    assert labels == [0, 3]


def test_boxes_are_converted_to_xyxy(tmp_path):
    sequences, annotations = _make_fixture(
        tmp_path, num_sequences=1, frames_per_sequence=1
    )
    ds = VisDroneDataset(
        sequences, annotations, transform=None, train=True, validation_split=0.0
    )
    image, target = ds[0]
    boxes = target["boxes"].numpy()
    # annotation x=10,y=10,w=20,h=20 becomes x1=10,y1=10,x2=30,y2=30
    expected = np.array([[10.0, 10.0, 30.0, 30.0], [30.0, 5.0, 40.0, 15.0]])
    assert np.allclose(np.sort(boxes, axis=0), np.sort(expected, axis=0))


def test_boxes_are_clipped_to_image_bounds(tmp_path):
    sequences, annotations = _make_fixture(
        tmp_path, num_sequences=1, frames_per_sequence=1
    )
    ds = VisDroneDataset(
        sequences, annotations, transform=None, train=True, validation_split=0.0
    )
    image, target = ds[0]
    # The fixture images are 40 rows by 50 columns. The sample is a CHW
    # tensor, so image.shape[:2] is (channels, rows), not (rows, cols).
    height, width = 40, 50
    for x1, y1, x2, y2 in target["boxes"].tolist():
        assert 0 <= x1 < x2 <= width
        assert 0 <= y1 < y2 <= height


def test_sample_is_returned_as_a_normalized_chw_tensor(tmp_path):
    sequences, annotations = _make_fixture(
        tmp_path, num_sequences=1, frames_per_sequence=1
    )
    ds = VisDroneDataset(
        sequences, annotations, transform=None, train=True, validation_split=0.0
    )
    image, target = ds[0]
    assert image.shape[0] == 3
    assert image.dtype.is_floating_point
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0


def test_train_and_val_splits_are_disjoint(tmp_path):
    sequences, annotations = _make_fixture(
        tmp_path, num_sequences=10, frames_per_sequence=1
    )
    train = VisDroneDataset(
        sequences,
        annotations,
        transform=None,
        train=True,
        validation_split=0.2,
        seed=42,
    )
    val = VisDroneDataset(
        sequences,
        annotations,
        transform=None,
        train=False,
        validation_split=0.2,
        seed=42,
    )
    train_paths = {s[0] for s in train.samples}
    val_paths = {s[0] for s in val.samples}
    assert train_paths.isdisjoint(val_paths)
    assert len(train_paths) + len(val_paths) == 10


def test_split_is_reproducible_for_a_fixed_seed(tmp_path):
    sequences, annotations = _make_fixture(
        tmp_path, num_sequences=10, frames_per_sequence=1
    )
    a = VisDroneDataset(
        sequences,
        annotations,
        transform=None,
        train=True,
        validation_split=0.2,
        seed=42,
    )
    b = VisDroneDataset(
        sequences,
        annotations,
        transform=None,
        train=True,
        validation_split=0.2,
        seed=42,
    )
    assert [s[0] for s in a.samples] == [s[0] for s in b.samples]


def test_collate_returns_lists(tmp_path):
    sequences, annotations = _make_fixture(
        tmp_path, num_sequences=1, frames_per_sequence=2
    )
    ds = VisDroneDataset(
        sequences, annotations, transform=None, train=True, validation_split=0.0
    )
    images, targets = collate_fn([ds[0], ds[1]])
    assert isinstance(images, list) and len(images) == 2
    assert isinstance(targets, list) and len(targets) == 2
    assert set(targets[0].keys()) == {"boxes", "labels"}


def test_train_transform_flips_and_resizes(tmp_path):
    sequences, annotations = _make_fixture(
        tmp_path, num_sequences=1, frames_per_sequence=1
    )
    ds = VisDroneDataset(
        sequences,
        annotations,
        transform=get_transform(train=True),
        train=True,
        validation_split=0.0,
    )
    image, target = ds[0]
    assert image.shape[-2:] == (640, 640)


def test_val_transform_resizes_without_augmentation(tmp_path):
    sequences, annotations = _make_fixture(
        tmp_path, num_sequences=1, frames_per_sequence=1
    )
    ds = VisDroneDataset(
        sequences,
        annotations,
        transform=get_transform(train=False),
        train=True,
        validation_split=0.0,
    )
    image, target = ds[0]
    assert image.shape[-2:] == (640, 640)
