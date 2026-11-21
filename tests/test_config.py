"""Pin the behavior of the config module and the VisDrone class list."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visdrone_surveillance import config


def test_class_list_matches_visdrone_taxonomy():
    assert config.CLASSES == [
        "ignored regions",
        "pedestrian",
        "people",
        "bicycle",
        "car",
        "van",
        "truck",
        "tricycle",
        "awning-tricycle",
        "bus",
        "motor",
        "others",
    ]


def test_num_classes_constant_is_stale_but_unused():
    # NUM_CLASSES says 10 while CLASSES holds 12 entries. Every call site
    # builds the model with len(CLASSES), so NUM_CLASSES is dead. Pinned so a
    # future cleanup knows the constant does not drive model construction.
    assert config.NUM_CLASSES == 10
    assert len(config.CLASSES) == 12


def test_model_class_count_is_derived_from_class_list():
    # The dataset remaps VisDrone category N to label N-1, so labels run 0-10.
    # The model is built with len(CLASSES) == 12, leaving one untrained output.
    # Saved checkpoints were trained at 12; changing this invalidates them.
    assert len(config.CLASSES) == 12
    assert max(config.CLASSES.index(c) for c in config.CLASSES) == 11


def test_dataset_paths_are_derived_from_dataset_root():
    assert config.SEQUENCES_PATH.endswith(
        os.path.join("VisDrone2019-VID-train", "sequences")
    )
    assert config.ANNOTATIONS_PATH.endswith(
        os.path.join("VisDrone2019-VID-train", "annotations")
    )


def test_rtmp_output_url_is_built_from_server_and_key():
    assert config.RTMP_OUTPUT_URL == f"{config.RTMP_SERVER}/{config.RTMP_STREAM_KEY}"


def test_training_defaults():
    assert config.BATCH_SIZE == 8
    assert config.LEARNING_RATE == 0.001
    assert config.NUM_EPOCHS == 20
    assert config.VALIDATION_SPLIT == 0.2
    assert config.SEED == 42


def test_thresholds():
    assert config.CONFIDENCE_THRESHOLD == 0.5
    assert config.NMS_THRESHOLD == 0.4
