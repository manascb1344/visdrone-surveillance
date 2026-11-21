"""Smoke tests for the ``visdrone-surveil`` CLI.

These only exercise argument parsing and help output; no training,
inference, or video processing is performed.
"""

import pytest

from visdrone_surveillance.cli import build_parser, main


def _help_text(capsys):
    out = capsys.readouterr()
    return out.out + out.err


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["--help"], ["train", "rtmp-demo", "surveil", "follow"]),
        (["train", "--help"], ["--batch_size", "--learning_rate", "--num_epochs"]),
        (
            ["rtmp-demo", "--help"],
            ["--model_path", "--input_video", "--output_video", "--confidence"],
        ),
        (
            ["surveil", "--help"],
            [
                "--video",
                "--camera",
                "--config",
                "--output",
                "--rtmp",
                "--rtmp-url",
                "--disable-alpr",
                "--disable-classification",
                "--disable-tracking",
                "--disable-speed",
                "--resolution",
                "--fps",
                "--headless",
                "--model-path",
                "--device",
                "--confidence",
                "--speed-limit",
            ],
        ),
        (
            ["follow", "--help"],
            [
                "--steps",
                "--seed",
                "--output",
                "--max-forward",
                "--max-yaw-rate",
                "--reacquire-timeout",
            ],
        ),
    ],
)
def test_help_exits_zero_and_lists_flags(argv, expected, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 0

    text = _help_text(capsys)
    for flag in expected:
        assert flag in text, f"missing {flag!r} in help output"


def test_top_level_help_lists_all_subcommands(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    text = _help_text(capsys)
    assert "train" in text
    assert "rtmp-demo" in text
    assert "surveil" in text
    assert "follow" in text


def test_parser_defaults_match_legacy_scripts():
    parser = build_parser()

    train_args = parser.parse_args(["train"])
    assert train_args.batch_size == 8
    assert train_args.learning_rate == 0.001
    assert train_args.num_epochs == 20

    demo_args = parser.parse_args(
        ["rtmp-demo", "--model_path", "m.pt", "--input_video", "in.mp4"]
    )
    assert demo_args.output_video is None
    assert demo_args.confidence == 0.5

    surveil_args = parser.parse_args(["surveil"])
    assert surveil_args.camera == 0
    assert surveil_args.config == "surveillance_config.json"
    assert surveil_args.rtmp is False
    assert surveil_args.rtmp_url == "rtmp://localhost/live/stream"
    assert surveil_args.resolution == "1280x720"
    assert surveil_args.fps == 30.0
    assert surveil_args.headless is False
    assert surveil_args.device is None
    assert surveil_args.confidence == 0.5
    assert surveil_args.speed_limit == 70.0
    for flag in (
        "disable_alpr",
        "disable_classification",
        "disable_tracking",
        "disable_speed",
    ):
        assert getattr(surveil_args, flag) is False


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    text = _help_text(capsys)
    assert "usage:" in text


def test_train_subcommand_dispatches_to_train_model(monkeypatch):
    """The ``train`` subcommand calls training.train.train_model with the args."""
    calls = {}

    def fake_train_model(args):
        calls["args"] = args

    import visdrone_surveillance.training.train_frcnn as train_module

    monkeypatch.setattr(train_module, "train_model", fake_train_model)
    assert main(["train", "--num_epochs", "3"]) == 0
    assert calls["args"].num_epochs == 3
