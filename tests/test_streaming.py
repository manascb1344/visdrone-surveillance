"""Smoke/characterization tests for the RTMP streaming package.

These are offline: no network, no GPU, no model weights. RTMP connections are
either never opened (fake URLs, hooks driven directly) or stubbed at the
``cv2.VideoCapture`` boundary.

Only observable behavior is pinned. Where the current implementation is
broken the test is marked ``xfail(strict=True)`` with the bug named, so the
failure is tracked rather than hidden.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main")
)

from visdrone_surveillance.streaming import (  # noqa: E402
    RTMPClient,
    RTMPHandler,
    RTMPStreamManager,
    VisDroneRTMPClient,
)
from visdrone_surveillance.streaming import stream_ingestion  # noqa: E402

FAKE_URL = "rtmp://127.0.0.1:1/live/fake"


def _frame(value=0):
    """A tiny fake frame with a distinguishable payload."""
    return np.full((4, 4, 3), value, dtype=np.uint8)


# --------------------------------------------------------------------------
# Package surface
# --------------------------------------------------------------------------


def test_streaming_package_exports_public_names():
    import visdrone_surveillance.streaming as streaming

    for name in (
        "RTMPHandler",
        "RTMPClient",
        "VisDroneRTMPClient",
        "RTMPStreamManager",
    ):
        assert hasattr(streaming, name), name
        assert name in streaming.__all__


def test_rtmp_client_is_an_abstract_base_class():
    with pytest.raises(TypeError):
        RTMPClient(FAKE_URL)


def test_visdrone_client_is_a_subclass_of_rtmp_client():
    assert issubclass(VisDroneRTMPClient, RTMPClient)


def test_stream_ingestion_uses_relative_import_of_rtmp_client():
    # The module was written for a different package layout; it must resolve
    # RTMPClient through the package-relative import.
    assert stream_ingestion.RTMPClient is RTMPClient


# --------------------------------------------------------------------------
# VisDroneRTMPClient: construction (no connection is attempted)
# --------------------------------------------------------------------------


def test_client_construction_does_not_connect():
    client = VisDroneRTMPClient(FAKE_URL)
    assert client.rtmp_url == FAKE_URL
    assert client.is_connected is False
    assert client.cap is None
    assert client.frame_count == 0
    assert client.is_running is False


def test_client_keeps_configured_buffer_size():
    client = VisDroneRTMPClient(FAKE_URL, buffer_size=5)
    assert client.buffer_size == 5
    assert client.frame_buffer.maxsize == 5


def test_client_starts_with_empty_stats():
    client = VisDroneRTMPClient(FAKE_URL)
    assert client.stats == {
        "fps": 0.0,
        "dropped_frames": 0,
        "reconnections": 0,
        "uptime": 0.0,
        "last_activity": 0.0,
    }


# --------------------------------------------------------------------------
# VisDroneRTMPClient: frame buffering
# --------------------------------------------------------------------------


def test_on_frame_queues_the_frame():
    client = VisDroneRTMPClient(FAKE_URL)
    frame = _frame(7)
    client.on_frame(frame)
    assert client.frame_buffer.qsize() == 1


def test_on_frame_stores_a_copy_not_the_same_object():
    client = VisDroneRTMPClient(FAKE_URL)
    frame = _frame(3)
    client.on_frame(frame)
    stored = client.frame_buffer.get_nowait()
    assert stored is not frame
    assert np.array_equal(stored, frame)


def test_buffer_drops_oldest_frame_when_full():
    client = VisDroneRTMPClient(FAKE_URL, buffer_size=2)
    for i in range(3):
        client.on_frame(_frame(i))
    assert client.frame_buffer.qsize() == 2
    assert client.stats["dropped_frames"] == 1
    # The oldest frame was dropped; the two newest survive.
    assert np.array_equal(client.frame_buffer.get_nowait(), _frame(1))
    assert np.array_equal(client.frame_buffer.get_nowait(), _frame(2))


def test_get_latest_frame_returns_newest_frame():
    client = VisDroneRTMPClient(FAKE_URL)
    client.on_frame(_frame(1))
    client.on_frame(_frame(2))
    latest = client.get_latest_frame()
    assert latest is not None
    assert np.array_equal(latest, _frame(2))


def test_get_latest_frame_on_empty_buffer_is_none():
    client = VisDroneRTMPClient(FAKE_URL)
    assert client.get_latest_frame() is None


def test_get_frame_batch_returns_requested_count():
    client = VisDroneRTMPClient(FAKE_URL)
    for i in range(4):
        client.on_frame(_frame(i))
    batch = client.get_frame_batch(2)
    assert len(batch) == 2
    assert np.array_equal(batch[0], _frame(0))


def test_get_frame_batch_on_empty_buffer_is_empty_list():
    client = VisDroneRTMPClient(FAKE_URL)
    assert client.get_frame_batch(3) == []


# --------------------------------------------------------------------------
# VisDroneRTMPClient: statistics
# --------------------------------------------------------------------------


def test_stats_report_buffer_usage_and_capacity():
    client = VisDroneRTMPClient(FAKE_URL, buffer_size=10)
    client.on_frame(_frame(1))
    stats = client.get_stats()
    assert stats["buffer_usage"] == 1
    assert stats["buffer_capacity"] == 10


def test_stats_include_stream_info():
    client = VisDroneRTMPClient(FAKE_URL)
    stats = client.get_stats()
    assert stats["stream_info"]["connected"] is False
    assert stats["stream_info"]["url"] == FAKE_URL
    assert stats["stream_info"]["frame_count"] == 0


def test_fps_is_populated_after_two_frames():
    client = VisDroneRTMPClient(FAKE_URL)
    client.on_frame(_frame(1))
    client.on_frame(_frame(2))
    assert client.stats["fps"] > 0.0


def test_last_activity_is_updated_on_frame():
    client = VisDroneRTMPClient(FAKE_URL)
    client.on_frame(_frame(1))
    assert client.stats["last_activity"] > 0.0


# --------------------------------------------------------------------------
# VisDroneRTMPClient: event callbacks
# --------------------------------------------------------------------------


def test_connect_event_callback_is_triggered():
    client = VisDroneRTMPClient(FAKE_URL)
    seen = []
    client.register_event_callback("connect", seen.append)
    client.on_connect()
    assert len(seen) == 1
    assert seen[0]["url"] == FAKE_URL


def test_disconnect_event_callback_is_triggered_with_uptime():
    client = VisDroneRTMPClient(FAKE_URL)
    seen = []
    client.register_event_callback("disconnect", seen.append)
    client.on_connect()
    client.on_disconnect()
    assert len(seen) == 1
    assert seen[0]["url"] == FAKE_URL
    assert seen[0]["uptime"] >= 0.0


def test_frame_event_callback_is_triggered():
    client = VisDroneRTMPClient(FAKE_URL)
    seen = []
    client.register_event_callback("frame", seen.append)
    client.on_frame(_frame(1))
    assert len(seen) == 1
    assert "frame_count" in seen[0]


def test_multiple_callbacks_for_the_same_event_all_run():
    client = VisDroneRTMPClient(FAKE_URL)
    first, second = [], []
    client.register_event_callback("frame", first.append)
    client.register_event_callback("frame", second.append)
    client.on_frame(_frame(1))
    assert len(first) == 1
    assert len(second) == 1


def test_callback_exception_is_swallowed():
    client = VisDroneRTMPClient(FAKE_URL)

    def boom(_data):
        raise RuntimeError("callback blew up")

    client.register_event_callback("frame", boom)
    # The error is logged, not propagated.
    client.on_frame(_frame(1))
    assert client.frame_buffer.qsize() == 1


# --------------------------------------------------------------------------
# RTMPClient base behavior that does not require a connection
# --------------------------------------------------------------------------


def test_read_frame_without_connection_returns_failure():
    client = VisDroneRTMPClient(FAKE_URL)
    success, frame = client.read_frame()
    assert success is False
    assert frame is None


def test_get_stream_info_without_connection():
    client = VisDroneRTMPClient(FAKE_URL)
    info = client.get_stream_info()
    assert info == {"connected": False, "url": FAKE_URL, "frame_count": 0}


def test_disconnect_without_connection_is_a_noop():
    client = VisDroneRTMPClient(FAKE_URL)
    client.disconnect()
    assert client.is_connected is False


def test_read_frame_counts_frames_and_triggers_on_frame(monkeypatch):
    client = VisDroneRTMPClient(FAKE_URL)
    seen = []
    client.register_event_callback("frame", seen.append)

    class _Cap:
        def read(self):
            return True, _frame(1)

        def release(self):
            pass

    client.cap = _Cap()
    client.is_connected = True
    success, frame = client.read_frame()
    assert success is True
    assert frame is not None
    assert client.frame_count == 1
    assert len(seen) == 1


# --------------------------------------------------------------------------
# RTMPStreamManager (no network: start() is stubbed)
# --------------------------------------------------------------------------


@pytest.fixture
def manager(monkeypatch):
    """A manager whose clients never attempt a real connection."""
    monkeypatch.setattr(stream_ingestion.VisDroneRTMPClient, "start", lambda self: True)
    monkeypatch.setattr(stream_ingestion.VisDroneRTMPClient, "stop", lambda self: None)
    return RTMPStreamManager()


def test_manager_starts_empty(manager):
    assert manager.get_active_streams() == []
    assert manager.get_latest_frames() == {}
    assert manager.get_all_stats() == {}


def test_add_stream_registers_the_client(manager):
    assert manager.add_stream("cam1", FAKE_URL) is True
    assert manager.get_active_streams() == ["cam1"]
    client = manager.get_stream("cam1")
    assert isinstance(client, VisDroneRTMPClient)
    assert client.rtmp_url == FAKE_URL


def test_add_stream_with_duplicate_id_replaces_it(manager):
    manager.add_stream("cam1", FAKE_URL)
    manager.add_stream("cam1", "rtmp://127.0.0.1:1/live/other")
    assert manager.get_active_streams() == ["cam1"]
    assert manager.get_stream("cam1").rtmp_url == "rtmp://127.0.0.1:1/live/other"


def test_get_stream_for_unknown_id_is_none(manager):
    assert manager.get_stream("nope") is None


def test_remove_stream_removes_the_client(manager):
    manager.add_stream("cam1", FAKE_URL)
    manager.remove_stream("cam1")
    assert manager.get_active_streams() == []
    assert manager.get_stream("cam1") is None


def test_remove_unknown_stream_is_a_noop(manager):
    manager.remove_stream("nope")
    assert manager.get_active_streams() == []


def test_get_latest_frames_maps_stream_ids_to_frames(manager):
    manager.add_stream("cam1", FAKE_URL)
    manager.add_stream("cam2", FAKE_URL)
    manager.get_stream("cam1").on_frame(_frame(1))
    manager.get_stream("cam2").on_frame(_frame(2))
    frames = manager.get_latest_frames()
    assert set(frames) == {"cam1", "cam2"}
    assert np.array_equal(frames["cam1"], _frame(1))
    assert np.array_equal(frames["cam2"], _frame(2))


def test_get_latest_frames_skips_streams_without_frames(manager):
    manager.add_stream("cam1", FAKE_URL)
    manager.add_stream("cam2", FAKE_URL)
    manager.get_stream("cam2").on_frame(_frame(2))
    frames = manager.get_latest_frames()
    assert set(frames) == {"cam2"}


def test_get_all_stats_reports_every_stream(manager):
    manager.add_stream("cam1", FAKE_URL)
    stats = manager.get_all_stats()
    assert set(stats) == {"cam1"}
    assert stats["cam1"]["buffer_capacity"] == 30


def test_stop_all_streams_clears_the_manager(manager):
    manager.add_stream("cam1", FAKE_URL)
    manager.stop_all_streams()
    assert manager.get_active_streams() == []


# --------------------------------------------------------------------------
# RTMPHandler (model weights are never loaded here)
# --------------------------------------------------------------------------


def test_rtmp_handler_is_importable_from_the_package():
    assert RTMPHandler.__name__ == "RTMPHandler"


def test_rtmp_handler_connect_rtmp_uses_the_configured_url(capsys):
    from visdrone_surveillance import config

    handler = RTMPHandler.__new__(RTMPHandler)  # skip model loading
    assert handler.connect_rtmp() is True
    assert handler.rtmp_url == config.RTMP_OUTPUT_URL

    handler = RTMPHandler.__new__(RTMPHandler)
    assert handler.connect_rtmp("rtmp://example/live") is True
    assert handler.rtmp_url == "rtmp://example/live"


# --------------------------------------------------------------------------
# Known bugs (pinned, not fixed)
# --------------------------------------------------------------------------


def test_frame_event_reports_the_frame_count():
    client = VisDroneRTMPClient(FAKE_URL)
    seen = []
    client.register_event_callback("frame", seen.append)

    class _Cap:
        def __init__(self):
            self.calls = 0

        def read(self):
            self.calls += 1
            return True, _frame(1)

        def release(self):
            pass

    client.cap = _Cap()
    client.is_connected = True
    client.read_frame()
    client.read_frame()
    assert len(seen) == 2
    assert seen[0]["frame_count"] == 1
    assert seen[1]["frame_count"] == 2


@pytest.mark.xfail(
    reason=(
        "Bug: VisDroneRTMPClient exposes a 'reconnections' counter in its "
        "stats, but handle_reconnection in the base RTMPClient never "
        "increments it, so the counter stays at 0 forever. Tracked for a "
        "separate fix."
    ),
    strict=True,
)
def test_reconnections_counter_is_incremented(monkeypatch):
    client = VisDroneRTMPClient(FAKE_URL, reconnect_delay=0.0)
    monkeypatch.setattr(client, "disconnect", lambda: None)
    monkeypatch.setattr(client, "connect", lambda: True)
    client.handle_reconnection()
    assert client.stats["reconnections"] == 1


@pytest.mark.xfail(
    reason=(
        "Bug: VisDroneRTMPClient.get_latest_frame is documented to return "
        "the most recent frame 'without removing it', but it drains every "
        "buffered frame and only puts back frames[:-1]. With a single "
        "buffered frame the buffer is left empty, so the frame is consumed "
        "after all. Tracked for a separate fix."
    ),
    strict=True,
)
def test_get_latest_frame_does_not_consume_the_buffer():
    client = VisDroneRTMPClient(FAKE_URL)
    client.on_frame(_frame(1))
    client.get_latest_frame()
    assert client.frame_buffer.qsize() == 1
