"""Mission state machine for autonomous vehicle following.

The pilot wires the tracker and the guidance controller together and
decides what the drone should do while the target is picked, tracked,
temporarily lost, or gone for good.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable, Optional, Sequence, Tuple

from ..follow.tracker import SingleTargetTracker
from ..types import Detection, TargetState, VelocityCommand
from .guidance import GuidanceController


class PilotState(Enum):
    """States of the follow mission."""

    SELECT = "select"
    LOCK = "lock"
    FOLLOW = "follow"
    REACQUIRE = "reacquire"
    LOST = "lost"
    ABORT = "abort"


class FollowPilot:
    """Drive a drone so it follows the vehicle the operator picked.

    Args:
        controller: The guidance controller producing velocity commands.
        lock_frames: Number of hover frames spent in ``LOCK`` before
            following starts.
        max_missed: Consecutive missed frames after which the tracker
            counts the target as lost.
        reacquire_timeout: Seconds spent searching before giving up.
        reacquire_speed_scale: Speed factor applied while searching.
    """

    def __init__(
        self,
        controller: GuidanceController,
        lock_frames: int = 2,
        max_missed: int = 5,
        reacquire_timeout: float = 3.0,
        reacquire_speed_scale: float = 0.4,
    ) -> None:
        self.controller = controller
        self.lock_frames = int(lock_frames)
        self.max_missed = int(max_missed)
        self.reacquire_timeout = float(reacquire_timeout)
        self.reacquire_speed_scale = float(reacquire_speed_scale)

        self._state = PilotState.SELECT
        self._tracker: Optional[SingleTargetTracker] = None
        self._lock_remaining = 0
        self._reacquire_start: Optional[float] = None
        self._last_command = VelocityCommand()
        self._search_yaw = 0.0
        self._search_direction = 1.0

    @property
    def state(self) -> PilotState:
        """The current mission state."""
        return self._state

    @property
    def tracker(self) -> Optional[SingleTargetTracker]:
        """The tracker, or ``None`` while no target is selected."""
        return self._tracker

    @property
    def target_state(self) -> Optional[TargetState]:
        """The current target estimate, or ``None`` while no target is selected."""
        return self._tracker.state if self._tracker is not None else None

    def set_target(self, detection: Detection, timestamp: float) -> None:
        """Accept the operator's pick and start locking onto it."""
        self._tracker = SingleTargetTracker(detection, timestamp)
        self.controller.reset()
        self._lock_remaining = self.lock_frames
        self._reacquire_start = None
        self._last_command = VelocityCommand()
        self._state = PilotState.LOCK

    def step(
        self,
        detections: Iterable[Detection],
        timestamp: float,
        image_size: Optional[Tuple[float, float]] = None,
        dt: float = 0.05,
    ) -> VelocityCommand:
        """Advance the mission by one frame and return the command.

        Every state handles every input without raising. In ``SELECT``
        the drone hovers; in ``LOCK`` it hovers for a couple of frames;
        in ``FOLLOW`` the guidance controller drives; in ``REACQUIRE``
        the last direction is flown at reduced speed while searching
        until the reacquire timeout, measured in timestamp seconds, has
        expired; in ``LOST`` and ``ABORT`` the drone holds position.
        """
        frame: Sequence[Detection] = list(detections or ())
        if image_size is not None:
            self.controller.image_size = (float(image_size[0]), float(image_size[1]))

        if self._state == PilotState.SELECT or self._tracker is None:
            self._last_command = VelocityCommand()
            return self._last_command

        if self._state == PilotState.LOST:
            self._last_command = VelocityCommand()
            return self._last_command

        if self._state == PilotState.ABORT:
            self._last_command = VelocityCommand()
            return self._last_command

        if self._state == PilotState.LOCK:
            self._tracker.update(frame, timestamp)
            self._lock_remaining -= 1
            if self._lock_remaining > 0:
                self._last_command = VelocityCommand()
                return self._last_command
            self._state = PilotState.FOLLOW

        if self._state == PilotState.FOLLOW:
            state = self._tracker.update(frame, timestamp)
            if self._tracker.lost(self.max_missed):
                self._state = PilotState.REACQUIRE
                self._reacquire_start = timestamp
                self._search_yaw = 0.0
                self._search_direction = 1.0
                return self._search_command(dt)
            command = self.controller.update(state, dt)
            self._last_command = command
            return command

        # REACQUIRE: the tracker keeps consuming detections, so a match
        # here brings the mission straight back to FOLLOW.
        state = self._tracker.update(frame, timestamp)
        if state.missed == 0:
            self.controller.reset()
            self._state = PilotState.FOLLOW
            self._reacquire_start = None
            command = self.controller.update(state, dt)
            self._last_command = command
            return command
        # The timeout runs on the frame clock, not on the summed dt, so a
        # slow frame cannot stretch the search window.
        elapsed = timestamp - (self._reacquire_start or timestamp)
        if elapsed < 0.0:
            elapsed = 0.0
        if elapsed >= self.reacquire_timeout:
            self._state = PilotState.LOST
            self._last_command = VelocityCommand()
            return self._last_command
        return self._search_command(dt)

    def _search_command(self, dt: float) -> VelocityCommand:
        """Fly the last heading at reduced speed while sweeping the yaw."""
        self._search_yaw += self._search_direction * dt
        if abs(self._search_yaw) > 0.5:
            self._search_direction = -self._search_direction
            self._search_yaw = -self._search_yaw
        scale = self.reacquire_speed_scale
        command = VelocityCommand(
            forward=self._last_command.forward * scale,
            yaw_rate=self._last_command.yaw_rate * scale
            + self._search_direction * 0.2 * scale,
            climb=self._last_command.climb * scale,
        )
        command = command.clamped(
            self.controller.max_forward,
            self.controller.max_yaw_rate,
            self.controller.max_climb,
        )
        self._last_command = command
        return command

    def abort(self) -> None:
        """Stop the mission and hold position."""
        self._state = PilotState.ABORT
        self._last_command = VelocityCommand()

    def reset(self) -> None:
        """Return to the ``SELECT`` state and forget the target."""
        self._state = PilotState.SELECT
        self._tracker = None
        self._lock_remaining = 0
        self._reacquire_start = None
        self._last_command = VelocityCommand()
        self._search_yaw = 0.0
        self._search_direction = 1.0
        self.controller.reset()
