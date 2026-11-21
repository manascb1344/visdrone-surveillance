"""PID guidance that turns a target estimate into a velocity command.

Sign conventions
----------------
The camera looks along the drone heading, so the image is a view of
what is in front of the drone.

``yaw_rate`` is positive when the drone turns to the right, that is,
toward a target that sits right of the image center. A positive image
offset (target to the right) therefore produces a positive yaw rate.

``forward`` is positive to move toward the target. The range error is
``(target_scale - desired_scale) / desired_scale``: a target that looks
bigger than desired is closer than desired, so the error is positive
and the drone must slow down or back off. The controller therefore
negates the range error before applying the forward gain, so a positive
range error yields a negative (backward) command.

``climb`` is positive up. A target above the image center (negative
vertical offset in image coordinates, since y grows downward) makes the
drone climb.
"""

from __future__ import annotations

from typing import Tuple

from ..types import TargetState, VelocityCommand


class GuidanceController:
    """PID controller on yaw, range, and vertical errors.

    Args:
        image_size: ``(width, height)`` of the camera frame in pixels.
        desired_scale: Target box size in pixels that corresponds to the
            wanted standoff distance.
        kp: Proportional gain.
        ki: Integral gain.
        kd: Derivative gain.
        max_forward: Maximum forward speed in m/s.
        max_yaw_rate: Maximum yaw rate in rad/s.
        max_climb: Maximum climb speed in m/s.
    """

    def __init__(
        self,
        image_size: Tuple[float, float],
        desired_scale: float,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        max_forward: float = 5.0,
        max_yaw_rate: float = 1.5,
        max_climb: float = 2.0,
    ) -> None:
        self.image_size = (float(image_size[0]), float(image_size[1]))
        self.desired_scale = float(desired_scale)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.max_forward = float(max_forward)
        self.max_yaw_rate = float(max_yaw_rate)
        self.max_climb = float(max_climb)

        self._yaw_integral = 0.0
        self._range_integral = 0.0
        self._climb_integral = 0.0
        self._prev_yaw_error: float = None
        self._prev_range_error: float = None
        self._prev_climb_error: float = None

    def reset(self) -> None:
        """Clear the integrators and the derivative history."""
        self._yaw_integral = 0.0
        self._range_integral = 0.0
        self._climb_integral = 0.0
        self._prev_yaw_error = None
        self._prev_range_error = None
        self._prev_climb_error = None

    def update(self, state: TargetState, dt: float) -> VelocityCommand:
        """Compute the clamped velocity command for one frame."""
        dt = float(dt)
        if dt <= 0.0:
            dt = 1e-3

        width, height = self.image_size
        cx, cy = state.center

        # Yaw error: horizontal offset from the image center, normalized.
        yaw_error = (cx - width / 2.0) / max(width / 2.0, 1e-6)
        # Range error: positive means the target is too close.
        desired = self.desired_scale if self.desired_scale > 0 else 1.0
        range_error = (state.scale - desired) / desired
        # Vertical error: positive means the target is below the center.
        climb_error = (cy - height / 2.0) / max(height / 2.0, 1e-6)

        self._yaw_integral += yaw_error * dt
        self._range_integral += range_error * dt
        self._climb_integral += climb_error * dt

        # Anti-windup: the integral is bounded so a long-standing error
        # cannot accumulate enough authority to overshoot when the error
        # finally clears.
        self._yaw_integral = self._clamp(self._yaw_integral, -1.0, 1.0)
        self._range_integral = self._clamp(self._range_integral, -1.0, 1.0)
        self._climb_integral = self._clamp(self._climb_integral, -1.0, 1.0)

        yaw_deriv = self._derivative(yaw_error, self._prev_yaw_error, dt)
        range_deriv = self._derivative(range_error, self._prev_range_error, dt)
        climb_deriv = self._derivative(climb_error, self._prev_climb_error, dt)

        self._prev_yaw_error = yaw_error
        self._prev_range_error = range_error
        self._prev_climb_error = climb_error

        yaw_rate = (
            self.kp * yaw_error + self.ki * self._yaw_integral + self.kd * yaw_deriv
        )

        # The range error is positive when the target is too close, so it
        # is negated to make forward motion approach the wanted distance.
        forward = -(
            self.kp * range_error
            + self.ki * self._range_integral
            + self.kd * range_deriv
        )

        # The vertical error is positive when the target is below the
        # center, so it is negated to climb toward a higher target.
        climb = -(
            self.kp * climb_error
            + self.ki * self._climb_integral
            + self.kd * climb_deriv
        )

        command = VelocityCommand(
            forward=forward,
            yaw_rate=yaw_rate,
            climb=climb,
        )
        return command.clamped(self.max_forward, self.max_yaw_rate, self.max_climb)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _derivative(error: float, previous: float, dt: float) -> float:
        if previous is None:
            return 0.0
        return (error - previous) / dt
