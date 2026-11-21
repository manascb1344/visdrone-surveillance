"""Kinematic ground-plane world used by the follow-loop simulation.

The world is a flat plane measured in meters. Vehicles move along simple
parametric paths (constant velocity straight lines or gentle constant-curvature
arcs). There is no physics engine: vehicle state is advanced with closed-form
kinematics so that a given seed and command sequence always reproduces the
same trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Palette used for the rendered frames. Values are BGR tuples so they can be
# handed to OpenCV without conversion.
VEHICLE_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (60, 180, 240),
    (80, 80, 235),
    (90, 200, 120),
    (200, 120, 60),
    (180, 120, 220),
    (60, 210, 210),
)


@dataclass
class Vehicle:
    """A single ground vehicle in the world.

    ``heading`` is the travel direction in radians measured from the +x axis.
    ``turn_rate`` in radians per second describes the path behavior: a value of
    zero gives a constant-velocity straight line, a small nonzero value gives a
    gentle turn.
    """

    id: int
    x: float
    y: float
    heading: float = 0.0
    speed: float = 8.0
    length: float = 4.5
    width: float = 1.9
    color: Tuple[int, int, int] = VEHICLE_COLORS[0]
    turn_rate: float = 0.0

    @property
    def position(self) -> Tuple[float, float]:
        """Ground position as an ``(x, y)`` tuple in meters."""
        return (self.x, self.y)

    @property
    def size(self) -> Tuple[float, float]:
        """Vehicle footprint as ``(length, width)`` in meters."""
        return (self.length, self.width)

    def corners(self) -> List[Tuple[float, float]]:
        """Return the four ground-plane corners of the vehicle footprint."""
        fx = math.cos(self.heading)
        fy = math.sin(self.heading)
        rx = -fy
        ry = fx
        hl = self.length / 2.0
        hw = self.width / 2.0
        return [
            (self.x + fx * hl + rx * hw, self.y + fy * hl + ry * hw),
            (self.x + fx * hl - rx * hw, self.y + fy * hl - ry * hw),
            (self.x - fx * hl - rx * hw, self.y - fy * hl - ry * hw),
            (self.x - fx * hl + rx * hw, self.y - fy * hl + ry * hw),
        ]

    def copy(self) -> "Vehicle":
        return replace(self)


class World:
    """A deterministic 2D ground plane populated with vehicles.

    The world owns a seeded generator, so creating two worlds with the same
    seed and stepping them with the same commands yields identical state.
    """

    def __init__(
        self,
        seed: int = 0,
        num_vehicles: int = 6,
        bounds: Tuple[float, float, float, float] = (-40.0, 40.0, -40.0, 40.0),
        speed_range: Tuple[float, float] = (4.0, 12.0),
    ) -> None:
        """Create the world and spawn the initial vehicles.

        Args:
            seed: Seed for the internal numpy generator.
            num_vehicles: How many vehicles to spawn.
            bounds: ``(x_min, x_max, y_min, y_max)`` of the drivable area.
            speed_range: Inclusive speed bounds in meters per second.
        """
        self.seed = int(seed)
        self.bounds = bounds
        self.speed_range = speed_range
        self._rng = np.random.default_rng(self.seed)
        self.vehicles: List[Vehicle] = []
        self.time = 0.0
        self._spawn(num_vehicles)

    def _spawn(self, num_vehicles: int) -> None:
        x_min, x_max, y_min, y_max = self.bounds
        for i in range(num_vehicles):
            x = float(self._rng.uniform(x_min + 5.0, x_max - 5.0))
            y = float(self._rng.uniform(y_min + 5.0, y_max - 5.0))
            heading = float(self._rng.uniform(0.0, 2.0 * math.pi))
            speed = float(self._rng.uniform(*self.speed_range))
            color = VEHICLE_COLORS[i % len(VEHICLE_COLORS)]
            # One third of the traffic turns gently, the rest drives straight.
            turn_rate = 0.0
            if i % 3 == 2:
                turn_rate = float(self._rng.uniform(-0.12, 0.12))
            self.vehicles.append(
                Vehicle(
                    id=i,
                    x=x,
                    y=y,
                    heading=heading,
                    speed=speed,
                    length=float(self._rng.uniform(4.0, 5.5)),
                    width=float(self._rng.uniform(1.7, 2.1)),
                    color=color,
                    turn_rate=turn_rate,
                )
            )

    def reset(
        self, seed: Optional[int] = None, num_vehicles: Optional[int] = None
    ) -> None:
        """Respawn the world from scratch.

        Args:
            seed: New seed. Defaults to the seed given at construction.
            num_vehicles: New vehicle count. Defaults to the current count.
        """
        if seed is not None:
            self.seed = int(seed)
        count = len(self.vehicles) if num_vehicles is None else num_vehicles
        self._rng = np.random.default_rng(self.seed)
        self.vehicles = []
        self.time = 0.0
        self._spawn(count)

    def step(self, dt: float) -> None:
        """Advance every vehicle by ``dt`` seconds.

        Vehicles are reflected off the world bounds so that they never leave
        the drivable area.
        """
        x_min, x_max, y_min, y_max = self.bounds
        for v in self.vehicles:
            v.heading += v.turn_rate * dt
            v.x += v.speed * math.cos(v.heading) * dt
            v.y += v.speed * math.sin(v.heading) * dt
            if v.x < x_min:
                v.x = 2.0 * x_min - v.x
                v.heading = math.pi - v.heading
            elif v.x > x_max:
                v.x = 2.0 * x_max - v.x
                v.heading = math.pi - v.heading
            if v.y < y_min:
                v.y = 2.0 * y_min - v.y
                v.heading = -v.heading
            elif v.y > y_max:
                v.y = 2.0 * y_max - v.y
                v.heading = -v.heading
        self.time += dt

    @property
    def states(self) -> List[Vehicle]:
        """Snapshot of the current ground truth vehicle states."""
        return [v.copy() for v in self.vehicles]

    def vehicle_states(self) -> List[Vehicle]:
        """Ground truth vehicle states as independent copies."""
        return self.states

    def in_bounds(self, pos: Sequence[float]) -> bool:
        """Return True when ``pos`` lies inside the world bounds."""
        x_min, x_max, y_min, y_max = self.bounds
        return x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max
