"""Scenario wiring that glues the world and the drone camera together.

A scenario is the smallest unit the follow loop can be validated against: it
owns a world and a camera, steps both from a velocity command, and returns
the rendered frame, the observations, and the ground truth vehicle states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple

import numpy as np

from ..types import Detection, VelocityCommand
from .camera import DroneCamera
from .world import World


@dataclass
class StepResult:
    """Everything produced by one simulation step."""

    frame: np.ndarray
    observations: List[Detection]
    world_states: List[object]


@dataclass
class SimScenario:
    """World plus camera scenario.

    Attributes:
        world: The simulated ground plane.
        camera: The simulated drone camera.
        dt: Default time step in seconds.
    """

    world: World = field(default_factory=World)
    camera: DroneCamera = field(default_factory=DroneCamera)
    dt: float = 0.05

    def reset(self, seed: int = 0) -> "SimScenario":
        """Reset the world and the camera to their initial state.

        Args:
            seed: Seed for the world generator.

        Returns:
            This scenario, to allow chaining.
        """
        self.world.reset(seed=seed)
        self.camera.set_position((0.0, 0.0, self.camera.z))
        self.camera.yaw = 0.0
        return self

    def step(
        self, command: VelocityCommand, dt: float = None
    ) -> Tuple[np.ndarray, List[Detection], List[object]]:
        """Advance the simulation by one step.

        Args:
            command: Body-frame velocity command for the drone.
            dt: Time step in seconds. Defaults to the scenario ``dt``.

        Returns:
            ``(frame, observations, world_states)``.
        """
        step_dt = self.dt if dt is None else float(dt)
        self.camera.apply(command, step_dt)
        self.world.step(step_dt)
        frame = self.camera.render(self.world)
        observations = self.camera.observe(self.world)
        return frame, observations, self.world.states

    def run(
        self,
        trajectory: Sequence[VelocityCommand],
        dt: float = None,
        seed: int = 0,
    ) -> List[Tuple[np.ndarray, List[Detection], List[object]]]:
        """Run a fixed command trajectory and collect the history.

        Args:
            trajectory: Ordered velocity commands to apply.
            dt: Time step in seconds. Defaults to the scenario ``dt``.
            seed: Seed used for the initial world reset.

        Returns:
            One ``(frame, observations, world_states)`` tuple per command.
        """
        self.reset(seed=seed)
        history: List[Tuple[np.ndarray, List[Detection], List[object]]] = []
        for command in trajectory:
            history.append(self.step(command, dt))
        return history
