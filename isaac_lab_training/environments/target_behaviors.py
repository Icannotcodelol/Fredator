"""Target motion behavior implementations for the LaserFind project."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple

import numpy as np


@dataclass
class TargetState:
    """Simple container describing the target pose and velocity."""

    position: np.ndarray
    velocity: np.ndarray


class TargetBehavior:
    """Base class for all target behaviors."""

    name: str = "base"

    def reset(self, rng: random.Random) -> TargetState:
        raise NotImplementedError

    def update(self, state: TargetState, dt: float, rng: random.Random) -> TargetState:
        raise NotImplementedError


class StationaryBehavior(TargetBehavior):
    name = "stationary"

    def reset(self, rng: random.Random) -> TargetState:
        position = random_point_on_sphere(rng, radius=rng.uniform(3.0, 10.0))
        velocity = rng.uniform(-0.1, 0.1) * random_unit_vector(rng)
        return TargetState(position=position, velocity=velocity)

    def update(self, state: TargetState, dt: float, rng: random.Random) -> TargetState:
        jitter = rng.uniform(-0.1, 0.1) * random_unit_vector(rng)
        velocity = state.velocity * 0.95 + jitter * 0.05
        position = state.position + velocity * dt
        return TargetState(position=position, velocity=velocity)


class ConstantVelocityBehavior(TargetBehavior):
    name = "constant_velocity"

    def reset(self, rng: random.Random) -> TargetState:
        position = random_point_on_sphere(rng, radius=rng.uniform(3.0, 10.0))
        direction = random_unit_vector(rng)
        speed = rng.uniform(1.0, 3.0)
        velocity = direction * speed
        return TargetState(position=position, velocity=velocity)

    def update(self, state: TargetState, dt: float, rng: random.Random) -> TargetState:
        position = state.position + state.velocity * dt
        return TargetState(position=position, velocity=state.velocity)


class CircularOrbitBehavior(TargetBehavior):
    name = "circular_orbit"

    def reset(self, rng: random.Random) -> TargetState:
        radius = rng.uniform(3.0, 7.0)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        height = rng.uniform(0.5, 3.0)
        position = np.array([radius * math.cos(angle), radius * math.sin(angle), height])
        angular_velocity = rng.uniform(0.3, 0.7)
        velocity = np.array([
            -radius * angular_velocity * math.sin(angle),
            radius * angular_velocity * math.cos(angle),
            0.0,
        ])
        return TargetState(position=position, velocity=velocity)

    def update(self, state: TargetState, dt: float, rng: random.Random) -> TargetState:
        radius = np.linalg.norm(state.position[:2])
        angle = math.atan2(state.position[1], state.position[0])
        angular_velocity = np.linalg.norm(state.velocity[:2]) / max(radius, 1e-6)
        angle += angular_velocity * dt
        position = np.array([radius * math.cos(angle), radius * math.sin(angle), state.position[2]])
        velocity = np.array([
            -radius * angular_velocity * math.sin(angle),
            radius * angular_velocity * math.cos(angle),
            0.0,
        ])
        return TargetState(position=position, velocity=velocity)


class FigureEightBehavior(TargetBehavior):
    name = "figure_eight"

    def reset(self, rng: random.Random) -> TargetState:
        self._omega = rng.uniform(0.2, 0.5)
        self._amplitude = np.array([
            rng.uniform(2.0, 4.0),
            rng.uniform(1.5, 3.0),
            rng.uniform(0.5, 1.5),
        ])
        self._phase = rng.uniform(0.0, 2.0 * math.pi)
        return self.update(TargetState(np.zeros(3), np.zeros(3)), 0.0, rng)

    def update(self, state: TargetState, dt: float, rng: random.Random) -> TargetState:
        t = getattr(self, "_time", 0.0) + dt
        self._time = t
        omega = self._omega
        A, B, C = self._amplitude
        phase = self._phase

        position = np.array([
            A * math.sin(omega * t + phase),
            B * math.sin(2 * omega * t + phase),
            2.0 + C * math.sin(omega * t + phase),
        ])
        velocity = np.array([
            A * omega * math.cos(omega * t + phase),
            2 * B * omega * math.cos(2 * omega * t + phase),
            C * omega * math.cos(omega * t + phase),
        ])
        return TargetState(position=position, velocity=velocity)


class RandomJinkingBehavior(TargetBehavior):
    name = "random_jinking"

    def reset(self, rng: random.Random) -> TargetState:
        self._time = 0.0
        position = random_point_on_sphere(rng, radius=rng.uniform(3.0, 10.0))
        velocity = rng.uniform(0.5, 2.0) * random_unit_vector(rng)
        self._accel = np.zeros(3)
        return TargetState(position=position, velocity=velocity)

    def update(self, state: TargetState, dt: float, rng: random.Random) -> TargetState:
        self._time += dt
        if int(self._time / 0.2) != int((self._time - dt) / 0.2):
            self._accel = rng.uniform(-5.0, 5.0) * random_unit_vector(rng)
        velocity = state.velocity + self._accel * dt
        speed = np.linalg.norm(velocity)
        if speed > 3.0:
            velocity *= 3.0 / speed
        position = state.position + velocity * dt
        return TargetState(position=position, velocity=velocity)


class ApproachBehavior(TargetBehavior):
    name = "approach"

    def reset(self, rng: random.Random) -> TargetState:
        position = random_point_on_sphere(rng, radius=rng.uniform(5.0, 10.0))
        direction_to_origin = -normalize(position)
        speed = rng.uniform(2.0, 4.0)
        velocity = direction_to_origin * speed
        return TargetState(position=position, velocity=velocity)

    def update(self, state: TargetState, dt: float, rng: random.Random) -> TargetState:
        position = state.position + state.velocity * dt
        return TargetState(position=position, velocity=state.velocity)


class SpiralApproachBehavior(TargetBehavior):
    name = "spiral_approach"

    def reset(self, rng: random.Random) -> TargetState:
        self._angular_velocity = rng.uniform(0.4, 0.8)
        radius = rng.uniform(5.0, 10.0)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        height = rng.uniform(1.0, 4.0)
        position = np.array([radius * math.cos(angle), radius * math.sin(angle), height])
        velocity = np.zeros(3)
        self._time = 0.0
        self._initial_radius = radius
        return TargetState(position=position, velocity=velocity)

    def update(self, state: TargetState, dt: float, rng: random.Random) -> TargetState:
        self._time += dt
        radius = max(self._initial_radius - 0.5 * self._time, 1.0)
        angle = math.atan2(state.position[1], state.position[0]) + self._angular_velocity * dt
        height = max(state.position[2] - 0.2 * dt, 0.5)
        position = np.array([radius * math.cos(angle), radius * math.sin(angle), height])
        tangential_speed = radius * self._angular_velocity
        radial_speed = -0.5
        vertical_speed = -0.2
        velocity = np.array([
            tangential_speed * -math.sin(angle) + radial_speed * math.cos(angle),
            tangential_speed * math.cos(angle) + radial_speed * math.sin(angle),
            vertical_speed,
        ])
        return TargetState(position=position, velocity=velocity)


def build_behavior_library() -> Dict[str, TargetBehavior]:
    """Return a mapping of behavior names to instances."""

    return {
        behavior.name: behavior
        for behavior in [
            StationaryBehavior(),
            ConstantVelocityBehavior(),
            CircularOrbitBehavior(),
            FigureEightBehavior(),
            RandomJinkingBehavior(),
            ApproachBehavior(),
            SpiralApproachBehavior(),
        ]
    }


def random_point_on_sphere(rng: random.Random, radius: float) -> np.ndarray:
    direction = random_unit_vector(rng)
    return direction * radius


def random_unit_vector(rng: random.Random) -> np.ndarray:
    phi = rng.uniform(0.0, 2.0 * math.pi)
    costheta = rng.uniform(-1.0, 1.0)
    theta = math.acos(costheta)
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)
    return np.array([x, y, z])


def normalize(vector: Iterable[float]) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return vec
    return vec / norm


def sample_behavior(behavior_weights: Dict[str, float], rng: random.Random) -> str:
    """Sample a behavior name using the provided weights."""

    names = list(behavior_weights.keys())
    weights = np.array(list(behavior_weights.values()), dtype=np.float64)
    weights /= weights.sum()
    return rng.choices(names, weights=weights, k=1)[0]


def get_behavior_factory() -> Callable[[str], TargetBehavior]:
    """Return a function that instantiates behaviors by name."""

    library = build_behavior_library()

    def factory(name: str) -> TargetBehavior:
        if name not in library:
            raise KeyError(f"Unknown behavior: {name}")
        # Return a fresh instance to avoid state leakage between environments
        return type(library[name])()

    return factory


__all__ = [
    "TargetState",
    "TargetBehavior",
    "StationaryBehavior",
    "ConstantVelocityBehavior",
    "CircularOrbitBehavior",
    "FigureEightBehavior",
    "RandomJinkingBehavior",
    "ApproachBehavior",
    "SpiralApproachBehavior",
    "build_behavior_library",
    "sample_behavior",
    "get_behavior_factory",
]
