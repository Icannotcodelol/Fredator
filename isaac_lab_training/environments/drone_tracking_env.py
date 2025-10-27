"""Gymnasium environment approximating the LaserFind drone tracking task."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .curriculum_scheduler import CurriculumScheduler, default_curriculum_scheduler
from .reward_functions import compute_reward, reward_breakdown, reward_to_scalar
from .target_behaviors import TargetBehavior, TargetState, get_behavior_factory, sample_behavior


@dataclass
class GimbalState:
    angles: np.ndarray  # [pan, tilt]
    velocities: np.ndarray  # [pan_dot, tilt_dot]


class DroneTrackingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        curriculum: Optional[CurriculumScheduler] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dt = 0.01
        self.max_steps = 3000
        self.max_distance = 20.0
        self.max_angular_velocity = math.pi  # 180 deg/s
        self.pan_limits = (-math.pi, math.pi)
        self.tilt_limits = (-math.pi / 2.0, math.pi / 2.0)
        self.curriculum = curriculum or default_curriculum_scheduler()
        self.behavior_factory = get_behavior_factory()
        self.behavior: Optional[TargetBehavior] = None
        self.target_state: Optional[TargetState] = None
        self.gimbal_state = GimbalState(angles=np.zeros(2), velocities=np.zeros(2))
        self.episode_step = 0
        self.global_step = 0
        self._rng = random.Random()
        if seed is not None:
            self.reset(seed=seed)

        obs_low = np.full(14, -np.inf, dtype=np.float32)
        obs_high = np.full(14, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def seed(self, seed: Optional[int] = None) -> None:
        self._rng.seed(seed)
        np.random.seed(seed)

    def _select_behavior(self) -> TargetBehavior:
        weights = self.curriculum.behavior_weights(self.global_step)
        name = sample_behavior(weights, self._rng)
        return self.behavior_factory(name)

    def _reset_gimbal(self) -> None:
        pan = math.radians(self._rng.uniform(-10.0, 10.0))
        tilt = math.radians(self._rng.uniform(-10.0, 10.0))
        self.gimbal_state = GimbalState(
            angles=np.array([pan, tilt], dtype=np.float64),
            velocities=np.zeros(2, dtype=np.float64),
        )

    def _gimbal_origin(self) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def _target_vector(self) -> np.ndarray:
        assert self.target_state is not None
        return self.target_state.position - self._gimbal_origin()

    def _desired_angles(self) -> Tuple[float, float]:
        vec = self._target_vector()
        horizontal_distance = math.hypot(vec[0], vec[1])
        pan = math.atan2(vec[1], vec[0])
        tilt = math.atan2(vec[2], horizontal_distance)
        return pan, tilt

    def _angle_error(self, desired: float, actual: float) -> float:
        error = desired - actual
        return (error + math.pi) % (2 * math.pi) - math.pi

    def _distance_to_target(self) -> float:
        return float(np.linalg.norm(self._target_vector()))

    def _tracking_error(self, pan_error: float, tilt_error: float) -> float:
        return math.sqrt(pan_error**2 + tilt_error**2)

    def _update_target(self) -> None:
        assert self.behavior is not None
        assert self.target_state is not None
        self.target_state = self.behavior.update(self.target_state, self.dt, self._rng)

    def _update_gimbal(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)
        commanded_velocity = action * self.max_angular_velocity
        new_velocities = commanded_velocity
        new_angles = self.gimbal_state.angles + new_velocities * self.dt
        new_angles[0] = float(np.clip(new_angles[0], *self.pan_limits))
        new_angles[1] = float(np.clip(new_angles[1], *self.tilt_limits))
        self.gimbal_state = GimbalState(angles=new_angles, velocities=new_velocities)

    def _get_observation(self, pan_error: float, tilt_error: float) -> np.ndarray:
        assert self.target_state is not None
        obs = np.zeros(14, dtype=np.float32)
        obs[0:3] = self.target_state.position.astype(np.float32)
        obs[3:6] = self.target_state.velocity.astype(np.float32)
        obs[6:8] = self.gimbal_state.angles.astype(np.float32)
        obs[8:10] = self.gimbal_state.velocities.astype(np.float32)
        obs[10] = float(pan_error)
        obs[11] = float(tilt_error)
        obs[12] = self._distance_to_target()
        obs[13] = self._tracking_error(pan_error, tilt_error)
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.seed(seed)
        self.episode_step = 0
        self.behavior = self._select_behavior()
        self.target_state = self.behavior.reset(self._rng)
        self._reset_gimbal()
        pan_desired, tilt_desired = self._desired_angles()
        pan_error = self._angle_error(pan_desired, self.gimbal_state.angles[0])
        tilt_error = self._angle_error(tilt_desired, self.gimbal_state.angles[1])
        observation = self._get_observation(pan_error, tilt_error)
        info = {
            "behavior": getattr(self.behavior, "name", "unknown"),
            "curriculum_complexity": self.curriculum.complexity(self.global_step),
        }
        return observation, info

    def step(self, action):
        if self.behavior is None or self.target_state is None:
            raise RuntimeError("Environment must be reset before stepping.")

        self._update_gimbal(action)
        self._update_target()

        pan_desired, tilt_desired = self._desired_angles()
        pan_error = self._angle_error(pan_desired, self.gimbal_state.angles[0])
        tilt_error = self._angle_error(tilt_desired, self.gimbal_state.angles[1])
        obs = self._get_observation(pan_error, tilt_error)

        terms = compute_reward(pan_error, tilt_error, np.asarray(action), self.gimbal_state.velocities)
        reward = reward_to_scalar(terms)

        self.episode_step += 1
        self.global_step += 1

        distance = self._distance_to_target()
        terminated = False
        truncated = False
        if distance > self.max_distance:
            truncated = True
        if self.episode_step >= self.max_steps:
            truncated = True

        info = {
            "reward_terms": reward_breakdown(terms),
            "behavior": getattr(self.behavior, "name", "unknown"),
            "distance": distance,
            "pan_error": pan_error,
            "tilt_error": tilt_error,
        }

        if truncated or terminated:
            self.behavior = None
            self.target_state = None

        return obs, reward, terminated, truncated, info


__all__ = ["DroneTrackingEnv"]
