"""Reward computation utilities for the LaserFind environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class RewardTerms:
    tracking_error: float
    action_magnitude: float
    stability_bonus: float
    velocity_penalty: float

    def total(self) -> float:
        return (
            -10.0 * self.tracking_error
            - 0.1 * self.action_magnitude
            + 1.0 * self.stability_bonus
            - 0.05 * self.velocity_penalty
        )


def compute_reward(
    angle_error_pan: float,
    angle_error_tilt: float,
    action: np.ndarray,
    gimbal_velocity: np.ndarray,
) -> RewardTerms:
    """Compute shaped reward according to the specification."""

    tracking_error = float(np.sqrt(angle_error_pan**2 + angle_error_tilt**2))
    action_magnitude = float(np.linalg.norm(action))
    stability_bonus = 1.0 if tracking_error < 0.1 else 0.0
    velocity_penalty = float(np.sum(np.square(gimbal_velocity)))
    return RewardTerms(
        tracking_error=tracking_error,
        action_magnitude=action_magnitude,
        stability_bonus=stability_bonus,
        velocity_penalty=velocity_penalty,
    )


def reward_to_scalar(terms: RewardTerms) -> float:
    return terms.total()


def reward_breakdown(terms: RewardTerms) -> Dict[str, float]:
    return {
        "tracking_error": -10.0 * terms.tracking_error,
        "action_penalty": -0.1 * terms.action_magnitude,
        "stability_bonus": 1.0 * terms.stability_bonus,
        "velocity_penalty": -0.05 * terms.velocity_penalty,
        "total": terms.total(),
    }


__all__ = ["RewardTerms", "compute_reward", "reward_to_scalar", "reward_breakdown"]
