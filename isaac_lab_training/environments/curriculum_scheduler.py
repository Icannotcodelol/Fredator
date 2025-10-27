"""Curriculum scheduler for sampling target behaviors during training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class CurriculumStage:
    start_step: int
    end_step: int
    behavior_distribution: Dict[str, float]
    complexity: float


class CurriculumScheduler:
    """Sample behaviors based on the current training step."""

    def __init__(self, stages: List[CurriculumStage]):
        if not stages:
            raise ValueError("At least one curriculum stage is required")
        self._stages = sorted(stages, key=lambda stage: stage.start_step)

    def stage_for_step(self, global_step: int) -> CurriculumStage:
        for stage in self._stages:
            if stage.start_step <= global_step < stage.end_step:
                return stage
        return self._stages[-1]

    def behavior_weights(self, global_step: int) -> Dict[str, float]:
        stage = self.stage_for_step(global_step)
        return stage.behavior_distribution

    def complexity(self, global_step: int) -> float:
        return self.stage_for_step(global_step).complexity


DEFAULT_STAGES: List[CurriculumStage] = [
    CurriculumStage(0, 1_000_000, {"stationary": 1.0}, 0.0),
    CurriculumStage(1_000_000, 3_000_000, {"stationary": 0.7, "constant_velocity": 0.3}, 0.2),
    CurriculumStage(
        3_000_000,
        5_000_000,
        {"constant_velocity": 0.5, "circular_orbit": 0.3, "stationary": 0.2},
        0.4,
    ),
    CurriculumStage(
        5_000_000,
        7_000_000,
        {"circular_orbit": 0.4, "figure_eight": 0.3, "constant_velocity": 0.3},
        0.6,
    ),
    CurriculumStage(
        7_000_000,
        9_000_000,
        {
            "figure_eight": 0.3,
            "random_jinking": 0.3,
            "circular_orbit": 0.2,
            "constant_velocity": 0.2,
        },
        0.8,
    ),
    CurriculumStage(
        9_000_000,
        10_000_000,
        {
            "stationary": 1.0,
            "constant_velocity": 1.0,
            "circular_orbit": 1.0,
            "figure_eight": 1.0,
            "random_jinking": 1.0,
            "approach": 1.0,
            "spiral_approach": 1.0,
        },
        1.0,
    ),
]


def default_curriculum_scheduler() -> CurriculumScheduler:
    return CurriculumScheduler(DEFAULT_STAGES)


__all__ = ["CurriculumStage", "CurriculumScheduler", "DEFAULT_STAGES", "default_curriculum_scheduler"]
