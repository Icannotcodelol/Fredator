import math
import random

import numpy as np
import pytest

from isaac_lab_training.environments.target_behaviors import (
    ApproachBehavior,
    CircularOrbitBehavior,
    ConstantVelocityBehavior,
    FigureEightBehavior,
    RandomJinkingBehavior,
    SpiralApproachBehavior,
    StationaryBehavior,
)


@pytest.mark.parametrize(
    "behavior_cls",
    [
        StationaryBehavior,
        ConstantVelocityBehavior,
        CircularOrbitBehavior,
        FigureEightBehavior,
        RandomJinkingBehavior,
        ApproachBehavior,
        SpiralApproachBehavior,
    ],
)
def test_behaviors_produce_finite_values(behavior_cls):
    rng = random.Random(0)
    behavior = behavior_cls()
    state = behavior.reset(rng)
    assert np.isfinite(state.position).all()
    assert np.isfinite(state.velocity).all()

    for _ in range(10):
        state = behavior.update(state, 0.01, rng)
        assert np.isfinite(state.position).all()
        assert np.isfinite(state.velocity).all()


def test_circular_orbit_preserves_radius():
    rng = random.Random(0)
    behavior = CircularOrbitBehavior()
    state = behavior.reset(rng)
    radius = np.linalg.norm(state.position[:2])
    state = behavior.update(state, 0.1, rng)
    assert pytest.approx(radius, rel=1e-2) == np.linalg.norm(state.position[:2])
