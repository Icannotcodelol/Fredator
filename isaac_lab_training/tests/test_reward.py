import numpy as np

from isaac_lab_training.environments.reward_functions import RewardTerms, compute_reward


def test_reward_components_follow_spec():
    action = np.array([0.5, -0.5])
    gimbal_velocity = np.array([0.2, -0.1])
    terms = compute_reward(0.05, 0.02, action, gimbal_velocity)
    assert isinstance(terms, RewardTerms)
    assert terms.stability_bonus == 1.0
    assert terms.tracking_error > 0
    assert terms.total() == -10.0 * terms.tracking_error - 0.1 * terms.action_magnitude + 1.0 * terms.stability_bonus - 0.05 * terms.velocity_penalty


def test_reward_penalizes_large_errors():
    action = np.zeros(2)
    gimbal_velocity = np.zeros(2)
    near_terms = compute_reward(0.05, 0.05, action, gimbal_velocity)
    far_terms = compute_reward(0.5, 0.5, action, gimbal_velocity)
    assert near_terms.total() > far_terms.total()
