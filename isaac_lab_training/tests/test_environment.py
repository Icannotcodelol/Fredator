import numpy as np

from isaac_lab_training.environments.drone_tracking_env import DroneTrackingEnv


def test_environment_reset_shape():
    env = DroneTrackingEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (14,)
    assert "behavior" in info


def test_environment_step_returns_expected_shapes():
    env = DroneTrackingEnv()
    env.reset(seed=0)
    action = np.zeros(2, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (14,)
    assert isinstance(reward, float)
    assert terminated in (True, False)
    assert truncated in (True, False)
    assert "reward_terms" in info


def test_episode_truncates_after_max_steps():
    env = DroneTrackingEnv()
    env.reset(seed=0)
    action = np.zeros(2, dtype=np.float32)
    truncated = False
    for _ in range(env.max_steps):
        _, _, _, truncated, _ = env.step(action)
        if truncated:
            break
    assert truncated
