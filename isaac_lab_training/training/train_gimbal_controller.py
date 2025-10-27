"""PPO training script for the LaserFind gimbal controller."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ..environments.drone_tracking_env import DroneTrackingEnv


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def make_env(seed: int | None = None) -> gym.Env:
    env = DroneTrackingEnv()
    if seed is not None:
        env.reset(seed=seed)
    return env


def build_model(config: Dict[str, Any], vec_env: VecNormalize) -> PPO:
    policy_kwargs = dict(
        net_arch=dict(pi=config.get("hidden_layers", [256, 256, 128]), vf=config.get("hidden_layers", [256, 256, 128])),
        activation_fn=torch.nn.ReLU if config.get("activation", "relu").lower() == "relu" else torch.nn.Tanh,
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=config.get("learning_rate", 3e-4),
        n_steps=config.get("n_steps", 2048),
        batch_size=config.get("batch_size", 2048),
        n_epochs=config.get("n_epochs", 10),
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
        clip_range=config.get("clip_range", 0.2),
        ent_coef=config.get("ent_coef", 0.0),
        vf_coef=config.get("vf_coef", 0.5),
        max_grad_norm=config.get("max_grad_norm", 0.5),
        policy_kwargs=policy_kwargs,
        verbose=1,
    )
    return model


def train(config_path: Path, output_dir: Path, total_timesteps: int | None = None, seed: int | None = None) -> Path:
    config = load_config(config_path)
    if total_timesteps is None:
        total_timesteps = config.get("total_timesteps", 10_000_000)

    os.makedirs(output_dir, exist_ok=True)
    log_dir = output_dir / "tensorboard"
    os.makedirs(log_dir, exist_ok=True)

    def _env_fn():
        return make_env(seed)

    vec_env = DummyVecEnv([_env_fn for _ in range(config.get("num_envs", 1))])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=config.get("clip_obs", 10.0), clip_reward=config.get("clip_reward", 10.0))

    model = build_model(config, vec_env)

    checkpoint_callback = CheckpointCallback(save_freq=100_000 // max(config.get("num_envs", 1), 1), save_path=str(output_dir / "checkpoints"), name_prefix="ppo_gimbal")
    new_logger = configure(str(log_dir), ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    model_path = output_dir / "ppo_gimbal_final"
    model.save(str(model_path))
    vec_env.save(str(output_dir / "vecnormalize.pkl"))

    return model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LaserFind gimbal controller using PPO")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"), help="Path to the training configuration YAML file")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ppo_gimbal"), help="Directory to store checkpoints and logs")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override total timesteps from config")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config, args.output_dir, args.total_timesteps, args.seed)
