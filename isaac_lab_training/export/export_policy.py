"""Export trained PPO policy to ONNX for deployment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize


def load_model(model_path: Path, vecnormalize_path: Path | None = None) -> Tuple[PPO, VecNormalize | None]:
    model = PPO.load(str(model_path))
    vecnorm = None
    if vecnormalize_path is not None and vecnormalize_path.exists():
        vecnorm = VecNormalize.load(str(vecnormalize_path), None)
        vecnorm.training = False
    return model, vecnorm


def export_policy(model_path: Path, output_path: Path, vecnormalize_path: Path | None = None) -> None:
    model, vecnorm = load_model(model_path, vecnormalize_path)
    policy = model.policy

    obs_dim = policy.observation_space.shape[0]
    dummy_obs = torch.zeros(1, obs_dim)

    torch.onnx.export(
        policy,
        dummy_obs,
        str(output_path),
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch_size"}, "action": {0: "batch_size"}},
        opset_version=14,
    )

    if vecnorm is not None:
        stats = {
            "obs_mean": vecnorm.obs_rms.mean.tolist(),
            "obs_std": vecnorm.obs_rms.var**0.5,
            "clip_obs": vecnorm.clip_obs,
        }
        stats_path = output_path.with_suffix(".stats.json")
        stats["obs_std"] = np.sqrt(stats["obs_std"]).tolist() if isinstance(stats["obs_std"], np.ndarray) else stats["obs_std"]
        with stats_path.open("w", encoding="utf-8") as file:
            json.dump(stats, file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PPO policy to ONNX")
    parser.add_argument("model_path", type=Path, help="Path to the trained PPO model")
    parser.add_argument("output", type=Path, help="Output ONNX file path")
    parser.add_argument("--vecnorm", type=Path, default=None, help="Path to VecNormalize statistics file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_policy(args.model_path, args.output, args.vecnorm)
