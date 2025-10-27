"""Validate exported ONNX policy by comparing outputs with the PyTorch policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from stable_baselines3 import PPO


def load_stats(stats_path: Path | None) -> dict | None:
    if stats_path is None or not stats_path.exists():
        return None
    with stats_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate(model_path: Path, onnx_path: Path, stats_path: Path | None = None, num_samples: int = 32, atol: float = 1e-4) -> bool:
    model = PPO.load(str(model_path))
    model.policy.eval()

    session = ort.InferenceSession(str(onnx_path))

    obs_dim = model.observation_space.shape[0]
    rng = np.random.default_rng(0)

    stats = load_stats(stats_path)
    obs_mean = np.zeros(obs_dim)
    obs_std = np.ones(obs_dim)
    if stats is not None:
        obs_mean = np.asarray(stats.get("obs_mean", obs_mean))
        obs_std = np.asarray(stats.get("obs_std", obs_std))

    for _ in range(num_samples):
        obs = rng.normal(size=(1, obs_dim)).astype(np.float32)
        normalized = (obs - obs_mean) / np.maximum(obs_std, 1e-6)
        torch_obs = torch.as_tensor(normalized)
        with torch.no_grad():
            torch_out = model.policy(torch_obs)
        onnx_out = session.run(None, {"observation": normalized})[0]
        if not np.allclose(torch_out.numpy(), onnx_out, atol=atol):
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ONNX export")
    parser.add_argument("model_path", type=Path, help="Path to the trained PPO model")
    parser.add_argument("onnx_path", type=Path, help="Path to the exported ONNX model")
    parser.add_argument("--stats", type=Path, default=None, help="Normalization statistics JSON file")
    parser.add_argument("--num-samples", type=int, default=32, help="Number of random samples to compare")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for comparison")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = validate(args.model_path, args.onnx_path, args.stats, args.num_samples, args.atol)
    if not success:
        raise SystemExit("ONNX validation failed: outputs diverge")
