"""
Monte Carlo evaluator for PPO policy under the same randomized conditions as run_mc_pid.py.

Usage:
  python rl/run_mc_eval.py \
    --model runs/ppo_multi_stage1/checkpoints/ppo_multi_4000000_steps.zip \
    --vecnorm runs/ppo_multi_stage1/vecnormalize_final.pkl \
    --num-drones 5 --trials 100 --steps 3000 --deterministic

Outputs:
  - logs/mc_<timestamp>.csv          : per-trial summary
  - logs/mc_summary_<timestamp>.json : aggregate stats
"""

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from sim.envs.multi_drone_quad_env import MultiDroneQuadEnv


def random_initial_conditions(num_drones: int = 5) -> Dict[str, np.ndarray]:
    return {
        "pos_jitter": np.random.uniform(-0.3, 0.3, size=(num_drones, 3)),
        "yaw_jitter": np.random.uniform(-0.4, 0.4, size=num_drones),
        "vel_jitter": np.random.uniform(-0.2, 0.2, size=(num_drones, 3)),
        "obstacle_jitter": np.random.uniform(-0.4, 0.4, size=3),
        "dynamic_jitter": np.random.uniform(-0.4, 0.4, size=3),
    }


class ResetOptionsWrapper(gym.Wrapper):
    """
    Ensures every env.reset() passes options from options_fn(), even when called via SB3 VecEnv.
    """
    def __init__(self, env: gym.Env, options_fn: Callable[[], Dict]):
        super().__init__(env)
        self._options_fn = options_fn

    def reset(self, **kwargs):
        opts = self._options_fn()
        kwargs["options"] = opts
        return self.env.reset(**kwargs)


def auto_find_vecnorm(model_path: Path) -> Optional[Path]:
    model_dir = model_path.parent
    run_dir = model_dir.parent
    candidates = [
        model_dir / "vecnormalize_final.pkl",
        run_dir / "vecnormalize_final.pkl",
        run_dir / "checkpoints" / "vecnormalize_final.pkl",
        model_dir / "vec_normalize.pkl",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def make_trial_env(num_drones: int, gui: bool, options_fn: Callable[[], Dict]) -> DummyVecEnv:
    def _thunk():
        base = MultiDroneQuadEnv(num_drones=num_drones, gui=gui)
        return ResetOptionsWrapper(base, options_fn)
    return DummyVecEnv([_thunk])


def run_one_trial(model: PPO, vec, steps: int) -> Tuple[float, float, bool, float]:
    """
    Returns:
        mean_form_error, min_dyn_distance_min, collided_any, total_reward
    """
    base_env = vec.venv.envs[0] if isinstance(vec, VecNormalize) else vec.envs[0]
    obs = vec.reset()
    errs, dists = [], []
    total_reward = 0.0
    collided = False

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, _ = vec.step(action)
        total_reward += float(rewards[0])

        mfe = float(base_env.last_metrics.get("mean_form_error", np.nan))
        mdd = float(base_env.last_metrics.get("min_dyn_distance", np.nan))
        errs.append(mfe); dists.append(mdd)
        collided = collided or bool(base_env.collision_happened)

        if bool(dones[0]):
            obs = vec.reset()

    mean_form_error = float(np.nanmean(errs)) if errs else float("nan")
    min_dyn_distance_min = float(np.nanmin(dists)) if dists else float("nan")
    return mean_form_error, min_dyn_distance_min, collided, total_reward


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RL Monte Carlo evaluation with PID-like randomizations.")
    p.add_argument("--model", type=str, required=True, help="Path to PPO .zip checkpoint")
    p.add_argument("--vecnorm", type=str, default="", help="Optional path to VecNormalize .pkl (auto if omitted)")
    p.add_argument("--num-drones", type=int, default=5)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--gui", action="store_true")
    p.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"[INFO] Loading model: {model_path}")
    model = PPO.load(str(model_path))

    vecnorm_path = Path(args.vecnorm).expanduser().resolve() if args.vecnorm else auto_find_vecnorm(model_path)
    if vecnorm_path and vecnorm_path.exists():
        print(f"[INFO] VecNormalize: {vecnorm_path}")
    else:
        print("[WARN] No VecNormalize stats found; evaluating WITHOUT obs normalization.")
        vecnorm_path = None

    os.makedirs("logs", exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path("logs") / f"mc_{stamp}.csv"
    json_path = Path("logs") / f"mc_summary_{stamp}.json"

    results = []
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "trial", "seed", "mean_form_error", "min_dyn_distance_min", "collision", "success", "total_reward"
        ])

        for t in range(args.trials):
            seed = (int(dt.datetime.now().timestamp() * 1e6) + t) % (2**31 - 1)
            np.random.seed(seed)

            opts_fn = lambda: random_initial_conditions(args.num_drones)

            vec = make_trial_env(args.num_drones, args.gui, opts_fn)

            if vecnorm_path:
                vec = VecNormalize.load(str(vecnorm_path), vec)
                vec.training = False
                vec.norm_reward = False

            model.set_env(vec)
            base_env = vec.venv.envs[0] if isinstance(vec, VecNormalize) else vec.envs[0]
            base_env.reset(seed=seed)

            mfe, mdd_min, collided, total_reward = run_one_trial(model, vec, args.steps)
            success = (not collided) and (mfe < 2.0)

            writer.writerow([t, seed, f"{mfe:.6f}", f"{mdd_min:.6f}", int(collided), int(success), f"{total_reward:.6f}"])
            results.append((mfe, mdd_min, collided, success, total_reward))

            try:
                vec.close()
            except Exception:
                pass

    mfes = np.array([r[0] for r in results], dtype=np.float64)
    mdds = np.array([r[1] for r in results], dtype=np.float64)
    col = np.array([int(r[2]) for r in results], dtype=np.int32)
    succ = np.array([int(r[3]) for r in results], dtype=np.int32)
    trew = np.array([r[4] for r in results], dtype=np.float64)

    summary = {
        "trials": int(args.trials),
        "steps_per_trial": int(args.steps),
        "num_drones": int(args.num_drones),
        "mean_form_error_mean": float(np.nanmean(mfes)),
        "mean_form_error_std": float(np.nanstd(mfes)),
        "min_dyn_distance_min_mean": float(np.nanmean(mdds)),
        "min_dyn_distance_min_std": float(np.nanstd(mdds)),
        "collision_rate": float(col.mean()),
        "success_rate": float(succ.mean()),
        "success_definition": "collision==0 AND mean_form_error<2.0",
        "mean_total_reward": float(np.nanmean(trew)),
        "model": str(model_path),
        "vecnorm": str(vecnorm_path) if vecnorm_path else "",
        "timestamp": stamp,
    }

    with open(json_path, "w") as jf:
        json.dump(summary, jf, indent=2)

    print(f"[OK] MC CSV  → {csv_path}")
    print(f"[OK] MC JSON → {json_path}")
    print(
        f"Summary: success_rate={summary['success_rate']:.3f} | "
        f"collision_rate={summary['collision_rate']:.3f} | "
        f"mfe={summary['mean_form_error_mean']:.3f}±{summary['mean_form_error_std']:.3f}"
    )


if __name__ == "__main__":
    main()
