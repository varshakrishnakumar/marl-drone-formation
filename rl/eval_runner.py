"""
rl/eval_runner.py

One-command evaluation for MultiDroneQuadEnv:
- PPO + VecNormalize
- PID baseline

Example usages
--------------

# PPO only
python -m rl.eval_runner \
  --agent ppo \
  --model_path path/to/final_model.zip \
  --vecnorm_path path/to/vecnormalize_final.pkl \
  --scenario obstacle_field \
  --episodes 100 \
  --out_dir data/eval

# PID only
python -m rl.eval_runner \
  --agent pid \
  --scenario obstacle_field \
  --episodes 100 \
  --out_dir data/eval

# Both PPO and PID
python -m rl.eval_runner \
  --agent both \
  --model_path path/to/final_model.zip \
  --vecnorm_path path/to/vecnormalize_final.pkl \
  --scenario all \
  --episodes 100 \
  --out_dir data/eval
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.envs.multi_drone_quad_env import MultiDroneQuadEnv
from sim.envs.eval_scenarios import EVAL_SCENARIOS
from sim.controllers.pid_formation_controller import PIDFormationController


ALGOS = {"ppo": PPO, "sac": SAC}


# ---------- utilities ---------- #

def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_vec_env_for_eval(options: dict, num_drones: int, seed: int,
                          vecnorm_path: str | None):
    """
    Wrap MultiDroneQuadEnv in DummyVecEnv (+VecNormalize if provided).
    """
    def make_env():
        def _thunk():
            env = MultiDroneQuadEnv(num_drones=num_drones, gui=False)
            obs, _ = env.reset(seed=seed, options=options)
            # ignore obs, env is now initialized
            return env
        return _thunk

    venv = DummyVecEnv([make_env()])

    if vecnorm_path:
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False

    return venv


# ---------- PPO / SAC evaluation ---------- #

def eval_ppo(
    model_path: str,
    algo: str,
    vecnorm_path: str | None,
    scenario: str,
    n_episodes: int,
    base_seed: int,
    num_drones: int,
) -> List[Dict]:
    if scenario not in EVAL_SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. "
                         f"Available: {list(EVAL_SCENARIOS.keys())}")

    algo = algo.lower()
    if algo not in ALGOS:
        raise ValueError(f"Unknown algo '{algo}'. Expected one of {list(ALGOS.keys())}.")
    ModelCls = ALGOS[algo]

    model = ModelCls.load(model_path, device="cpu")

    results: List[Dict] = []

    for ep in range(n_episodes):
        seed = base_seed + ep
        set_global_seeds(seed)

        options = EVAL_SCENARIOS[scenario]
        venv = make_vec_env_for_eval(options, num_drones, seed, vecnorm_path)

        obs = venv.reset()
        done = False
        total_reward = 0.0
        steps = 0

        # metrics live inside underlying env
        env: MultiDroneQuadEnv = venv.envs[0]

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = venv.step(action)
            done = bool(dones[0])
            total_reward += float(rewards[0])
            steps += 1

        metrics = infos[0].get("metrics", {}) or {}

        results.append(
            {
                "agent": "ppo",
                "scenario": scenario,
                "episode": ep,
                "seed": seed,
                "return": total_reward,
                "ep_len": steps,
                "mean_form_error": float(metrics.get("mean_form_error", np.nan)),
                "max_form_error": float(metrics.get("max_form_error", np.nan)),
                "min_dyn_distance": float(metrics.get("min_dyn_distance", np.nan)),
                # if you add these to last_metrics, they will appear:
                "min_pairwise_sep": float(metrics.get("min_pairwise_sep", np.nan)),
                "min_static_distance": float(metrics.get("min_static_distance", np.nan)),
                "collision": float(metrics.get("collision", 0.0)),
                "success": float(metrics.get("success", np.nan)),
            }
        )

        venv.close()

    return results


# ---------- PID evaluation ---------- #

def eval_pid(
    scenario: str,
    n_episodes: int,
    base_seed: int,
    num_drones: int,
) -> List[Dict]:
    if scenario not in EVAL_SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. "
                         f"Available: {list(EVAL_SCENARIOS.keys())}")

    controller = PIDFormationController()
    results: List[Dict] = []

    for ep in range(n_episodes):
        seed = base_seed + ep
        set_global_seeds(seed)

        env = MultiDroneQuadEnv(num_drones=num_drones, gui=False)
        options = EVAL_SCENARIOS[scenario]
        obs_flat, _ = env.reset(seed=seed, options=options)

        # obs from env is flattened (global_obs_dim,)
        per_drone = env.per_drone_obs_dim
        obs_all = obs_flat.reshape(env.num_drones, per_drone)

        controller.reset(env.num_drones)

        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            actions = controller.act(env, obs_all)
            obs_flat, reward, terminated, truncated, info = env.step(actions.flatten())
            done = bool(terminated or truncated)

            total_reward += float(reward)
            steps += 1

            obs_all = obs_flat.reshape(env.num_drones, per_drone)

        metrics = info.get("metrics", {}) or {}

        results.append(
            {
                "agent": "pid",
                "scenario": scenario,
                "episode": ep,
                "seed": seed,
                "return": total_reward,
                "ep_len": steps,
                "mean_form_error": float(metrics.get("mean_form_error", np.nan)),
                "max_form_error": float(metrics.get("max_form_error", np.nan)),
                "min_dyn_distance": float(metrics.get("min_dyn_distance", np.nan)),
                "min_pairwise_sep": float(metrics.get("min_pairwise_sep", np.nan)),
                "min_static_distance": float(metrics.get("min_static_distance", np.nan)),
                "collision": float(metrics.get("collision", 0.0)),
                "success": float(metrics.get("success", np.nan)),
            }
        )

        env.close()

    return results


# ---------- I/O helpers ---------- #

def save_results(results: List[Dict], out_dir: Path, label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not results:
        return

    # JSON
    json_path = out_dir / f"{label}.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)

    # CSV
    csv_path = out_dir / f"{label}.csv"
    fieldnames = list(results[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # NPZ / NPY
    npy_path = out_dir / f"{label}.npy"
    np.save(npy_path, np.array(results, dtype=object), allow_pickle=True)

    print(f"[INFO] Saved {len(results)} episodes to {out_dir / (label + '.csv')}")


# ---------- CLI ---------- #

def main():
    parser = argparse.ArgumentParser(description="Evaluation runner: PPO + PID.")
    parser.add_argument(
        "--agent",
        type=str,
        default="ppo",
        choices=["ppo", "pid", "both"],
        help="Which agent(s) to evaluate.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to PPO/SAC .zip model (for agent=ppo/both).",
    )
    parser.add_argument(
        "--vecnorm_path",
        type=str,
        default=None,
        help="Path to VecNormalize .pkl (optional).",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "sac"],
        help="RL algorithm used for the model.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=list(EVAL_SCENARIOS.keys()) + ["all"],
        help="Scenario name or 'all'.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Episodes per scenario per agent.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base seed; episode seed = base_seed + ep.",
    )
    parser.add_argument(
        "--num_drones",
        type=int,
        default=5,
        help="Number of drones.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/eval_runs",
        help="Directory to store results.",
    )

    args = parser.parse_args()

    out_root = Path(args.out_dir)
    scenarios = (
        list(EVAL_SCENARIOS.keys())
        if args.scenario == "all"
        else [args.scenario]
    )

    for scen in scenarios:
        print(f"\n[INFO] Scenario: {scen}")

        # PPO
        if args.agent in ("ppo", "both"):
            if not args.model_path:
                raise ValueError("model_path is required when agent includes 'ppo'.")
            res_ppo = eval_ppo(
                model_path=args.model_path,
                algo=args.algo,
                vecnorm_path=args.vecnorm_path,
                scenario=scen,
                n_episodes=args.episodes,
                base_seed=args.base_seed,
                num_drones=args.num_drones,
            )
            save_results(res_ppo, out_root / "ppo", f"{scen}_ppo")

        # PID
        if args.agent in ("pid", "both"):
            res_pid = eval_pid(
                scenario=scen,
                n_episodes=args.episodes,
                base_seed=args.base_seed,
                num_drones=args.num_drones,
            )
            save_results(res_pid, out_root / "pid", f"{scen}_pid")


if __name__ == "__main__":
    main()
