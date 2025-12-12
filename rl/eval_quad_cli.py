"""
rl/eval_quad_cli.py

One-command evaluation + plots for MultiDroneQuadEnv.

Usage examples:
---------------
# Evaluate PPO model on obstacle field only (100 episodes)
python -m rl.eval_quad_cli \
    --model_path data/checkpoints/ppo_quad.zip \
    --scenario obstacle_field \
    --algo ppo \
    --episodes 100 \
    --out_dir data/eval/ppo_quad

# Evaluate on ALL standard scenarios and make plots
python -m rl.eval_quad_cli \
    --model_path data/checkpoints/ppo_quad.zip \
    --scenario all \
    --algo ppo \
    --episodes 100 \
    --out_dir data/eval/ppo_quad \
    --make_plots
"""

import argparse
import os
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from sim.envs.multi_drone_quad_env import MultiDroneQuadEnv
from sim.envs.eval_scenarios import EVAL_SCENARIOS

from stable_baselines3 import PPO, SAC

ALGOS = {
    "ppo": PPO,
    "sac": SAC,
}


# ----------------- utilities ----------------- #

def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(
    model_path: str,
    algo: str,
    scenario: str,
    n_episodes: int,
    base_seed: int,
    num_drones: int,
) -> List[Dict]:
    """Run n_episodes of a trained model in a given scenario; return per-episode metrics."""
    if scenario not in EVAL_SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Available: {list(EVAL_SCENARIOS.keys())}"
        )

    algo = algo.lower()
    if algo not in ALGOS:
        raise ValueError(f"Unknown algo '{algo}'. Expected one of {list(ALGOS.keys())}.")

    model_cls = ALGOS[algo]
    model = model_cls.load(model_path, device="cpu")

    results: List[Dict] = []

    for ep in range(n_episodes):
        seed = base_seed + ep
        set_global_seeds(seed)

        env = MultiDroneQuadEnv(num_drones=num_drones, gui=False)
        options = EVAL_SCENARIOS[scenario]

        obs, _ = env.reset(seed=seed, options=options)

        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            total_reward += float(reward)
            steps += 1

        metrics = info.get("metrics", {}) or {}
        results.append(
            {
                "episode": ep,
                "seed": seed,
                "return": total_reward,
                "ep_len": steps,
                "mean_form_error": float(metrics.get("mean_form_error", np.nan)),
                "max_form_error": float(metrics.get("max_form_error", np.nan)),
                "min_dyn_distance": float(metrics.get("min_dyn_distance", np.nan)),
                "collision": float(metrics.get("collision", 0.0)),
            }
        )

        env.close()

    return results


def save_results(results: List[Dict], out_dir: Path, scenario: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    json_path = out_dir / f"{scenario}_results.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)

    # Save as CSV
    csv_path = out_dir / f"{scenario}_results.csv"
    if results:
        fieldnames = list(results[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    # Also save as NumPy for quick loading in notebooks
    npy_path = out_dir / f"{scenario}_results.npy"
    np.save(npy_path, np.array(results, dtype=object), allow_pickle=True)

    print(f"[INFO] Saved {len(results)} episodes for '{scenario}' to {out_dir}")


def make_plots(results: List[Dict], out_dir: Path, scenario: str) -> None:
    """Basic plots: return distribution, formation error, and collision rate."""
    if not results:
        print(f"[WARN] No results for scenario '{scenario}', skipping plots.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    returns = np.array([r["return"] for r in results], dtype=np.float32)
    mean_form = np.array([r["mean_form_error"] for r in results], dtype=np.float32)
    max_form = np.array([r["max_form_error"] for r in results], dtype=np.float32)
    collisions = np.array([r["collision"] for r in results], dtype=np.float32)

    # 1) Returns histogram
    plt.figure()
    plt.hist(returns, bins=20)
    plt.xlabel("Episode return")
    plt.ylabel("Count")
    plt.title(f"Return distribution – {scenario}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{scenario}_returns_hist.png", dpi=200)
    plt.close()

    # 2) Formation error (mean/max) boxplot
    plt.figure()
    plt.boxplot(
        [mean_form, max_form],
        labels=["mean_form_error", "max_form_error"],
        showmeans=True,
    )
    plt.ylabel("Error [m]")
    plt.title(f"Formation error – {scenario}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{scenario}_formation_error_box.png", dpi=200)
    plt.close()

    # 3) Collision rate bar
    collision_rate = float(np.mean(collisions > 0.5))  # > 0.5 → treat as "true"
    plt.figure()
    plt.bar(["collision_rate"], [collision_rate])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Fraction of episodes")
    plt.title(f"Collision rate – {scenario}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{scenario}_collision_rate.png", dpi=200)
    plt.close()

    print(f"[INFO] Saved plots for '{scenario}' to {out_dir}")


# ----------------- CLI entrypoint ----------------- #

def main():
    parser = argparse.ArgumentParser(
        description="One-command evaluation + plots for MultiDroneQuadEnv."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model (.zip from SB3).",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "sac"],
        help="RL algorithm used to train the model.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=list(EVAL_SCENARIOS.keys()) + ["all"],
        help="Which evaluation scenario to run (or 'all').",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes per scenario.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base seed; episode seed = base_seed + ep_idx.",
    )
    parser.add_argument(
        "--num_drones",
        type=int,
        default=5,
        help="Number of drones in the environment.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/eval",
        help="Directory to save metrics and plots.",
    )
    parser.add_argument(
        "--make_plots",
        action="store_true",
        help="If set, generate PNG plots for each scenario.",
    )

    args = parser.parse_args()

    out_root = Path(args.out_dir)

    if args.scenario == "all":
        scenarios = list(EVAL_SCENARIOS.keys())
    else:
        scenarios = [args.scenario]

    print(f"[INFO] Evaluating model '{args.model_path}' with algo={args.algo}")
    print(f"[INFO] Scenarios: {scenarios}")
    print(f"[INFO] Episodes per scenario: {args.episodes}")
    print(f"[INFO] Output dir: {out_root}")

    for scen in scenarios:
        print(f"\n[INFO] --- Scenario: {scen} ---")
        scen_dir = out_root / scen

        results = evaluate_model(
            model_path=args.model_path,
            algo=args.algo,
            scenario=scen,
            n_episodes=args.episodes,
            base_seed=args.base_seed,
            num_drones=args.num_drones,
        )

        save_results(results, scen_dir, scen)

        if args.make_plots:
            make_plots(results, scen_dir, scen)


if __name__ == "__main__":
    main()
