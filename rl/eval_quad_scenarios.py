# rl/eval_quad_scenarios.py

import random
import numpy as np
import torch

from sim.envs.multi_drone_quad_env import MultiDroneQuadEnv
from sim.envs.eval_scenarios import EVAL_SCENARIOS
from stable_baselines3 import PPO  # or SAC, etc.

def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_model(
    model_path: str,
    scenario: str,
    n_episodes: int = 100,
    base_seed: int = 42,
    num_drones: int = 5,
):
    if scenario not in EVAL_SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. "
                         f"Choose from {list(EVAL_SCENARIOS.keys())}.")

    # load policy once
    model = PPO.load(model_path, device="cpu")

    results = []
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
            done = terminated or truncated

            total_reward += float(reward)
            steps += 1

        metrics = info.get("metrics", {})
        results.append({
            "episode": ep,
            "seed": seed,
            "return": total_reward,
            "ep_len": steps,
            "mean_form_error": float(metrics.get("mean_form_error", np.nan)),
            "max_form_error": float(metrics.get("max_form_error", np.nan)),
            "min_dyn_distance": float(metrics.get("min_dyn_distance", np.nan)),
            "collision": float(metrics.get("collision", 0.0)),
        })

        env.close()

    return results
