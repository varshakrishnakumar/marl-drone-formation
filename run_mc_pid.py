# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 10:26:23 2025

@author: ronak

Monte Carlo script to run environment many times with added randomness and record results
Randomness: moving objects' position, wind? 
"""
import numpy as np
import pandas as pd

from sim.envs.multi_drone_quad_env import MultiDroneQuadEnv
from sim.controllers.baseline_pid import FormationPIDController


# randomize initial conditions
def random_initial_conditions(num_drones=5):
    return {
        "pos_jitter": np.random.uniform(-0.3, 0.3, size=(num_drones, 3)),
        "yaw_jitter": np.random.uniform(-0.4, 0.4, size=num_drones),
        "vel_jitter": np.random.uniform(-0.2, 0.2, size=(num_drones, 3)),
        "obstacle_jitter": np.random.uniform(-0.4, 0.4, size=3),
        "dynamic_jitter": np.random.uniform(-0.4, 0.4, size=3),
    }


def run_multiple_episodes(
    num_episodes=100,
    max_steps=1000,
    gui=False,
    summary_csv="pid_summary.csv"
):
    summary_rows = []

    for ep in range(num_episodes):
        print(f"Starting Episode {ep}")

        env = MultiDroneQuadEnv(num_drones=5, gui=gui)
        controller = FormationPIDController(env)

        #obs, info = env.reset()
        reset_options = random_initial_conditions(env.num_drones)
        obs, info = env.reset(options=reset_options)

        controller.reset()

        done = False
        truncated = False
        step = 0
        total_reward = 0.0

        while not (done or truncated) and step < max_steps:
            action = controller(obs)
            obs, reward, done, truncated, info = env.step(action)

            total_reward += reward
            step += 1

        # Record single summary row
        summary_rows.append({
            "episode": ep,
            "total_reward": total_reward,
            **env.last_metrics
        })

        env.close()
        print(f"Episode {ep} completed - total reward: {total_reward:.3f}")

    # Save only the summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary saved to {summary_csv}")


if __name__ == "__main__":
    run_multiple_episodes(num_episodes=20, gui=False)
