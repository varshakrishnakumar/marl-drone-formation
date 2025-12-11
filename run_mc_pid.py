# -*- coding: utf-8 -*-
"""
Monte Carlo PID runner — per-episode CSV logs + IC metadata with units.
Restores behaviors requested by Ronak:
  • Save one CSV per episode
  • Row 0 contains IC metadata (with units)
  • Rows 1..T contain time-history (same columns)
  • Summary CSV remains separate
"""

import numpy as np
import pandas as pd
import time

from sim.envs.multi_drone_quad_env import MultiDroneQuadEnv
from sim.controllers.baseline_pid import FormationPIDController

EPISODE_HORIZON = 2000


# -------------------------------------------------------------
# Random initial conditions (MATCH collaborator version + IC)
# -------------------------------------------------------------
def random_initial_conditions(num_drones=5):
    chase_modes = ["nearest", "round_robin", "random", "most_error"]

    return {
        "pos_jitter": np.random.uniform(-0.3, 0.3, size=(num_drones, 3)),
        "yaw_jitter": np.random.uniform(-0.4, 0.4, size=num_drones),
        "vel_jitter": np.random.uniform(-0.2, 0.2, size=(num_drones, 3)),
        "obstacle_jitter": np.random.uniform(-0.4, 0.4, size=3),
        "dynamic_jitter": np.random.uniform(-0.4, 0.4, size=3),

        # Collaborator's new fields
        "disable_dynamic": False,
        "chase_mode": np.random.choice(chase_modes),
    }


# -------------------------------------------------------------
# Flatten initial conditions into a single dict with units
# -------------------------------------------------------------
def flatten_ic(ic, num_drones):
    flat = {}

    for i in range(num_drones):
        flat[f"ic_pos_jitter_d{i}_x_m"] = ic["pos_jitter"][i, 0]
        flat[f"ic_pos_jitter_d{i}_y_m"] = ic["pos_jitter"][i, 1]
        flat[f"ic_pos_jitter_d{i}_z_m"] = ic["pos_jitter"][i, 2]

        flat[f"ic_yaw_jitter_d{i}_rad"] = ic["yaw_jitter"][i]

        flat[f"ic_vel_jitter_d{i}_x_mps"] = ic["vel_jitter"][i, 0]
        flat[f"ic_vel_jitter_d{i}_y_mps"] = ic["vel_jitter"][i, 1]
        flat[f"ic_vel_jitter_d{i}_z_mps"] = ic["vel_jitter"][i, 2]

    flat["ic_obstacle_jitter_x_m"] = ic["obstacle_jitter"][0]
    flat["ic_obstacle_jitter_y_m"] = ic["obstacle_jitter"][1]
    flat["ic_obstacle_jitter_z_m"] = ic["obstacle_jitter"][2]

    flat["ic_dynamic_jitter_x_m"] = ic["dynamic_jitter"][0]
    flat["ic_dynamic_jitter_y_m"] = ic["dynamic_jitter"][1]
    flat["ic_dynamic_jitter_z_m"] = ic["dynamic_jitter"][2]

    # new collaborator fields
    flat["ic_disable_dynamic_flag"] = int(ic["disable_dynamic"])
    flat["ic_chase_mode"] = ic["chase_mode"]

    return flat


# -------------------------------------------------------------
# Main multi-episode runner (WITH PER-EP CSVs)
# -------------------------------------------------------------
def run_multiple_episodes(
    num_episodes=100,
    max_steps=EPISODE_HORIZON,
    gui=False,
    summary_csv="pid_summary.csv"
):
    summary_rows = []

    for ep in range(num_episodes):
        print(f"\n=== Starting Episode {ep} ===")

        env = MultiDroneQuadEnv(num_drones=5, gui=gui, max_steps=max_steps)
        controller = FormationPIDController(env)

        ic = random_initial_conditions(env.num_drones)
        ic_flat = flatten_ic(ic, env.num_drones)

        obs, info = env.reset(options=ic)
        controller.reset()

        step = 0
        total_reward = 0.0
        done = False
        truncated = False

        # Per-episode time-history rows
        rows = []

        while not (done or truncated) and step < max_steps:
            action = controller(obs)

            obs_pd = obs.reshape(env.num_drones, env.per_drone_obs_dim)
            leader = obs_pd[0]

            pos = leader[0:3]
            vel = leader[3:6]
            eul = leader[6:9]
            ang = leader[9:12]
            desired_pos = env.get_desired_positions()[0]

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            rows.append({
                "step": step,
                "reward": reward,

                # Metrics
                "mean_form_error_m": env.last_metrics["mean_form_error"],
                "mean_z_error_m": env.last_metrics["mean_z_error"],
                "min_dyn_distance_m": env.last_metrics["min_dyn_distance"],
                "collision_flag": env.last_metrics["collision"],

                # Leader state
                "x_m": pos[0], "y_m": pos[1], "z_m": pos[2],
                "vx_mps": vel[0], "vy_mps": vel[1], "vz_mps": vel[2],
                "roll_rad": eul[0], "pitch_rad": eul[1], "yaw_rad": eul[2],
                "wx_rps": ang[0], "wy_rps": ang[1], "wz_rps": ang[2],

                # Leader desired
                "x_des_m": desired_pos[0],
                "y_des_m": desired_pos[1],
                "z_des_m": desired_pos[2],
            })

            step += 1

        # ---- Construct per-episode CSV ----
        df_hist = pd.DataFrame(rows)

        # One IC row with same columns
        df_ic_row = {col: np.nan for col in df_hist.columns}
        df_ic_row.update(ic_flat)
        df_ic = pd.DataFrame([df_ic_row])

        df_out = pd.concat([df_ic, df_hist], ignore_index=True)

        filename = f"pid_log_ep{ep}.csv"
        df_out.to_csv(filename, index=False)
        print(f"Saved episode log → {filename}")

        # ---- Summary ----
        summary_rows.append({
            "episode": ep,
            "total_reward": total_reward,
            **env.last_metrics,
            "ic_chase_mode": ic["chase_mode"],
        })

        env.close()

    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"\nSummary saved → {summary_csv}")


if __name__ == "__main__":
    run_multiple_episodes(num_episodes=100, gui=False, max_steps=EPISODE_HORIZON)
