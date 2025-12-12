# -*- coding: utf-8 -*-
"""
PID curriculum runner using YAML stages
  • One CSV per stage
  • One MP4 per stage
  • No RL reward logic (PID only)
"""

import os
import time
import numpy as np
import pandas as pd
import pybullet as p
import yaml

from sim.envs.multi_drone_quad_env import MultiDroneQuadEnv
from sim.controllers.baseline_pid import FormationPIDController
from systems.video_recorder import VideoRecorder


YAML_PATH = r"C:\Users\ronak\anaconda3\envs\marl\marl-drone-formation\systems\configs\ppo_multi_curriculum.yaml"
LOG_DIR   = r"C:\Users\ronak\anaconda3\envs\marl\marl-drone-formation\logs\PIDStagesLogs"

os.makedirs(LOG_DIR, exist_ok=True)


# Load curriculum
with open(YAML_PATH, "r") as f:
    cfg = yaml.safe_load(f)

stages = cfg["stages"]


#  Safety: kill any existing PyBullet connections 
while p.isConnected():
    p.disconnect()


# Create env + PID controller 
env = MultiDroneQuadEnv(num_drones=5, gui=False)
pid = FormationPIDController(env)

MAX_STAGE_STEPS = 12000

#for PID run, skip stage 6
SKIP_STAGE_NAMES = {"stage6_threat_extreme"}



# Run stages
for k, stage in enumerate(stages):
    stage_name = stage["name"]
    print(f"\n=== STAGE {k}: {stage_name} ===")

    if stage_name in SKIP_STAGE_NAMES:
        print(f"Skipping stage {k}: {stage_name}")
        continue
    # Disable RL-specific reward logic explicitly
    stage_opts = dict(stage)
    stage_opts.update({
        "use_stl_reward": False,
        "reward_scale": 0.0,
        "collision_penalty": 0.0,
        "formation_break_penalty": 0.0,
    })


    # Reset environment for this stage
    obs, _ = env.reset(options=stage_opts)
    pid.reset()


    # Video recorder
    video_path = os.path.join(LOG_DIR, f"stage{k:02d}_{stage_name}.mp4")
    recorder = VideoRecorder(video_path, fps=60)


    # CSV logging setup
    rows = []
    T = min(int(stage["total_timesteps"]), MAX_STAGE_STEPS)


    done = False
    truncated = False

    try:
        for step in range(T):
            action = pid(obs)

            # leader state before step 
            obs_pd = obs.reshape(env.num_drones, env.per_drone_obs_dim)
            leader = obs_pd[0]

            pos = leader[0:3]
            vel = leader[3:6]
            eul = leader[6:9]
            ang = leader[9:12]
            des = env.get_desired_positions()[0]

            obs, _, done, truncated, info = env.step(action)

            # log row
            rows.append({
                "stage": stage_name,
                "step": step,

                # Metrics
                "mean_form_error_m": env.last_metrics.get("mean_form_error", np.nan),
                "mean_z_error_m": env.last_metrics.get("mean_z_error", np.nan),
                "min_dyn_distance_m": env.last_metrics.get("min_dyn_distance", np.nan),
                "collision_flag": env.last_metrics.get("collision", 0.0),

                # Leader state
                "x_m": pos[0], "y_m": pos[1], "z_m": pos[2],
                "vx_mps": vel[0], "vy_mps": vel[1], "vz_mps": vel[2],
                "roll_rad": eul[0], "pitch_rad": eul[1], "yaw_rad": eul[2],
                "wx_rps": ang[0], "wy_rps": ang[1], "wz_rps": ang[2],

                # Desired
                "x_des_m": des[0],
                "y_des_m": des[1],
                "z_des_m": des[2],
            })

            # record frame
            frame = env.render(mode="rgb_array")
            recorder.add_frame(frame)

            if done or truncated:
                print(f"Stage {stage_name} ended early at step {step}")
                break

    finally:
        recorder.close()

    # Save CSV for this stage
    df = pd.DataFrame(rows)
    csv_path = os.path.join(LOG_DIR, f"stage{k:02d}_{stage_name}.csv")
    df.to_csv(csv_path, index=False)

    print(f"Saved CSV   → {csv_path}")
    print(f"Saved video → {video_path}")
    print(f"Final metrics:", env.last_metrics)


env.close()
print("\nAll PID stages complete.")
