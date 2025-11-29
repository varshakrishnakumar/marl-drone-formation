import argparse
import csv
import datetime
import json
import os
import time

import numpy as np

import matplotlib.pyplot as plt

from stable_baselines3 import PPO
import pybullet as p

from sim.envs.multi_drone_quad_env import MultiDroneQuadEnv
from systems.video_recorder import VideoRecorder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate PPO with GUI, video recording, CSV + JSON logging, and live plots."
    )

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-drones", type=int, default=5)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--gui", action="store_true")

    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--video-fps", type=int, default=30)

    return parser.parse_args()


# ----------------------------------------------------------------------
# Live Plot Dashboard
# ----------------------------------------------------------------------
class LivePlotDashboard:
    def __init__(self, num_drones):
        self.num_drones = num_drones
        self.tracking_errors = [[] for _ in range(num_drones)]
        self.z_errors = [[] for _ in range(num_drones)]
        self.team_rewards = []

        self.fig, self.axs = plt.subplots(3, 1, figsize=(7, 8))
        plt.tight_layout()

    def update(self, obs_all, desired_positions, reward):
        for i in range(self.num_drones):
            pos = obs_all[i][0:3]
            des = desired_positions[i]

            err = np.linalg.norm(pos - des)
            self.tracking_errors[i].append(err)
            self.z_errors[i].append(abs(pos[2] - des[2]))

        self.team_rewards.append(reward)
        self._redraw()

    def _redraw(self):
        for ax in self.axs:
            ax.clear()

        # Tracking error
        for i in range(self.num_drones):
            self.axs[0].plot(self.tracking_errors[i], label=f"Drone {i}")
        self.axs[0].set_title("Tracking Error (‖pos - desired‖)")
        self.axs[0].legend()

        # Height error
        for i in range(self.num_drones):
            self.axs[1].plot(self.z_errors[i], label=f"Drone {i}")
        self.axs[1].set_title("Height Error |z - z_des|")
        self.axs[1].legend()

        # Reward
        self.axs[2].plot(self.team_rewards, color="purple")
        self.axs[2].set_title("Team Reward")

        plt.pause(0.001)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    print(f"Loading model: {args.model}")
    model = PPO.load(args.model)

    # Environment
    env = MultiDroneQuadEnv(num_drones=args.num_drones, gui=args.gui)
    obs, _ = env.reset()

    # Logging filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"eval_log_{timestamp}.csv"
    json_path = f"eval_log_{timestamp}.json"

    os.makedirs("logs", exist_ok=True)

    csv_file = open(f"logs/{csv_path}", "w", newline="")
    csv_writer = csv.writer(csv_file)

    # CSV header
    header = ["step", "team_reward", "collision"]
    for i in range(args.num_drones):
        header += [
            f"d{i}_px", f"d{i}_py", f"d{i}_pz",
            f"d{i}_des_x", f"d{i}_des_y", f"d{i}_des_z",
            f"d{i}_track_err",
        ]
    csv_writer.writerow(header)

    print(f"CSV logging → logs/{csv_path}")

    # JSON log (rich data)
    json_log = []

    # Video
    recorder = None
    if args.video:
        recorder = VideoRecorder(args.video, fps=args.video_fps)
        print(f"Recording video → {args.video}")

    dashboard = LivePlotDashboard(args.num_drones)
    dt = 1.0 / args.fps

    print("Starting evaluation...")

    for step in range(args.steps):

        # ---- RL ACTION ----
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # ---- Extract data ----
        obs_all = env._get_all_obs()

        desired_positions = env.get_desired_positions()

        # ---- CSV LOGGING ----
        row = [step, reward, int(env.collision_happened)]

        for i in range(args.num_drones):
            pos = obs_all[i][0:3]
            des = desired_positions[i]
            err = np.linalg.norm(pos - des)

            row.extend([pos[0], pos[1], pos[2],
                        des[0], des[1], des[2],
                        err])

        csv_writer.writerow(row)

        # ---- JSON LOGGING ----
        json_log.append({
            "step": step,
            "team_reward": float(reward),
            "collision": bool(env.collision_happened),
            "positions": obs_all[:, 0:3].tolist(),
            "desired_positions": desired_positions.tolist(),
            "actions": action.tolist(),
        })

        # ---- LIVE DASHBOARD ----
        dashboard.update(obs_all, desired_positions, reward)

        # ---- RESET IF END ----
        if terminated or truncated:
            obs, _ = env.reset()

        # ---- VIDEO ----
        if recorder:
            leader_pos, _ = p.getBasePositionAndOrientation(env.drone_ids[0])

            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=leader_pos,
                distance=3.0,
                yaw=45,
                pitch=-20,
                roll=0,
                upAxisIndex=2,
            )

            projection_matrix = p.computeProjectionMatrixFOV(
                fov=70, aspect=1.0, nearVal=0.01, farVal=20
            )

            w, h, rgb, depth, seg = p.getCameraImage(
                width=720,
                height=720,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL if args.gui else p.ER_TINY_RENDERER
            )

            if not recorder.add_frame(rgb[:, :, :3]):
                print("[WARN] Video recorder closed unexpectedly; stopping recording.")
                recorder = None

        time.sleep(dt)

    if recorder:
        recorder.close()
        print(f"Video saved → {args.video}")

    # ---- SAVE JSON LOG ----
    with open(f"logs/{json_path}", "w") as jf:
        json.dump(json_log, jf, indent=2)

    print(f"JSON logging → logs/{json_path}")

    # Close CSV
    csv_file.close()

    print("ACTION SAMPLE:", np.mean(np.abs(action)))

    print("Evaluation finished.")


if __name__ == "__main__":
    main()
