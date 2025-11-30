import argparse
import time
import os

import gymnasium as gym
from stable_baselines3 import PPO

from sim.envs.multi_drone_env import MultiDroneEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate PPO model for MultiDroneEnv with GUI."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained PPO model (.zip).",
    )

    parser.add_argument(
        "--num-drones",
        type=int,
        default=5,
        help="Number of drones to evaluate with (default: 5).",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=3000,
        help="Number of simulation steps to run.",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Render FPS for sleep timing.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    print(f"Loading model: {args.model}")
    model = PPO.load(args.model)

    env = MultiDroneEnv(num_drones=args.num_drones, gui=True)
    obs, _ = env.reset()

    dt = 1.0 / args.fps

    print("Starting evaluation... Press Ctrl+C to quit.")

    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Environment signaled termination. Resetting...")
            obs, _ = env.reset()

        time.sleep(dt)

    print("Evaluation finished.")


if __name__ == "__main__":
    main()
