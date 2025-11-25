import os
import time
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from sim.envs.multi_drone_quad_env import MultiDroneQuadEnv


def make_env(gui=True):
    def _init():
        return MultiDroneQuadEnv(num_drones=5, gui=gui)
    return _init


def record_video(model_path, out_dir="videos_eval", num_steps=3000):

    os.makedirs(out_dir, exist_ok=True)

    env = DummyVecEnv([make_env(gui=True)])

    # Wrap with video recorder
    env = VecVideoRecorder(
        env,
        video_folder=out_dir,
        record_video_trigger=lambda step: step == 0,
        video_length=num_steps,
        name_prefix="ppo_eval"
    )

    model = PPO.load(model_path)

    obs = env.reset()
    for step in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs = env.reset()

    env.close()
    print(f"Video saved to: {out_dir}")


if __name__ == "__main__":
    record_video("models/YRUN_NAME/ppo_final_model")
