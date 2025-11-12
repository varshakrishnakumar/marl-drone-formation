from stable_baselines3 import PPO
from sim.envs.single_drone_env import SingleDroneEnv
import time
import pybullet as p

# --- Load environment in GUI mode ---
env = SingleDroneEnv(gui=True)

# --- Load trained PPO model ---
model = PPO.load("ppo_single_hover", env)

# --- Start recording ---
video_path = "ppo_hover_demo.mp4"
p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_path)
print(f"[INFO] Recording started: {video_path}")

p.resetDebugVisualizerCamera(
    cameraDistance=3.0,
    cameraYaw=60,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0.5]
)

# --- Evaluate / visualize ---
obs, _ = env.reset()
for step in range(600):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, term, trunc, _ = env.step(action)

    if term or trunc:
        obs, _ = env.reset()

    time.sleep(1/240)

# --- Stop recording ---
p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)
print(f"[INFO] Recording finished. Video saved to {video_path}")

env.close()
