import os
TARGET_DIR = r"C:\Users\ronak\anaconda3\envs\marl\marl-drone-formation"
if os.getcwd() != TARGET_DIR:
    os.chdir(TARGET_DIR)

from sim.envs.multi_drone_quad_env import MultiDroneQuadEnv
from sim.controllers.baseline_pid import FormationPIDController
import numpy as np
import pybullet as p
from systems.video_recorder import VideoRecorder




while p.isConnected():
    p.disconnect()


def run_baseline(num_episodes=1, gui=True):
    env = MultiDroneQuadEnv(num_drones=5, gui=gui)
    controller = FormationPIDController(env)

    # --- START VIDEO RECORDER ---
    recorder = VideoRecorder("videos/run1.mp4", fps=60)
    try:
        for ep in range(num_episodes):
            obs, info = env.reset()
            controller.reset()
    
            ep_reward = 0.0
            done = False
            truncated = False
    
            while not (done or truncated):
                action = controller(obs)
                obs, reward, done, truncated, info = env.step(action)
                ep_reward += reward
    
                # --- RECORD FRAME ---
                w, h, rgba, _, _ = p.getCameraImage(
                    width=640,
                    height=480,
                    renderer=p.ER_BULLET_HARDWARE_OPENGL
                )[0:5]
    
                rgb = np.reshape(rgba, (h, w, 4))[:, :, :3]   # drop alpha
                recorder.add_frame(rgb)
    
            print(f"Episode {ep}: total reward = {ep_reward:.3f}, metrics = {env.last_metrics}")
    
        # --- CLOSE VIDEO RECORDER ---
    
    finally:
        recorder.close()
        env.close()
        print("Recorder closed cleanly.")


if __name__ == "__main__":
    run_baseline()
