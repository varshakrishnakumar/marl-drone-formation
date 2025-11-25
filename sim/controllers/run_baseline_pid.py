from sim.envs.multi_drone_quad_env import MultiDroneQuadEnv
from sim.controllers.baseline_pid import FormationPIDController
import numpy as np

def run_baseline(num_episodes=5, gui=True):
    env = MultiDroneQuadEnv(num_drones=5, gui=gui)
    controller = FormationPIDController(env)

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

        print(f"Episode {ep}: total reward = {ep_reward:.3f}, metrics = {env.last_metrics}")

    env.close()

if __name__ == "__main__":
    run_baseline()
