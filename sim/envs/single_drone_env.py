import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os, time
import pandas as pd


class SingleDroneEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, gui=True):
        super().__init__()
        self.gui = gui
        self.time_step = 1/240
        self.max_steps = 1000
        self.target_z = 1.0
        self.kp = 10.0
        self.log_data = []

        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        asset_path = os.path.join(base_dir, "../assets/crazyflie/cf_assets/cf2x.urdf")
        self.plane = p.loadURDF("plane.urdf")
        self.drone = p.loadURDF(asset_path, [0, 0, 0.5])

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self.step_count = 0
        self.mass = p.getDynamicsInfo(self.drone, -1)[0]
        self.hover_force = self.mass * 9.81

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetBasePositionAndOrientation(self.drone, [0, 0, 0.5], [0, 0, 0, 1])
        p.resetBaseVelocity(self.drone, [0, 0, 0], [0, 0, 0])
        self.step_count = 0
        init_z = 0.5 + np.random.uniform(-0.1, 0.1)
        p.resetBasePositionAndOrientation(self.drone, [0, 0, init_z], [0, 0, 0, 1])
        return self._get_obs(), {}


    def _get_obs(self):
        pos, vel = p.getBasePositionAndOrientation(self.drone)[0], p.getBaseVelocity(self.drone)[0]
        z, z_dot = pos[2], vel[2]
        err = self.target_z - z
        return np.array([z, z_dot, err], dtype=np.float32)

    def step(self, action):
        """Apply an action, advance physics, and compute reward."""

        thrust_ratio = float(np.clip(action[0], 0, 1))
        thrust = thrust_ratio * 2 * self.hover_force

        p.applyExternalForce(
            self.drone, -1,
            [0, 0, thrust],
            [0, 0, 0],
            p.LINK_FRAME
        )

        p.stepSimulation()
        if self.gui:
            time.sleep(self.time_step)

        obs = self._get_obs()
        z, z_dot, err = obs

        reward = 1.0 - abs(err)

        reward -= 0.1 * abs(z_dot)

        reward = float(np.clip(reward, -1, 1))

        terminated = z < 0.1 or z > 3.0
        truncated = self.step_count >= self.max_steps
        self.step_count += 1

        self.log_data.append({
            "step": self.step_count,
            "z": z,
            "error": err,
            "reward": reward
        })

        info = {}
        return obs, reward, terminated, truncated, info


    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect(self.cid)
    
    def save_log(self, filename="hover_log.csv"):
        df = pd.DataFrame(self.log_data)
        df.to_csv(filename, index=False)
        print(f"[INFO] Log saved to {filename}")


if __name__ == "__main__":
    env = SingleDroneEnv(gui=True)
    obs = env.reset()
    for _ in range(500):
        env.step([0.5])
    env.close()
