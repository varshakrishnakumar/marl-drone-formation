import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os, time

class SingleDroneEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, gui=True):
        super().__init__()
        self.gui = gui
        self.time_step = 1/240
        self.max_steps = 1000
        self.target_z = 1.0
        self.kp = 10.0

        # Connect to PyBullet
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)

        # Load plane + drone
        base_dir = os.path.dirname(os.path.abspath(__file__))
        asset_path = os.path.join(base_dir, "../assets/crazyflie/cf_assets/cf2x.urdf")
        self.plane = p.loadURDF("plane.urdf")
        self.drone = p.loadURDF(asset_path, [0, 0, 0.5])

        # Observation: [z, z_dot, error]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        # Action: normalized thrust (0 to 1)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self.step_count = 0
        self.mass = p.getDynamicsInfo(self.drone, -1)[0]
        self.hover_force = self.mass * 9.81

    def reset(self):
        p.resetBasePositionAndOrientation(self.drone, [0, 0, 0.5], [0, 0, 0, 1])
        p.resetBaseVelocity(self.drone, [0, 0, 0], [0, 0, 0])
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        pos, vel = p.getBasePositionAndOrientation(self.drone)[0], p.getBaseVelocity(self.drone)[0]
        z, z_dot = pos[2], vel[2]
        err = self.target_z - z
        return np.array([z, z_dot, err], dtype=np.float32)

    def step(self, action):
        thrust_ratio = np.clip(action[0], 0, 1)
        thrust = thrust_ratio * 2 * self.hover_force  # allow up to 2x hover

        p.applyExternalForce(self.drone, -1, [0, 0, thrust], [0, 0, 0], p.LINK_FRAME)
        p.stepSimulation()
        if self.gui:
            time.sleep(self.time_step)

        obs = self._get_obs()
        z = obs[0]
        done = z < 0.1 or z > 3.0 or self.step_count >= self.max_steps
        reward = -abs(obs[2])  # penalize height error
        self.step_count += 1
        return obs, reward, done, {}

    def render(self, mode="human"):
        pass  # GUI handled automatically

    def close(self):
        p.disconnect(self.cid)

if __name__ == "__main__":
    env = SingleDroneEnv(gui=True)
    obs = env.reset()
    for _ in range(500):
        env.step([0.5])  # apply half hover thrust
    env.close()
