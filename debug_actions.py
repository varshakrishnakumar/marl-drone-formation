import time
import numpy as np
import pybullet as p

from sim.envs.multi_drone_env import MultiDroneEnv

def debug_actions(num_drones=5, steps=1000, gui=True):
    env = MultiDroneEnv(num_drones=num_drones, gui=gui)
    obs, _ = env.reset()

    print("\n==============================")
    print(" LIVE DRONE ACTION DEBUGGER")
    print("==============================")
    print(" Columns: Drone0  Drone1  Drone2  Drone3  Drone4\n")

    for t in range(steps):
        # FORCE all drones to use SAME hover thrust artificially
        # so we can check whether physics is symmetric
        action = np.ones(num_drones) * 0.5

        obs, reward, terminated, truncated, info = env.step(action)

        # Record Z heights for all drones
        heights = []
        for i in range(num_drones):
            pos, _ = p.getBasePositionAndOrientation(env.drone_ids[i])
            heights.append(pos[2])

        print(f"Step {t:04d}:  " + "  ".join(f"{h:.3f}" for h in heights))

        time.sleep(1.0/60)

        if terminated or truncated:
            print("Episode ended early.")
            break

    print("\nDone debugging.\n")


if __name__ == "__main__":
    debug_actions()
