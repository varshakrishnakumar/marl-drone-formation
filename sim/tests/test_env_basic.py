import time
import numpy as np
import matplotlib.pyplot as plt
from sim.envs.multi_drone_quad_env import MultiDroneQuadEnv


def main():
    num_drones = 5
    steps = 1500
    gui = True

    print("Creating environment...")
    env = MultiDroneQuadEnv(num_drones=num_drones, gui=gui)

    print("Resetting environment...")
    obs, _ = env.reset()

    # Create arrays to track high-level behavior
    tracking_errors = [[] for _ in range(num_drones)]
    height_errors = [[] for _ in range(num_drones)]
    team_rewards = []

    plt.ion()
    fig, axs = plt.subplots(3, 1, figsize=(7, 8))

    print("Running zero-action evaluation...")
    for t in range(steps):

        # ACTION = all zeros (no RL)
        action = np.zeros(env.action_space.shape, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        # We extract per-drone states directly from obs
        obs_all = obs.reshape(num_drones, -1)
        desired_positions = env.get_desired_positions()

        # Logging errors
        for i in range(num_drones):
            pos = obs_all[i][0:3]
            des = desired_positions[i]

            e = np.linalg.norm(pos - des)
            z_e = abs(pos[2] - des[2])

            tracking_errors[i].append(e)
            height_errors[i].append(z_e)

        team_rewards.append(reward)

        # Live plotting every 10 frames
        if t % 10 == 0:
            axs[0].clear()
            for i in range(num_drones):
                axs[0].plot(tracking_errors[i], label=f"Drone {i}")
            axs[0].set_title("Tracking Error")
            axs[0].legend()

            axs[1].clear()
            for i in range(num_drones):
                axs[1].plot(height_errors[i], label=f"Drone {i}")
            axs[1].set_title("Height Error |z - z_des|")
            axs[1].legend()

            axs[2].clear()
            axs[2].plot(team_rewards, color="purple")
            axs[2].set_title("Team Reward")

            plt.pause(0.001)

        if terminated or truncated:
            print("Episode ended early. Resetting...")
            obs, _ = env.reset()

        # Slow down slightly to see motion more clearly
        time.sleep(1 / 200)

    print("\nFinished zero-action evaluation.")
    input("Press ENTER to close...")
    env.close()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
