import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


LOG_PATH = "logs/playground_runs.npy"


def main():
    data = np.load(LOG_PATH)   # shape = (T, num_drones, 3)
    T, N, _ = data.shape

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for drone_id in range(N):
        x = data[:, drone_id, 0]
        y = data[:, drone_id, 1]
        z = data[:, drone_id, 2]
        ax.plot(x, y, z, label=f"Drone {drone_id}")

    ax.set_title("Drone Trajectories")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
