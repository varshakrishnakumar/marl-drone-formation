import numpy as np
import matplotlib.pyplot as plt


# simple example offset spec
FORMATION_OFFSETS = {
    0: np.array([0,0,0]),
    1: np.array([0.5,0.3,0]),
    2: np.array([0.5,-0.3,0])
}

LOG_PATH = "logs/playground_runs.npy"


def main():
    data = np.load(LOG_PATH)   # shape = (T, N, 3)
    T, N, _ = data.shape

    leader_traj = data[:, 0, :]  # drone 0 is leader

    plt.figure()
    for drone_id in range(1, N):
        desired = leader_traj + FORMATION_OFFSETS[drone_id]
        actual  = data[:, drone_id, :]
        error = np.linalg.norm(actual - desired, axis=1)
        plt.plot(error, label=f"Drone {drone_id}")

    plt.title("Formation Error Over Time")
    plt.xlabel("Time step")
    plt.ylabel("Error [m]")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
