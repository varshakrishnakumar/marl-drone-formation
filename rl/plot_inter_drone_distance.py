import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


LOG_PATH = "logs/playground_runs.npy"


def main():
    data = np.load(LOG_PATH)
    T, N, _ = data.shape

    pairs = list(combinations(range(N), 2))

    plt.figure()
    for (i, j) in pairs:
        dist = np.linalg.norm(data[:, i, :] - data[:, j, :], axis=1)
        plt.plot(dist, label=f"{i}-{j}")

    plt.title("Inter-Drone Distance")
    plt.xlabel("Time step")
    plt.ylabel("Distance [m]")
    plt.axhline(0.2, color="r", linestyle="--", label="Min Safe Dist")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
