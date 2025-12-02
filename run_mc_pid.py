import numpy as np
from multi_drone_quad_env import make_env
from baseline_pid import FormationPIDController
from run_utils import run_multiple_episodes, summarize_episode_metrics


def random_initial_conditions(num_drones=5):
    # modes supported by _pick_chase_target(): "random", "round_robin", "nearest", "most_error"
    chase_modes = ["nearest", "round_robin", "random", "most_error"]

    return {
        "pos_jitter": np.random.uniform(-0.3, 0.3, size=(num_drones, 3)),
        "yaw_jitter": np.random.uniform(-0.4, 0.4, size=num_drones),
        "vel_jitter": np.random.uniform(-0.2, 0.2, size=(num_drones, 3)),
        "obstacle_jitter": np.random.uniform(-0.4, 0.4, size=3),
        "dynamic_jitter": np.random.uniform(-0.4, 0.4, size=3),

        # **Make sphere active & randomize chase behavior per episode**
        "disable_dynamic": False,                      # ensure the chaser is ON
        "chase_mode": np.random.choice(chase_modes),   # random mode each episode
    }



def main():
    n_episodes = 25
    max_steps = 1000

    env = make_env()
    controller = FormationPIDController()

    all_metrics = []
    for ep in range(n_episodes):
        print(f"Episode {ep+1}/{n_episodes}")
        options = random_initial_conditions()
        obs = env.reset(options=options)
        metrics = run_multiple_episodes(
            env,
            controller,
            num_episodes=1,
            max_steps=max_steps,
            render=False,
            options=options,
        )
        all_metrics.append(metrics[0])

    summary = summarize_episode_metrics(all_metrics)
    print("\n==== PID Evaluation Summary ====")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
