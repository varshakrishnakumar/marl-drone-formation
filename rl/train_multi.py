import argparse
import datetime
import os

import gymnasium as gym
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.buffers import RolloutBuffer

from sim.envs.multi_drone_quad_env import MultiDroneQuadEnv
from sim.envs.callbacks import CustomMetricsCallback
from stable_baselines3.common.buffers import RolloutBuffer


# -----------------------------------------------------------------------------
# Environment factory
# -----------------------------------------------------------------------------
def make_env(num_drones=5, gui=False, seed_offset=0):
    def _init():
        env = MultiDroneQuadEnv(num_drones=num_drones, gui=gui)
        env.reset(seed=seed_offset)
        return env
    return _init


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script for MultiDroneQuadEnv (Multi-Drone Formation + Obstacle Avoidance)"
    )

    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total PPO training timesteps.")

    parser.add_argument("--num-drones", type=int, default=5,
                        help="Number of drones.")

    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments. "
                             "Use 1 on macOS; use 4-16 on Linux for speed.")

    parser.add_argument("--run-name", type=str, default=None,
                        help="Optional name for logging/model directories.")

    parser.add_argument("--normalize", action="store_true",
                        help="Use VecNormalize (recommended for PPO).")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to a previous PPO model to continue training.")

    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override the PPO learning rate (float).")

    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help=("Optional KL threshold for early stopping. "
              "Increase to reduce clipping or set to 0 to disable."),
    )

    parser.add_argument("--hyper", type=str, default=None,
                        help="JSON string with hyperparameter overrides.")


    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main training logic
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    import json

    # Apply hyperparameter overrides
    hyperparams = {}
    if args.hyper:
        hyperparams = json.loads(args.hyper)
        print("Applying hyperparameter overrides:", hyperparams)


    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"marl_run_{timestamp}"

    log_dir = os.path.join("logs", run_name)
    models_dir = os.path.join("models", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Build vectorized training environment
    # -------------------------------------------------------------------------
    print(f"\nCreating {args.n_envs} training environments...")

    if args.n_envs == 1:
        env_fns = [make_env(args.num_drones, gui=False)]
        vec_env = DummyVecEnv(env_fns)
    else:
        env_fns = [make_env(args.num_drones, gui=False, seed_offset=i)
                   for i in range(args.n_envs)]
        vec_env = SubprocVecEnv(env_fns)

    # Optional normalization
    if args.normalize:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )
        print("Using VecNormalize.")

    # -------------------------------------------------------------------------
    # PPO Model (tuned for drone dynamics)
    # -------------------------------------------------------------------------
    print("Initializing PPO model...\n")

    target_kl_override = hyperparams.get("target_kl", args.target_kl)
    if target_kl_override is not None and target_kl_override <= 0:
        target_kl_override = None

    learning_rate = hyperparams.get(
        "learning_rate",
        args.learning_rate if args.learning_rate is not None else 2.5e-4,
    )

    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    if args.load_model:
        print(f"Loading previous model: {args.load_model}")
        model = PPO.load(args.load_model, env=vec_env)
        model.set_logger(logger)

        if target_kl_override is not None:
            model.target_kl = target_kl_override

        if learning_rate is not None:
            model.learning_rate = learning_rate
            model.lr_schedule = get_schedule_fn(model.learning_rate)

        buffer_reset_needed = False

        if "gamma" in hyperparams:
            model.gamma = hyperparams["gamma"]
            buffer_reset_needed = True
        if "n_steps" in hyperparams:
            model.n_steps = hyperparams["n_steps"]
            buffer_reset_needed = True

        if buffer_reset_needed:
            model.rollout_buffer = RolloutBuffer(
                model.n_steps,
                model.observation_space,
                model.action_space,
                model.device,
                gae_lambda=model.gae_lambda,
                gamma=model.gamma,
                n_envs=model.n_envs,
            )

    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=learning_rate,
            n_steps=hyperparams.get("n_steps", 2048 // args.n_envs),
            gamma=hyperparams.get("gamma", 0.995),
            batch_size=128,
            n_epochs=10,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=log_dir,
            verbose=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            target_kl=target_kl_override,
        )
        model.set_logger(logger)

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=models_dir,
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    metrics_callback = CustomMetricsCallback(log_freq=200)

    # -------------------------------------------------------------------------
    # Train!
    # -------------------------------------------------------------------------
    print(f"Starting PPO training for {args.timesteps:,} timesteps...")
    print(f"Logs → {log_dir}")
    print(f"Models → {models_dir}\n")

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, metrics_callback],
        progress_bar=True
    )


    # -------------------------------------------------------------------------
    # Save final model + normalization stats
    # -------------------------------------------------------------------------
    model_path = os.path.join(models_dir, "ppo_final_model")
    model.save(model_path)

    if args.normalize:
        vec_env.save(os.path.join(models_dir, "vec_normalize.pkl"))

    print("\nTraining complete!")
    print(f"Final model saved to: {model_path}")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
