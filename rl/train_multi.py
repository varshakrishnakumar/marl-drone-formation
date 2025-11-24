import os
import argparse
import datetime

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from sim.envs.multi_drone_env import MultiDroneEnv


def make_env(num_drones=5, gui=False):
    """
    Returns a function that creates a MultiDroneEnv.
    Used by DummyVecEnv to construct parallel instances.
    """
    def _init():
        return MultiDroneEnv(num_drones=num_drones, gui=gui)
    return _init


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimized training script for MultiDroneEnv."
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=300_000,
        help="Total PPO training timesteps."
    )

    parser.add_argument(
        "--num-drones",
        type=int,
        default=5,
        help="Number of drones."
    )

    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel vector environments. Use 1 for macOS speed."
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name for logs/model folders."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"marl_run_{timestamp}"

    log_dir = os.path.join("logs", run_name)
    models_dir = os.path.join("models", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)


    env_fns = [make_env(num_drones=args.num_drones, gui=False)
               for _ in range(args.n_envs)]
    train_env = DummyVecEnv(env_fns)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=1024,          # smaller for stability on macOS
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=log_dir,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=models_dir,
        name_prefix="ppo_marl_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # ----------------------------------------------------------------------
    # Train
    # ----------------------------------------------------------------------
    print(f"Starting optimized PPO training for {args.timesteps} timesteps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback],
        progress_bar=True
    )

    # ----------------------------------------------------------------------
    # Save final model
    # ----------------------------------------------------------------------
    final_model_path = os.path.join(models_dir, "ppo_marl_final")
    model.save(final_model_path)

    print(f"Training complete.")
    print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
