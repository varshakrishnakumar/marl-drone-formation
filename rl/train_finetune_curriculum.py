from __future__ import annotations
import os
import argparse
from typing import Dict, Any, Callable

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

from sim.envs.multi_drone_quad_env import MultiDroneQuadEnv

class ResetOptionsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, options_fn: Callable[[], Dict[str, Any]]):
        super().__init__(env)
        self._options_fn = options_fn

    def reset(self, **kwargs):
        opts = self._options_fn() if self._options_fn else None
        if opts:
            kwargs["options"] = opts
        return self.env.reset(**kwargs)

def make_env(seed: int, gui: bool, options_fn: Callable[[], Dict[str, Any]]):
    def _thunk():
        env = MultiDroneQuadEnv(gui=gui)
        env = ResetOptionsWrapper(env, options_fn)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _thunk

def parse_args():
    p = argparse.ArgumentParser("PPO fine-tune curriculum (A: no sphere, B: moving leader, C: full)")
    p.add_argument("--load_model", type=str, required=True, help="path to previous final_model.zip")
    p.add_argument("--load_vecnorm", type=str, required=True, help="path to previous vecnormalize_final.pkl")
    p.add_argument("--log_dir", type=str, default="runs/ppo_multi_stage1_diamond_ft")
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gui", action="store_true")

    p.add_argument("--n_steps", type=int, default=3072)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--ent_coef", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--target_kl", type=float, default=0.02)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--steps_A", type=int, default=500_000)
    p.add_argument("--steps_B", type=int, default=700_000)
    p.add_argument("--steps_C", type=int, default=800_000)
    p.add_argument("--formation_spacing", type=float, default=0.6)

    return p.parse_args()

def main():
    args = parse_args()
    set_random_seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    def opts_A():
        return {
            "disable_dynamic": True,
            "leader_speed_scale": 0.0,
        }

    def opts_B():
        return {
            "disable_dynamic": True,
            "leader_speed_scale": 1.0,
        }

    def opts_C():
        return {
            "disable_dynamic": False,
            "leader_speed_scale": 1.0,
        }

    current_opts_fn = [opts_A]

    def options_router():
        return current_opts_fn[0]()

    thunks = [make_env(seed=args.seed + i, gui=args.gui and i == 0, options_fn=options_router)
              for i in range(args.n_envs)]
    vec = SubprocVecEnv(thunks, start_method="forkserver")
    vec = VecNormalize.load(args.load_vecnorm, vec)
    vec.training = True
    vec.norm_reward = True

    model = PPO.load(
    args.load_model,
    env=vec,
    device="cuda" if torch.cuda.is_available() else "cpu",
    custom_objects=dict(
        learning_rate=args.learning_rate,
        clip_range=args.clip_range,
    ),
    print_system_info=False,
)
    from stable_baselines3.common.logger import configure
    model.set_logger(configure(args.log_dir, ["stdout", "tensorboard"]))
    print("Logging to", args.log_dir)


    model.learning_rate = args.learning_rate
    model.n_epochs = args.n_epochs
    model.ent_coef = args.ent_coef
    model.gamma = args.gamma
    model.gae_lambda = args.gae_lambda
    model.clip_range = args.clip_range
    model.target_kl = args.target_kl
    model.max_grad_norm = args.max_grad_norm
    if model.n_steps != args.n_steps:
        model.n_steps = args.n_steps
        model._setup_model()


    updates_per_save = max(1, (args.n_envs * args.n_steps) // (256 * args.n_envs))
    ckpt_cb = CheckpointCallback(
        save_freq=updates_per_save,
        save_path=ckpt_dir,
        name_prefix="ft",
        save_vecnormalize=True,
    )

    current_opts_fn[0] = opts_A
    print("[STAGE A] leader static, no sphere — enter & hold diamond")
    model.learn(total_timesteps=args.steps_A, callback=[ckpt_cb], reset_num_timesteps=False)
    model.save(os.path.join(args.log_dir, "stageA_model.zip"))
    vec.save(os.path.join(args.log_dir, "stageA_vecnormalize.pkl"))

    current_opts_fn[0] = opts_B
    print("[STAGE B] leader moving, no sphere — hold while moving")
    model.learn(total_timesteps=args.steps_B, callback=[ckpt_cb], reset_num_timesteps=False)
    model.save(os.path.join(args.log_dir, "stageB_model.zip"))
    vec.save(os.path.join(args.log_dir, "stageB_vecnormalize.pkl"))

    current_opts_fn[0] = opts_C
    print("[STAGE C] full task — avoid sphere while keeping diamond")
    model.learn(total_timesteps=args.steps_C, callback=[ckpt_cb], reset_num_timesteps=False)
    model.save(os.path.join(args.log_dir, "final_ft_model.zip"))
    vec.save(os.path.join(args.log_dir, "vecnormalize_final.pkl"))

    vec.close()
    print("[DONE] Fine-tune complete →", args.log_dir)

if __name__ == "__main__":
    main()
