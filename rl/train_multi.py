from __future__ import annotations

import os
import re
import sys
import yaml
import signal
import argparse
from glob import glob
from typing import Any, Dict, Optional
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import gymnasium as gym
from sim.envs.multi_drone_quad_env import MultiDroneQuadEnv
import time
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class ResetOptionsWrapper(gym.Wrapper):
    """Injects reset(options=...) on every env.reset(). Why: enforce stage options reliably across resets."""
    def __init__(self, env: gym.Env, options: dict):
        super().__init__(env)
        self._opt = options or {}
    def reset(self, **kwargs):
        kwargs["options"] = {**self._opt, **kwargs.get("options", {})}
        return self.env.reset(**kwargs)


def make_env(
    seed: int,
    gui: bool = False,
    *,
    num_drones: int = 5,
    max_steps: int = 2000,
    leader_speed_scale: float = 0.8,
    spawn_in_formation: bool = True,
    disable_dynamic: bool = False,
    **env_overrides,
):
    def _thunk():
        base = MultiDroneQuadEnv(num_drones=num_drones, gui=gui, max_steps=max_steps)

        eval_opts = dict(
            leader_speed_scale=leader_speed_scale,
            spawn_in_formation=spawn_in_formation,
            disable_dynamic=disable_dynamic,
            debug_diamond=False,
        )

        for k, v in env_overrides.items():
            if v is not None:
                eval_opts[k] = v

        env = ResetOptionsWrapper(base, eval_opts)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _thunk




def save_config(path: str, cfg: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)


def find_latest_checkpoint(dir_path: str) -> Optional[str]:
    candidates = glob(os.path.join(dir_path, "ppo_multi_*_steps.zip"))
    if not candidates:
        return None
    def steps(p: str) -> int:
        m = re.search(r"_(\d+)_steps\.zip$", p)
        return int(m.group(1)) if m else -1
    candidates.sort(key=steps)
    return candidates[-1]


def attach_eval_env(
    train_vec: VecNormalize,
    gamma: float,
    *,
    use_train_options: bool = True,
    leader_speed_scale: float = 0.3,
    spawn_in_formation: bool = True,
    disable_dynamic: bool = True,
    env_overrides: Optional[Dict[str, Any]] = None,
) -> VecNormalize:
    env_overrides = env_overrides or {}
    if use_train_options:
        thunk = make_env(
            seed=10_000,
            gui=False,
            leader_speed_scale=leader_speed_scale,
            spawn_in_formation=spawn_in_formation,
            disable_dynamic=disable_dynamic,
            **env_overrides,
        )
    else:
        thunk = make_env(seed=10_000, gui=False)

    eval_env = SubprocVecEnv([thunk])
    eval_vec = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=gamma)
    eval_vec.obs_rms = train_vec.obs_rms
    eval_vec.training = False
    return eval_vec



def graceful_save(model: PPO, vec: VecNormalize, out_dir: str, name: str = "final") -> None:
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, f"{name}_model"))
    vec.save(os.path.join(out_dir, f"vecnormalize_{name}.pkl"))
    print(f"[INFO] Saved model+vecnorm to {out_dir} ({name}).", flush=True)

class ProgressLoggerCallback(BaseCallback):
    """
    Prints a simple progress bar and rolling metrics pulled from env `infos`.
    Also logs scalars to TensorBoard under `train/*`.
    """
    def __init__(self, total_timesteps: int, log_every: int = 10000,
                 window: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self.log_every = int(log_every)
        self.window = int(window)
        self._last_log = 0
        self._t0 = time.time()

        # rolling windows
        self._mfe = deque(maxlen=window)   # mean formation error
        self._mdd = deque(maxlen=window)   # min dyn distance
        self._col = deque(maxlen=window)   # collision indicator
        self._rew = deque(maxlen=window)   # reward
        self._stl = deque(maxlen=window)   # STL robustness margin (optional)

    def _on_step(self) -> bool:
        # rolling reward
        rewards = self.locals.get("rewards", None)
        if rewards is not None:
            for r in np.atleast_1d(rewards):
                self._rew.append(float(r))

        # metrics from env.info
        infos = self.locals.get("infos", [])
        for info in infos or []:
            m = info.get("metrics", {})
            if "mean_form_error" in m:
                self._mfe.append(float(m["mean_form_error"]))
            if "min_dyn_distance" in m:
                self._mdd.append(float(m["min_dyn_distance"]))
            if "collision" in m:
                self._col.append(float(m["collision"]))
            # this is safe even if env doesn't provide stl_margin
            if "stl_margin" in m:
                self._stl.append(float(m["stl_margin"]))

        steps = int(self.model.num_timesteps)
        if steps - self._last_log >= self.log_every:
            self._last_log = steps
            pct = steps / max(1, self.total_timesteps)
            bar_len = 28
            filled = int(bar_len * pct)
            bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"

            mean_rew = np.mean(self._rew) if len(self._rew) else float("nan")
            mean_mfe = np.mean(self._mfe) if len(self._mfe) else float("nan")
            mean_mdd = np.mean(self._mdd) if len(self._mdd) else float("nan")
            col_rate = np.mean(self._col) if len(self._col) else float("nan")
            mean_stl = np.mean(self._stl) if len(self._stl) else float("nan")

            elapsed = time.time() - self._t0
            msg = (
                f"{bar} {pct:5.1%}  "
                f"steps={steps:,}  "
                f"r≈{mean_rew: .3f}  "
                f"mfe≈{mean_mfe: .3f}  "
                f"mdd≈{mean_mdd: .3f}  "
                f"coll_rate≈{col_rate: .3f}  "
                f"stl≈{mean_stl: .3f}  "
                f"t={elapsed:,.0f}s"
            )
            print(msg, flush=True)

            # log to TensorBoard
            self.logger.record("train/progress_percent", pct)
            if len(self._rew):
                self.logger.record("train/roll_reward_mean", float(mean_rew))
            if len(self._mfe):
                self.logger.record("train/mfe_mean_roll", float(mean_mfe))
            if len(self._mdd):
                self.logger.record("train/min_dyn_dist_roll", float(mean_mdd))
            if len(self._col):
                self.logger.record("train/collision_rate_roll", float(col_rate))
            if len(self._stl):
                self.logger.record("train/stl_margin_roll", float(mean_stl))

        return True

    
class PeriodicSaveCallback(BaseCallback):
    """Hard-save model + vecnorm every `freq` timesteps, independent of EvalCallback."""
    def __init__(self, freq: int, save_dir: str, vecnorm: VecNormalize, prefix: str = "ppo_multi", verbose: int = 1):
        super().__init__(verbose)
        self.freq = int(freq)
        self.save_dir = save_dir
        self.vecnorm = vecnorm
        self.prefix = prefix
        self._last = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.freq <= 0:
            return True
        if (self.num_timesteps - self._last) >= self.freq:
            steps = int(self.num_timesteps)
            model_path = os.path.join(self.save_dir, f"{self.prefix}_{steps}_steps.zip")
            vec_path   = os.path.join(self.save_dir, f"{self.prefix}_vecnormalize_{steps}_steps.pkl")
            try:
                self.model.save(model_path)
                self.vecnorm.save(vec_path)
                print(f"[SAVE] checkpoint @ {steps:,} → {model_path}", flush=True)
            except Exception as e:
                print(f"[WARN] periodic save failed at {steps:,}: {e}", flush=True)
            self._last = steps
        return True

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--log_dir", type=str, default="runs/ppo_multi")
    p.add_argument("--total_timesteps", type=int, default=4_000_000)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gui", action="store_true")
    p.add_argument("--start_method", type=str, default="forkserver",
                   choices=["fork", "forkserver", "spawn"])
    p.add_argument("--num_drones", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=2000)


    p.add_argument("--n_steps", type=int, default=2048)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--ent_coef", type=float, default=1e-3)
    p.add_argument("--vf_coef", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--target_kl", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=0.5)

    p.add_argument("--progress_log_every", type=int, default=10000,
                   help="Print progress/metrics every N env steps.")

    p.add_argument("--save_every_steps", type=int, default=500_000)
    p.add_argument("--eval_every_steps", type=int, default=250_000)
    p.add_argument("--eval_episodes", type=int, default=5)

    p.add_argument("--load", type=str, default="")
    p.add_argument("--load_latest", action="store_true")
    p.add_argument("--load_vecnorm", type=str, default="")

    p.add_argument("--leader_speed_scale", type=float, default=0.3,
                   help="Leader trajectory speed scale (0 freezes).")
    p.add_argument("--spawn_in_formation", action="store_true",
                   help="Spawn drones at formation slots on reset.")
    p.add_argument("--disable_dynamic", action="store_true",
                   help="Disable chasing sphere during training.")
    p.add_argument("--eval_use_train_options", action="store_true",
                   help="Evaluate with the same reset options used in training.")

    args, unknown = p.parse_known_args()

    passthrough = {}
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok[2:]

        # flag-only
        if i + 1 >= len(unknown) or unknown[i + 1].startswith("--"):
            passthrough[key] = True
            i += 1
            continue

        raw = unknown[i + 1]
        i += 2

        if isinstance(raw, str) and raw.lower() in ("true", "false"):
            passthrough[key] = (raw.lower() == "true")
        else:
            try:
                passthrough[key] = float(raw)
            except ValueError:
                passthrough[key] = raw

    args.passthrough = passthrough

    return args




def main():
    args = parse_args()
    set_random_seed(args.seed)

    os.makedirs(args.log_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    env_kwargs = dict(
        num_drones=args.num_drones,
        max_steps=args.max_steps,
        leader_speed_scale=args.leader_speed_scale,
        spawn_in_formation=args.spawn_in_formation,
        disable_dynamic=args.disable_dynamic,
        **args.passthrough,
    )



    env_kwargs.update(getattr(args, "passthrough", {}))


    thunks = [
        make_env(
            seed=args.seed + i,
            gui=(args.gui and i == 0),
            **env_kwargs,
        )
        for i in range(args.n_envs)
    ]
    vec = SubprocVecEnv(thunks, start_method=args.start_method)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=args.gamma)

    if args.load_vecnorm and os.path.isfile(args.load_vecnorm):
        try:
            loaded_vec = VecNormalize.load(args.load_vecnorm, vec)
            vec.obs_rms = loaded_vec.obs_rms
            vec.ret_rms = loaded_vec.ret_rms
            print(f"[INFO] Loaded VecNormalize stats from {args.load_vecnorm}", flush=True)
        except Exception as e:
            print(f"[WARN] Could not load VecNormalize from {args.load_vecnorm}: {e}", flush=True)

    n_steps_per_env = int(args.n_steps)
    if n_steps_per_env <= 0:
        raise ValueError("--n_steps must be > 0")
    total_rollout = n_steps_per_env * args.n_envs
    batch_size = max(1024, total_rollout // 4)

    policy_kwargs = dict(
    net_arch=[256, 256],
    activation_fn=torch.nn.Tanh,
    ortho_init=True,
    log_std_init=-2.0,
    )


    device = "cuda" if torch.cuda.is_available() else "cpu"

    model: PPO
    load_path = args.load
    if args.load_latest:
        latest = find_latest_checkpoint(ckpt_dir)
        if latest:
            load_path = latest
            print(f"[INFO] --load_latest resolved to: {load_path}", flush=True)
        else:
            print("[WARN] --load_latest requested but no checkpoints found; starting fresh.", flush=True)

    if load_path:
        print(f"[INFO] Loading PPO from {load_path}", flush=True)
        model = PPO.load(load_path, env=vec, device=device)
        model.set_env(vec)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec,
            learning_rate=args.learning_rate,
            n_steps=n_steps_per_env,
            batch_size=batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            tensorboard_log=args.log_dir,
            verbose=1,
            policy_kwargs=policy_kwargs,
            device=device,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_every_steps,
        save_path=ckpt_dir,
        name_prefix="ppo_multi",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_vec = attach_eval_env(
        vec,
        gamma=args.gamma,
        use_train_options=args.eval_use_train_options,
        leader_speed_scale=args.leader_speed_scale,
        spawn_in_formation=args.spawn_in_formation,
        disable_dynamic=args.disable_dynamic,
        env_overrides=args.passthrough,
    )

    eval_cb = EvalCallback(
        eval_env=eval_vec,
        best_model_save_path=ckpt_dir,
        log_path=args.log_dir,
        eval_freq=args.eval_every_steps,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )
    progress_cb = ProgressLoggerCallback(
        total_timesteps=args.total_timesteps,
        log_every=args.progress_log_every,
        window=max(2000, args.n_steps)
    )

    periodic_cb = PeriodicSaveCallback(
        freq=args.save_every_steps,
        save_dir=ckpt_dir,
        vecnorm=vec,
        prefix="ppo_multi",
    )

    # Snapshot config for reproducibility
    run_cfg = vars(args).copy()
    run_cfg.update(
        dict(
            computed_total_rollout=total_rollout,
            batch_size=batch_size,
            device=device,
            policy_net_arch=[256, 256],
        )
    )
    save_config(os.path.join(args.log_dir, "config.yaml"), run_cfg)
    print(
        f"[INFO] n_envs={args.n_envs}  n_steps(per-env)={n_steps_per_env}  "
        f"rollout={total_rollout}  batch_size={batch_size}  device={device}  "
        f"target_kl={args.target_kl}  ent_coef={args.ent_coef}",
        flush=True,
    )

    callbacks = [checkpoint_cb, eval_cb, progress_cb, periodic_cb]

    def _signal_handler(signum, frame):
        print(f"\n[WARN] Caught signal {signum}. Saving and exiting...", flush=True)
        graceful_save(model, vec, args.log_dir, name="interrupt")
        vec.close()
        eval_vec.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    finally:
        graceful_save(model, vec, args.log_dir, name="final")
        vec.close()
        eval_vec.close()



if __name__ == "__main__":
    main()
