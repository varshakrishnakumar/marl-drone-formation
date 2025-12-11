import argparse
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from sim.envs.multi_drone_quad_env import MultiDroneQuadEnv


class ResetOptionsWrapper(gym.Wrapper):
    def __init__(self, env, options: dict):
        super().__init__(env)
        self._options = options or {}

    def reset(self, **kwargs):
        opts = dict(self._options)
        if "options" in kwargs and kwargs["options"] is not None:
            opts.update(kwargs["options"])
        kwargs["options"] = opts
        return self.env.reset(**kwargs)


def make_env(gui: bool, stage_options: dict):
    def _thunk():
        base = MultiDroneQuadEnv(
            num_drones=5,
            gui=gui,
            max_steps=2000,
        )
        env = ResetOptionsWrapper(base, stage_options)
        env = Monitor(env)
        return env
    return _thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Path to SB3 .zip model (e.g. best_model.zip)")
    parser.add_argument("--vecnorm", required=True,
                        help="Path to VecNormalize .pkl (e.g. vecnormalize_final.pkl)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI")
    args = parser.parse_args()

    # Stage0-style options (match your YAML stage0 as much as you care to)
    # Stage4-style options (match your stage4_threat_hard YAML)
    stage_opts = dict(
        leader_speed_scale=1.0,
        spawn_in_formation=True,
        disable_dynamic=False,
        chase_mode="most_error",
        dynamic_aggression=0.80,
        retarget_interval=1.6,
        retarget_on_close=True,
        retarget_close_dist=0.50,

        dynamic_repulse_radius=1.10,
        dynamic_repulse_gain=1.00,

        threat_radius=1.4,
        danger_radius=1.0,
        safe_radius=2.2,
        evade_gain=0.7,
        avoid_scale=3.5,
        safe_bonus=3.0,
        form_under_threat_gain=1.45,

        obstacle_repulse_gain=0.9,
        static_clear_gain=0.55,

        form_w_mean=2.8,
        form_w_max=0.60,
        form_var_gain=0.35,
        huber_delta=0.45,
        alt_w=0.6,

        formation_spacing=0.78,
        min_sep=0.62,
        sep_radius=0.68,
        sep_gain=2.2,
        sep_hysteresis=0.05,
        sep_force=1.15,

        max_roll_deg=10,
        max_pitch_deg=10,
        max_yaw_rate_deg=60,
        thrust_delta_scale=0.28,
    )


    venv = DummyVecEnv([make_env(gui=args.gui, stage_options=stage_opts)])


    # Load VecNormalize stats
    venv = VecNormalize.load(args.vecnorm, venv)
    venv.training = False      # eval mode
    venv.norm_reward = False   # report raw rewards

    # Load model
    model = PPO.load(args.model, env=venv)

    for ep in range(args.episodes):
        obs = venv.reset()
        done = [False]
        ep_rew = 0.0
        ep_len = 0

        mfe_hist = []
        coll_hist = []

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, done, infos = venv.step(action)
            ep_rew += float(rew[0])
            ep_len += 1

            info0 = infos[0]
            metrics = info0.get("metrics", {})
            if "mean_form_error" in metrics:
                mfe_hist.append(float(metrics["mean_form_error"]))
            if "collision" in metrics:
                coll_hist.append(float(metrics["collision"]))

        mfe_mean = float(np.mean(mfe_hist)) if mfe_hist else np.nan
        coll_rate = float(np.mean(coll_hist)) if coll_hist else 0.0

        print(
            f"EP {ep:02d} | len={ep_len:4d}  R={ep_rew:6.3f}  "
            f"mfe_mean={mfe_mean:6.3f}  coll_rate={coll_rate:5.3f}"
        )


if __name__ == "__main__":
    main()
