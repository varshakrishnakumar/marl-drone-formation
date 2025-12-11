
import argparse
import csv
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from sim.envs.multi_drone_quad_env import MultiDroneQuadEnv


class ResetOptionsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, options: dict):
        super().__init__(env)
        self._opt = options or {}

    def reset(self, **kwargs):
        kwargs["options"] = self._opt
        return self.env.reset(**kwargs)


def unwrap_env(e: Any):
    """Return the underlying base env (handles DummyVecEnv, VecNormalize, and gym wrappers)."""
    if hasattr(e, "envs"):
        e = e.envs[0]
    if hasattr(e, "venv"):
        e = e.venv.envs[0]
    while hasattr(e, "env"):
        e = e.env
    if hasattr(e, "unwrapped"):
        e = e.unwrapped
    return e


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate PPO with GUI/video, CSV+JSON logging, live plots, and thresholds.")
    mex = ap.add_mutually_exclusive_group(required=True)
    mex.add_argument("--model", type=str, help="Path to .zip checkpoint")
    mex.add_argument("--zero-policy", action="store_true", help="Fly with zero actions (no model)")
    ap.add_argument("--vecnorm", type=str, default="", help="Path to vecnormalize_*.pkl")
    ap.add_argument("--formation-spacing", type=float, default=None)
    ap.add_argument("--episodes", type=int, default=20,help="Number of episodes to run (default: 20)")
    ap.add_argument("--num-drones", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--video", type=str, default="")
    ap.add_argument("--video-fps", type=int, default=30)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--leader-speed-scale", type=float, default=1.0)
    ap.add_argument("--spawn-in-formation", action="store_true")
    ap.add_argument("--disable-dynamic", action="store_true")
    ap.add_argument("--debug-diamond", action="store_true")
    ap.add_argument("--max-mfe", type=float, default=2.0,
                    help="Fail if any episode mean_form_error >= this.")
    ap.add_argument("--forbid-collision", action="store_true",
                    help="Fail if any collision occurs in any episode.")
    ap.add_argument("--min-sep", type=float)
    ap.add_argument("--obstacle-repulse-gain", type=float)
    ap.add_argument("--static-clear-gain", type=float)
    ap.add_argument("--sep-hysteresis", type=float)
    ap.add_argument("--thrust-delta-scale", type=float)
    ap.add_argument("--max-roll-deg", type=float)
    ap.add_argument("--max-pitch-deg", type=float)
    ap.add_argument("--max-yaw-rate-deg", type=float)
    return ap.parse_args()


def _ts() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _auto_find_vecnorm(model_path: Path) -> Optional[Path]:
    model_dir = model_path.parent
    run_dir = model_dir.parent
    candidates = [
        run_dir / "vecnormalize_final.pkl",
        model_dir / "vecnormalize_final.pkl",
        run_dir / "checkpoints" / "vecnormalize_final.pkl",
        model_dir / "vec_normalize.pkl",
    ]
    for c in candidates:
        if c.exists():
            return c
    ckpt_vecs = sorted((run_dir / "checkpoints").glob("ppo_multi_vecnormalize_*_steps.pkl"))
    return ckpt_vecs[-1] if ckpt_vecs else None


class LivePlotDashboard:
    def __init__(self, num_drones: int):
        self.num_drones = num_drones
        self.tracking_errors = [[] for _ in range(num_drones)]
        self.z_errors = [[] for _ in range(num_drones)]
        self.team_rewards = []
        self.mean_form_err = []
        self.min_dyn_dist = []
        plt.ion()
        self.fig, self.axs = plt.subplots(4, 1, figsize=(8, 9))
        self.fig.tight_layout()

    def update(self, obs_all, desired_positions, reward, min_dyn_distance):
        errs = []
        for i in range(self.num_drones):
            pos = obs_all[i][0:3]
            des = desired_positions[i]
            err = float(np.linalg.norm(pos - des))
            self.tracking_errors[i].append(err)
            self.z_errors[i].append(abs(float(pos[2] - des[2])))
            errs.append(err)
        self.team_rewards.append(float(reward))
        self.mean_form_err.append(float(np.mean(errs)))
        self.min_dyn_dist.append(float(min_dyn_distance))
        self._redraw()

    def _redraw(self):
        for ax in self.axs:
            ax.clear()
        for i in range(self.num_drones):
            self.axs[0].plot(self.tracking_errors[i], label=f"d{i}")
        self.axs[0].set_title("Tracking Error ‖pos - des‖")
        self.axs[0].legend(ncol=5, fontsize=7)
        for i in range(self.num_drones):
            self.axs[1].plot(self.z_errors[i], label=f"d{i}")
        self.axs[1].set_title("Height Error |z - z_des|")
        self.axs[2].plot(self.team_rewards)
        self.axs[2].set_title("Team Reward")
        self.axs[3].plot(self.mean_form_err, label="mean_form_error")
        self.axs[3].plot(self.min_dyn_dist, label="min_dyn_distance")
        self.axs[3].legend(fontsize=8)
        self.axs[3].set_title("Formation vs Sphere Distance")
        plt.pause(0.001)


def main():
    args = parse_args()

    eval_opts = dict(
    leader_speed_scale=args.leader_speed_scale,
    spawn_in_formation=args.spawn_in_formation,
    disable_dynamic=args.disable_dynamic,
    debug_diamond=args.debug_diamond,
    )
    if args.formation_spacing is not None: eval_opts["formation_spacing"] = args.formation_spacing
    if args.min_sep is not None:           eval_opts["min_sep"] = args.min_sep
    if args.sep_hysteresis is not None:    eval_opts["sep_hysteresis"] = args.sep_hysteresis
    if args.thrust_delta_scale is not None:eval_opts["thrust_delta_scale"] = args.thrust_delta_scale
    if args.max_roll_deg is not None:      eval_opts["max_roll_deg"] = args.max_roll_deg
    if args.max_pitch_deg is not None:     eval_opts["max_pitch_deg"] = args.max_pitch_deg
    if args.max_yaw_rate_deg is not None:  eval_opts["max_yaw_rate_deg"] = args.max_yaw_rate_deg
    
    if args.obstacle_repulse_gain is not None:
        eval_opts["obstacle_repulse_gain"] = args.obstacle_repulse_gain
    if args.static_clear_gain is not None:
        eval_opts["static_clear_gain"] = args.static_clear_gain


    print("[EVAL_OPTS]", eval_opts, flush=True)  

    def make_env():
        base = MultiDroneQuadEnv(num_drones=args.num_drones, gui=args.gui)
        return ResetOptionsWrapper(base, eval_opts)

    vec = DummyVecEnv([make_env])

    if args.zero_policy:
        base_env = unwrap_env(vec)
        act_dim = int(np.prod(base_env.action_space.shape))

        def predict(obs, deterministic=True):
            n_env = int(obs.shape[0])
            return np.zeros((n_env, act_dim), dtype=np.float32), None

        model = None
    else:
        model_path = Path(args.model).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        vecnorm_path = Path(args.vecnorm).expanduser().resolve() if args.vecnorm else _auto_find_vecnorm(model_path)
        if vecnorm_path and vecnorm_path.exists():
            print(f"[INFO] Restoring VecNormalize from: {vecnorm_path}")
            vec = VecNormalize.load(str(vecnorm_path), vec)
            vec.training = False
            vec.norm_reward = False
        else:
            print("[WARN] No VecNormalize stats found; evaluating WITHOUT obs normalization.")

        print(f"[INFO] Loading model: {model_path}")
        model = PPO.load(str(model_path), env=vec)

        def predict(obs, deterministic=True):
            return model.predict(obs, deterministic=deterministic)

    os.makedirs("logs", exist_ok=True)
    stamp = _ts()
    csv_path = Path("logs") / f"eval_{stamp}.csv"
    json_path = Path("logs") / f"eval_{stamp}.json"

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    header = ["episode", "ep_step", "global_step", "team_reward", "collision", "mean_form_error", "min_dyn_distance"]
    for i in range(args.num_drones):
        header += [f"d{i}_px", f"d{i}_py", f"d{i}_pz",
                   f"d{i}_des_x", f"d{i}_des_y", f"d{i}_des_z", f"d{i}_track_err"]
    csv_writer.writerow(header)
    print(f"[INFO] CSV → {csv_path}")

    recorder = None
    if args.video:
        try:
            from systems.video_recorder import VideoRecorder
            recorder = VideoRecorder(args.video, fps=args.video_fps)
            print(f"[INFO] Recording video → {args.video} @ {args.video_fps} FPS")
        except Exception as e:
            print(f"[WARN] Video init failed: {e}")

    dashboard = LivePlotDashboard(args.num_drones) if args.plot else None
    dt_sec = 1.0 / max(1e-6, args.fps)

    episodes = []
    ep_metrics = {"mean_form_error": [], "min_dyn_distance": [], "collision": []}
    collisions_total = 0

    step_log = []

    obs = vec.reset()
    base_env = unwrap_env(vec)

    try:
        step = 0
        episode_idx = 0
        while episode_idx < args.episodes and step < args.steps:
            action, _ = predict(obs, deterministic=args.deterministic)
            obs, rewards, dones, infos = vec.step(action)

            reward = float(rewards[0])
            done = bool(dones[0])
            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos

            m = (info0 or {}).get("metrics", {})
            mean_form_error = float(m.get("mean_form_error", np.nan))
            min_dyn_distance = float(m.get("min_dyn_distance", np.inf))
            collided = bool(m.get("collision", 0.0))
            collisions_total += int(collided)

            obs_all = base_env._get_all_obs()
            desired_positions = base_env.get_desired_positions()
            
            row = [episode_idx,                     # which episode
                len(ep_metrics["mean_form_error"]),  # step index within this episode
                step,                             # global step across all episodes
                reward,
                int(collided),
                mean_form_error,
                min_dyn_distance]
            for i in range(args.num_drones):
                pos = obs_all[i][0:3]; des = desired_positions[i]
                err = float(np.linalg.norm(pos - des))
                row.extend([pos[0], pos[1], pos[2], des[0], des[1], des[2], err])
            csv_writer.writerow(row)


            step_log.append({
                "step": step,
                "reward": reward,
                "collision": bool(collided),
                "mean_form_error": mean_form_error,
                "min_dyn_distance": min_dyn_distance,
            })

            if dashboard:
                dashboard.update(obs_all, desired_positions, reward, min_dyn_distance)

            if args.gui:
                for des in desired_positions:
                    top = (des + np.array([0, 0, 0.25], dtype=np.float32)).tolist()
                    p.addUserDebugLine(des.tolist(), top, [0, 1, 0], lifeTime=0.1)

            if recorder:
                leader_pos, _ = p.getBasePositionAndOrientation(base_env.drone_ids[0])
                view = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=leader_pos, distance=3.2, yaw=40, pitch=-25, roll=0, upAxisIndex=2
                )
                proj = p.computeProjectionMatrixFOV(fov=70, aspect=1.0, nearVal=0.01, farVal=30)
                w, h, rgb, *_ = p.getCameraImage(
                    width=720, height=720, viewMatrix=view, projectionMatrix=proj,
                    renderer=p.ER_BULLET_HARDWARE_OPENGL if args.gui else p.ER_TINY_RENDERER
                )
                if hasattr(recorder, "add_frame") and not recorder.add_frame(rgb[:, :, :3]):
                    print("[WARN] Video recorder closed; stopping recording.")
                    recorder = None

            ep_metrics["mean_form_error"].append(mean_form_error)
            ep_metrics["min_dyn_distance"].append(min_dyn_distance)
            ep_metrics["collision"].append(int(collided))

            if done:
                arr_mfe = np.array(ep_metrics["mean_form_error"], dtype=np.float32)
                arr_mdd = np.array(ep_metrics["min_dyn_distance"], dtype=np.float32)
                arr_col = np.array(ep_metrics["collision"], dtype=np.int32)
                epi = {
                    "steps": int(arr_mfe.size),
                    "mfe_mean": float(np.nanmean(arr_mfe)) if arr_mfe.size else float("nan"),
                    "mfe_p90": float(np.nanpercentile(arr_mfe, 90)) if arr_mfe.size else float("nan"),
                    "mdd_min": float(np.nanmin(arr_mdd)) if arr_mdd.size else float("nan"),
                    "collision_any": int(np.any(arr_col > 0)),
                    "collision_rate": float(np.mean(arr_col)) if arr_col.size else 0.0,
                }
                episodes.append(epi)
                ep_metrics = {"mean_form_error": [], "min_dyn_distance": [], "collision": []}
                episode_idx += 1
                if episode_idx >= args.episodes:
                    break

                obs = vec.reset()
                base_env = unwrap_env(vec)

            step += 1
            if args.gui or dashboard or recorder:
                time.sleep(dt_sec)


    except KeyboardInterrupt:
        print("\n[WARN] Interrupted; saving logs...")

    finally:
        summary = {
            "episodes": len(episodes),
            "mfe_mean_avg": float(np.nanmean([e["mfe_mean"] for e in episodes])) if episodes else float("nan"),
            "mdd_min_avg": float(np.nanmean([e["mdd_min"] for e in episodes])) if episodes else float("nan"),
            "collisions_any_total": int(sum(e["collision_any"] for e in episodes)),
            "per_episode": episodes,
            "thresholds": {"max_mfe": args.max_mfe, "forbid_collision": args.forbid_collision},
            "outputs": {"csv": str(csv_path), "json": str(json_path)},
        }
        with open(json_path, "w") as jf:
            json.dump(summary, jf, indent=2)
        print(f"[INFO] JSON → {json_path}")
        csv_file.close()
        print(f"[DONE] Steps: {len(step_log)} | Episodes: {len(episodes)} | Collisions(total steps): {collisions_total}")

        if recorder:
            try:
                recorder.close()
                print(f"[INFO] Video saved → {args.video}")
            except Exception as e:
                print(f"[WARN] Failed to close video: {e}")

        if dashboard:
            print("[INFO] Close the plot window to exit.")
            plt.ioff()
            plt.show()

    fail = False
    if episodes:
        fail |= any(e["mfe_mean"] >= args.max_mfe for e in episodes)
        if args.forbid_collision:
            fail |= any(e["collision_any"] > 0 for e in episodes)

    if fail:
        print("[FAIL] thresholds violated.")
        raise SystemExit(1)
    else:
        print("[PASS] thresholds satisfied.")
        raise SystemExit(0)


if __name__ == "__main__":
    main()
