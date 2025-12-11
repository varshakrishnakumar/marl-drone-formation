import os
import csv
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder

from sim.envs.multi_drone_quad_env import MultiDroneQuadEnv


FIELDNAMES = [
    "time_s",
    "leader_pos_x_m","leader_pos_y_m","leader_pos_z_m",
    "leader_vel_x_mps","leader_vel_y_mps","leader_vel_z_mps",
    "leader_roll_rad","leader_pitch_rad","leader_yaw_rad",
    "leader_wx_radps","leader_wy_radps","leader_wz_radps",
    "leader_des_x_m","leader_des_y_m","leader_des_z_m",
    "leader_pos_err_x_m","leader_pos_err_y_m","leader_pos_err_z_m",
    "leader_pos_err_norm_m",
    "leader_roll_cmd_unitless","leader_pitch_cmd_unitless","leader_yaw_rate_cmd_unitless",
    "thrust_cmd","Fz_pd_N","thrust_total_N",
    "followers_mean_pos_err_m","followers_std_pos_err_m",
    "reward",
    "dist_leader_drone1_m","dist_target_leader_drone1_m","too_close_leader_drone1_flag",
    "dist_leader_drone2_m","dist_target_leader_drone2_m","too_close_leader_drone2_flag",
    "dist_leader_drone3_m","dist_target_leader_drone3_m","too_close_leader_drone3_flag",
    "dist_leader_drone4_m","dist_target_leader_drone4_m","too_close_leader_drone4_flag",
    "thrust_cmd_drone0_unitless",
    "thrust_cmd_drone1_unitless",
    "thrust_cmd_drone2_unitless",
    "thrust_cmd_drone3_unitless",
    "thrust_cmd_drone4_unitless",
    "collisions","mean_form_error","max_form_error","mean_z_error",
    "episode","step_in_episode",
]


def make_eval_env(
    num_drones: int,
    leader_speed_scale: float,
    spawn_in_formation: bool,
    disable_dynamic: bool,
    max_steps: int,
):
    def _thunk():
        env = MultiDroneQuadEnv(
            num_drones=num_drones,
            gui=False,
            max_steps=max_steps,
            render_mode="rgb_array",
        )
        eval_opts = dict(
            leader_speed_scale=leader_speed_scale,
            spawn_in_formation=spawn_in_formation,
            disable_dynamic=disable_dynamic,
        )
        env.reset(options=eval_opts)
        return env

    return DummyVecEnv([_thunk])


def run_eval(
    model_path: str,
    vecnorm_path: str,
    csv_path: str,
    video_path: str | None,
    plot_dir: str | None,
    num_episodes: int = 3,
    max_steps: int = 3000,
    num_drones: int = 5,
    leader_speed_scale: float = 0.0,
    spawn_in_formation: bool = True,
    disable_dynamic: bool = True,
):
    """
    Run evaluation for a single (model, vecnorm) pair.
    """

    base_env = make_eval_env(
        num_drones=num_drones,
        leader_speed_scale=leader_speed_scale,
        spawn_in_formation=spawn_in_formation,
        disable_dynamic=disable_dynamic,
        max_steps=max_steps,
    )
    eval_env = VecNormalize.load(vecnorm_path, base_env)
    eval_env.training = False
    eval_env.norm_reward = False

    # Enable trace logging inside the env if available
    if hasattr(eval_env, "venv") and hasattr(eval_env.venv, "envs"):
        for e in eval_env.venv.envs:
            if hasattr(e, "enable_trace"):
                e.enable_trace = True

    # Optional video
    if video_path is not None:
        video_folder = os.path.dirname(video_path)
        Path(video_folder).mkdir(parents=True, exist_ok=True)
        name_prefix = os.path.splitext(os.path.basename(video_path))[0]

        eval_env = VecVideoRecorder(
            eval_env,
            video_folder=video_folder,
            record_video_trigger=lambda step: step == 0,
            video_length=max_steps,
            name_prefix=name_prefix,
        )

    model = PPO.load(model_path, env=eval_env)

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    times = []
    mean_fe = []
    max_fe = []
    z_err = []
    collisions = []

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        obs = eval_env.reset()
        for ep in range(num_episodes):
            done = False
            step_in_ep = 0

            while not done and step_in_ep < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = eval_env.step(action)

                info = infos[0]
                trace = info.get("trace_row")

                # --- always create a row so CSV is never empty ---
                row = {k: np.nan for k in FIELDNAMES}
                row["episode"] = ep
                row["step_in_episode"] = step_in_ep
                row["reward"] = float(rewards[0])

                if trace is not None:
                    # Fill in any fields that are present in trace_row
                    for k, v in trace.items():
                        if k in row:
                            row[k] = v

                    t = trace.get("time_s", float(step_in_ep))
                    times.append(t)
                    mean_fe.append(trace.get("mean_form_error", np.nan))
                    max_fe.append(trace.get("max_form_error", np.nan))
                    z_err.append(trace.get("mean_z_error", np.nan))
                    collisions.append(trace.get("collisions", np.nan))
                else:
                    # Fallback: approximate time and collisions from what we have
                    times.append(float(step_in_ep))
                    mean_fe.append(np.nan)
                    max_fe.append(np.nan)
                    z_err.append(np.nan)
                    collisions.append(float(info.get("collisions", 0.0)))

                writer.writerow(row)

                done = bool(dones[0])
                step_in_ep += 1
            obs = eval_env.reset()

    eval_env.close()

    if plot_dir is not None:
        import matplotlib.pyplot as plt

        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        times_arr = np.array(times)
        mean_fe_arr = np.array(mean_fe)
        max_fe_arr = np.array(max_fe)
        z_err_arr = np.array(z_err)
        coll_arr = np.array(collisions)

        prefix = Path(csv_path).stem
        title_prefix = Path(model_path).stem

        # Formation error plot
        plt.figure()
        plt.plot(times_arr, mean_fe_arr, label="mean FE")
        plt.plot(times_arr, max_fe_arr, label="max FE")
        plt.axhline(2.0, linestyle="--", label="target=2.0")
        plt.xlabel("time (s)")
        plt.ylabel("formation error (m)")
        plt.title(f"{title_prefix} eval: formation error vs time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{prefix}_formation_error.png", dpi=200)
        plt.close()

        # Altitude error plot
        plt.figure()
        plt.plot(times_arr, z_err_arr)
        plt.xlabel("time (s)")
        plt.ylabel("mean |z - target_z| (m)")
        plt.title(f"{title_prefix} eval: altitude error vs time")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{prefix}_z_error.png", dpi=200)
        plt.close()

        # Collision flag plot
        plt.figure()
        plt.plot(times_arr, coll_arr, drawstyle="steps-post")
        plt.xlabel("time (s)")
        plt.ylabel("collision flag")
        plt.title(f"{title_prefix} eval: collisions")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{prefix}_collisions.png", dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser()

    # Mutually exclusive: single-model or multi-run mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        type=str,
        help="Path to a single PPO model .zip to evaluate",
    )
    group.add_argument(
        "--runs_root",
        type=str,
        help="Root directory containing multiple run subfolders to auto-evaluate",
    )

    parser.add_argument("--vecnorm", type=str, help="VecNormalize .pkl (single-model mode)")

    # Single-model outputs
    parser.add_argument("--csv_out", type=str, help="CSV output path (single-model mode)")
    parser.add_argument(
        "--video_out",
        type=str,
        default=None,
        help="Optional video output path (single-model mode)",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        help="Directory for plots (single-model mode)",
    )

    # Multi-run outputs
    parser.add_argument(
        "--csv_root",
        type=str,
        default="logs/eval_csv",
        help="Root directory for CSVs when using --runs_root",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default="logs/eval_videos",
        help="Root directory for videos when using --runs_root",
    )
    parser.add_argument(
        "--plot_root",
        type=str,
        default="logs/eval_plots",
        help="Root directory for plots when using --runs_root",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="If set, record a video for each model in multi-run mode",
    )

    # Common eval options
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--num_drones", type=int, default=5)

    # Env options (can tune per stage if you want)
    parser.add_argument("--leader_speed_scale", type=float, default=0.0)
    parser.add_argument(
        "--spawn_in_formation",
        action="store_true",
        default=True,
        help="Spawn followers already in formation",
    )
    parser.add_argument(
        "--no_spawn_in_formation",
        dest="spawn_in_formation",
        action="store_false",
        help="Disable spawn_in_formation",
    )
    parser.add_argument(
        "--disable_dynamic",
        action="store_true",
        default=True,
        help="Disable dynamic threats/obstacles",
    )
    parser.add_argument(
        "--enable_dynamic",
        dest="disable_dynamic",
        action="store_false",
        help="Enable dynamic threats/obstacles",
    )

    args = parser.parse_args()

    # ---------- SINGLE-MODEL MODE ----------
    if args.model is not None:
        if args.vecnorm is None:
            parser.error("--vecnorm is required when using --model")
        if args.csv_out is None:
            parser.error("--csv_out is required when using --model")
        if args.plot_dir is None:
            parser.error("--plot_dir is required when using --model")

        run_eval(
            model_path=args.model,
            vecnorm_path=args.vecnorm,
            csv_path=args.csv_out,
            video_path=args.video_out,
            plot_dir=args.plot_dir,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            num_drones=args.num_drones,
            leader_speed_scale=args.leader_speed_scale,
            spawn_in_formation=args.spawn_in_formation,
            disable_dynamic=args.disable_dynamic,
        )
        return

    # ---------- MULTI-RUN MODE ----------
    runs_root = Path(args.runs_root)
    csv_root = Path(args.csv_root)
    video_root = Path(args.video_root)
    plot_root = Path(args.plot_root)

    csv_root.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)
    if args.record_video:
        video_root.mkdir(parents=True, exist_ok=True)

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue

        run_name = run_dir.name

        # Find VecNormalize
        vecnorm_path = None
        for cand in ["vecnormalize_final.pkl", "vecnormalize_interrupt.pkl", "vecnormalize.pkl"]:
            p = run_dir / cand
            if p.exists():
                vecnorm_path = str(p)
                break

        if vecnorm_path is None:
            print(f"[SKIP] {run_name}: no vecnormalize .pkl found")
            continue

        # Find model to evaluate
        model_path = None
        for cand in ["final_model.zip", "interrupt_model.zip", "best_model.zip"]:
            p = run_dir / cand
            if p.exists():
                model_path = str(p)
                break

        # Fallback: latest checkpoint
        if model_path is None:
            ckpt_dir = run_dir / "checkpoints"
            if ckpt_dir.exists():
                ckpts = sorted(ckpt_dir.glob("*.zip"))
                if ckpts:
                    model_path = str(ckpts[-1])

        if model_path is None:
            print(f"[SKIP] {run_name}: no model zip or checkpoints found")
            continue

        base_tag = f"{run_name}_{Path(model_path).stem}"

        csv_out = csv_root / f"{base_tag}.csv"
        video_out = None
        if args.record_video:
            video_out = video_root / f"{base_tag}.mp4"

        print(f"[INFO] Evaluating {base_tag}")
        run_eval(
            model_path=model_path,
            vecnorm_path=vecnorm_path,
            csv_path=str(csv_out),
            video_path=str(video_out) if video_out is not None else None,
            plot_dir=str(plot_root),
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            num_drones=args.num_drones,
            leader_speed_scale=args.leader_speed_scale,
            spawn_in_formation=args.spawn_in_formation,
            disable_dynamic=args.disable_dynamic,
        )

    print("[DONE] Multi-run evaluation complete.")


if __name__ == "__main__":
    main()
