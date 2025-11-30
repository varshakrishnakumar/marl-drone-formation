"""
Stage-aware watcher for eval_marl.py.
- Picks newest run under runs/ppo_multi (or --run path).
- Auto-switches eval flags: sphere OFF only for stage1, ON otherwise.
- Logs scalars to TensorBoard (runs/eval_tb by default).
"""
import argparse, os, sys, time, json, glob, math, subprocess
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception as e:
    print(f"[ERR] TensorBoard writer import failed: {e}")
    print("pip install torch torchvision tensorboard")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ROOT / "runs" / "ppo_multi"
DEFAULT_TB = ROOT / "runs" / "eval_tb"

def newest_run(base: Path) -> Path | None:
    cand = [p for p in base.glob("*") if p.is_dir()]
    return sorted(cand, key=lambda p: p.stat().st_mtime, reverse=True)[0] if cand else None

def newest_ckpt(run_dir: Path) -> Path | None:
    ck = list((run_dir / "checkpoints").glob("*.zip"))
    return sorted(ck, key=lambda p: p.stat().st_mtime, reverse=True)[0] if ck else None

def stage_mode(run_dir: Path) -> str:
    name = run_dir.name.lower()
    if "stage1" in name and "easy" in name:
        return "stage1_easy"
    if "stage2" in name:
        return "stage2_threat"
    if "stage3" in name:
        return "stage3_polish"
    return "threat_on"

def run_eval(model_path: Path, tb_logdir: Path, n_drones: int, steps: int, fps: float,
             deterministic: bool, extra_eval_flags: list[str]) -> dict:
    """Run eval_marl.py, return parsed JSON summary (or {})."""
    cmd = [
        sys.executable, str(ROOT / "rl" / "eval_marl.py"),
        "--model", str(model_path),
        "--num-drones", str(n_drones),
        "--steps", str(steps),
        "--video-fps", "30",
    ] + (["--deterministic"] if deterministic else []) + extra_eval_flags

    print("[watch] eval:", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    print(f"[watch] eval exit code={rc} (ignored)")

    logs = sorted((ROOT / "logs").glob("eval_*.json"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return {}
    try:
        return json.loads(logs[-1].read_text())
    except Exception:
        return {}

def log_scalars(tb_logdir: Path, summary: dict):
    """Write eval metrics to TB."""
    w = SummaryWriter(log_dir=str(tb_logdir))
    t = int(time.time())
    def _get(path, default=float("nan")):
        cur = summary
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    mfe = float(_get(["mfe_mean_avg"], float("nan")))
    mdd = float(_get(["mdd_min_avg"], float("nan")))
    coll = float(_get(["collisions_any_total"], 0.0))

    w.add_scalar("eval/mfe_mean_avg", mfe, t)
    w.add_scalar("eval/min_dyn_distance", mdd, t)
    w.add_scalar("eval/collisions_any_total", coll, t)
    w.flush()
    w.close()
    print(f"[watch] scalars → TB: mfe={mfe:.3f}, min_dyn_dist={mdd:.3f}, collisions_any={coll:g}")

def build_eval_flags(stage: str, leader_speed_scale: float, spawn_in_formation: bool,
                     disable_dynamic_override: bool | None) -> list[str]:
    """Compose eval flags for eval_marl.py (gymnasium side uses dashes)."""
    flags = [
        "--leader-speed-scale", str(leader_speed_scale),
    ]
    if spawn_in_formation:
        flags.append("--spawn-in-formation")

    if disable_dynamic_override is not None:
        dd = disable_dynamic_override
    else:
        dd = (stage == "stage1_easy")

    if dd:
        flags.append("--disable-dynamic")

    flags.append("--debug-diamond")
    flags += ["--max-mfe", "2.0", "--forbid-collision"]
    return flags

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="",
                    help="Run directory to watch (defaults to newest under runs/ppo_multi).")
    ap.add_argument("--tb", type=str, default=str(DEFAULT_TB),
                    help="TensorBoard log dir for eval scalars.")
    ap.add_argument("--poll", type=float, default=120.0, help="Seconds between checks.")
    ap.add_argument("--eval-steps", type=int, default=2000)
    ap.add_argument("--num-drones", type=int, default=5)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--deterministic", action="store_true")

    ap.add_argument("--leader-speed-scale", type=float, default=0.8)
    ap.add_argument("--spawn-in-formation", action="store_true", default=True)
    ap.add_argument("--disable-dynamic", dest="disable_dynamic_override", action="store_true")
    ap.add_argument("--no-disable-dynamic", dest="disable_dynamic_override", action="store_false")
    ap.set_defaults(disable_dynamic_override=None)
    args = ap.parse_args()

    runs_base = DEFAULT_RUNS
    run_dir = Path(args.run).resolve() if args.run else newest_run(runs_base)
    if not run_dir or not run_dir.exists():
        print(f"[ERR] No run directory found (base={runs_base}).")
        sys.exit(2)

    tb_dir = Path(args.tb).resolve()
    tb_dir.mkdir(parents=True, exist_ok=True)

    print(f"[watch] monitoring {run_dir}")
    last_seen: Path | None = None

    while True:
        ckpt = newest_ckpt(run_dir)
        if ckpt and ckpt != last_seen:
            stg = stage_mode(run_dir)
            flags = build_eval_flags(
                stage=stg,
                leader_speed_scale=args.leader_speed_scale,
                spawn_in_formation=args.spawn_in_formation,
                disable_dynamic_override=args.disable_dynamic_override,
            )
            summary = run_eval(
                model_path=ckpt,
                tb_logdir=tb_dir,
                n_drones=args.num_drones,
                steps=args.eval_steps,
                fps=args.fps,
                deterministic=args.deterministic,
                extra_eval_flags=flags,
            )
            if summary:
                log_scalars(tb_dir, summary)
            last_seen = ckpt
        else:
            print("[watch] no new checkpoint; sleeping…")
        time.sleep(max(5.0, float(args.poll)))

if __name__ == "__main__":
    main()
