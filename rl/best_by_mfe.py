
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional

DEFAULT_STEPS = 3000
DEFAULT_NUM_DRONES = 5
DEFAULT_SPEED = 0.8
DEFAULT_MIN_MDD = 1.1

def run_eval(
    model: str,
    steps: int,
    num_drones: int,
    speed: float,
    spawn: bool,
    deterministic: bool,
    disable_dynamic: bool,
    formation_spacing: float | None = None,
) -> dict:
    cmd = [
        sys.executable, "rl/eval_marl.py",
        "--model", model,
        "--steps", str(steps),
        "--num-drones", str(num_drones),
        "--leader-speed-scale", str(speed),
        "--spawn-in-formation" if spawn else "--no-spawn-in-formation",
        "--deterministic" if deterministic else "",
        "--disable-dynamic" if disable_dynamic else "",
        "--max-mfe", "2.0",
        "--forbid-collision",
    ]
    if formation_spacing is not None:
        cmd += ["--formation-spacing", str(formation_spacing)]
    cmd = [c for c in cmd if c]

    subprocess.run(cmd, check=False)

    logs = sorted(glob.glob("logs/eval_*.json"), key=os.path.getmtime)
    if not logs:
        raise RuntimeError("No eval JSON found. Did eval_marl.py run?")
    with open(logs[-1], "r") as f:
        return json.load(f)
    
def find_checkpoints(run_dir: Path) -> List[Path]: 
    ckpt_dir = run_dir / "checkpoints" 
    cks = sorted(ckpt_dir.glob("ppo_multi_*_steps.zip")) 
    bm = ckpt_dir / "best_model.zip" 
    if bm.exists(): 
        cks.append(bm) 
    return cks

def main():
    ap = argparse.ArgumentParser(description="Pick best checkpoint by MFE (zero-collision only).")
    ap.add_argument("run_dir", type=str, help="Path to a training run directory under runs/ppo_multi/...")

    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--num-drones", type=int, default=DEFAULT_NUM_DRONES)

    ap.add_argument("--leader-speed-scale", "--leader_speed_scale", dest="leader_speed_scale",
                    type=float, default=DEFAULT_SPEED)

    ap.add_argument("--min-mdd", type=float, default=DEFAULT_MIN_MDD, help="Min acceptable min_dyn_distance")

    spawn_group = ap.add_mutually_exclusive_group()
    spawn_group.add_argument("--spawn-in-formation", dest="spawn_in_formation", action="store_true")
    spawn_group.add_argument("--no-spawn-in-formation", dest="spawn_in_formation", action="store_false")
    ap.set_defaults(spawn_in_formation=True)

    ap.add_argument("--disable-dynamic", "--disable_dynamic", dest="disable_dynamic", action="store_true")

    ap.add_argument("--formation-spacing", "--formation_spacing", dest="formation_spacing", type=float, default=None)

    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    ckpts = find_checkpoints(run_dir)
    if not ckpts:
        print(f"[ERR] No checkpoints in {run_dir}/checkpoints", file=sys.stderr)
        sys.exit(2)

    print(f"[INFO] Evaluating {len(ckpts)} checkpoints in {run_dir.name}")
    rows: List[Tuple[str, float, float, int]] = []
    for ck in ckpts:
        print(f"  - {ck.name}")
        s = run_eval(
            model=str(ck),
            steps=args.steps,
            num_drones=args.num_drones,
            speed=args.leader_speed_scale,
            spawn=args.spawn_in_formation,
            deterministic=(not args.stochastic),
            disable_dynamic=args.disable_dynamic,
            formation_spacing=args.formation_spacing,
        )
        mfe = float(s.get("mfe_mean_avg", float("nan")))
        mdd = float(s.get("mdd_min_avg", float("nan")))
        col = int(s.get("collisions_any_total", 1))
        rows.append((ck.name, mfe, mdd, col))
        print(f"    mfe={mfe:.3f}  mdd_min={mdd:.3f}  collisions_any={col}")

    out_csv = run_dir / "checkpoints" / "best_by_mfe_scores.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "mfe_mean_avg", "mdd_min_avg", "collisions_any_total"])
        w.writerows(rows)
    print(f"[INFO] Wrote {out_csv}")

    valid = [(n, mfe, mdd, col) for (n, mfe, mdd, col) in rows if col == 0 and (mdd >= args.min_mdd)]
    valid.sort(key=lambda r: (r[1], r[2]))
    if not valid:
        print("[WARN] No zero-collision checkpoint with acceptable min_dyn_distance. See CSV.", file=sys.stderr)
        sys.exit(1)

    best_name = valid[0][0]
    src = run_dir / "checkpoints" / best_name
    dst = run_dir / "checkpoints" / "best_by_mfe.zip"
    import shutil
    shutil.copy2(src, dst)
    print(f"[OK] Best-by-MFE → {dst.name} (from {best_name})")


if __name__ == "__main__":
    main()
