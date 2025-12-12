"""
rl/checkpoint_leaderboard.py

Checkpoint leaderboard for PPO models evaluated with rl.eval_runner.

Assumed directory structure:
----------------------------
data/eval_runs/
    checkpoint_A/
        ppo/
            obstacle_field_ppo.csv
    checkpoint_B/
        ppo/
            obstacle_field_ppo.csv
    ...

Each CSV file must come from rl.eval_runner and contain at least:
    - episode
    - collision (0/1 or numeric)
    - mean_form_error
    - ep_len

This script computes, per checkpoint:
    - collision_rate        (primary safety metric)
    - mean_form_error       (accuracy)
    - mean_ep_len           (robustness / survival time)

and prints ranked tables.

Usage examples:
---------------
# Leaderboard for a single scenario
python -m rl.checkpoint_leaderboard \
    --root_dir data/eval_runs \
    --scenario obstacle_field

# If you used a custom root:
python -m rl.checkpoint_leaderboard \
    --root_dir my_eval_runs \
    --scenario obstacle_field
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_results_csv(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.is_file():
        return rows
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(arr: List[Dict], key: str) -> np.ndarray:
    out = []
    for r in arr:
        v = r.get(key, "")
        try:
            out.append(float(v))
        except (ValueError, TypeError):
            out.append(np.nan)
    return np.asarray(out, dtype=np.float32)


def summarize_checkpoint(
    csv_path: Path,
) -> Tuple[float, float, float, int]:
    """
    Returns:
        collision_rate, mean_form_error, mean_ep_len, n_episodes
    """
    rows = load_results_csv(csv_path)
    if not rows:
        return np.nan, np.nan, np.nan, 0

    coll = to_float(rows, "collision")
    form = to_float(rows, "mean_form_error")
    ep_len = to_float(rows, "ep_len")

    mask = ~np.isnan(coll) & ~np.isnan(form) & ~np.isnan(ep_len)
    if np.sum(mask) == 0:
        return np.nan, np.nan, np.nan, 0

    coll = coll[mask]
    form = form[mask]
    ep_len = ep_len[mask]

    collision_rate = float(np.mean(coll > 0.5))
    mean_form_error = float(np.mean(form))
    mean_ep_len = float(np.mean(ep_len))
    n_episodes = coll.size

    return collision_rate, mean_form_error, mean_ep_len, n_episodes


def print_table(title: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        print(f"[WARN] No entries for {title}")
        return

    print(f"\n=== {title} ===")
    # determine column widths
    headers = list(rows[0].keys())
    col_widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(row[h])))

    # header
    header_line = "  ".join(h.ljust(col_widths[h]) for h in headers)
    sep_line = "  ".join("-" * col_widths[h] for h in headers)
    print(header_line)
    print(sep_line)

    # rows
    for row in rows:
        line = "  ".join(str(row[h]).ljust(col_widths[h]) for h in headers)
        print(line)


def main():
    parser = argparse.ArgumentParser(
        description="Checkpoint leaderboard from eval_runner PPO results."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="data/eval_runs",
        help="Root directory under which each checkpoint has its own subfolder.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Scenario name used in eval_runner (e.g., obstacle_field).",
    )
    parser.add_argument(
        "--agent_subdir",
        type=str,
        default="ppo",
        help="Subdirectory for the agent inside each checkpoint folder (default: ppo).",
    )
    parser.add_argument(
        "--export_best",
        action="store_true",
        help="If set, write best checkpoint names (by safety/accuracy/survival) to a JSON/text file in root_dir.",
    )

    args = parser.parse_args()

    root = Path(args.root_dir)
    if not root.is_dir():
        print(f"[ERROR] root_dir does not exist or is not a directory: {root}")
        return

    scenario = args.scenario
    agent_subdir = args.agent_subdir

    # Each immediate subdirectory under root is treated as a separate "checkpoint run"
    checkpoint_dirs = [
        d for d in root.iterdir() if d.is_dir()
    ]
    if not checkpoint_dirs:
        print(f"[ERROR] No checkpoint subdirectories found under {root}")
        return

    summary_rows = []
    for run_dir in checkpoint_dirs:
        # expect: <root>/<checkpoint_name>/<agent_subdir>/<scenario>_ppo.csv
        csv_name = f"{scenario}_ppo.csv"
        csv_path = run_dir / agent_subdir / csv_name
        collision_rate, mean_form_error, mean_ep_len, n_episodes = summarize_checkpoint(csv_path)
        if n_episodes == 0:
            print(f"[WARN] No usable data for checkpoint '{run_dir.name}' (missing or empty {csv_path})")
            continue

        summary_rows.append(
            {
                "checkpoint": run_dir.name,
                "collision_rate": f"{collision_rate:.3f}",
                "mean_form_error": f"{mean_form_error:.3f}",
                "mean_ep_len": f"{mean_ep_len:.1f}",
                "episodes": str(n_episodes),
            }
        )

    if not summary_rows:
        print("[ERROR] No valid checkpoints found with data. Nothing to rank.")
        return

    # ---- Rankings ----

    # 1) Primary: lowest collision rate, then lowest formation error
    rows_safety = sorted(
        summary_rows,
        key=lambda r: (float(r["collision_rate"]), float(r["mean_form_error"])),
    )

    # 2) Best formation error
    rows_accuracy = sorted(
        summary_rows,
        key=lambda r: float(r["mean_form_error"]),
    )

    # 3) Best survival time (mean ep_len)
    rows_survival = sorted(
        summary_rows,
        key=lambda r: -float(r["mean_ep_len"]),
    )

    print_table(
        f"Leaderboard by safety (collision_rate ↑ safety) – scenario={scenario}",
        rows_safety,
    )
    print_table(
        f"Leaderboard by accuracy (mean_form_error ↓ better) – scenario={scenario}",
        rows_accuracy,
    )
    print_table(
        f"Leaderboard by robustness (mean_ep_len ↑ better) – scenario={scenario}",
        rows_survival,
    )
        # ---- Determine winners and optionally export ----
    best_safety = rows_safety[0]
    best_accuracy = rows_accuracy[0]
    best_survival = rows_survival[0]

    print("\n=== Recommended best-model candidates (scenario={}) ===".format(scenario))
    print(f"By safety    (lowest collision_rate, then mean_form_error): {best_safety['checkpoint']}")
    print(f"By accuracy  (lowest mean_form_error):                     {best_accuracy['checkpoint']}")
    print(f"By robustness(highest mean_ep_len):                       {best_survival['checkpoint']}")

    if args.export_best:
        import json

        best_dict = {
            "scenario": scenario,
            "by_safety": {
                "checkpoint": best_safety["checkpoint"],
                "collision_rate": float(best_safety["collision_rate"]),
                "mean_form_error": float(best_safety["mean_form_error"]),
                "mean_ep_len": float(best_safety["mean_ep_len"]),
            },
            "by_accuracy": {
                "checkpoint": best_accuracy["checkpoint"],
                "collision_rate": float(best_accuracy["collision_rate"]),
                "mean_form_error": float(best_accuracy["mean_form_error"]),
                "mean_ep_len": float(best_accuracy["mean_ep_len"]),
            },
            "by_survival": {
                "checkpoint": best_survival["checkpoint"],
                "collision_rate": float(best_survival["collision_rate"]),
                "mean_form_error": float(best_survival["mean_form_error"]),
                "mean_ep_len": float(best_survival["mean_ep_len"]),
            },
        }

        root = Path(args.root_dir)
        json_path = root / f"best_models_{scenario}.json"
        txt_path = root / f"best_models_{scenario}.txt"

        with json_path.open("w") as f:
            json.dump(best_dict, f, indent=2)

        with txt_path.open("w") as f:
            f.write(f"Scenario: {scenario}\n")
            f.write("Best by safety   : {checkpoint}\n".format(**best_safety))
            f.write("Best by accuracy : {checkpoint}\n".format(**best_accuracy))
            f.write("Best by survival : {checkpoint}\n".format(**best_survival))

        print(f"[INFO] Exported best-model summary to {json_path} and {txt_path}")


if __name__ == "__main__":
    main()
