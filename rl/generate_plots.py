"""
rl/generate_plots.py

Create report-ready figures from evaluation results produced by rl.eval_runner.

Figures (per scenario):
    - formation error vs "time" (episode index) with mean ± std band
    - collision probability (bar plot)
    - survival curve (episode length distribution)
    - safety vs accuracy scatter (mean_form_error vs min_dyn_distance)

Usage example
-------------
python -m rl.generate_plots \
    --results_dir data/eval_runs \
    --figures_dir figures \
    --scenario obstacle_field \
    --agents ppo pid
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import csv


def load_results_csv(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.is_file():
        print(f"[WARN] Missing results file: {path}")
        return rows

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(arr, key: str) -> np.ndarray:
    out = []
    for r in arr:
        v = r.get(key, "")
        try:
            out.append(float(v))
        except (ValueError, TypeError):
            out.append(np.nan)
    return np.asarray(out, dtype=np.float32)


def formation_error_vs_episode(
    data_by_agent: Dict[str, List[Dict]],
    scenario: str,
    out_dir: Path,
    canonical_path: Path | None = None,
):
    """
    Plot mean_form_error vs episode index with ±1 std band for each agent.
    This is a proxy for "vs time": episode index is the horizontal axis.
    """
    if not data_by_agent:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    for agent, rows in data_by_agent.items():
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: int(r["episode"]))
        mean_err = to_float(rows_sorted, "mean_form_error")
        episodes = np.arange(len(mean_err))

        mu = np.nanmean(mean_err)
        sigma = np.nanstd(mean_err)

        plt.plot(episodes, mean_err, label=f"{agent} (per-episode)")
        plt.hlines(mu, episodes[0], episodes[-1], linestyles="dashed", label=f"{agent} mean")
        plt.fill_between(
            episodes,
            mu - sigma,
            mu + sigma,
            alpha=0.2,
        )

    plt.xlabel("Episode index")
    plt.ylabel("Mean formation error [m]")
    plt.title(f"Formation error vs episode – {scenario}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{scenario}_formation_error_vs_episode.png", dpi=200)
    if canonical_path is not None:
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(canonical_path, format="pdf", dpi=300)
    plt.close()
    print(f"[INFO] Saved formation-error figure for {scenario} -> {out_dir}")


def collision_probability_bar(
    data_by_agent: Dict[str, List[Dict]],
    scenario: str,
    out_dir: Path,
    canonical_path: Path | None = None,
):
    """
    Bar plot of collision probability per agent for a given scenario.
    """
    if not data_by_agent:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    agents = []
    probs = []
    for agent, rows in data_by_agent.items():
        if not rows:
            continue
        col = to_float(rows, "collision")
        prob = float(np.nanmean(col > 0.5))
        agents.append(agent)
        probs.append(prob)

    if not agents:
        return

    x = np.arange(len(agents))

    plt.figure()
    plt.bar(x, probs)
    plt.xticks(x, agents)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Collision probability")
    plt.title(f"Collision rate – {scenario}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{scenario}_collision_probability.png", dpi=200)
    if canonical_path is not None:
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(canonical_path, format="pdf", dpi=300)
    plt.close()
    print(f"[INFO] Saved collision-probability figure for {scenario} -> {out_dir}")


def survival_curve(
    data_by_agent: Dict[str, List[Dict]],
    scenario: str,
    out_dir: Path,
    canonical_path: Path | None = None,
):
    """
    Plot survival curves (P(ep_len >= t)) per agent.
    Uses episode length as "time-to-failure" proxy.
    """
    if not data_by_agent:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()

    for agent, rows in data_by_agent.items():
        if not rows:
            continue
        ep_len = to_float(rows, "ep_len")
        ep_len = ep_len[~np.isnan(ep_len)]
        if ep_len.size == 0:
            continue

        # sort lengths and build empirical survival curve
        t_sorted = np.sort(ep_len)
        n = len(t_sorted)
        surv = 1.0 - np.arange(n) / float(n)

        plt.step(t_sorted, surv, where="post", label=agent)

    plt.xlabel("Episode length [steps]")
    plt.ylabel("Survival probability")
    plt.title(f"Survival curve – {scenario}")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{scenario}_survival_curve.png", dpi=200)
    if canonical_path is not None:
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(canonical_path, format="pdf", dpi=300)
    plt.close()
    print(f"[INFO] Saved survival curve for {scenario} -> {out_dir}")


def safety_vs_accuracy_scatter(
    data_by_agent: Dict[str, List[Dict]],
    scenario: str,
    out_dir: Path,
    canonical_path: Path | None = None,
):
    """
    Scatter plot: x = mean_form_error (accuracy), y = min_dyn_distance (safety).
    Each point is an episode; different colors per agent.
    """
    if not data_by_agent:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "ppo": "tab:blue",
        "pid": "tab:orange",
    }

    plt.figure()
    for agent, rows in data_by_agent.items():
        if not rows:
            continue
        acc = to_float(rows, "mean_form_error")
        saf = to_float(rows, "min_dyn_distance")
        mask = ~np.isnan(acc) & ~np.isnan(saf)
        if np.sum(mask) == 0:
            continue
        plt.scatter(
            acc[mask],
            saf[mask],
            alpha=0.5,
            label=agent,
            s=20,
            # color optional; map known agents for readability
            c=colors.get(agent, None),
        )

    plt.xlabel("Mean formation error [m] (↓ better)")
    plt.ylabel("Min. dynamic obstacle distance [m] (↑ safer)")
    plt.title(f"Safety vs accuracy – {scenario}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{scenario}_safety_vs_accuracy.png", dpi=200)
    if canonical_path is not None:
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(canonical_path, format="pdf", dpi=300)
    plt.close()
    print(f"[INFO] Saved safety-vs-accuracy scatter for {scenario} -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate report-ready plots from eval_runner results."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="data/eval_runs",
        help="Directory where eval_runner stored PPO/PID CSVs.",
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default="figures",
        help="Directory to store generated figures.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Scenario name (hover_formation, leader_tracking, obstacle_field, or 'all').",
    )
    parser.add_argument(
        "--agents",
        type=str,
        nargs="+",
        default=["ppo", "pid"],
        help="Which agents to include (subset of: ppo, pid).",
    )
    parser.add_argument(
        "--primary_scenario",
        type=str,
        default="obstacle_field",
        help="Scenario whose figures become the canonical report figures.",
    )
    parser.add_argument(
        "--canonical_dir",
        type=str,
        default="docs/figures",
        help="Directory for drop-in report figures (canonical names).",
    )
    parser.add_argument(
        "--write_canonical",
        action="store_true",
        help="If set, also write canonical PDFs (fig_*.pdf) for primary_scenario.",
    )


    args = parser.parse_args()

    results_root = Path(args.results_dir)
    figs_root = Path(args.figures_dir)

    scenarios = []
    if args.scenario == "all":
        # look for subfiles like "<scenario>_ppo.csv" under ppo/
        ppo_dir = results_root / "ppo"
        if ppo_dir.is_dir():
            for csv_path in ppo_dir.glob("*_ppo.csv"):
                name = csv_path.stem.replace("_ppo", "")
                scenarios.append(name)
        scenarios = sorted(set(scenarios))
    else:
        scenarios = [args.scenario]

    if not scenarios:
        print("[WARN] No scenarios found. Did you run rl.eval_runner first?")
        return

    for scen in scenarios:
        print(f"[INFO] Generating plots for scenario: {scen}")
        data_by_agent: Dict[str, List[Dict]] = {}

        for agent in args.agents:
            agent_dir = results_root / agent
            if agent == "ppo":
                csv_path = agent_dir / f"{scen}_ppo.csv"
            elif agent == "pid":
                csv_path = agent_dir / f"{scen}_pid.csv"
            else:
                csv_path = agent_dir / f"{scen}_{agent}.csv"

            rows = load_results_csv(csv_path)
            if rows:
                data_by_agent[agent] = rows

        if not data_by_agent:
            print(f"[WARN] No data for scenario '{scen}'. Skipping.")
            continue

        scen_fig_dir = figs_root / scen

        # Decide if this scenario should also update canonical report figures
        canonical_dir = Path(args.canonical_dir)
        use_canonical = args.write_canonical and (scen == args.primary_scenario)

        canon_form = canonical_dir / "fig_formation_error.pdf"      if use_canonical else None
        canon_coll = canonical_dir / "fig_collision_rate.pdf"       if use_canonical else None
        canon_surv = canonical_dir / "fig_survival_curve.pdf"       if use_canonical else None
        canon_scat = canonical_dir / "fig_safety_vs_accuracy.pdf"   if use_canonical else None

        formation_error_vs_episode(
            data_by_agent,
            scen,
            scen_fig_dir,
            canonical_path=canon_form,
        )
        collision_probability_bar(
            data_by_agent,
            scen,
            scen_fig_dir,
            canonical_path=canon_coll,
        )
        survival_curve(
            data_by_agent,
            scen,
            scen_fig_dir,
            canonical_path=canon_surv,
        )
        safety_vs_accuracy_scatter(
            data_by_agent,
            scen,
            scen_fig_dir,
            canonical_path=canon_scat,
        )



if __name__ == "__main__":
    main()
