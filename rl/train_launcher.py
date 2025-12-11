
import argparse
import os
import sys
import yaml
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent

def _resolve(path_like: str | Path) -> str:
    p = Path(path_like)
    if p.is_absolute():
        return str(p)
    cwd_candidate = (Path.cwd() / p).resolve()
    return str(cwd_candidate if cwd_candidate.exists() else (HERE / p).resolve())

def _load_yaml(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f) or {}

def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update({k: v for k, v in override.items() if v is not None})
    return out

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def run_stage(train_script: str, stage_cfg: Dict[str, Any], extra_cli: List[str], env: Dict[str, str]) -> None:
    flags = _flatten_flags(stage_cfg)
    cmd = [sys.executable, train_script, *flags, *extra_cli]
    print("\n[LAUNCH]", " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _flatten_flags(cfg: Dict[str, Any]) -> List[str]:
    """Map a flat dict to CLI flags. Keep underscores to match argparse."""
    flags: List[str] = []
    for k, v in cfg.items():
        key = f"--{k}"
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                flags.append(key)
        elif isinstance(v, (list, tuple)):
            for item in v:
                flags.extend([key, str(item)])
        else:
            flags.extend([key, str(v)])
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="systems/ppo_multi.yaml")
    ap.add_argument("--train_script", type=str, default="train_multi.py")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, help="CLI overrides appended to each stage")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cfg_path = _resolve(args.config)
    train_script = _resolve(args.train_script)
    if not Path(cfg_path).exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not Path(train_script).exists():
        raise FileNotFoundError(f"Train script not found: {train_script}")

    base_cfg = _load_yaml(cfg_path)

    env = os.environ.copy()
    env_cfg = base_cfg.get("env")
    if isinstance(env_cfg, dict):
        for k, v in env_cfg.items():
            env[str(k)] = str(v)

    stages: List[Tuple[str, Dict[str, Any]]] = []
    if "stages" in base_cfg:
        common = base_cfg.get("common", {})
        for i, st in enumerate(base_cfg["stages"]):
            name = st.get("name", f"stage{i+1}")
            merged = _merge(common, {k: v for k, v in st.items() if k != "name"})
            stages.append((name, merged))
    else:
        stages.append(("stage", base_cfg))

    extra_cli = args.extra or []

    prev_log_dir: str | None = None
    for idx, (stage_name, scfg) in enumerate(stages, start=1):
        log_dir = scfg.get("log_dir", "runs/ppo_multi")
        auto_ts = bool(scfg.pop("auto_timestamp", True))
        resume_from_prev = bool(scfg.pop("resume_from_prev", idx > 1))

        if auto_ts:
            log_dir = str(Path(log_dir) / f"{_timestamp()}_{stage_name}")
        scfg["log_dir"] = log_dir

        # Resolve any user-provided load/load_vecnorm early
        for key in ("load", "load_vecnorm"):
            if key in scfg and scfg[key] is not None:
                scfg[key] = _resolve(scfg[key])

        # Allow CUDA device via YAML
        if "cuda_visible_devices" in scfg:
            env["CUDA_VISIBLE_DEVICES"] = str(scfg.pop("cuda_visible_devices"))

        print(f"\n=== Stage {idx}: {stage_name} ===")
        print(f"config: {cfg_path}")
        print(f"train_script: {train_script}")
        print(f"log_dir: {log_dir}")

        # Auto-resume from previous stage if requested
        if resume_from_prev and prev_log_dir:
            ckpt_dir = Path(prev_log_dir) / "checkpoints"
            load_path = None

            # Prefer best_by_mfe, fallback to best_model, then last ppo_multi_*_steps.zip
            for name in ("best_by_mfe.zip", "best_model.zip"):
                p = ckpt_dir / name
                if p.exists():
                    load_path = p
                    break
            if load_path is None:
                steps_zips = sorted(ckpt_dir.glob("ppo_multi_*_steps.zip"))
                if steps_zips:
                    load_path = steps_zips[-1]

            if load_path:
                scfg["load"] = _resolve(load_path)
                print(f"resume: --load {scfg['load']}")
            else:
                print("[WARN] No checkpoint in previous stage; starting fresh.")

            # Try to find vecnormalize stats from previous stage
            for v in [
                Path(prev_log_dir) / "vecnormalize_final.pkl",
                ckpt_dir / "vecnormalize_final.pkl",
            ]:
                if v.exists():
                    scfg["load_vecnorm"] = _resolve(v)
                    break

        if args.dry_run:
            print("[DRY RUN] Flags:", " ".join(_flatten_flags(scfg)))
        else:
            run_stage(train_script, scfg, extra_cli, env)

        prev_log_dir = log_dir

    print("\n[OK] All stages finished.")

if __name__ == "__main__":
    main()
