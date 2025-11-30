# filepath: tools/watch_and_eval.sh
#!/usr/bin/env bash
set -euo pipefail

export KMP_DUPLICATE_LIB_OK=TRUE   # <<< macOS OpenMP fix

RUNS_DIR="runs/ppo_multi"
EVAL_STEPS=2000
NDRONES=5
SLEEP_SEC=180
TB_LOGDIR="runs/eval_tb"

mkdir -p "$TB_LOGDIR" logs

log_scalar_py='
import sys, time
from torch.utils.tensorboard import SummaryWriter
tag, value, logdir = sys.argv[1], float(sys.argv[2]), sys.argv[3]
w = SummaryWriter(log_dir=logdir)
w.add_scalar(tag, value, global_step=int(time.time()))
w.flush(); w.close()
'

last_ckpt=""

echo "[watch] monitoring $RUNS_DIR"
while true; do
  run_dir=$(ls -td "${RUNS_DIR}"/* 2>/dev/null | head -n1 || true)
  if [ -z "$run_dir" ]; then echo "[watch] no runs yet"; sleep "$SLEEP_SEC"; continue; fi

  ckpt=$(ls -t "$run_dir"/checkpoints/*.zip 2>/dev/null | head -n1 || true)
  if [ -z "$ckpt" ]; then echo "[watch] no checkpoint yet in $run_dir"; sleep "$SLEEP_SEC"; continue; fi

  if [ "$ckpt" != "$last_ckpt" ]; then
    echo "[watch] evaluating $ckpt"
    stamp=$(date +%Y%m%d_%H%M%S)

    # 1) run eval (ignore exit code; thresholds may fail)
    set +e
    # python rl/eval_marl.py \
    #   --model "$ckpt" \
    #   --num-drones "$NDRONES" \
    #   --steps "$EVAL_STEPS" \
    #   --leader-speed-scale 1.0 \
    #   --spawn-in-formation \
    #   --debug-diamond \
    #   --max-mfe 2.0 --forbid-collision \
    #   >/dev/null
    python rl/eval_marl.py \
      --model "$ckpt" \
      --num-drones "$NDRONES" \
      --steps "$EVAL_STEPS" \
      --leader-speed-scale 0.3 \
      --spawn-in-formation \
    #  --disable-dynamic \   # << turn the sphere OFF for Stage-1 eval
      --debug-diamond \
      --max-mfe 2.0 --forbid-collision \
      > /dev/null
    rc=$?
    set -e
    echo "[watch] eval exit code=$rc (ignored)"

    # 2) parse latest JSON
    mfe=$(python - <<'PY'
import json, glob, math
paths=sorted(glob.glob("logs/eval_*.json"))
if not paths: print("nan"); raise SystemExit
with open(paths[-1]) as f: s=json.load(f)
v=s.get("mfe_mean_avg", float("nan"))
print("nan" if (v is None or (isinstance(v,float) and math.isnan(v))) else v)
PY
) || mfe="nan"
    echo "[watch] eval/mfe_mean_avg=$mfe"

    # 3) write TB scalars (uses torch's SummaryWriter; KMP fix is exported above)
    set +e
    python -c "$log_scalar_py" "eval/mfe_mean_avg" "$mfe" "$TB_LOGDIR"
    # optional: also min distance & collisions
    python - <<'PY' "$TB_LOGDIR"
import json, glob, time, sys
from torch.utils.tensorboard import SummaryWriter
logdir=sys.argv[1]
paths=sorted(glob.glob("logs/eval_*.json"))
if not paths: raise SystemExit
with open(paths[-1]) as f: s=json.load(f)
mdd=float(s.get("mdd_min_avg", float("nan")))
col=float(s.get("collisions_any_total", 0.0))
t=int(time.time()); w=SummaryWriter(log_dir=logdir)
w.add_scalar("eval/min_dyn_distance", mdd, t)
w.add_scalar("eval/collisions_any_total", col, t)
w.flush(); w.close()
PY
    set -e

    last_ckpt="$ckpt"
  else
    echo "[watch] newest checkpoint unchanged; sleeping…"
  fi
  sleep "$SLEEP_SEC"
done
