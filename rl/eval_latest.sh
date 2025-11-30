set -euo pipefail

# Defaults
RUN_DIR="runs/ppo_multi_stage1"
NUM_DRONES=5
EVAL_STEPS=3000
DET="--deterministic"
GUI=""
PLOT=""
PY=python

usage() {
  cat <<EOF
Usage: $(basename "$0") [--run_dir <path>] [--num-drones N] [--steps N] [--gui] [--plot] [--no-det]
Examples:
  $(basename "$0") --run_dir runs/ppo_multi_stage1 --gui --plot
EOF
  exit 1
}

# Args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_dir) RUN_DIR="$2"; shift 2 ;;
    --num-drones) NUM_DRONES="$2"; shift 2 ;;
    --steps) EVAL_STEPS="$2"; shift 2 ;;
    --gui) GUI="--gui"; shift ;;
    --plot) PLOT="--plot"; shift ;;
    --no-det) DET=""; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

CKPT_DIR="$RUN_DIR/checkpoints"
[[ -d "$CKPT_DIR" ]] || { echo "[ERR] No checkpoints dir: $CKPT_DIR"; exit 2; }

# 1) Pick model: prefer best_model.zip, else newest checkpoint zip
if [[ -f "$CKPT_DIR/best_model.zip" ]]; then
  MODEL="$CKPT_DIR/best_model.zip"
else
  MODEL="$(ls -t "$CKPT_DIR"/*.zip 2>/dev/null | head -n1 || true)"
fi
[[ -n "${MODEL:-}" && -f "$MODEL" ]] || { echo "[ERR] No model .zip found in $CKPT_DIR"; exit 3; }

# Extract step count if model is a step checkpoint
STEP_TAG=""
if [[ "$MODEL" =~ _([0-9]+)_steps\.zip$ ]]; then
  STEP_TAG="${BASH_REMATCH[1]}"
fi

# 2) Pick VecNormalize:
# prefer final in run dir, then final in checkpoints, else matching-step vecnorm
VECNORM=""
if [[ -f "$RUN_DIR/vecnormalize_final.pkl" ]]; then
  VECNORM="$RUN_DIR/vecnormalize_final.pkl"
elif [[ -f "$CKPT_DIR/vecnormalize_final.pkl" ]]; then
  VECNORM="$CKPT_DIR/vecnormalize_final.pkl"
elif [[ -n "$STEP_TAG" && -f "$CKPT_DIR/ppo_multi_vecnormalize_${STEP_TAG}_steps.pkl" ]]; then
  VECNORM="$CKPT_DIR/ppo_multi_vecnormalize_${STEP_TAG}_steps.pkl"
else
  # fallback: newest per-step vecnorm
  LATEST_VEC="$(ls -t "$CKPT_DIR"/ppo_multi_vecnormalize_*_steps.pkl 2>/dev/null | head -n1 || true)"
  if [[ -n "$LATEST_VEC" ]]; then VECNORM="$LATEST_VEC"; fi
fi

echo "[INFO] Run dir         : $RUN_DIR"
echo "[INFO] Model           : $MODEL"
if [[ -n "$VECNORM" ]]; then
  echo "[INFO] VecNormalize    : $VECNORM"
else
  echo "[WARN] VecNormalize not found; evaluating WITHOUT obs normalization."
fi
echo "[INFO] Drones/Steps    : $NUM_DRONES / $EVAL_STEPS"
echo

# 3) Run evaluator
CMD=("$PY" "eval_marl.py" \
  --model "$MODEL" \
  --num-drones "$NUM_DRONES" \
  --steps "$EVAL_STEPS" \
  $DET $GUI $PLOT)

# Add vecnorm flag if present
if [[ -n "$VECNORM" ]]; then
  CMD+=(--vecnorm "$VECNORM")
fi

echo "[RUN]" "${CMD[@]}"
exec "${CMD[@]}"
