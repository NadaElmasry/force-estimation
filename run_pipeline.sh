#!/usr/bin/env bash
# ---------------------------------------------------------------
#  run_pipeline.sh  —  sequential Train→Eval→Train→Eval on RunPod
# ---------------------------------------------------------------
set -e                                  # abort on first error
set -o pipefail
export WANDB_MODE=offline               # optional: disable WANDB sync if no internet

# --------------------------  USER CONFIG  ----------------------
DATA_ROOT="/runpod/data"                # where your roll_out/ & images/ live
GPU="${CUDA_VISIBLE_DEVICES:-0}"        # honour existing CUDA env, else GPU0
EPOCHS=50
BATCH=32
TRAIN_RUNS="1 2 3 4 5 7 8 11"
VAL_RUNS="6 9 12"
TEST_RUNS="10 13 14 15"                 # will be evaluated afterwards
# -----------------------  helper functions  --------------------
train () {
  local EXP=$1
  local SEQ=$2
  echo -e "\n================  TRAIN  ($EXP)  ================\n"
  python right_camera_force_estimator.py \
      --name "$EXP" \
      --dataset mixed \
      --architecture vit \
      --type vs \
      --recurrency \
      --seq-length "$SEQ" \
      --batch-size "$BATCH" \
      --epochs "$EPOCHS" \
      --train-force-runs $TRAIN_RUNS \
      --test-force-runs  $VAL_RUNS \
      --use_custom \
      --gpu "$GPU"
}

evaluate () {
  local EXP=$1
  local SEQ=$2
  local CKPT="checkpoints/${EXP}/best_overall_model.pth"
  echo -e "\n================  EVAL  ($EXP)  ================\n"
  python evaluate_force_estimator.py \
      --checkpoint "$CKPT" \
      --runs $TEST_RUNS \
      --architecture vit \
      --type vs \
      --recurrency \
      --seq-length "$SEQ" \
      --dataset mixed \
      --data-root "$DATA_ROOT" \
      --device "cuda:$GPU"
}

# -----------------------  run the pipeline  --------------------
train  "vit_vs_len5" 5
evaluate "vit_vs_len5" 5

train  "vit_vs_len10" 10
evaluate "vit_vs_len10" 10

echo -e "\n[✓]  Pipeline finished — results in checkpoints/ and eval_outputs/\n"
