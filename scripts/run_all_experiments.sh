#!/usr/bin/env bash
# ---------------------------------------------------------------
#  Complete experiment suite for:
#  1. Vision-only with ViT
#  2. Vision+Force with ViT+LSTM
#  
#  This script handles both training and evaluation for all
#  configurations and sequence lengths.
# ---------------------------------------------------------------
set -e  # abort on first error

# Configuration
DATA_ROOT="${DAFOES_DATA:-data}"        # use env var or default to 'data'
TRAIN_RUNS="1 2 3 4 5 7 8 11"                    # training runs
VAL_RUNS="6 9 12"                          # validation runs
TEST_RUNS="10 13 14 15"                       # test runs
SEQ_LENGTHS="5 10 50"                   # sequence lengths to test
EPOCHS=150                              # number of epochs
BATCH=32                                # batch size
WANDB_PROJECT="force_estimation_final"   # W&B project name

# Ensure directories exist
mkdir -p checkpoints
mkdir -p evaluation_results

# Function to run training
train_model() {
    local MODE=$1
    local SEQ_LEN=$2
    local EXP_NAME=$3
    
    echo -e "\n=====================================
Training:
- Mode: $MODE
- Sequence length: $SEQ_LEN
- Experiment: $EXP_NAME
====================================="

    python src/trainwbcv.py \
        --name "$EXP_NAME" \
        --mode "$MODE" \
        --dataset mixed \
        --data-root "$DATA_ROOT" \
        --architecture vit \
        --seq-length "$SEQ_LEN" \
        --batch-size "$BATCH" \
        --epochs "$EPOCHS" \
        --train-force-runs $TRAIN_RUNS \
        --test-force-runs $VAL_RUNS \
        --wandb-project "$WANDB_PROJECT"
}

# Function to run evaluation
evaluate_model() {
    local MODE=$1
    local SEQ_LEN=$2
    local EXP_NAME=$3
    local CKPT="checkpoints/${EXP_NAME}/best_overall_model.pth"
    
    echo -e "\n=====================================
Evaluating:
- Mode: $MODE
- Sequence length: $SEQ_LEN
- Experiment: $EXP_NAME
- Checkpoint: $CKPT
====================================="

    python src/evaluate.py \
        --checkpoint "$CKPT" \
        --mode "$MODE" \
        --runs $TEST_RUNS \
        --architecture vit \
        --seq-length "$SEQ_LEN" \
        --data-root "$DATA_ROOT" \
        --output-dir "evaluation_results/${EXP_NAME}" \
        --wandb-project "${WANDB_PROJECT}_eval" \
        --save-predictions \
        --save-smooth-predictions \
        --save-ground-truth \
        --save-smooth-ground-truth \
        --save-run-info \
        --feature-analysis
}

# Function to run experiment (training + evaluation)
run_experiment() {
    local MODE=$1
    local SEQ_LEN=$2
    local EXP_NAME=$3
    
    echo -e "\n=============================================
Starting experiment:
- Mode: $MODE
- Sequence length: $SEQ_LEN
- Experiment name: $EXP_NAME
============================================="
    
    # Record start time
    local start_time=$(date +%s)
    
    # Run training
    train_model "$MODE" "$SEQ_LEN" "$EXP_NAME"
    
    # Run evaluation
    evaluate_model "$MODE" "$SEQ_LEN" "$EXP_NAME"
    
    # Calculate duration
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(( (duration % 3600) / 60 ))
    
    echo -e "\nExperiment completed in ${hours}h ${minutes}m"
}

# Create experiment tracking file
TRACKING_FILE="experiment_tracking.txt"
echo "Experiment Suite Tracking" > $TRACKING_FILE
echo "=======================" >> $TRACKING_FILE
date >> $TRACKING_FILE
echo -e "\nConfigurations:" >> $TRACKING_FILE
echo "- Training runs: $TRAIN_RUNS" >> $TRACKING_FILE
echo "- Validation runs: $VAL_RUNS" >> $TRACKING_FILE
echo "- Test runs: $TEST_RUNS" >> $TRACKING_FILE
echo "- Sequence lengths: $SEQ_LENGTHS" >> $TRACKING_FILE
echo "- Epochs: $EPOCHS" >> $TRACKING_FILE
echo "- Batch size: $BATCH" >> $TRACKING_FILE
echo -e "\nExperiments:" >> $TRACKING_FILE

# Calculate total experiments
total_experiments=$((2 * $(echo $SEQ_LENGTHS | wc -w)))
current_experiment=0

# Run Vision-only (ViT) experiments
for seq in $SEQ_LENGTHS; do
    ((current_experiment++))
    exp_name="vit_vision_seq${seq}"
    echo -e "\nExperiment $current_experiment/$total_experiments: $exp_name"
    echo -e "\nStarting experiment: $exp_name" >> $TRACKING_FILE
    start_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "Start time: $start_time" >> $TRACKING_FILE
    
    run_experiment "vision" "$seq" "$exp_name"
    
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" >> $TRACKING_FILE
    echo "----------------------------------------" >> $TRACKING_FILE
done

# Run Vision+Force (ViT+LSTM) experiments
for seq in $SEQ_LENGTHS; do
    ((current_experiment++))
    exp_name="vit_lstm_seq${seq}"
    echo -e "\nExperiment $current_experiment/$total_experiments: $exp_name"
    echo -e "\nStarting experiment: $exp_name" >> $TRACKING_FILE
    start_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "Start time: $start_time" >> $TRACKING_FILE
    
    run_experiment "vision+force" "$seq" "$exp_name"
    
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" >> $TRACKING_FILE
    echo "----------------------------------------" >> $TRACKING_FILE
done

echo -e "\n✓ All experiments completed!"
echo -e "\nResults structure:"
echo "checkpoints/"
for seq in $SEQ_LENGTHS; do
    echo "├── vit_vision_seq${seq}/"
    echo "│   └── best_overall_model.pth"
    echo "├── vit_lstm_seq${seq}/"
    echo "│   └── best_overall_model.pth"
done

echo -e "\nevaluation_results/"
for seq in $SEQ_LENGTHS; do
    echo "├── vit_vision_seq${seq}/"
    echo "│   ├── forces_run*.csv"
    echo "│   ├── run_info.csv"
    echo "│   ├── force_plots/"
    echo "│   └── evaluation_results.txt"
    echo "├── vit_lstm_seq${seq}/"
    echo "│   ├── forces_run*.csv"
    echo "│   ├── run_info.csv"
    echo "│   ├── force_plots/"
    echo "│   └── evaluation_results.txt"
done

echo -e "\nForce data format in CSV files:"
echo "run_id,force_type,axis,values"
echo "7,predicted,x,value1;value2;value3;..."
echo "7,predicted,y,value1;value2;value3;..."
echo "7,ground_truth,x,value1;value2;value3;..."
echo "..."

echo -e "\nExperiment tracking saved in: $TRACKING_FILE" 