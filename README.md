# DaFoEs: Dynamic and Force Estimation System

A deep learning system for estimating forces from visual and force data using Vision Transformers (ViT) and LSTM networks.

## System Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- At least 16GB RAM
- At least 50GB free disk space for datasets and model checkpoints

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DaFoEs.git
cd DaFoEs
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
export DAFOES_DATA=/path/to/your/data  # Optional: defaults to ./data
```

## Data Organization

The data should be organized in the following structure:
```
data/
├── run1/
│   ├── images/
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   └── ...
│   └── forces.csv
├── run2/
└── ...
```

## Running Experiments

### Quick Start

To run all experiments (both Vision-only and Vision+Force):

```bash
chmod +x scripts/run_all_experiments.sh
./scripts/run_all_experiments.sh
```

This will:
1. Train and evaluate Vision-only (ViT) models with sequence lengths 5, 10, and 50
2. Train and evaluate Vision+Force (ViT+LSTM) models with sequence lengths 5, 10, and 50
3. Save all results and models

### Output Structure

```
checkpoints/                            # Saved models
├── vit_vision_seq5/
│   └── best_overall_model.pth
├── vit_lstm_seq5/
│   └── best_overall_model.pth
└── ...

evaluation_results/                     # Evaluation results
├── vit_vision_seq5/
│   ├── forces_run*.csv                # Force predictions
│   ├── run_info.csv                   # Run metadata
│   ├── force_plots/                   # Visualization plots
│   └── evaluation_results.txt         # Metrics summary
└── ...
```

### Experiment Configuration

The default configuration in `scripts/run_all_experiments.sh`:
- Training runs: 1, 2, 3, 4, 5, 7, 8, 11
- Validation runs: 6, 9, 12
- Test runs: 10, 13, 14, 15
- Sequence lengths: 5, 10, 50
- Training epochs: 150
- Batch size: 32

You can modify these parameters in the script as needed.

## Monitoring Experiments

1. **Terminal Output**: Shows real-time progress of each experiment

2. **Weights & Biases**: 
   - Training metrics: `force_estimation_final` project
   - Evaluation metrics: `force_estimation_final_eval` project

3. **Experiment Tracking**:
   - `experiment_tracking.txt`: Detailed logs of all experiments
   - Individual evaluation results in `evaluation_results/*/evaluation_results.txt`

## Results Format

Force predictions are saved in CSV files with the format:
```
run_id,force_type,axis,values
7,predicted,x,1.234;1.236;1.238;...
7,predicted,y,1.345;1.347;1.349;...
7,ground_truth,x,1.232;1.234;1.236;...
...
```

## Troubleshooting

1. **CUDA Out of Memory**:
   - Reduce batch size in `scripts/run_all_experiments.sh`
   - Use shorter sequence lengths

2. **Missing Data**:
   - Ensure data path is set correctly in `DAFOES_DATA`
   - Check data structure matches expected format

3. **Training Issues**:
   - Check GPU availability with `nvidia-smi`
   - Verify all dependencies are installed correctly
