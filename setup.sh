#!/usr/bin/env bash
# ---------------------------------------------------------------
#  DaFoEs Setup Script
#  This script sets up the environment for DaFoEs
# ---------------------------------------------------------------
set -e  # abort on first error

echo "Setting up environment..."

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python $required_version or higher is required (you have $python_version)"
    exit 1
fi

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: NVIDIA GPU driver not found. Training will be slow without GPU acceleration."
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p data
mkdir -p checkpoints
mkdir -p evaluation_results

# Set up environment variables
echo "Setting up environment variables..."
if [[ "$SHELL" == */zsh ]]; then
    rcfile="$HOME/.zshrc"
elif [[ "$SHELL" == */bash ]]; then
    rcfile="$HOME/.bashrc"
else
    echo "Warning: Unsupported shell. Please manually set DAFOES_DATA environment variable."
    rcfile=""
fi

if [[ -n "$rcfile" ]]; then
    if ! grep -q "export DAFOES_DATA" "$rcfile"; then
        echo "export DAFOES_DATA=$(pwd)/data" >> "$rcfile"
        echo "Added DAFOES_DATA environment variable to $rcfile"
    fi
fi

# Make scripts executable
echo "Making scripts executable..."
chmod +x scripts/*.sh

echo -e "\n✓ Setup completed successfully!"
echo -e "\nTo get started:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo "2. Place your data in the data/ directory"
echo "3. Run experiments:"
echo "   ./scripts/run_all_experiments.sh"
echo -e "\nFor more information, see README.md" 