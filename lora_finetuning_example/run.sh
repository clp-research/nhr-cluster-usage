#!/bin/bash
#SBATCH --partition=gpu-a100:test
#SBATCH --job-name=finetune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=run_%j.out
#SBATCH --error=run_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR-EMAIL

# Print job information
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=================================================="
echo ""

# Load required modules
#module purge
#module load HLRNenv
#module load sw.a100
#module load gcc/11.3.0

# Load Python module (adjust version as needed)
# Check available modules with: module avail python
module load python/3.9.21  # or appropriate version

# Set up virtual environment
VENV_DIR="finetune_venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python -m venv $VENV_DIR
    echo "Virtual environment created successfully."
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

pip install --upgrade pip

echo "installing requirements"
pip install torch
pip install transformers
pip install datasets
pip install peft
pip install accelerate
pip install scikit-learn

echo ""
echo "Installation completed!"
echo ""

# Print Python and PyTorch information
echo "Python version:"
python --version
echo ""

echo "PyTorch installation check:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# Print GPU information using nvidia-smi
echo "GPU Information (nvidia-smi):"
nvidia-smi
echo ""

# Run the GPU test script
echo "Running GPU test script..."
echo "=================================================="
python finetune.py


