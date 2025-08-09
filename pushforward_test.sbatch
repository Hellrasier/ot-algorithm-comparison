#!/bin/bash -l
#SBATCH --job-name=pushforward-test
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# ── 0. ALWAYS start clean ────────────────────────────────────────────────
module purge            # no system CUDA modules -> keep MODULEPATH empty

# ── 1. Expose your *own* CUDA/cuDNN install (skip if Conda provides both) ─
export CUDA_HOME="$HOME/cuda-12.6"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# ── 2. Activate Conda env with the Python stack you need ────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py311                # contains CuPy/PyTorch/JAX/etc.

# ── 3. Move to the submit directory and add project root to PYTHONPATH ──
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:$PYTHONPATH"

# ── 4. (Optional) sanity check that CUDA is visible inside the job ──────
nvcc --version  || echo "nvcc not found (OK if you don't need it)"
nvidia-smi      || true

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=1.0

python -m pushforward_test_python
