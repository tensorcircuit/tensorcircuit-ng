#!/usr/bin/env bash
#SBATCH --job-name=bf16-l3
#SBATCH --partition=a100            # adjust to cluster partition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/bf16_%A_%a.out
#SBATCH --error=logs/bf16_%A_%a.err
set -euo pipefail

REPO=${REPO:-$HOME/tensorcircuit-ng}
MODE=${MODE:-end-to-end}            # end-to-end | micro
BACKENDS=${BACKENDS:-jax,pytorch,tensorflow,cupy}
TRIALS=${TRIALS:-5}
GPU=${GPU:-0}

# activate env: E2 unified (ENV=tcng-l3), or E1 per-backend (ENV=tcng-l3-<backend>)
if [[ -n "${CONDA_PREFIX:-}" ]]; then :; else
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "${ENV:-tcng-l3}"
fi

cd "$REPO"
mkdir -p results logs

if [[ "$MODE" == "micro" ]]; then
  python applications/benchmarks/bench_bf16_gpu.py run \
    --backends "$BACKENDS" --mode micro --repo "$REPO" \
    --trials "$TRIALS" --gpu "$GPU" --out "results/micro_${SLURM_ARRAY_TASK_ID:-0}.csv"
else
  python applications/benchmarks/bench_bf16_gpu.py run \
    --backends "$BACKENDS" --mode end-to-end \
    --circuits ghz,brickwork,qaoa-ising \
    --mem-ns 16,18,20,22,24,26 --speed-ns 20,22,24 \
    --repo "$REPO" --trials "$TRIALS" --gpu "$GPU" \
    --timeout 1800 --out "results/e2e_${SLURM_ARRAY_TASK_ID:-0}.csv"
fi
