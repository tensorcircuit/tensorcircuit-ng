#!/usr/bin/env bash
# E1 (fallback): one conda env per backend, if E2 hits CUDA/dep conflicts.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
CUDA=${CUDA:-11.8}
REPO=$(pwd)
for BE in jax pytorch tensorflow cupy; do
  ENV="tcng-l3-$BE"
  conda create -y -n "$ENV" python=3.10
  conda activate "$ENV"
  pip install ml_dtypes cotengra autoray opt_einsum
  case "$BE" in
    jax)        pip install "jax[cuda${CUDA//./}]" ;;
    pytorch)    pip install torch --index-url "https://download.pytorch.org/whl/cu$(( ${CUDA/./} ))" ;;
    tensorflow) pip install tensorflow ;;
    cupy)       pip install cupy-cuda$(( ${CUDA/./} )) ;;
  esac
  pip install -e "$REPO"
  conda deactivate
done
echo "E1 envs tcng-l3-{jax,pytorch,tensorflow,cupy} ready"
