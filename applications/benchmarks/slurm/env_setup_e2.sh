#!/usr/bin/env bash
# E2: single unified conda env with jax+torch+tf+cupy GPU stacks on one CUDA major.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
ENV=${1:-tcng-l3}
CUDA=${CUDA:-11.8}   # pin one CUDA major; adjust to cluster default
REPO=$(pwd)
conda create -y -n "$ENV" python=3.10
conda activate "$ENV"
# pin-compatible GPU stacks (example pins; adjust to cluster CUDA)
pip install "jax[cuda${CUDA//./}]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install torch --index-url "https://download.pytorch.org/whl/cu$(( ${CUDA/./} ))"
pip install tensorflow cupy-cuda$(( ${CUDA/./} ))
pip install ml_dtypes cotengra autoray opt_einsum
pip install -e "$REPO"
echo "E2 env '$ENV' ready; smoke-test with: python -c 'import jax,torch,tensorflow,cupy'"
