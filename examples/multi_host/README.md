## single host multi controller

- `multicontroller_vqe.py`: one end-to-end script.

- `pathfinding.py` + `multicontroller_vqe_with_path.py`: path search is separated to save the GPU time.

## multiple host managed by slurm

- `pathfind.py` + `slurm_vqe_with_path.py`: used in a slurm cluster. The slurm batch script is `slurm_submit.sh`.
