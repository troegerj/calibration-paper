#!/bin/bash
#SBATCH --partition=gpu_irmb
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=IPINN2D
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:ampere:1

## Build command
## singularity build --fakeroot --force calibrationpaper.sif app/.devcontainer/container.def

SCRIPT=inverse_pinn_2D_linearelasticity_withnoise_4e-04.py

srun singularity run \
 --cleanenv \
 --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
 --nv \
 --nvccli \
 --bind output:/data/output \
 calibrationpaper.sif \
 python /home/davanton/development/CalibrationPaper/AAO/PINN/app/${SCRIPT}
