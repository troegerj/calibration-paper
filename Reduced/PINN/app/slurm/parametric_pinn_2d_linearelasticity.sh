#!/bin/bash
#SBATCH --partition=gpu_irmb
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --job-name=PPINN2D
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:ampere:1

## Build command
## singularity build --fakeroot --force calibrationpaper.sif app/.devcontainer/container.def

SCRIPT=parametric_pinn_2d_linearelasticity.py

srun singularity run \
 --cleanenv \
 --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
 --nv \
 --nvccli \
 calibrationpaper.sif \
 python3 /home/davanton/development/CalibrationPaper/Reduced/PINN/app/${SCRIPT}