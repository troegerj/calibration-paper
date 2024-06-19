#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=CalPaper
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

##singularity build --fakeroot --force calibrationpinn.sif app/.devcontainer/calibrationpinn.def

SCRIPT=inverse_pinn_2D_linearelasticity_withnoise_4e-04.py

srun singularity run \
 --cleanenv \
 --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
 --nv \
 --nvccli \
 --bind output:/data/output \
 calibrationpaper.sif \
 python3 /home/davanton/development/CalibrationPaper/AAO/PINN/app/${SCRIPT}