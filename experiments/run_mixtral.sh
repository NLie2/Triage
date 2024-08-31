#!/bin/bash
#SBATCH --job-name=triage_mixtral
#SBATCH --partition=single
#SBATCH --time=20:00:00
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=nathalie.kirch.nk@gmail.com
#SBATCH --mail-type=END
#SBATCH --output=logs/run_mixtral

conda init
conda activate base

python run_mixtral.py