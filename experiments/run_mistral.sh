#!/bin/bash
#SBATCH --job-name=triage_misrral
#SBATCH --partition=single
#SBATCH --time=5:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=nathalie.kirch.nk@gmail.com
#SBATCH --mail-type=END
#SBATCH --output=logs/run_mistral

conda activate base

python run_mistral.py