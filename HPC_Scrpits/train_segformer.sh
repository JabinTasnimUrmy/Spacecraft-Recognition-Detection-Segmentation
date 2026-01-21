#!/bin/bash
#SBATCH --job-name="CVIA-Task2"
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 24
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=05:30:00
#SBATCH --output=slurm_%j.out

# --- ENVIRONMENT SETUP ---
# Clean modules
module purge

# Only the essential line to make 'micromamba' work as a command in scripts
eval "$(micromamba shell hook --shell bash)"

# Activate your environment
micromamba activate CVIA

# --- EXECUTION ---
echo "--- Starting process ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Cores: $SLURM_CPUS_PER_TASK"

# Run Python
cd $SCRATCH/spacecraft_eta25/Segmentation/SegFormer
srun python TrainSegformer.py --epochs 200 --patience 20 --suffix Segformer100Epochs --batch_size 16 --custom_loss

echo "--- Process finished at $(date) ---"
