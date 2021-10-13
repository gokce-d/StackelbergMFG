#!/bin/bash
#SBATCH --job-name=tf2-test                 # create a short name for your job
#SBATCH --nodes=1                           # node count
#SBATCH --ntasks=1                          # total number of tasks across all nodes
#SBATCH --cpus-per-task=1                   # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                            # total memory per node (4G per cpu-core is default)
#SBATCH --gres=gpu:tesla_v100:1             # number of gpus per node
#SBATCH --time=10:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-user=gokced@princeton.edu

module purge
module load anaconda3
conda activate tf2-gpu

python main-fbsde-solver_terminal.py
