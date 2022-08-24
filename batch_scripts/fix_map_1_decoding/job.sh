#!/bin/bash
#SBATCH -t 5:59:59
#SBATCH --array=1-121 ### 11 participants * 11 ROIs
# SBATCH --array=1 ### for testing
#SBATCH --partition normal  ### update for your server
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=25GB


echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

module load Apps/Matlab/R2016b ### update for your server

matlab -nosplash -nodisplay -r "cd ../../fixation_map_reconstruction;fixation_map_direct_decoding($SLURM_ARRAY_TASK_ID); quit"
