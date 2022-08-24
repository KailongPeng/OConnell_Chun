#!/bin/bash
#SBATCH -t 5:59:59
#SBATCH --array=1-11 ### 1 network * 11 ROIs
# SBATCH --array=1 ### for testing
#SBATCH --partition normal  ### update for your server
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=25GB


echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

module load Apps/Matlab/R2016b ### update for your server

matlab -nosplash -nodisplay -r "cd ../../model_based_reconstruction;extract_decoded_spatial_priority_maps_average($SLURM_ARRAY_TASK_ID); quit"
