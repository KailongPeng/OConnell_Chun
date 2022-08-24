all scripts written for MATLAB R2016b

for the spatial attention models, run these two scripts:
1. eyetracking/calculate_empirical_gold_and_center_bias_models.m
2. eyetracking/computational_spatial_attention_models.m
these can be run in a normal desktop matlab environment

to run the fMRI analyses, we recommend a cluster environment with slurm 
use the scripts in batch_scripts to launch jobs:
1. cd into the desired analysis directory
2. run 'sbatch job.sh' to submit jobs. 
   note: you will need to modify each job.sh in two places (slurm partition name, matlab import)
3. wait for all jobs to finish before moving to next step.

there are three parts to each analysis: 
1. decoding (transforming fMRI activity into fixation map or CNN space)
2. recon (reconstructing fixation or spatial priority maps from transformed fMRI activity)
3. pred (predicting eye movement patterns using reconstructions)

the progression for each fMRI analysis:

fixation map decoding
1. batch_scripts/fix_map_1_decoding
   runs: fixation_map_reconstruction/fixation_map_direct_decoding.m
2. batch_scripts/fix_map_2_recon
   runs: fixation_map_reconstruction;extract_decoded_fixation_maps.m
3. batch_scripts/fix_map_3_pred
   runs: fixation_map_reconstruction;fix_map_recon_fix_predict.m

model-based decoding
1. batch_scripts/model_based_1_decoding
   runs: model_based_reconstruction/model_based_decoding.m
2. batch_scripts/model_based_average_2_recon    
   runs: model_based_reconstruction/extract_decoded_spatial_priority_maps_average.m
3. batch_scripts/model_based_average_3_pred
   runs: model_based_reconstruction/model_based_recon_fix_predict_average_model.m

model-based decoding defaults to run with the main cnn featured in the paper, vgg16-places365. if you would like to run the analyses for vgg16-ilsvrc or vgg16-face, then there are alternative cost_functions variables at the top of each step's matlab script that you can uncomment. note you will need to adjust the number of slurm_array jobs in the job.sh scripts to account for the additional models.

results/results.m produces Figs. 1 & 2 from the paper. this can also be run in a desktop matlab environment. results files in outputs/computational_model_files and outputs/fixation_prediction are included, so this can be run out-of-the-box

note: the outputs from the decoding and recon steps are quite large. after running all the included analyses, the total size of the experiment directory is 500GB
