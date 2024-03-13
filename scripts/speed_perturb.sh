#!/usr/bin/env bash
#SBATCH --array=1-100
#SBATCH --job-name=speed_perturb
#SBATCH --partition=<NAME-OF-PARTITION>
#SBATCH --cpus-per-task=1
#SBATCH --error=<PATH-TO-THE-REPO>/logs/speed_perturb.err
#SBATCH --output=<PATH-TO-THE-REPO>/logs/speed_perturb.out

# INSTRUCTIONS TO USE THIS SCRIPT
# - in the SBATCH parameters above, replace <NAME-OF-PARTITION> with the name of your SLURM partition
# - in the SBATCH parameters above, replace <PATH-TO-THE-REPO> with the path to MMCSG repository
# - below, replace $PATH_TO_MINICODA with path to your miniconda
# - below, replace $PATH_TO_THE_REPO with the path to the MMCSG repository
# - submit this script to SLURM using `sbatch speed_perturb.sh`

# IF YOUR COMPUTATIONAL CLUSTER DIFFERS
# - please create a script to correctly submit to your computational cluster
# - the script should call the Python script as below, 
#   specifying the index of partition and number of overall partitions 

. $PATH_TO_MINICONDA/etc/profile.d/conda.sh && conda deactivate && conda activate chime_recipe_2

cd $PATH_TO_THE_REPO

task_id=$((SLURM_ARRAY_TASK_ID-1))

python -m scripts.speed_perturb speed_perturb.i_split=${task_id} speed_perturb.n_splits=100
