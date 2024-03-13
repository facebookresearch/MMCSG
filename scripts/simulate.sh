#!/usr/bin/env bash
#SBATCH --array=1-100
#SBATCH --job-name=simulate_data
#SBATCH --partition=<NAME-OF-PARTITION>
#SBATCH --cpus-per-task=1
#SBATCH --error=<PATH-TO-THE-REPO>/logs/simulate.err
#SBATCH --output=<PATH-TO-THE-REPO>/logs/simulate.out

# INSTRUCTIONS TO USE THIS SCRIPT
# - in the SBATCH parameters above, replace <NAME-OF-PARTITION> with the name of your SLURM partition
# - in the SBATCH parameters above, replace <PATH-TO-THE-REPO> with the path to MMCSG repository
# - below, replace $PATH_TO_MINICODA with path to your miniconda
# - below, replace $PATH_TO_THE_REPO with the path to the MMCSG repository
# - submit this script to SLURM using `sbatch simulate.sh`

# IF YOUR COMPUTATIONAL CLUSTER DIFFERS
# - please create a script to correctly submit to your computational cluster
# - the script should call the Python script as below, specifying the index of partition

. $PATH_TO_MINICONDA/etc/profile.d/conda.sh && conda deactivate && conda activate chime_recipe_2

cd $PATH_TO_THE_REPO

task_id=$((SLURM_ARRAY_TASK_ID-1))

python -m scripts.simulate simulate.i_split=${task_id}
