# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

mkdir logs

python -m scripts.prepare_tedlium_data
python -m scripts.prepare_librispeech_alignments

## This step should be run on a computational cluster. 
## The bash script below shows an example of how this can be done with SLURM.
## Please modify the script to work in your environment.
sbatch scripts/speed_perturb.sh
##

python -m scripts.prepare_data_for_simulation

## This step should be run on a computational cluster. 
## The bash script below shows an example of how this can be done with SLURM.
## Please modify the script to work in your environment.
sbatch scripts/simulate.sh
##

python -m scripts.prepare_simulated_data
python -m scripts.prepare_tokenizer
python -m scripts.train_asr
python -m scripts.prepare_data
python -m scripts.finetune_multichannel_asr
python -m scripts.inference inference.checkpoint_dir=exp/base_from_scratch/finetune_multichannel/checkpoints inference.output_dir=exp/base_from_scratch/inference
python -m scripts.evaluate evaluate.hypotheses_dir=exp/base_from_scratch/inference evaluate.result_dir=exp/base_from_scratch/results
