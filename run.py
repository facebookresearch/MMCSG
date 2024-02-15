# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from omegaconf import OmegaConf
import logging
import hydra
from scripts.prepare_data import prepare_data
from scripts.finetune_asr import finetune_asr
from scripts.inference import inference
from scripts.evaluate import evaluate


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    if cfg.prepare_data.run:
        prepare_data(cfg.prepare_data)

    if cfg.finetune_asr.run:
        finetune_asr(cfg.finetune_asr)

    if cfg.inference.run:
        inference(cfg.inference)

    if cfg.evaluate.run:
        evaluate(cfg.evaluate)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
