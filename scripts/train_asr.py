# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra

import pytorch_lightning as pl
from nemo.utils.exp_manager import exp_manager
from src.model import MultichannelEncDecHybridRNNTCTCBPEModel


def train_asr(cfg):
    pl.seed_everything(111, workers=True)
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = MultichannelEncDecHybridRNNTCTCBPEModel(
        cfg=cfg.model, trainer=trainer
    )
    trainer.fit(asr_model)


@hydra.main(
    version_base=None, config_path="../config", config_name="main_from_scratch"
)
def main(cfg):
    train_asr(cfg.train_asr)


if __name__ == "__main__":
    main()
