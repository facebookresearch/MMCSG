# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import hydra

import pytorch_lightning as pl
from nemo.utils.exp_manager import exp_manager
from src.model import MultichannelEncDecHybridRNNTCTCBPEModel
from .inference import find_best_checkpoint


def initialize(model, pretrained_model):
    new_state = model.state_dict()
    for name, param in pretrained_model.state_dict().items():
        new_state[name].data.copy_(param)


def finetune_multichannel_asr(cfg):
    pl.seed_everything(111, workers=True)
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model_path = find_best_checkpoint(Path(cfg.initialize_from))
    pretrained_model = (
        MultichannelEncDecHybridRNNTCTCBPEModel.load_from_checkpoint(
            model_path
        )
    )
    asr_model = MultichannelEncDecHybridRNNTCTCBPEModel(
        cfg=cfg.model, trainer=trainer
    )
    initialize(asr_model, pretrained_model)
    trainer.fit(asr_model)


@hydra.main(
    version_base=None, config_path="../config", config_name="main_from_scratch"
)
def main(cfg):
    finetune_multichannel_asr(cfg.finetune_multichannel_asr)


if __name__ == "__main__":
    main()
