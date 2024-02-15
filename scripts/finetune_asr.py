# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path
import hydra

import pytorch_lightning as pl
from nemo.utils.exp_manager import exp_manager
import nemo.collections.asr as nemo_asr

from src.model import MultichannelEncDecHybridRNNTCTCBPEModel

import sentencepiece as spm
# Necessary for manually extending the sentence-piece model with custom tokens.
# https://github.com/google/sentencepiece/issues/121
sys.path.insert(0, spm.__path__[0])
import sentencepiece_model_pb2 as model_proto


def modify_tokenizer(pretrained_model, tokenizer_dir):
    '''Adds speaker tokens into tokenizer.

    The tokens are pre-pended, i.e. will have indices 0 and 1.
    The added tokens are »0 and »1 for SELF/OTHER, i.e. the wearer
    of the glasses and the conversatinal partner, respectively.

    The modification of the sentencepiece model follows what is described
    in here: https://github.com/google/sentencepiece/issues/121
    '''
    m = model_proto.ModelProto()
    m.ParseFromString(
        pretrained_model.decoding.tokenizer.tokenizer.serialized_model_proto()
    )
    tokenizer_dir.mkdir(exist_ok=True, parents=True)

    new_token0 = m.SentencePiece()
    new_token0.piece = "▁»0"
    new_token0.score = 0.0
    m.pieces.insert(0, new_token0)
    new_token1 = m.SentencePiece()
    new_token1.piece = "▁»1"
    new_token1.score = 0.0
    m.pieces.insert(1, new_token1)

    with open(tokenizer_dir / "tokenizer.model", "wb") as fw:
        fw.write(m.SerializeToString())

    with open(tokenizer_dir / "vocab.txt", "w") as fw:
        for piece in m.pieces:
            token = piece.piece
            if token.startswith("▁"):
                fw.write(f"{token[1:]}\n")
            else:
                fw.write(f"##{token}\n")


def initialize(model, pretrained_model):
    '''Initializes parameters of a multi-channel, multi-speaker model.

    The original pretrained model accepts single-channel audio at the input
    and its tokenizer does not contain speaker tokens »0 and »1.
    The multi-channel model accepts multi-channel recording that is first
    passed through a beamformer with multiple output beams. These beams are
    then fed into the model. The tokenizer of the multi-channel model also
    contains the speaker tokens.

    The layers that thus need to be modified are:
    - input layer of the encoder (extended for multiple-beam input)
    - RNN-T decoder input (extended for two additional tokens)
    - CTC decoded output (extended for two additional tokens)
    - RNN-T joiner output (extended for two additional tokens)
    '''
    new_state = model.state_dict()
    for name, param in pretrained_model.state_dict().items():
        if name == "encoder.pre_encode.conv.0.weight":
            # initialize input convolutional layer to average input channels
            num_channels = new_state[name].shape[1]
            for c in range(num_channels):
                new_state[name][:, c].data.copy_(
                    1 / num_channels * param[:, 0]
                )
        elif name == "decoder.prediction.embed.weight":
            # initialize input layer of RNN-T decoder with zero weight
            # for speaker tokens
            new_state[name][:2].data.fill_(0.0)
            new_state[name][2:].data.copy_(param)
        elif name == "ctc_decoder.decoder_layers.0.weight":
            # initialize weights of CTC decoder output layer for speaker tokens
            # with average weights of other tokens
            new_state[name][:2].data.copy_(param.mean(dim=0))
            new_state[name][2:].data.copy_(param)
        elif name == "ctc_decoder.decoder_layers.0.bias":
            # initialize bias of CTC decoder output layer for speaker tokens
            # with max of other tokens except blank
            new_state[name][:2].data.copy_(param[:-1].max() + param.std())
            new_state[name][2:].data.copy_(param)
        elif name == "joint.joint_net.2.weight":
            # initialize weights of RNN-T joiner output layer
            # for speaker tokens with average weights of other tokens
            new_state[name][:2].data.copy_(param.mean(dim=0))
            new_state[name][2:].data.copy_(param)
        elif name == "joint.joint_net.2.bias":
            # initialize bias of RNN-T joiner output layer for speaker tokens
            # with max of other tokens except blank
            new_state[name][:2].data.copy_(param[:-1].max() + param.std())
            new_state[name][2:].data.copy_(param)
        else:
            new_state[name].data.copy_(param)


def finetune_asr(cfg):
    pl.seed_everything(111, workers=True)
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    pretrained_model = nemo_asr.models.ASRModel.restore_from(
        restore_path=cfg.pretrained_model_path
    )
    modify_tokenizer(pretrained_model, Path(cfg.model.tokenizer.dir))
    asr_model = MultichannelEncDecHybridRNNTCTCBPEModel(
        cfg=cfg.model, trainer=trainer
    )
    initialize(asr_model, pretrained_model)
    trainer.fit(asr_model)


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg):
    finetune_asr(cfg.finetune_asr)


if __name__ == "__main__":
    main()
