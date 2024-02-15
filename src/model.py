# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models import (
    EncDecHybridRNNTCTCBPEModel,
)
from nemo.collections.asr.data.audio_to_text import _speech_collate_fn
from typing import Optional, Dict
from nemo.core.neural_types import (
    NeuralType,
    LengthsType,
    SpectrogramType,
    AudioSignal,
)


def _multichannel_speech_collate_fn(batch, pad_id):
    '''
    Transposes first item in the batch, i.e. the multi-channel audio,
    so that it works well with the _speech_collate_fn function.
    '''
    batch = [[b[0].T, *b[1:]] for b in batch]
    return _speech_collate_fn(batch, pad_id)


class MultichannelEncDecHybridRNNTCTCBPEModel(EncDecHybridRNNTCTCBPEModel):
    '''Modifies EncDecHybridRNNTCTCBPEModel to accept multi-channel input.

    Necessary modifications are:
    - definition of input dimensions, which is checked during forward pass
    - multichannel collate function

    Note that other modifications to the model are done in the corresponding
    modules, as specified in the config file. I.e. preprocessor and encoder
    modules are defined in `src/preprocessor.py` and `src/encoder.py` files.
    '''

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        input_signal_eltype = AudioSignal(
            freq=self.preprocessor._sample_rate
        )

        return {
            "input_signal": NeuralType(
                ("B", "C", "T"), input_signal_eltype, optional=True
            ),
            "input_signal_length": NeuralType(
                tuple("B"), LengthsType(), optional=True
            ),
            "processed_signal": NeuralType(
                ("B", "C", "D", "T"), SpectrogramType(), optional=True
            ),
            "processed_signal_length": NeuralType(
                tuple("B"), LengthsType(), optional=True
            ),
        }

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        dataloader = super()._setup_dataloader_from_config(config)
        if dataloader is None:
            return None
        dataloader.collate_fn = lambda batch: _multichannel_speech_collate_fn(
            batch, pad_id=dataloader.dataset.manifest_processor.pad_id
        )
        return dataloader
