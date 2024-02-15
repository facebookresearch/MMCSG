# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from nemo.core.neural_types import (
    AudioSignal,
    LengthsType,
    MelSpectrogramType,
    NeuralType,
)
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor,
)


def load_bf_weights(weights_path):
    weights = np.load(weights_path)
    weights = np.transpose(weights, (1, 0, 2))
    return torch.tensor(
        weights, dtype=torch.float32
    )  # [out_channels, in_channels, kernel_size]


class Beamformer(torch.nn.Module):
    def __init__(self, weights_path: str, ldFFTOrder: int = 10):
        super().__init__()
        self.w = load_bf_weights(
            weights_path=weights_path,
        )
        self.out_channels, self.in_channels, self.kernel_size = self.w.shape
        self.N = 2**ldFFTOrder
        self.W = torch.fft.rfft(self.w, n=self.N, axis=-1, norm=None)

    def forward(self, x):
        B = x.shape[0]
        # left padding
        x_padded = torch.cat(
            [
                torch.zeros(
                    [B, self.in_channels, self.kernel_size - 1],
                    device=x.device,
                ),
                x,
            ],
            axis=-1,
        )

        # reshape x into frames
        length = x_padded.shape[-1]
        chunk_length = self.N - self.kernel_size + 1
        x_padded = torch.cat(
            [
                x_padded,
                torch.zeros(
                    [
                        B,
                        self.in_channels,
                        chunk_length - length % chunk_length,
                    ],
                    device=x_padded.device,
                ),
            ],
            axis=-1,
        )
        x_framed = x_padded.reshape(
            B, self.in_channels, -1, chunk_length
        ).permute(0, 2, 1, 3)

        # beamforming
        X = torch.fft.rfft(
            x_framed[:, :, None], n=self.N, axis=-1, norm="ortho"
        )
        self.W = self.W.to(X.device)
        Y = (X * self.W).sum(axis=-2)
        y_framed = torch.fft.irfft(Y, n=self.N, axis=-1, norm="ortho")

        # reshape y from frames to signal
        # extra care for tails of convolutions overflowing into future windows
        y_framed_valid = y_framed[..., :chunk_length]
        for i in range((self.kernel_size - 1) // chunk_length + 1):
            leftover = y_framed[
                :, : -i - 1, :, (i + 1) * chunk_length : (i + 2) * chunk_length
            ]
            y_framed_valid[:, i + 1 :, :, : leftover.shape[-1]] += leftover
        y_padded = y_framed_valid.permute(0, 2, 1, 3).reshape(
            [B, self.out_channels, -1]
        )
        y_padded = y_padded[..., :length]

        # remove left padding
        y = y_padded[..., self.kernel_size - 1 :]
        return y


class AudioToBeamformedToMelSpectrogramPreprocessor(
    AudioToMelSpectrogramPreprocessor
):
    '''Extends feature extraction with beamforming and multi-channel features.

    Multi-channel signal is accepted at the input and passed through
    a fixed beamformer. The beamformer outputs multiple beams (into multiple
    directions). Features are extracted from each of these beams and stacked.
    '''
    @property
    def input_types(self):
        return {
            "input_signal": NeuralType(
                ("B", "C", "T"),
                AudioSignal(freq=self._sample_rate),
            ),
            "length": NeuralType(tuple("B"), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "processed_signal": NeuralType(
                ("B", "C", "D", "T"), MelSpectrogramType()
            ),
            "processed_length": NeuralType(tuple("B"), LengthsType()),
        }

    def __init__(
        self,
        beamforming_weights_path,
        scale=1.0,
        sample_rate=16000,
        normalize="per_feature",
        window_size=0.02,
        window_stride=0.01,
        window="hann",
        features=64,
        n_fft=None,
        frame_splicing=1,
        dither=1e-5,
        pad_to=16,
    ):
        super().__init__(
            sample_rate=sample_rate,
            normalize=normalize,
            window_size=window_size,
            window_stride=window_stride,
            window=window,
            features=features,
            n_fft=n_fft,
            frame_splicing=frame_splicing,
            dither=dither,
            pad_to=pad_to,
        )
        self.scale = scale
        self.beamformer = Beamformer(beamforming_weights_path)

    def get_features(self, input_signal, length):
        input_signal *= self.scale
        beamformed_signals = self.beamformer(input_signal)
        features = []
        for channel in range(self.beamformer.out_channels):
            features_1channel, length_out = self.featurizer(
                beamformed_signals[:, channel], length
            )
            features.append(features_1channel)
        features_multichannel = torch.stack(features, dim=1)
        return features_multichannel, length_out
