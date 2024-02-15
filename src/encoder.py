# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from nemo.collections.asr.parts.submodules.subsampling import (
    ConvSubsampling,
    calc_length,
)
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
import torch.nn as nn
from nemo.core.neural_types import (
    ChannelType,
    LengthsType,
    NeuralType,
    SpectrogramType,
)
from collections import OrderedDict
from nemo.collections.asr.parts.submodules.causal_convs import CausalConv2D


class MultichannelConformerEncoder(ConformerEncoder):
    '''Extends encoder to multi-channel input.

    The only change is at the very input convolutional layer, which is extended
    from single-channel to `in_channels` channels. This is implemented in the
    MultichannelConvSubsampling class.
    '''
    def __init__(
        self,
        feat_in,
        n_layers,
        d_model,
        in_channels,
        feat_out=-1,
        causal_downsampling=False,
        subsampling="striding",
        subsampling_factor=4,
        subsampling_conv_chunking_factor=1,
        subsampling_conv_channels=-1,
        reduction=None,
        reduction_position=None,
        reduction_factor=1,
        ff_expansion_factor=4,
        self_attention_model="rel_pos",
        n_heads=4,
        att_context_size=None,
        att_context_probs=None,
        att_context_style="regular",
        xscaling=True,
        untie_biases=True,
        pos_emb_max_len=5000,
        conv_kernel_size=31,
        conv_norm_type="batch_norm",
        conv_context_size=None,
        dropout=0.1,
        dropout_pre_encoder=0.1,
        dropout_emb=0.1,
        dropout_att=0.0,
        stochastic_depth_drop_prob: float = 0.0,
        stochastic_depth_mode: str = "linear",
        stochastic_depth_start_layer: int = 1,
        global_tokens: int = 0,
        global_tokens_spacing: int = 1,
        global_attn_separate: bool = False,
    ):
        super().__init__(
            feat_in,
            n_layers,
            d_model,
            feat_out,
            causal_downsampling,
            subsampling,
            subsampling_factor,
            subsampling_conv_chunking_factor,
            subsampling_conv_channels,
            reduction,
            reduction_position,
            reduction_factor,
            ff_expansion_factor,
            self_attention_model,
            n_heads,
            att_context_size,
            att_context_probs,
            att_context_style,
            xscaling,
            untie_biases,
            pos_emb_max_len,
            conv_kernel_size,
            conv_norm_type,
            conv_context_size,
            dropout,
            dropout_pre_encoder,
            dropout_emb,
            dropout_att,
            stochastic_depth_drop_prob,
            stochastic_depth_mode,
            stochastic_depth_start_layer,
            global_tokens,
            global_tokens_spacing,
            global_attn_separate,
        )
        # Subsampling
        assert subsampling == "dw_striding" and subsampling_factor > 1, (
            "The multi-channel encoder is now limited to dw_striding"
            " subsampling with subsampling factor > 1"
        )
        if subsampling_conv_channels == -1:
            subsampling_conv_channels = d_model
        self.pre_encode = MultichannelConvSubsampling(
            subsampling=subsampling,
            subsampling_factor=subsampling_factor,
            feat_in=feat_in,
            feat_out=d_model,
            in_channels=in_channels,
            conv_channels=subsampling_conv_channels,
            subsampling_conv_chunking_factor=subsampling_conv_chunking_factor,
            activation=nn.ReLU(True),
            is_causal=causal_downsampling,
        )

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "audio_signal": NeuralType(
                    ("B", "C", "D", "T"), SpectrogramType()
                ),
                "length": NeuralType(tuple("B"), LengthsType()),
                "cache_last_channel": NeuralType(
                    ("D", "B", "T", "D"), ChannelType(), optional=True
                ),
                "cache_last_time": NeuralType(
                    ("D", "B", "D", "T"), ChannelType(), optional=True
                ),
                "cache_last_channel_len": NeuralType(
                    tuple("B"), LengthsType(), optional=True
                ),
            }
        )

    @property
    def input_types_for_export(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "audio_signal": NeuralType(
                    ("B", "C", "D", "T"), SpectrogramType()
                ),
                "length": NeuralType(tuple("B"), LengthsType()),
                "cache_last_channel": NeuralType(
                    ("B", "D", "T", "D"), ChannelType(), optional=True
                ),
                "cache_last_time": NeuralType(
                    ("B", "D", "D", "T"), ChannelType(), optional=True
                ),
                "cache_last_channel_len": NeuralType(
                    tuple("B"), LengthsType(), optional=True
                ),
            }
        )

    def forward_internal(
        self,
        audio_signal,
        length,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
    ):
        B, C, D, T = audio_signal.shape
        return super().forward_internal(
            audio_signal.permute((0, 2, 3, 1)),
            length,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
        )


class MultichannelConvSubsampling(ConvSubsampling):
    '''Extends ConvSubsampling module to accept multi-channel input.

    The only change is made to the input convolutional layer `self.conv[0]`
    which is extended to have `in_channels` channels at the input instead of
    a single channel.
    '''
    def __init__(
        self,
        subsampling,
        subsampling_factor,
        feat_in,
        feat_out,
        conv_channels,
        in_channels,
        subsampling_conv_chunking_factor=1,
        activation=nn.ReLU(),
        is_causal=False,
    ):
        assert (
            subsampling == "dw_striding"
        ), "Currently only dw_striding subsampling is implemented"
        super(MultichannelConvSubsampling, self).__init__(
            subsampling,
            subsampling_factor,
            feat_in,
            feat_out,
            conv_channels,
            subsampling_conv_chunking_factor,
            activation,
            is_causal,
        )

        padding = self.conv[0].padding
        stride = self.conv[0].stride
        kernel_size = self.conv[0].kernel_size
        if isinstance(self.conv[0], CausalConv2D):
            padding = None
            kernel_size = self.conv[0].kernel_size[0]
            stride = self.conv[0].stride[0]
        self.conv[0] = self.conv[0].__class__(
            in_channels,
            self.conv[0].out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=self.conv[0].groups,
        )

    def forward(self, x, lengths):
        x = x.permute(0, 3, 1, 2)
        lengths = calc_length(
            lengths,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )

        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).reshape(b, t, -1))

        return x, lengths
