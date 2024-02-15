# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

from multi_talker_overlap_transform import OverlapMultiTalkerTransform

from sample_apply_rir_transform import SampleApplyRIRTransform

from sample_noise_transform import SampleNoiseRemoteTransform
from snr_adjustment_transform import SNRAdjustmentMCTransform

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
#   Compose
# ------------------------------------------------------------------------------

TRANSFORM_REGISTRY = {
    "SampleNoiseRemoteTransform": SampleNoiseRemoteTransform,
    "OverlapMultiTalkerTransform": OverlapMultiTalkerTransform,
    "SampleApplyRIRTransform": SampleApplyRIRTransform,
    "SNRAdjustmentMCTransform": SNRAdjustmentMCTransform,
}


class Compose:
    """Composes several transforms together.

    Args:
        transforms (list of ``DataTransform`` objects): transforms to compose.

    Example:
        >>> Compose([
        >>>     SampleNoiseRemoteTransform(),
        >>>     OverlapMultiTalkerTransform(),
        >>>     SampleApplyRIRTransform(),
        >>>     SNRAdjustmentMCTransform(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        if transform is not None:
            self.transforms.append(transform)

    @classmethod
    def setup_from_config(cls, config):
        transforms = []
        for name, parameters in config:
            transform = TRANSFORM_REGISTRY[name](**parameters)
            transforms.append(transform)
        return cls(transforms=transforms)
