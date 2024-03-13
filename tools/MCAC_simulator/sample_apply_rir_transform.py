# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import pickle
from typing import List

import numpy as np
import scipy.signal as sps

from sample_noise_transform import load_column

logger = logging.getLogger(__name__)


def generate_multi_data(src, rir, sample_rate=16000):
    """
    generate multi-channel data funtion
    args:
        src: L
        rir: L x C
    return:
        wav_tgt: L x C
    """
    channels = rir.shape[1]
    rir_len = rir.shape[0]
    wav_tgt = np.zeros([channels, src.shape[0] + rir_len - 1])

    for i in range(channels):
        wav_tgt[i] = sps.oaconvolve(src, rir[:, i])
    # L x C
    wav_tgt = np.transpose(wav_tgt)
    wav_tgt = wav_tgt[: src.shape[0]]
    return wav_tgt


class SampleApplyRIRTransform(object):
    """
    This transform is used to simple the rirs from manifold and
    then apply the rir to the clean and noise sources.
    """

    def __init__(
        self,
        rir_datasets: List[tuple] = None,
        fs: int = 16000,
        random_seed=None,
        num_noise_sources: int = 1,
    ):
        """
        Note that this transform is adapted from SampleNoiseRemoteTransform!!
        This tranform should be run after sample noise tranform
        Args:
            rir_datasets: list of (path, col_id, id, weight) where
                path: path to a TSV file
                col_id: column ID of the rir handle in the TSV file
                id: integer ID of the rir dataset. it will be used by
                    subsequent data augmentation transforms to identify
                    the source of the rir.
                weight: float value of how often this rir dataset will be sampled.
                        Values should be positive but do not need to be summed to one.
            fs: desired sampling rate
            random_seed: seed for the random number generator
            num_noise_sources: sample number of noise sources in rir

        Note:
            All elements in a single dataset have the same chance being sampled
        """
        super().__init__()
        self.fs = fs
        self.rir_lists = [
            load_column(path, col_id) for path, col_id, _, _ in rir_datasets
        ]
        # misc params
        self.dataset_ids = [i for _, _, i, _ in rir_datasets]
        # normalize
        weights = [weight for _, _, _, weight in rir_datasets]
        weight_sum = sum(weights)
        self.dataset_weights = [weight / weight_sum for weight in weights]
        # pyre-fixme[4]: Attribute must be annotated.
        self.rng = np.random.default_rng(random_seed)
        # pyre-fixme[4]: Attribute must be annotated.
        self.selected_meta = {}
        self.num_noise_sources = num_noise_sources
        logger.info("Weights for rir datasets:")
        for (path, _, _, _), weight in zip(rir_datasets, self.dataset_weights):
            logger.info(f"\t{weight:.3f}\t{path}")

    def _sample_and_load_rir(self):
        # choose rir dataset based on their weights
        list_id = self.rng.choice(range(len(self.rir_lists)), p=self.dataset_weights)
        # sample one rir from the dataset
        rir_meta_path = self.rng.choice(self.rir_lists[list_id])
        # load rir signal
        # (#mic, #src, sample), the order of src: sound sources->noise sources->mouth
        try:
            # rirs_meta = np.load(str(rir_meta_path), allow_pickle=True)
            with open(rir_meta_path, "rb") as f:
                rirs_meta = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load {rir_meta_path} from rir dataset")
            raise e

        return rirs_meta, self.dataset_ids[list_id]

    def __call__(self, data):
        """
        Args:
            data: list of np array, including clean sources (should be more than 2)
            and noise sources sampled by OverlapMultiTalkerTransform
            a decorator to supprot mult-channel input ("data") where "data" is a list like:
            [mixed_audio, ctm, sot, noise_id]

            class MixedAudio:
            mixed_audio: Optional[np.ndarray] = None
            near_field_audio: Optional[np.ndarray] = None
            far_field_audio: Optional[np.ndarray] = None
            distractor_audio: Optional[np.ndarray] = None
            noise_audio: Optional[np.ndarray] = None
            metadata: Optional[Dict] = None
            length: int = -1
        Returns
            a list:
                [
                    mc_sp0_arr_near_end,
                    mc_sp1_arr_far_end,
                    mc_sp2_arr_distractor,
                    (noise, id),
                    selected meta
                ]
                the last element is the selected sound/noise/mouth rir meta information
        """

        # sample rir
        rirs, rir_dataset_id = self._sample_and_load_rir()
        mixed_audio, alignment, sot, noise_id = data
        wearer_audio, ft_audio, distractor_audio, noise = (
            mixed_audio.near_field_audio,
            mixed_audio.far_field_audio,
            mixed_audio.distractor_audio,
            mixed_audio.noise_audio,
        )
        processed_data = []
        # sample far-talker sound source rirs
        selected_far_sound_rir = {}
        sound_rir_key = self.rng.choice(list(rirs["mic_to_ft_sound_source"]))
        sound_far_rir = rirs["mic_to_ft_sound_source"][sound_rir_key]
        selected_far_sound_rir.update(
            {str(sound_rir_key): rirs["mic_to_ft_sound_source"][sound_rir_key]}
        )
        self.selected_meta.update({"sound_far_rirs": selected_far_sound_rir})

        # sample dsitractor
        selected_distractor_sound_rir = {}
        sound_rir_key = self.rng.choice(list(rirs["mic_to_distractor_sound_source"]))
        sound_distractor_rir = rirs["mic_to_distractor_sound_source"][sound_rir_key]
        selected_distractor_sound_rir.update(
            {str(sound_rir_key): rirs["mic_to_distractor_sound_source"][sound_rir_key]}
        )
        self.selected_meta.update(
            {"sound_distrctor_rirs": selected_distractor_sound_rir}
        )
        # sample noise source rirs
        noise_rirs = []
        selected_noise_rirs = {}
        for _ in range(self.num_noise_sources):
            noise_rir_key = self.rng.choice(list(rirs["mic_to_noise_source"]))
            noise_rirs.append(rirs["mic_to_noise_source"][noise_rir_key])
            selected_noise_rirs.update(
                {str(noise_rir_key): rirs["mic_to_noise_source"][noise_rir_key]}
            )
        self.selected_meta.update({"noise_rirs": selected_noise_rirs})

        mouth_rir_key = self.rng.choice(list(rirs["mic_to_mouth_source"]))
        mouth_rir = rirs["mic_to_mouth_source"][mouth_rir_key]
        self.selected_meta.update({"mouth_rir": {str(mouth_rir_key): mouth_rir}})
        # add microphone locations
        self.selected_meta.update({"mic_geometry": rirs["mic_geometry"]})
        self.selected_meta.update({"room_dim": rirs["room_dim"]})

        # for near-end audio
        near_clean = None
        if wearer_audio is not None:
            near_clean = generate_multi_data(
                wearer_audio, mouth_rir["rir"].T, sample_rate=self.fs
            )
            near_clean = near_clean.astype("float32")
        processed_data.append(near_clean)

        # for the far_end audio
        far_clean = None
        if ft_audio is not None:
            far_clean = generate_multi_data(
                ft_audio,
                sound_far_rir["rir"].T,
                sample_rate=self.fs,
            )
            far_clean = far_clean.astype("float32")
        processed_data.append(far_clean)

        # for distractor audio
        distractor_clean = None
        if distractor_audio is not None:
            distractor_clean = generate_multi_data(
                distractor_audio,
                sound_distractor_rir["rir"].T,
                sample_rate=self.fs,
            )
            distractor_clean = distractor_clean.astype("float32")

        processed_data.append(distractor_clean)
        processed_noise = []
        if noise is not None:
            # current only consider one noise source
            if len(noise.shape) > 1 and noise.shape[1] > 1:
                # multi-channel noise and bypass the simulation for noise
                RIR_ch = noise_rirs[0]["rir"].shape[0]
                assert (
                    noise.shape[1] == RIR_ch
                ), f"when using multi-channel noise, the numner of channels of noise source much match simulated RIR {noise.shape[1]} != {RIR_ch}"
                mul_noise = noise
            else:
                mul_noise = generate_multi_data(
                    noise, noise_rirs[0]["rir"].T, sample_rate=self.fs
                )

            processed_noise.append((mul_noise.astype("float32"), noise_id))
        processed_data.extend(processed_noise)
        self.selected_meta.update({"alignment": alignment.decode()})
        self.selected_meta.update({"sot": sot.decode()})
        processed_data.append(self.selected_meta)

        return processed_data
