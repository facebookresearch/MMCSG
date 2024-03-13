# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# pyre-fixme[5]: Global expression must be annotated.
EPS = np.finfo(float).eps


class BaseSNRContainer:
    def __init__(self, params: Union[list, dict]):
        if self.__class__ is BaseSNRContainer:
            raise TypeError(
                "BaseSNRContainer is purely abstract, please don't instantiate."
            )
        super().__init__()

    def sample_snr(self):
        raise NotImplementedError(
            "Child SNRContainer should implement the smaple_snr method"
        )


class UniformSNRContainer(BaseSNRContainer):
    """
    The uniform noise sampler for snr specified using
    INPUTS:
        * params: List with three elements [start, stop, hop]

    """

    def __init__(self, params, seed=None):
        if type(params) is not list:
            raise TypeError(
                f"params should be of the type 'list' but current param is {params}"
            )
        assert (
            len(params) == 3
        ), f"The params should be of len 3 but current params are {params}"
        self.snr = np.arange(params[0], params[1] + 1, params[2])
        self.len = len(self.snr)
        self.rng = np.random.default_rng(seed)
        logging.info(f"[{self.__class__.__name__}] The SNR values are {self.snr}")

    def sample_snr(self):
        snr_idx = self.rng.integers(0, self.len, 1)[0]
        return self.snr[snr_idx]


class UserDefinedSNRContainer(BaseSNRContainer):
    """
    User defined distribution for the noise.
    INPUTS:
        * params: is dictionary of the format:
                {
                    snr: []
                    weight: []
                }
    """

    def __init__(self, params, seed=None):
        if type(params) is not dict:
            raise TypeError(
                f"params should be of the dict but current param is {params}"
            )

        self.snr = params["snr"]
        self.weight = params["weight"]
        self.rng = np.random.default_rng(seed)

        assert len(self.snr) > 0, f"[{self.__class__.__name__}] SNR cannot be empty"
        assert len(self.snr) == len(
            self.weight
        ), f"[{self.__class__.__name__}] Length of the snr and weight should be same"

        den = sum(self.weight)
        if den != 1.0:
            self.weight = [w / den for w in self.weight]

        logging.info(f"[{self.__class__.__name__}] The weights are {self.weight}")
        logging.info(f"[{self.__class__.__name__}] The SNR values are {self.snr}")

    def sample_snr(self):
        snr = self.rng.choice(self.snr, size=1, p=self.weight)[0]
        return snr


def snr_container_factory(
    param: Union[list, dict],
    seed: Any = None,
) -> BaseSNRContainer:
    if type(param) is list:
        return UniformSNRContainer(param, seed)
    elif type(param) is dict:
        return UserDefinedSNRContainer(param, seed)
    else:
        raise NotImplementedError(
            f"SNR container is not implemented for param of type {type(param)}"
        )


# ------------------------------------------------------------------------------
#   SNRAdjustmentMCTransform
# ------------------------------------------------------------------------------
class SNRAdjustmentMCTransform(object):
    """
    This tranform should be run after sample rir tranform. It expects that
    the noise data should be a tuple (noise_array, id)
    The id: of the noise will let this transform know how the snr should be sampled
            and applied. More on this in the input

    Input:
        * snr: List[List[3]] specifies different SNR samplers. Each snr sampler is
            described by a 3 tuple, 1st and 2nd element of the tuple are the limits of the
            sampler and the 3rd element is the step size.
        * simuated_fs_bounds: how do you scale the training data
        * distractor_fs_bounds: how do you scale the distractor data
        * ft_fs_bounds: how do you scale the conversation partner data
        * wearer_fs_bounds: how do you scale the device user data
        * clipping_threshold: threshold where clipping correction is applied.
        * add_noise: whether add noise into the simulated data
        * add_distractor_as_bcg_noise: whether add distractor as noise with given SNR,
            if not, directly add the distractor with the simulated conversation audio
        * distractor_snr: the given SNR range when adding distractor as background noise
        * far_field_only: this is for the far-field noise suppression task
        * near_field_only: this is for the near-field noise suppression task
        * add_sil_to_noise: controls if we randomly add silence between repeated copies of
            noise signals when the length of noise signals is less than the clean sources.
        * seed: random seed
        * use_alignment: whether to use alignment information for a more accurate SNR estimation.
        * spk1_id: speaker id for the first speaker (close-talk speaker aka SELF)
        * spk2_id: speaker id for the second speaker (far-field speaker aka OTHER)
    """

    def __init__(
        self,
        snr: List[Dict[str, List[float]]],  # List of list of SNR
        simulated_fs_bounds: Tuple[int, int] = (
            -45,
            -35,
        ),  # Target gain/vol values limits in dBFS
        distractor_fs_bounds: Tuple[int, int] = (-50, -45),
        ft_fs_bounds: Tuple[int, int] = (-45, -38),
        wearer_fs_bounds: Tuple[int, int] = (-38, -30),
        fs: int = 16000,
        clipping_threshold: float = 0.97,
        add_noise: bool = False,
        add_distractor_as_bcg_noise: bool = False,
        distractor_snr: Tuple[int, int] = (15, 25),
        add_sil_to_noise: bool = False,
        seed: Optional[int] = None,
        use_alignment: bool = False,
        spk1_id: str = "»0",
        spk2_id: str = "»1",
    ):
        self.rng = np.random.default_rng(seed)
        assert isinstance(snr, list), "data indices must be a dictionary"
        # Process the SNR dict
        self.snr = []
        for s in snr:
            self.snr.append(snr_container_factory(s, self.rng))
        self.fs = fs
        self.fs_limits = simulated_fs_bounds
        self.distractor_fs_limits = distractor_fs_bounds
        self.ft_fs_limits = ft_fs_bounds
        self.wearer_fs_limts = wearer_fs_bounds
        self.clipping_threshold = clipping_threshold
        self.add_noise = add_noise
        self.add_distractor_as_bcg_noise = add_distractor_as_bcg_noise
        self.distractor_snr = distractor_snr
        self.add_sil_to_noise = add_sil_to_noise
        self.use_alignment = use_alignment
        self.spk1_id = spk1_id
        self.spk2_id = spk2_id
        assert (
            self.clipping_threshold < 0.99
        ), "Clipping thereshold should be less than 0.99"

        # pyre-fixme[4]: Attribute must be annotated.
        self.string = (
            self.__class__.__name__
            + "("
            + "snr={}, ".format(snr)
            + "fs_bounds={}, ".format(simulated_fs_bounds)
            + "fs={}, ".format(self.fs)
            + "clipping_threshold={}, ".format(clipping_threshold)
            + ")"
        )

    def __repr__(self):
        return self.string

    def normalize(self, audio, target_level=-25):
        """Normalize the signal to the target level"""
        rms = math.sqrt(np.mean(audio**2))
        scalar = 10 ** (target_level / 20) / (rms + EPS)
        audio = audio * scalar
        return audio

    def scale_noise(self, speech_part, noise_part, snr):
        """Scale the Noise to achieve particular SNR"""

        energy_speech = float(np.sum(np.power(speech_part, 2))) + EPS
        energy_noise = float(np.sum(np.power(noise_part, 2))) + EPS  # prevent NaN
        scale = math.pow(10, -snr / 10.0) * energy_speech / energy_noise
        scale = math.sqrt(scale)
        return noise_part * scale

    def clipping_solver(self, audio, threshold=0.99):
        """
        This function helps scale down the audio to solve the clipping issue
        returnes the scale that can be used to scale clean speech
        """
        amplitude_max = np.max(np.abs(audio))
        scale = None
        if amplitude_max <= threshold:
            return audio, scale
        else:
            scale = (amplitude_max / threshold) + EPS
            scaled_audio = audio / scale
            return scaled_audio, scale

    def process_noise(self, noise: np.ndarray, lensrc: int) -> np.ndarray:
        """
        Padding noise if the length of noise is smaller than the signals
        """
        num_mic = noise.shape[1]
        lennoise = noise.shape[0]
        if lennoise == lensrc:
            return noise
        if lennoise < lensrc:
            if self.add_sil_to_noise:
                # randomly pad the noise signal 30% of the time
                if self.rng.integers(0, 3, 1) == 1:
                    noise = np.concatenate(
                        (
                            np.zeros((self.fs // 4, num_mic)),
                            noise,
                            np.zeros((self.fs // 4, num_mic)),
                        )
                    )
            # Count how many times we might have to tile the noise extend it by 2
            tile_count = int(np.ceil(lensrc / lennoise)) + 2
            noise = np.tile(noise, (tile_count, 1))
            lennoise = noise.shape[0]  # update noise length

        # At this point the noise is always greater than speech now trim and return data
        noise_st = self.rng.integers(0, lennoise - lensrc)
        noise_en = noise_st + lensrc
        ret_audio = noise[noise_st:noise_en, :]

        return ret_audio

    def compute_noise_scale(
        self, speech_part: np.ndarray, noise_part: np.ndarray, snr: float
    ) -> float:
        energy_speech = float(np.sum(np.power(speech_part, 2))) + EPS
        energy_noise = float(np.sum(np.power(noise_part, 2))) + EPS  # prevent NaN
        scale = math.pow(10, -snr / 10.0) * energy_speech / energy_noise
        scale = math.sqrt(scale)
        return scale

    def scale_noise_with_vad_wrapper(
        self,
        clean_src: np.ndarray,
        noise_src: np.ndarray,
        snr: float,
        vad_list: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
    ) -> np.ndarray:
        """Scale the Noise under consideration of VAD information to achieve target SNR"""
        noise_src = self.process_noise(noise_src, clean_src.shape[0])
        noise_scales = []
        for vad in vad_list:
            if vad is None or not vad.any():
                continue
            speech_part = clean_src[vad, ...]
            noise_part = noise_src[vad, ...]
            noise_scale = self.compute_noise_scale(speech_part, noise_part, snr)
            noise_scales.append(noise_scale)
        assert len(noise_scales) > 0, "No valid VAD information found"
        # We're using the minimum scale to assure that we're not adding noise
        # to any speaker with an SNR worse than the target SNR.
        noise_scale = min(noise_scales)
        # Save guard against the case when the noise is near zero for all speech parts
        # but not for the non-speech parts, which could then result in unrealistic
        # high noise scales.
        noise_scale_without_vad = self.compute_noise_scale(clean_src, noise_src, snr)
        if noise_scale_without_vad * 20 < noise_scale:
            noise_scale = noise_scale_without_vad
        return noise_src * noise_scale

    def scale_noise_wrapper(self, src, noise, snr):
        """
        scaling the noise signals with given SNR
        """
        noise = self.process_noise(noise, src.shape[0])
        assert (
            noise.shape[0] == src.shape[0]
        ), "Noise and speech should be same length here"

        return self.scale_noise(src, noise, snr)

    def get_paired_data(self, clean_src, noise_srcs, snr_list, vad_list):
        """
        mix noise and speech signals
        """
        noise_src = 0.0
        for i, noise_tpl in enumerate(noise_srcs):
            assert isinstance(
                noise_tpl, tuple
            ), "[SNRAdjustmentTransform] Noise should be a tuple"
            # Normalize the noise level before proecessing
            noise = self.normalize(noise_tpl[0])
            if any(vad is not None and vad.any() for vad in vad_list):
                scaled_noise = self.scale_noise_with_vad_wrapper(
                    clean_src, noise, snr_list[i], vad_list
                )
            else:
                scaled_noise = self.scale_noise_wrapper(clean_src, noise, snr_list[i])
            noise_src = noise_src + scaled_noise

        # If we have multiple noise sources in the room guard against the possibility
        # that the SNR ends up being too high or too low due to all the accumulation
        if len(snr_list) > 1:
            assert isinstance(noise_src, np.ndarray)
            if any(vad is not None and vad.any() for vad in vad_list):
                snr_ = self.rng.choice(snr_list)
                noise_src = self.scale_noise_with_vad_wrapper(
                    clean_src, noise_src, snr_, vad_list
                )
            else:
                noise_src = self.scale_noise(
                    clean_src,
                    noise_src,
                    self.rng.choice(snr_list),
                )
        # Get noisy data
        noisy_data = clean_src + noise_src
        # # Randomly pick the target max fulscale value
        fs_val = self.rng.integers(self.fs_limits[0], self.fs_limits[1])
        rmsnoisy = math.sqrt(np.mean((noisy_data**2)))
        scale = 10 ** (fs_val / 20) / (rmsnoisy + EPS)
        noisy_data = noisy_data * scale
        clean_src = clean_src * scale
        out_scale = scale
        noisy_data, scale = self.clipping_solver(noisy_data, self.clipping_threshold)
        if scale is not None:
            clean_src = clean_src / scale
            out_scale = out_scale / scale

        return clean_src, noisy_data, out_scale

    def sample_snr(self, noise_id):
        # of noise_id exceeds the len(snr) stick ot the last element
        noise_id = min(len(self.snr) - 1, noise_id)
        snr = self.snr[noise_id].sample_snr()

        return snr

    def get_speech_activity(
        self,
        spk1: Optional[np.ndarray],
        spk2: Optional[np.ndarray],
        meta: Dict[str, Any],
        collar: float = 0.025,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        # collar extends a speech segment by this amount on both sides
        # default is 0.025 seconds (25ms)
        spk1_vad, spk2_vad = None, None
        if self.use_alignment:
            ali_lines = [
                x for x in meta["alignment"].split("\n") if not x.endswith(" SIL")
            ]
            assert all(
                len(x.split()) == 5 for x in ali_lines
            ), f"Alignment format error! Alignment:\n{meta['alignment']}"
            spk1_ali, spk2_ali = [], []
            spk1_active, spk2_active = False, False
            for line in ali_lines:
                if line.endswith(" " + self.spk1_id):
                    spk1_active, spk2_active = True, False
                elif line.endswith(" " + self.spk2_id):
                    spk1_active, spk2_active = False, True
                else:
                    wrd_start, wrd_dur = line.split()[2:4]  # word start and dur in sec.
                    wrd_start, wrd_dur = float(wrd_start), float(wrd_dur)
                    wrd_start, wrd_dur = max(0, wrd_start - collar), wrd_dur + collar
                    # convert time in seconds to number of samples
                    start, dur = int(wrd_start * self.fs), int(wrd_dur * self.fs)
                    if spk1_active:
                        spk1_ali.append((start, dur))
                    elif spk2_active:
                        spk2_ali.append((start, dur))
            if spk1 is not None and spk1_ali:
                spk1_vad = np.zeros(spk1.shape[0], dtype=bool)
                for start, dur in spk1_ali:
                    stop = min(spk1_vad.shape[0], start + dur + 1)
                    spk1_vad[start:stop] = True
            if spk2 is not None and spk2_ali:
                spk2_vad = np.zeros(spk2.shape[0], dtype=bool)
                for start, dur in spk2_ali:
                    stop = min(spk2_vad.shape[0], start + dur + 1)
                    spk2_vad[start:stop] = True
        return (spk1_vad, spk2_vad)

    def __call__(self, data):
        """
        In this transform, we assume that the noise and clean sources are multi-channel data generated
        by sample rir transform.

        data:
        a list:
                [
                    mc_sp0_arr_near_end,
                    mc_sp1_arr_far_end,
                    mc_sp2_arr_distractor,
                    (noise, id),
                    selected meta
                ]

        Output (dict): {near_end_ref, far_end_ref, mix_noisy_sig, ...}
        """
        if data is None:
            return data
        meta = data[-1]
        processed_data = {}
        # normlize data
        distractor_fs_val = self.rng.integers(
            self.distractor_fs_limits[0], self.distractor_fs_limits[1]
        )
        ft_fs_val = self.rng.integers(self.ft_fs_limits[0], self.ft_fs_limits[1])
        wearer_fs_val = self.rng.integers(
            self.wearer_fs_limts[0], self.wearer_fs_limts[1]
        )

        # using clean instead of clean_early [i][0]
        spk1, spk2, distractor = None, None, None
        if data[0] is not None:
            spk1 = self.normalize(data[0], target_level=wearer_fs_val)
        if data[1] is not None:
            spk2 = self.normalize(data[1], target_level=ft_fs_val)
        if data[2] is not None:
            distractor = self.normalize(data[2], target_level=distractor_fs_val)

        # Get speech acitvity information for each speaker
        speech_activity = self.get_speech_activity(spk1, spk2, meta)

        # Generate audio mixure from each speaker and distractor
        if self.add_distractor_as_bcg_noise:
            if distractor is not None and spk1 is not None:
                distractor_snr = self.rng.integers(
                    self.distractor_snr[0], self.distractor_snr[1]
                )
                if any(vad is not None and vad.any() for vad in speech_activity):
                    if spk1 is None:
                        ref_audio = spk2
                    elif spk2 is None:
                        ref_audio = spk1
                    else:
                        ref_audio = spk1 + spk2
                    distractor = self.scale_noise_with_vad_wrapper(
                        ref_audio, distractor, distractor_snr, speech_activity
                    )
                else:
                    # if far-talk speech is available, use it as the reference for computing the distractor SNR
                    ref_spk = spk1 if spk2 is None else spk2
                    distractor = self.scale_noise_wrapper(
                        ref_spk, distractor, distractor_snr
                    )

        audio_list = [spk1, spk2, distractor]
        audio_mixture_clean = sum([x for x in audio_list if x is not None])

        # Add noise
        if self.add_noise:
            noise_data = [data[-2]]
            # sample snr in terms of noise id
            # currently only consider one noise source
            snr_list = []
            for _, noise_tpl in enumerate(noise_data):
                assert isinstance(
                    noise_tpl, tuple
                ), "[SNRAdjustmentTransform] Noise should be a tuple"
                snr = self.sample_snr(noise_tpl[1])
                snr_list.append(snr)

            audio_mixture_clean, audio_mixture_noisy, scale = self.get_paired_data(
                audio_mixture_clean, noise_data, snr_list, speech_activity
            )
        else:
            audio_mixture_noisy = audio_mixture_clean

        processed_data.update({"mix_mc_sig": audio_mixture_noisy})
        processed_data.update({"meta": meta})
        return processed_data
