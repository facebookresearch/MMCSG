# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np

from utils import generate_mixture, generate_single_speaker_speech

EPS = np.finfo(float).eps

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OverlapMultiTalkerTransform(object):
    """
    This transform allows us to simulate multi talker scenerio for multi-channel speech.
    Inputs:
        * segment_audio=False,
        * first_spk_tag="0",
        * second_spk_tag="1",
        * split_utt_threshold_min=0.1,
        * split_utt_threshold_max=0.5,
    """

    def __init__(
        self,
        add_mixture_prob=0.8,
        none_overlap_frac=0.2,
        shifting_fraction_min=0.0,
        shifting_fraction_max=1.0,
        fs=16000,
        prob_segment_audio=0.5,
        split_utt_threshold_min=0.1,
        split_utt_threshold_max=0.5,
        prob_add_distractor=0.8,
        first_spk_tag="0",
        second_spk_tag="1",
        distractor_overlap_ratio=None,
        seed=None,
        single_speaker_near_field_prob=0.5,
        add_spk_tag_at_all_spk_changes=True,
        speaker_tags_per_col=None,
        random_speaker_tags_per_col=False,
        drop_random_speaker_prob=(0.0,),
        spk_grouping=None,
        skip_speaker_tag=False,
    ):
        self.add_mixture_prob = add_mixture_prob
        self.none_overlap_frac = none_overlap_frac
        self.shifting_fraction_min = shifting_fraction_min
        self.shifting_fraction_max = shifting_fraction_max
        self.prob_segment_audio = prob_segment_audio
        self.split_utt_threshold_min = split_utt_threshold_min
        self.split_utt_threshold_max = split_utt_threshold_max
        self.prob_add_distractor = prob_add_distractor
        self.first_spk_tag = first_spk_tag
        self.second_spk_tag = second_spk_tag
        self.distractor_overlap_ratio = distractor_overlap_ratio
        self.rng = np.random.default_rng(seed)
        self.single_speaker_near_field_prob = single_speaker_near_field_prob
        # add_spk_tag_at_all_spk_changes: Controls whether to add a speaker tag label to
        # the transcription when the speaker changes but the speaker tag is not compared to the
        # previous speaker. This only has an effect when we provide multiple speakers per
        # speaker tag. Otherwise the speaker tag is always added regardless of this
        self.add_spk_tag_at_all_spk_changes = add_spk_tag_at_all_spk_changes
        # speaker_tags_per_col: can be used to define the speaker tag for each column.
        # If it is None, then the speaker tag is alternating between columns starting
        # with first_spk_tag.
        self.speaker_tags_per_col = speaker_tags_per_col
        # random_speaker_tags_per_col: If it is False and speaker_tags_per_col is not defined,
        # then speaker tags are alternating between first an second speaker tag
        # starting with first_spk_tag. If random_speaker_tags_per_col is True, then a random
        # speaker tag is attributed to each audio handle
        self.random_speaker_tags_per_col = random_speaker_tags_per_col
        # drop_random_speaker_prob: If more than two speakers are present, which means more than
        # one per speaker tag, then drop_random_speaker_prob can be used to define a probablity,
        # e.g., 0.1, for randomly dropping speakers per speaker tag but it assures that at least
        # one speaker per speaker tag is remaining. If multiple drop probabilities are specified,
        # then the length must match the number speakers such that a probablilty per speaker is provided.
        self.drop_random_speaker_prob = tuple(drop_random_speaker_prob)
        # spk_grouping: If not None, then this specifies a list of indexes to be used to select
        # a group of speaker utterances when generating the output mixture. For example, if the list
        # of speakers is [self1, other1, self2, other2] and spk_grouping = [[0, 1, 2], [1, 2, 3]],
        # then only the speaker handles [self1, other1, self2] or [other1, self2, other2] can be combined
        # in the generated audio mixtures. If None, then all speaker utterances can occure together,
        # which is similar to using spk_grouping = [[0, 1, 2, 3]] in the above example.
        self.spk_grouping = spk_grouping
        self.skip_speaker_tag = skip_speaker_tag

    def _get_shift_frac(self):
        if self.rng.random() <= self.none_overlap_frac:
            shift_frac = 1
        else:
            shift_frac = self.rng.uniform(
                self.shifting_fraction_min, self.shifting_fraction_max
            )
        return shift_frac

    def __call__(self, data):
        """
        data: [spk1, spk2, ..., spkN, distractor, alignment1, alignment2, ..., alignmentN]

        each elements:
        * spk1: first speaker's audio, numpy array
        * spk2: second speaker's audio, numpy array
        * ...
        * spkN: the Nth speaker's audio, numpy array
        * distractor: distractor audio, numpy array
        * alignment1: alignment in CTM format for spk1
        * alignment2: alignment in CTM format for spk2
        * ...
        * alignmentN: alignment in CTM format for spkN
        * noise: noise audio, numpy array


        Return [mixed_audio, ctm, sot, noise_id]
        class MixedAudio:
            mixed_audio: Optional[np.ndarray] = None
            near_field_audio: Optional[np.ndarray] = None
            far_field_audio: Optional[np.ndarray] = None
            distractor_audio: Optional[np.ndarray] = None
            noise_audio: Optional[np.ndarray] = None
            metadata: Optional[Dict] = None
            length: int = -1
        """
        num_spks = len(data) // 2 - 1
        audio_list = data[:num_spks]
        distractor = data[num_spks]
        alignments = data[num_spks + 1 : -1]
        noise, noise_id = data[-1]
        ctm_list = alignments
        spk_tags = [self.first_spk_tag, self.second_spk_tag]
        if self.speaker_tags_per_col is None:
            # assuming speaker tags are alternating
            spk_tag_list = [spk_tags[i % 2] for i in range(num_spks)]
        else:
            spk_tag_list = self.speaker_tags_per_col
        if self.random_speaker_tags_per_col:
            spk_tag_list = np.random.permutation(spk_tag_list).tolist()
        assert set(spk_tag_list) == set(spk_tags)

        # check whether we want to add distractor
        is_add_distractor = self.rng.random() <= self.prob_add_distractor
        if self.rng.random() > self.add_mixture_prob:  # no mix
            spk1_audio = audio_list[0]
            spk2 = audio_list[1]
            alignment1 = ctm_list[0]
            alignment2 = ctm_list[1]
            if not is_add_distractor:
                distractor = None
            if self.rng.random() < self.single_speaker_near_field_prob:
                mixed_audio, ctm, sot = generate_single_speaker_speech(
                    audio=spk1_audio,
                    ctm_blob=alignment1,
                    is_near_field=True,
                    spk_tag=self.first_spk_tag,
                    distractor_audio=distractor,
                    distractor_overlap_ratio=self.distractor_overlap_ratio,
                    skip_speaker_tag=self.skip_speaker_tag,
                )
            else:
                mixed_audio, ctm, sot = generate_single_speaker_speech(
                    audio=spk2,
                    ctm_blob=alignment2,
                    is_near_field=False,
                    spk_tag=self.second_spk_tag,
                    distractor_audio=distractor,
                    distractor_overlap_ratio=self.distractor_overlap_ratio,
                    skip_speaker_tag=self.skip_speaker_tag,
                )
        else:
            shift_frac = self._get_shift_frac()
            is_segment_audio = self.rng.random() <= self.prob_segment_audio
            if not is_add_distractor:
                distractor = None

            mixed_audio, ctm, sot = generate_mixture(
                audio_list=audio_list,
                ctm_list=ctm_list,
                shift_frac=shift_frac,
                distractor_audio=distractor,
                distractor_overlap_ratio=self.distractor_overlap_ratio,
                segment_audio=is_segment_audio,
                spk_tags=tuple(spk_tag_list),
                split_utt_threshold_min=self.split_utt_threshold_min,
                split_utt_threshold_max=self.split_utt_threshold_max,
                add_spk_tag_at_all_spk_changes=self.add_spk_tag_at_all_spk_changes,
                drop_random_speaker_prob=self.drop_random_speaker_prob,
                spk_grouping=self.spk_grouping,
                skip_speaker_tag=self.skip_speaker_tag,
            )
        mixed_audio.noise_audio = noise
        return [mixed_audio, ctm, sot, noise_id]
