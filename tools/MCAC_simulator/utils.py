# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SPK_TOKENS = {"»0": 0, "»1": 1}
SPEAKER_CHANGE_TOKEN = "»"


@dataclass
class MixedAudio:
    mixed_audio: Optional[np.ndarray] = None
    near_field_audio: Optional[np.ndarray] = None
    far_field_audio: Optional[np.ndarray] = None
    distractor_audio: Optional[np.ndarray] = None
    noise_audio: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None
    length: int = -1


class WordAlignment(NamedTuple):
    # start / duration are both in seconds
    name: str
    channel: str
    start: float
    duration: float
    word: str

    def __str__(self):
        return f"{self.name} {self.channel} {self.start:.2f} {self.duration:.2f} {self.word}"


def get_ctm_list(
    ctm: Union[bytes, str],
    spk_label: str = "default",
    shift: float = 0.0,
    fill_gap: bool = True,
    append_channel_id: Optional[str] = None,
) -> List[WordAlignment]:
    """
    Give a CTM blob (can be bytes or string), return a list of word-level alignments
    it will also fill in SIL if there is a gap > 0.1s between consecutive words
    the CTM format is <file> <channel> <start-time> <duration> <word>

    Returns: a list of WordAlignments

    Args:
        spk_label (optional):
            will change <file> to spk_label, for multi talker scenerio
        shift (optional):
            in seconds, change the beginning of the segment from 0 to shift
        fill_gap (optional):
            if filling SIL between two consecutive words
            when there is a gap that is > 0.1s is needed or not
    """
    if type(ctm) == bytes:
        ctm = ctm.decode("utf-8")
    if ctm[0] == ctm[-1] == '"':
        ctm = ctm[1:-1]
    if ctm[0] == ctm[-1] == "'":
        ctm = ctm[1:-1]
    lines = ctm.strip().split("\\n")
    res = []
    previous_word_end = 0
    for x in lines:
        if len(x.split(" ")) != 5:
            continue
        id, id2, word_start, duration, word = x.split(" ")
        if append_channel_id is not None:
            id2 += append_channel_id
        word_gap = float(word_start) - previous_word_end
        # Add silence if fill_gap is true
        if fill_gap and word_gap > 0.1:
            gap_start = float(previous_word_end) + shift
            res.append(WordAlignment(spk_label, id2, gap_start, word_gap, "SIL"))
        res.append(
            WordAlignment(
                spk_label, id2, float(word_start) + shift, float(duration), word
            )
        )
        previous_word_end = float(word_start) + float(duration)

    return res


def _apply_fadein(audio, sr, duration=3.0):
    # convert to audio indices (samples)
    length = int(duration * sr)
    end = min(length, audio.shape[0])

    # compute fade in curve
    # linear fade
    fade_curve = np.linspace(0, 1.0, end)

    # apply the curve
    audio[:end] = audio[:end] * fade_curve


def _generate_audio_mixures(
    audio_list: List[np.ndarray], start_secs: List[float], fs: int = 16000
) -> List[np.ndarray]:

    start_frames = [int(start_sec * fs) for start_sec in start_secs]
    total_frames = max(
        s_frame + len(audio) for s_frame, audio in zip(start_frames, audio_list)
    )
    output_audio_list = []
    # _apply_fadein(b, fs, 1.0)
    for start_frame, audio in zip(start_frames, audio_list):
        if start_frame > 0:
            audio = np.concatenate((np.zeros(start_frame), audio)).astype(np.float32)
        if len(audio) < total_frames:
            audio = np.concatenate((audio, np.zeros(total_frames - len(audio)))).astype(
                np.float32
            )
        output_audio_list.append(audio)
    return output_audio_list


def _mix_audio_with_overlap(
    a: np.ndarray, b: np.ndarray, start_sec: float, fs: int = 16000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Give two audios a and b, with a start_sec indicating where b starts.
    Then pad audios with silence accordingly and return padded audio a and b.
    So that a + b is the mixed audio where a starts at 0 and b starts at start_sec.

    Returns: [a, b], if you want the mixed result just do a + b

    Args:
        a: audio which starts at 0
        b: audio which starts at start_sec
        start_sec: the start sec of audio b
        fs (optional): default 16000
    """

    start_frame = int(start_sec * fs)
    total_frames = max(len(a), start_frame + len(b))
    # _apply_fadein(b, fs, 1.0)
    if len(a) < total_frames:
        a = np.concatenate((a, np.zeros(total_frames - len(a)))).astype(np.float32)
    if start_frame > 0:
        b = np.concatenate((np.zeros(start_frame), b)).astype(np.float32)
    if len(b) < total_frames:
        b = np.concatenate((b, np.zeros(total_frames - len(b)))).astype(np.float32)
    return a, b


def _get_ctm_sot_from_word_alignment_list(
    word_alignment_list: List[WordAlignment],
    speaker_change_token: str = SPEAKER_CHANGE_TOKEN,
    strip_channel_id_token: Optional[str] = None,
) -> Tuple[List[WordAlignment], List[str]]:
    """
    Give word_alignment_list, remove SIL and add speaker change token with
    speaker name into the alignment, as well as generating the transcription with speaker
    change tokens.
    The speaker change token will inserted when there is a speaker change. The start/end timestamp for speaker
    change token will be the same as the next word.

    Returns: word_alignment_list and transcription

    Args:
        word_alignment_list: list of WordAlignment, where WordAlignment.name indicates speaker
        speaker_change_token (optional): default », inserted when there is a speaker change
    """

    new_ctm = []
    new_sot = []
    previous_speaker = None
    previous_channel = None
    # Get rid of SIL
    word_alignment_list = [x for x in word_alignment_list if x.word != "SIL"]
    # First speaker tag token
    first_word = word_alignment_list[0]
    spk_tag_start, spk_tag_dur = first_word.start, first_word.duration
    previous_channel = first_word.channel
    channel = first_word.channel
    if strip_channel_id_token is not None:
        channel = first_word.channel.split(strip_channel_id_token)[0]

    new_ctm.append(
        WordAlignment(
            first_word.name,
            channel,
            spk_tag_start,
            spk_tag_dur,
            speaker_change_token + first_word.name,
        )
    )
    new_sot.append(speaker_change_token + first_word.name)
    for word_alignment in word_alignment_list:
        if previous_speaker is not None and (
            previous_speaker != word_alignment.name
            or (
                strip_channel_id_token is not None
                and previous_channel != word_alignment.channel
            )
        ):
            # Insert speaker tag token
            spk_tag = word_alignment.name
            spk_tag_start = word_alignment.start
            spk_tag_dur = word_alignment.duration
            channel = word_alignment.channel
            if strip_channel_id_token is not None:
                channel = channel.split(strip_channel_id_token)[0]
            new_ctm.append(
                WordAlignment(
                    word_alignment.name,
                    channel,
                    spk_tag_start,
                    spk_tag_dur,
                    speaker_change_token + spk_tag,
                )
            )
            new_sot.append(speaker_change_token + spk_tag)

        previous_channel = word_alignment.channel
        if strip_channel_id_token is not None:
            word_alignment = WordAlignment(
                word_alignment.name,
                word_alignment.channel.split(strip_channel_id_token)[0],
                word_alignment.start,
                word_alignment.duration,
                word_alignment.word,
            )
        new_ctm.append(word_alignment)
        new_sot.append(word_alignment.word)
        previous_speaker = word_alignment.name

    return new_ctm, new_sot


def _segment_audio_and_adjust_alignment(
    audio: np.ndarray, word_alignment_list: List[WordAlignment], fs: int = 16000
) -> Tuple[np.ndarray, List[WordAlignment]]:
    """
    Extract the audio (numpy array) according to the list of WordAlignments
    Also adjust the output alignment to start from 0

    Returns: extracted audio, adjusted WordAlignments

    Args:
        audio:
            need to be numpy array
        word alignment list:
            a list of WordAlignment
        fs (optional):
            frame rate, by default = 16000
    """
    first_word_start = word_alignment_list[0].start
    last_word_end = word_alignment_list[-1].start + word_alignment_list[-1].duration
    start_frame = int(first_word_start * fs)
    end_frame = int(last_word_end * fs)
    segmented_audio = audio[start_frame:end_frame].astype(np.float32)
    # re-adjust ctm
    adjusted_word_alignment_list = []
    for word_alignment in word_alignment_list:
        new_word_start = word_alignment.start - first_word_start
        adjusted_word_alignment_list.append(
            WordAlignment(
                word_alignment.name,
                word_alignment.channel,
                new_word_start,
                word_alignment.duration,
                word_alignment.word,
            )
        )
    return segmented_audio, adjusted_word_alignment_list


def get_segmented_audio(
    audio: np.ndarray,
    word_alignment_list: List[WordAlignment],
    spk_tag: str,
    split_utt_threshold_min: float,
    split_utt_threshold_max: float,
    segment_max_sil_duration: float = 1.5,
) -> List[Tuple[np.ndarray, List[WordAlignment], str]]:
    """
    Get a list of segmented audio with WordAlignments plus a speaker tag
    This function will sample a silence gap from (split_utt_threshold_min, split_utt_threshold_max)
    and segment the audio and adjust the word alignments accordingly.

    Returns: a list of (segmented audio, adjusted WordAlignments, speaker tag)

    Args:
        audio:
            need to be numpy array
        word alignment list:
            a list of WordAlignment
        spk_tag:
            an indentifier of speaker, going to pass to further functions
        split_utt_threshold_min: used to sample a split threshold, in seconds
        split_utt_threshold_max: used to sample a split threshold, in seconds
        segment_max_sil_duration: the maximum silence we want, in seconds
    """
    res = []
    current_ctm = []
    split_threshold = np.random.uniform(
        split_utt_threshold_min, split_utt_threshold_max
    )
    for i in range(0, len(word_alignment_list)):
        word_alignment = word_alignment_list[i]
        current_ctm.append(word_alignment)
        if word_alignment.word == "SIL":
            silence_dur = word_alignment.duration
            if i > 0 and silence_dur >= split_threshold:
                # Split SIL equally, half goes to previous utt and half goes to next utt.
                current_ctm = current_ctm[:-1]
                sil_dur = min(word_alignment.duration, segment_max_sil_duration)

                sil_first_half = WordAlignment(
                    word_alignment.name,
                    word_alignment.channel,
                    word_alignment.start,
                    sil_dur / 2,
                    word_alignment.word,
                )

                sil_second_half = WordAlignment(
                    word_alignment.name,
                    word_alignment.channel,
                    word_alignment.start + word_alignment.duration - sil_dur / 2,
                    sil_dur / 2,
                    word_alignment.word,
                )

                current_ctm.append(sil_first_half)
                segmented_audio, adjusted_ctm = _segment_audio_and_adjust_alignment(
                    audio, current_ctm
                )
                res.append((segmented_audio, adjusted_ctm, spk_tag))
                # Init with a SIL
                current_ctm = [sil_second_half]
    if current_ctm:
        segmented_audio, adjusted_ctm = _segment_audio_and_adjust_alignment(
            audio, current_ctm
        )
        res.append((segmented_audio, adjusted_ctm, spk_tag))
    return res


def _get_start_time_of_next_segment(
    shift_frac: float,
    segments_start_end: List[List[float]],
    utt_spker: str,
    previous_spker: str,
):
    # Decide the start time of the next segment
    current_shift_frac = shift_frac
    # If the current segment is from the same speaker as previous one, shift_frac = 1
    if utt_spker == previous_spker:
        current_shift_frac = 1
    start_time = (
        segments_start_end[-1][0]
        + (segments_start_end[-1][1] - segments_start_end[-1][0]) * current_shift_frac
    )
    # Start time can't be earlier before the end of the previous to last segment.
    if len(segments_start_end) > 1:
        start_time = max(start_time, segments_start_end[-2][1])
    return start_time


def merge_segments(
    mini_utt_list: List[List[Tuple[np.ndarray, List[WordAlignment], str]]],
    shift_frac: float,
    s_tag: str = "0",
    add_spk_tag_at_all_spk_changes: bool = True,
) -> Tuple[MixedAudio, List[WordAlignment], List[str]]:
    """
    Iteratively merge two segmented audios with shift_frac

    Returns: MixedAudio, alignment and serialized output transcription

    Args:
        mini_utt_s:
            start_utt (wearer)
        mini_utt_t:
            target_utt (partner)
    """
    # source audio / ctm and target audio / ctm
    # make a copy
    infos = [mini_utt.copy() for mini_utt in mini_utt_list]
    merged_ctm = []
    segments_start_end = []  # start and end of each segments
    target_audio = None
    previous_spker = ""
    s_audio = np.array([])  # audio of clean source, after mixture
    t_audio = np.array([])  # audio of clean target, after mixture
    active_info_indexes = list(range(len(infos)))
    # while len(s_info) + len(t_info) > 0:
    while sum(len(info) for info in infos) > 0:
        # Get next segment
        idx = np.random.permutation(active_info_indexes)[0]
        while len(active_info_indexes) > 0 and len(infos[idx]) == 0:
            active_info_indexes.remove(idx)
            idx = np.random.permutation(active_info_indexes)[0]
        target_list = infos[idx]
        utt_audio, utt_ctm, utt_spker = target_list.pop(0)
        utt_end = utt_ctm[-1].start + utt_ctm[-1].duration

        # Edge case where this is the first segment
        if len(merged_ctm) == 0:
            merged_ctm.append(utt_ctm)
            if utt_spker == s_tag:
                s_audio = utt_audio
            else:
                t_audio = utt_audio
            target_audio = utt_audio
            segments_start_end.append([0, utt_end])
            previous_spker = utt_spker
            continue

        start_time = _get_start_time_of_next_segment(
            shift_frac, segments_start_end, utt_spker, previous_spker
        )

        audio_list = _mix_audio_with_overlap(target_audio, utt_audio, start_time)

        target_audio = audio_list[0] + audio_list[1]

        if utt_spker == s_tag:
            audio_list = _mix_audio_with_overlap(s_audio, utt_audio, start_time)
            s_audio = audio_list[0] + audio_list[1]
        else:
            audio_list = _mix_audio_with_overlap(t_audio, utt_audio, start_time)
            t_audio = audio_list[0] + audio_list[1]

        # change ctm
        updated_ctm = []
        for word_alignment in utt_ctm:
            new_word_start = word_alignment.start + start_time
            updated_ctm.append(
                WordAlignment(
                    word_alignment.name,
                    word_alignment.channel,
                    new_word_start,
                    word_alignment.duration,
                    word_alignment.word,
                )
            )
        merged_ctm.append(updated_ctm)
        segments_start_end.append([start_time, start_time + utt_end])
        previous_spker = utt_spker

    new_ctm = [item for sublist in merged_ctm for item in sublist]
    sorted_ctm = sorted(
        new_ctm,
        key=lambda x: x.start + x.duration,
    )
    # make same length
    s_audio, t_audio = _mix_audio_with_overlap(s_audio, t_audio, 0)
    strip_channel_id_token = "::" if add_spk_tag_at_all_spk_changes else None
    new_ctm, new_sot = _get_ctm_sot_from_word_alignment_list(
        sorted_ctm,
        strip_channel_id_token=strip_channel_id_token,
    )
    return (
        MixedAudio(
            mixed_audio=target_audio,
            near_field_audio=s_audio,
            far_field_audio=t_audio,
            length=len(s_audio),
        ),
        new_ctm,
        new_sot,
    )


def merge_direct(
    audio_list: List[np.ndarray],
    ctm_blob_list: List[Union[bytes, str]],
    shift_fracs: List[float],
    spk_tags: Tuple[str, ...],
    fs: int = 16000,
    add_spk_tag_at_all_spk_changes: bool = True,
) -> Tuple[MixedAudio, List[WordAlignment], List[str]]:
    assert len(audio_list) == len(ctm_blob_list)
    assert len(audio_list) == len(shift_fracs) + 1
    # Randomize order
    order = np.random.permutation(len(audio_list))
    audio_list = [audio_list[i] for i in order]
    ctm_blob_list = [ctm_blob_list[i] for i in order]
    spk_tags = tuple(spk_tags[i] for i in order)

    start_sec = [0.0]
    for i in range(1, len(audio_list)):
        shift_sec = len(audio_list[i - 1]) * shift_fracs[i - 1] / fs
        start_sec.append(start_sec[-1] + shift_sec)

    audio_list = _generate_audio_mixures(audio_list, start_sec)

    # now deal with ctm and ref
    unsorted_ctm = []
    for i, (ctm_blob, spk_tag, shift) in enumerate(
        zip(ctm_blob_list, spk_tags, start_sec)
    ):
        append_channel_id = f"::{i}" if add_spk_tag_at_all_spk_changes else None
        unsorted_ctm += get_ctm_list(
            ctm_blob,
            spk_tag,
            shift=shift,
            append_channel_id=append_channel_id,
        )

    sorted_ctm = sorted(
        unsorted_ctm,
        key=lambda x: x.start + x.duration,
    )

    strip_channel_id_token = "::" if add_spk_tag_at_all_spk_changes else None
    new_ctm, new_sot = _get_ctm_sot_from_word_alignment_list(
        sorted_ctm,
        strip_channel_id_token=strip_channel_id_token,
    )

    near_field_audio = np.sum(
        [audio_list[i] for i in range(len(audio_list)) if spk_tags[i] == "0"],
        axis=0,
        dtype=np.float32,
    )
    far_field_audio = np.sum(
        [audio_list[i] for i in range(len(audio_list)) if spk_tags[i] == "1"],
        axis=0,
        dtype=np.float32,
    )
    return (
        MixedAudio(
            mixed_audio=np.sum(audio_list, axis=0, dtype=np.float32),
            near_field_audio=near_field_audio,
            far_field_audio=far_field_audio,
            length=len(audio_list[0]),
        ),
        new_ctm,
        new_sot,
    )


def _add_distractor(
    mixed_audio: MixedAudio,
    distractor_audio: np.ndarray,
    ctm: List[WordAlignment],
    overlap_ratio: Optional[float] = None,
) -> Tuple[MixedAudio, List[WordAlignment]]:
    """
    Add a distractor audio to mixed_audio, the start of the distractor audio
    is sampled between [-len(distractor_audio), len(mixed_audio)]
    which means distractor audio can start before mixed audio, and can end
    after mixed audio. The returned results audio length will be adjusted.

    e.g.

    |-----distractor---|
       |---mixed_audio------|

           |-----distractor---|
    |---mixed_audio------|

    We can specify an overlap ratio, if specified, the distractor will overlap with the main speech
    with the specified ratio, e.g. if overlap ratio = 0%, result can be
    |-----distractor---|
                         |---mixed_audio------|
    or
                          |-----distractor---|
    |---mixed_audio------|

    Current implementation didn't change the length of the audio, until we figure out
    an AR training strategy.

    Returns: MixedAudio, ctm

    Args:
        mixed_audio:
        distractor:
        ctm:
    """
    mixed_audio_len = mixed_audio.length
    distractor_audio_len = len(distractor_audio)
    start_frame = np.random.randint(-distractor_audio_len, mixed_audio_len)
    if overlap_ratio is not None:
        # either distractor starts first or mixed_audio starts first
        # if distractor starts first, we have -start_frame = distractor_len * (1 - overlap_ratio)
        # if mixed audio starts first, we have mix_len - start_frame = distractor_len * overlap_ratio
        start_frame = random.choice(
            [
                -distractor_audio_len * (1 - overlap_ratio),
                max(0, mixed_audio_len - overlap_ratio * distractor_audio_len),
            ]
        )
        start_frame = int(start_frame)
    total_len = max(
        start_frame + distractor_audio_len,
        distractor_audio_len,
        mixed_audio_len,
        mixed_audio_len - start_frame,
    )
    # if start frame < 0, then we need to adjust the ctm
    if start_frame < 0:
        shift_sec = -start_frame / 16000.0
        new_ctm = []
        for align in ctm:
            new_ctm.append(
                WordAlignment(
                    align.name,
                    align.channel,
                    align.start + shift_sec,
                    align.duration,
                    align.word,
                )
            )
        ctm = new_ctm
        # adjust the audio as well
        if mixed_audio.mixed_audio is not None:
            mixed_audio.mixed_audio = np.concatenate(
                (np.zeros(-start_frame), mixed_audio.mixed_audio)
            ).astype(np.float32)
        if mixed_audio.near_field_audio is not None:
            mixed_audio.near_field_audio = np.concatenate(
                (np.zeros(-start_frame), mixed_audio.near_field_audio)
            ).astype(np.float32)
        if mixed_audio.far_field_audio is not None:
            mixed_audio.far_field_audio = np.concatenate(
                (np.zeros(-start_frame), mixed_audio.far_field_audio)
            ).astype(np.float32)
    else:
        distractor_audio = np.concatenate(
            (np.zeros(start_frame), distractor_audio)
        ).astype(np.float32)

    # pad zero if needed
    if mixed_audio.mixed_audio is not None and len(mixed_audio.mixed_audio) < total_len:
        mixed_audio.mixed_audio = np.concatenate(
            (
                mixed_audio.mixed_audio,
                np.zeros(total_len - len(mixed_audio.mixed_audio)),
            )
        ).astype(np.float32)
    if (
        mixed_audio.near_field_audio is not None
        and len(mixed_audio.near_field_audio) < total_len
    ):
        mixed_audio.near_field_audio = np.concatenate(
            (
                mixed_audio.near_field_audio,
                np.zeros(total_len - len(mixed_audio.near_field_audio)),
            )
        ).astype(np.float32)
    if (
        mixed_audio.far_field_audio is not None
        and len(mixed_audio.far_field_audio) < total_len
    ):
        mixed_audio.far_field_audio = np.concatenate(
            (
                mixed_audio.far_field_audio,
                np.zeros(total_len - len(mixed_audio.far_field_audio)),
            )
        ).astype(np.float32)
    if len(distractor_audio) < total_len:
        distractor_audio = np.concatenate(
            (
                distractor_audio,
                np.zeros(total_len - len(distractor_audio)),
            )
        ).astype(np.float32)

    mixed_audio.distractor_audio = distractor_audio
    # pyre-ignore
    mixed_audio.mixed_audio = mixed_audio.mixed_audio + distractor_audio
    return mixed_audio, ctm


def generate_single_speaker_speech(
    audio: np.ndarray,
    ctm_blob: Union[bytes, str],
    is_near_field: bool = True,
    spk_tag: str = "0",
    distractor_audio: Optional[np.ndarray] = None,
    distractor_overlap_ratio: Optional[float] = None,
    skip_speaker_tag: bool = False,
) -> Tuple[MixedAudio, bytes, bytes]:

    """
    Generate single speaker speech, basically wrap the single speech to MixedAudio

    Args:
        audio: np.ndarray,
        ctm: Union[bytes, str],
        is_near_field: bool = True, if False then it's far-field audio
        spk_tag: str = "0",
    """

    if is_near_field:
        mixed_audio = MixedAudio(
            mixed_audio=audio,
            near_field_audio=audio,
            length=len(audio),
        )
    else:
        mixed_audio = MixedAudio(
            mixed_audio=audio,
            far_field_audio=audio,
            length=len(audio),
        )
    ctm_with_tag = get_ctm_list(ctm_blob, spk_tag)
    ctm, sot = _get_ctm_sot_from_word_alignment_list(ctm_with_tag)

    # add distractor if any
    if distractor_audio is not None:
        mixed_audio, ctm = _add_distractor(
            mixed_audio, distractor_audio, ctm, distractor_overlap_ratio
        )

    if skip_speaker_tag:
        ctm = [x for x in ctm if len(x.word) > 0 and x.word[0] != SPEAKER_CHANGE_TOKEN]
        sot = [x for x in sot if len(x) > 0 and x[0] != SPEAKER_CHANGE_TOKEN]
    # encode ctm and sot into bytes
    ctm = "\\n".join([str(x) for x in ctm]).encode("utf-8")
    sot = " ".join(sot).encode("utf-8")

    return mixed_audio, ctm, sot


def grouping_and_random_speaker_drop(
    drop_random_speaker_prob: Tuple[float, ...],
    speaker_tags: Tuple[str, ...],
    spk_grouping: Optional[List[List[int]]] = None,
) -> List[int]:
    """This function groups speakers and randomly drop speakers.
    If there are more speakers than speaker_tags, then randomly drop speakers per spk_tag
    given a probability but at least keep one speaker per spk_tag.
    """
    assert isinstance(
        drop_random_speaker_prob, tuple
    ), "drop_random_speaker_prob should be a tuple of floating point numbers"
    assert len(drop_random_speaker_prob) == 1 or len(drop_random_speaker_prob) == len(
        speaker_tags
    )
    if len(drop_random_speaker_prob) == 1:
        # apply the same drop probability to all speakers if only one prob is provided
        drop_random_speaker_prob = tuple(
            list(drop_random_speaker_prob) * len(speaker_tags)
        )
    num_spk_tags = len(set(speaker_tags))
    if spk_grouping is not None:
        # select a speaker group based on the provided list of indexes. For example, if
        # we have four speaker handles [self1, other1, self2, other2] and spk_grouping = [[0, 1, 3], [1, 2]]
        # then self1, other1, and other2 or self2 and other1 will be selected to be combined in the
        # generated mixture. Note, there must be at least one speaker per speaker tag.
        assert all(len(indexes) >= num_spk_tags for indexes in spk_grouping)
        indexes = spk_grouping[np.random.randint(len(spk_grouping))]
        if len({speaker_tags[i] for i in indexes}) != num_spk_tags:
            raise ValueError("There must be at least one speaker per speaker tag.")
    else:
        indexes = list(range(len(speaker_tags)))
    # apply the selected group of speakers
    speaker_drop_probs = [drop_random_speaker_prob[i] for i in indexes]
    spk_tags = [speaker_tags[i] for i in indexes]

    # randomly drop a speaker from a list of speakers given a list of probabilities
    if max(speaker_drop_probs) > 0 and len(spk_tags) > num_spk_tags:
        spk_tags_group = defaultdict(list)
        drop_spk_prop_group = defaultdict(list)
        for i, spk_tag in enumerate(spk_tags):
            spk_tags_group[spk_tag].append(i)
            drop_spk_prop_group[spk_tag].append(speaker_drop_probs[i])
        drop_spk_prop_group = {
            spk_tag: np.asarray(drop_spk_prop_group[spk_tag])
            for spk_tag in drop_spk_prop_group
        }
        rprop = {
            spk_tag: np.random.random(len(spk_tags_group[spk_tag]))
            for spk_tag in spk_tags_group
        }
        for spk_tag in spk_tags_group:
            L = len(spk_tags_group[spk_tag])
            if L > 1:
                idx_drop = rprop[spk_tag] < drop_spk_prop_group[spk_tag]
                if idx_drop.all():
                    # do not drop all speakers with the same speaker tag and
                    # keep at least one speaker per spk_tag
                    spk_tags_group[spk_tag] = [
                        spk_tags_group[spk_tag][rprop[spk_tag].argmin()]
                    ]
                else:
                    spk_tags_group[spk_tag] = [
                        spk_tags_group[spk_tag][i] for i in range(L) if not idx_drop[i]
                    ]
        idx2keep = sorted(
            i for spk_tag in spk_tags_group for i in spk_tags_group[spk_tag]
        )
    else:
        idx2keep = range(len(spk_tags))
    # map indexes back to original indexing, before group selection
    return [indexes[i] for i in idx2keep]


def generate_mixture(
    audio_list: List[np.ndarray],
    ctm_list: List[Union[bytes, str]],
    shift_frac: float,
    segment_audio: bool = False,
    segment_max_sil_duration: float = 1.5,
    spk_tags: Tuple[str, ...] = ("0", "1"),
    split_utt_threshold_min: float = 0.1,
    split_utt_threshold_max: float = 0.5,
    distractor_audio: Optional[np.ndarray] = None,
    distractor_overlap_ratio: Optional[float] = None,
    add_spk_tag_at_all_spk_changes: bool = True,
    drop_random_speaker_prob: Tuple[float, ...] = (0.0,),
    spk_grouping: Optional[List[List[int]]] = None,
    skip_speaker_tag: bool = False,
) -> Tuple[MixedAudio, bytes, bytes]:
    """
    Generate two speakers' speech, with the option to add a distractor's speech

    Returns:
        MixedAudio: The mixed audio as well as each clean audio for each speaker
        ctm_blob: The encoded wordalignment (ctm)
        sot_blob: The encoded serialized transcription (sot)

    Args:
        first_audio: np.ndarray, first audio, near field
        second_audio: np.ndarray, second audio, far field
        first_ctm: Union[bytes, str], CTM blob, can be in bytes of str
        second_ctm: Union[bytes, str], ditto
        shift_frac: float, indicating the amount of shift to apply to second audio. 1 means second audio starts after first audio,
            0 means second audio starts within the same time as first audio.
        segment_audio: bool = False, whether to merge directly or segment audios first and merge segments.
        segment_max_sil_duration: max silence we want when segmenting audios, in seconds.
        first_spk_tag: str = "0", speaker tag for self,
        second_spk_tag: str = "1", speaker tag for other,
        split_utt_threshold_min: float = 0.1, the min threshold used when sampling a silence gap to segment audios
        split_utt_threshold_max: float = 0.5, the max thrshold used when sampling a silence gap to segment audios
        distractor_audio: Optional[np.ndarray] = None, optional, if not null then it's used as a distractor audio.
        distractor_overlap_ratio: Optional[float] = None, if not specified, then distractor can appear at anywhere in the mix.
            If specified (more in evaluation case), distractor_overlap_ratio should be between 0 and 1 and indicates the
            percentage of distractor audio that is overlapped with the main speech.
        add_spk_tag_at_all_spk_changes: bool = True, whether to add a spk tag to all spk changes even when the spk_tag is
            the same compared to the previous speaker but a different speaker is added.
        drop_random_speaker_prob: Tuple[float, ...] = (0.0,), the probability to drop a speaker if multiple speakers per spk_tag are
            available. It is made sure that at least one speaker per spk_tag is kept. The drop probability per speaker can
            be defined by providing a sequence of drop probabilities, otherwise the same probability is applied to all speakers.
        spk_grouping: Optional[List[List[int]]] = None, a list of indexes, which are used to only select a group of utterances
            from the list of audio handles for generating the output mixture. For example, if the list of speaker utterances is
            [self1, other1, self2, other2] and the spk_grouping = [[0, 1, 2], [1, 2, 3]], then only the speaker handles
            [self1, other1, self2] or [other1, self2, other2] can be combined in the generated audio mixture.
        skip_speaker_tag: bool = False, whether to skip adding speaker tags to sot/ctm.
    """
    idx2keep = grouping_and_random_speaker_drop(
        drop_random_speaker_prob, spk_tags, spk_grouping
    )
    audio_list = [audio_list[idx] for idx in idx2keep]
    ctm_list = [ctm_list[idx] for idx in idx2keep]
    spk_tags = tuple(spk_tags[idx] for idx in idx2keep)

    if not segment_audio:
        mixed_audio, ctm, sot = merge_direct(
            audio_list,
            ctm_list,
            [shift_frac] * (len(audio_list) - 1),
            spk_tags,
            add_spk_tag_at_all_spk_changes=add_spk_tag_at_all_spk_changes,
        )
    else:
        audio_segs_list = []
        for i, (audio, ctm, spk_tag) in enumerate(zip(audio_list, ctm_list, spk_tags)):
            # To distinguish between different speakers with the same spk_tag, we append
            # an ID to the channel ID, which is removed at the end.
            append_channel_id = f"::{i}" if add_spk_tag_at_all_spk_changes else None
            audio_segs = get_segmented_audio(
                audio,
                get_ctm_list(ctm, spk_tag, append_channel_id=append_channel_id),
                spk_tag,
                split_utt_threshold_min,
                split_utt_threshold_max,
                segment_max_sil_duration,
            )
            audio_segs_list.append(audio_segs)
        mixed_audio, ctm, sot = merge_segments(
            audio_segs_list,
            shift_frac,
            add_spk_tag_at_all_spk_changes=add_spk_tag_at_all_spk_changes,
        )

    # add distractor if any
    if distractor_audio is not None:
        mixed_audio, ctm = _add_distractor(
            mixed_audio, distractor_audio, ctm, distractor_overlap_ratio
        )

    if skip_speaker_tag:
        ctm = [x for x in ctm if len(x.word) > 0 and x.word[0] != SPEAKER_CHANGE_TOKEN]
        sot = [x for x in sot if len(x) > 0 and x[0] != SPEAKER_CHANGE_TOKEN]
    # encode ctm and sot into bytes
    ctm = "\\n".join([str(x) for x in ctm]).encode("utf-8")
    sot = " ".join(sot).encode("utf-8")

    return mixed_audio, ctm, sot
