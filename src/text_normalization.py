# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import yaml


def remove_punctuation(s):
    return re.sub("[\)\(.=?\-,><\[\]+~#^!]", "", s)


def load_permitted_subs(permitted_subs_path):
    with open(permitted_subs_path) as f:
        permitted_subs = yaml.safe_load(f)
        permitted_subs = {
            remove_punctuation(key).lower(): remove_punctuation(value).lower()
            for key, value in permitted_subs.items()
        }
    return permitted_subs


def normalize_words(transcription, permitted_subs):
    """Text normalization of a sequence of words.

    Transforms everything into lower-case. Removes punctuation.
    Substitutes words according to the list of permitted substitutons
    to normalized variants.
    When substituting words, care needs to be taken to deal properly with
    timestamps. E.g. when a single word is replaced by multiple words,
    timestamps need to be distributed to all of the created words. When
    multiple words are merged into one, timestamp of the last of the original
    words is assigned.
    """
    if len(transcription) == 0:
        return transcription
    words, timestamps, idxs = zip(*transcription)
    words = [remove_punctuation(word).lower() for word in words]

    # first substitutions of single words (into single or multiple words)
    words_sub, timestamps_sub, idxs_sub = [], [], []
    for word, timestamp, idx in zip(words, timestamps, idxs):
        if word in permitted_subs:
            # handles also cases when word is replaced by multiple words
            for subword in permitted_subs[word].split(" "):
                if subword != "":
                    words_sub.append(subword)
                    timestamps_sub.append(timestamp)
                    idxs_sub.append(idx)
        else:
            words_sub.append(word)
            timestamps_sub.append(timestamp)
            idxs_sub.append(idx)
    words, timestamps, idxs = words_sub, timestamps_sub, idxs_sub

    # loop by two-word tuples to handle cases when multiple words are replaced
    # (in our list of substitutions, there are no more then two-word tuples)
    words_sub, timestamps_sub, idxs_sub = [], [], []
    i = 0
    while i < len(words) - 1:
        word_tuple = f"{words[i]} {words[i+1]}"
        if word_tuple in permitted_subs:
            for subword in permitted_subs[word_tuple].split(" "):
                if subword != "":
                    words_sub.append(subword)
                    timestamps_sub.append(timestamps[i + 1])
                    idxs_sub.append(idxs[i + 1])
            i += 2
        else:
            words_sub.append(words[i])
            timestamps_sub.append(timestamps[i])
            idxs_sub.append(idxs[i])
            i += 1
    if i == len(words) - 1:
        words_sub.append(words[i])
        timestamps_sub.append(timestamps[i])
        idxs_sub.append(idxs[i])

    return zip(words_sub, timestamps_sub, idxs_sub)

def normalize_transcriptions(utt2transcription, permitted_subs):
    """Wrapper around `normalize_words` for all utterances and both speakers."""
    utt2transcription_norm = {}
    for utt, transcription in utt2transcription.items():
        speaker0_transcription = sorted(
            [
                (word_timestamp_speaker[0], word_timestamp_speaker[1], idx)
                for idx, word_timestamp_speaker in enumerate(transcription)
                if word_timestamp_speaker[2] == 0
            ],
            key=lambda wti: wti[1],
        )
        speaker1_transcription = sorted(
            [
                (word_timestamp_speaker[0], word_timestamp_speaker[1], idx)
                for idx, word_timestamp_speaker in enumerate(transcription)
                if word_timestamp_speaker[2] == 1
            ],
            key=lambda wti: wti[1],
        )

        speaker0_transcription_norm = normalize_words(
            speaker0_transcription, permitted_subs
        )
        speaker1_transcription_norm = normalize_words(
            speaker1_transcription, permitted_subs
        )

        utt2transcription_norm[utt] = sorted(
            [
                (word, timestamp, 0, idx)
                for word, timestamp, idx in speaker0_transcription_norm
            ]
            + [
                (word, timestamp, 1, idx)
                for word, timestamp, idx in speaker1_transcription_norm
            ],
            key=lambda wtsi: wtsi[-1],
        )
        utt2transcription_norm[utt] = [(w,t,s) for w,t,s,i in utt2transcription_norm[utt]]

    return utt2transcription_norm
