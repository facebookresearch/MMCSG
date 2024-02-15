# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from pathlib import Path
import json
import hydra
import soundfile as sf
from scipy.signal import resample
import logging
from tqdm import tqdm


def get_word_segments(transcript_file):
    """Reads the word boundaries from the transcription files.

    The transcription file is expected to have a line per word of format:
    `{start}\t{end}\t{word}\t{speaker}`
    `start` and `end` are in seconds. `speaker` is 0/1 for SELF/OTHER, i.e.
    wearer of the glasses and conversational partner, respectively.
    """
    segments = []
    with open(transcript_file, "r") as f:
        for line in f:
            st, en, word, spk = line.strip().split("\t")
            st, en = float(st), float(en)
            segments.append((st, en, word.lower(), spk))
    segments = sorted(segments, key=lambda x: x[0])
    return segments


def get_chunks(segments, length):
    """Cuts a recording into chunks of approximate `length` in seconds.

    Each chunk is returned as list of word-segments to include in the chunk
    in the order of the words appearing. Chunks are cut respecting word
    boundaries, i.e. if `length` is 20.0 seconds, but 20.0s mark is
    in the middle of the word, the chunk will be extended by the length
    of the word, until the boundary does not interfere with any word segment.
    """
    candidate_end = length
    start = 0.0
    chunks = []

    while candidate_end < segments[-1][1]:
        segments_to_include = []
        for st, en, word, spk in segments:
            if en <= start:
                continue
            elif st >= start and en <= candidate_end:
                segments_to_include.append((st, en, word, spk))
            elif st >= candidate_end:
                break
            elif st >= start and en > candidate_end:
                candidate_end = en
                segments_to_include.append((st, en, word, spk))
        start = candidate_end
        candidate_end += length
        if segments_to_include != []:
            chunks.append(segments_to_include)
    return chunks


def write_audios(chunk_audio_dir, utt, chunks, s, fs):
    """Writes the chunked audios into `chunk_audio_dir`."""
    chunk_audio_dir.mkdir(parents=True, exist_ok=True)
    audio_files, durations = [], []
    for chunk in chunks:
        st = chunk[0][0]
        en = chunk[-1][1]
        s_chunk = s[int(st * fs) : int(en * fs)]
        filename = (
            chunk_audio_dir / f"{utt}_{round(st*1000)}_{round(en*1000)}.wav"
        )
        sf.write(filename, s_chunk, fs)
        audio_files.append(filename)
        durations.append(en - st)
    return audio_files, durations


def get_sot(chunks):
    """Composes SOT transcription for each chunk.

    Other than usual, our SOT (serialized-output-training) transcriptions have
    speaker labels before each word, as we found this forces the fine-tuning
    procedure to better learn about the speaker attribution.

    The speaker labels before each word are »0 or »1, denoting SELF and OTHER,
    respectively, SELF being the wearer of the glasses and OTHER
    the conversational partner.
    """
    sot_transcriptions = []
    for chunk in chunks:
        sot_transcript = ""
        for st, en, word, spk in chunk:
            sot_transcript += f"»{spk} {word} "
        sot_transcriptions.append(sot_transcript)
    return sot_transcriptions


def chunk_audios_transcriptions(
    utts,
    audio_dir,
    transcript_dir,
    chunk_audio_dir,
    chunk_length,
):
    """Chunks recordings, stored chunked audios and creates SOT transcriptions.

    Chunks are cut to be approximately `chunk_length` in seconds, but
    respecting word boundaries. SOT transcription for each chunk contains
    speaker labels »0 or »1 before each word. »0 denotes SELF (speaker
    wearing the glasses), »1 denotes OTHER (conversational partner).
    """
    audio_paths_all, durations_all, sot_transcriptions_all = [], [], []
    for utt in tqdm(utts):
        wavfile = audio_dir / f"{utt}.wav"
        s, fs = sf.read(wavfile)

        # resample to 16 kHz
        n_resampled = round(s.shape[0] * 16000 / fs)
        s = resample(s, n_resampled)
        fs = 16000

        word_segments = get_word_segments(transcript_dir / f"{utt}.tsv")
        chunks = get_chunks(word_segments, chunk_length)
        audio_paths, durations = write_audios(
            chunk_audio_dir, utt, chunks, s, fs
        )
        sot_transcriptions = get_sot(chunks)
        audio_paths_all.extend(audio_paths)
        durations_all.extend(durations)
        sot_transcriptions_all.extend(sot_transcriptions)

    return audio_paths_all, durations_all, sot_transcriptions_all


def create_manifest(
    utts,
    audio_dir,
    transcript_dir,
    manifest_file,
    chunk_audio_dir,
    chunk_length,
):
    """Prepares manifest files for fine-tuning of NeMO model.

    We fine-tune on chunks created from the training recordings, each of
    approximately `chunk_length` length. The chunks are cut to respect word
    boundaries.
    The resulting manifest file contains one line per chunk. Each line is in
    json format, with 'audio_filepath', 'duration', and 'text' fields.
    """
    logging.info('Creating chunks')
    audio_paths, durations, sot_transcriptions = chunk_audios_transcriptions(
        utts,
        audio_dir,
        transcript_dir,
        chunk_audio_dir,
        chunk_length,
    )
    manifest_file.parent.mkdir(exist_ok=True, parents=True)

    with open(manifest_file, "w") as fw:
        for audio_path, duration, sot_transcription in zip(
            audio_paths, durations, sot_transcriptions
        ):
            line = json.dumps({
                "audio_filepath": str(audio_path),
                "duration": duration,
                "text": sot_transcription,
            })
            fw.write(f"{line}\n")


def create_train_validation_split(audio_dir, train_ratio=0.9):
    """Split recordings into train and validation subsets.

    This is a very simple split that does not account for identity of speakers.
    However, in the baseline, validation is used solely for choosing the best
    epoch during training and this split seems to be sufficient for this.
    """
    utts = sorted([path.stem for path in audio_dir.glob("*.wav")])
    n_train_utts = int(round(len(utts) * train_ratio))
    train_utts = random.sample(utts, n_train_utts)
    valid_utts = [u for u in utts if u not in train_utts]
    return train_utts, valid_utts


def prepare_data(cfg):
    """Prepare data for fine-tuning of NeMO model and testing.

    First, the training recordings are split into train/validation subsets.
    For both of these subsets, recordings are split into chunks. Audio for
    each of these chunks is stored in `cfg.train_chunk_audio_dir` and
    `cfg.valid_chunk_audio_dir`. All chunks are then described in manifest
    files.
    """
    random.seed(111)
    logging.info('Spliting data into train/validation')
    train_utts, valid_utts = create_train_validation_split(
        Path(cfg.train_audio_dir), cfg.train_ratio
    )

    logging.info('Processing training data')
    create_manifest(
        train_utts,
        Path(cfg.train_audio_dir),
        Path(cfg.train_transcript_dir),
        Path(cfg.train_manifest_file),
        Path(cfg.train_chunk_audio_dir),
        cfg.chunk_length,
    )
    logging.info('Processing validation data')
    create_manifest(
        valid_utts,
        Path(cfg.train_audio_dir),
        Path(cfg.train_transcript_dir),
        Path(cfg.valid_manifest_file),
        Path(cfg.valid_chunk_audio_dir),
        cfg.chunk_length,
    )


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg):
    prepare_data(cfg.prepare_data)


if __name__ == "__main__":
    main()
