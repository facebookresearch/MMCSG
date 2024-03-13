# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import hydra
import soundfile as sf
import librosa
import torch
from torchaudio.pipelines import MMS_FA as bundle
from src.text_normalization import load_permitted_subs, normalize_words


def read_stm_segments(tedlium_stm_dir, permitted_subs_path):
    """Reads information about segments from STM files.

    TEDLIUM dataset provides long-form recordings of TED talks.
    The information about how to segment these is present in STM format.
    This function parses the STMs and returns information about the segments.
    It also normalizes the transcriptions so that they are ready for alignment.
    """
    permitted_subs = load_permitted_subs(permitted_subs_path)
    utt2segments = defaultdict(list)
    for stmpath in tedlium_stm_dir.glob("*.stm"):
        with open(stmpath) as f:
            for line in f:
                _, _, _, st, en, _, text = line.strip().split(maxsplit=6)
                st, en = float(st), float(en)
                # TEDLIUM has spaces before apostrophes, such as "it 's"
                text = text.replace("<unk>", "").replace(" '", "'")
                words = text.split()
                words_norm = [
                    w
                    for w, _, _ in normalize_words(
                        list(zip(words, range(len(words)), range(len(words)))),
                        permitted_subs,
                    )
                ]
                text_norm = " ".join(words_norm)
                utt2segments[stmpath.stem].append((st, en, text_norm))
    return utt2segments


def get_aligner():
    """Initializes torchaudio aligner.

    As in the following tutorial:
    https://pytorch.org/audio/main/tutorials/forced_alignment_for_multilingual_data_tutorial.html
    """
    device = torch.device("cuda")
    model = bundle.get_model()
    model.to(device)
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    return tokenizer, model, aligner, device


def align_segment(
    segment_path, waveform, fs, transcript, tokenizer, model, aligner, device
):
    """Force-aligns words of a TEDLIUM segment.

    As in the following tutorial:
    https://pytorch.org/audio/main/tutorials/forced_alignment_for_multilingual_data_tutorial.html
    """
    assert fs == bundle.sample_rate
    # Some transcriptions contain e.g. digits and other symbols not present
    # in the tokenizer of the aligner. Because there are not many of these,
    # we simply skip them.
    if not all([c in tokenizer.dictionary or c == " " for c in transcript]):
        logging.warning(
            f"Skipping {segment_path} because it contains "
            "characters not present in tokenizer"
        )
        return
    if transcript.strip() == "":
        logging.warning(f"Skipping {segment_path} because transcript empty")
        return
    words = transcript.split()
    ctm_items = []
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        num_frames = emission.shape[1]
        tokens = tokenizer(words)
        # segment is too short for the aligner to work properly
        if not check_token_length(tokens, num_frames):
            logging.warning(f"Skipping {segment_path} because it's too short")
            return
        token_spans = aligner(emission[0], tokens)
        ratio = waveform.shape[1] / num_frames
        for word_span, word in zip(token_spans, words):
            word_start = ratio * word_span[0].start / fs
            word_end = ratio * word_span[-1].end / fs
            ctm_items.append((word, word_start, word_end))

    return ctm_items


def segment_and_align_tedlium_data(
    utt2segments,
    tedlium_audio_dir,
    output_tedlium_segments_dir,
    permitted_subs_path,
    output_tedlium_alignments_dir,
):
    output_tedlium_segments_dir.mkdir(parents=True, exist_ok=True)
    output_tedlium_alignments_dir.mkdir(exist_ok=True, parents=True)

    tokenizer, model, aligner, device = get_aligner()

    logging.info("Segmenting and aligning TEDLIUM data")

    for utt, segments in tqdm(utt2segments.items()):
        s, fs = librosa.load(tedlium_audio_dir / f"{utt}.sph", sr=16000)
        for seg in segments:
            st, en, text = seg
            s_seg = s[int(round(st * fs)) : int(round(en * fs))]
            name = f"{utt}_{int(round(st*1000))}_{int(round(en*1000))}"
            sf.write(output_tedlium_segments_dir / f"{name}.wav", s_seg, fs)

            s_seg = torch.from_numpy(s_seg)[None].float()

            ctm_items = align_segment(
                name, s_seg, fs, text, tokenizer, model, aligner, device
            )

            if ctm_items:
                with open(
                    output_tedlium_alignments_dir / f"{name}.ctm", "w"
                ) as fw:
                    for word, start, end in ctm_items:
                        fw.write(f"1 A {start:.3f} {end-start:.3f} {word}\n")


def check_token_length(tokens, num_frames):
    """
    The aligner requires that the number of tokens in the transcription
    is larger than the number of input frames.
    """
    tokens_flat = [token for word in tokens for token in word]

    r = 0  # count blanks that need to be inserted between repeated tokens
    for i in range(1, len(tokens_flat)):
        if tokens_flat[i - 1] == tokens_flat[i]:
            r += 1
    return num_frames >= len(tokens_flat) + r


def prepare_tedlium_data(cfg):
    utt2segments = read_stm_segments(
        Path(cfg.tedlium_stm_dir), Path(cfg.permitted_subs_path)
    )
    segment_and_align_tedlium_data(
        utt2segments,
        Path(cfg.tedlium_audio_dir),
        Path(cfg.output_tedlium_segments_dir),
        Path(cfg.permitted_subs_path),
        Path(cfg.output_tedlium_alignments_dir),
    )


@hydra.main(
    version_base=None, config_path="../config", config_name="main_from_scratch"
)
def main(cfg):
    prepare_tedlium_data(cfg.prepare_tedlium_data)


if __name__ == "__main__":
    main()
