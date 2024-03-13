# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra
from pathlib import Path
import soundfile as sf
import random
import json
from tqdm import tqdm
from src.text_normalization import normalize_transcriptions
from src.text_normalization import load_permitted_subs

random.seed(111)


def create_train_validation_split(infos, train_ratio):
    n = len(infos)
    n_train = int(round(train_ratio * n))
    idx_train = random.sample(list(range(n)), n_train)
    idx_valid = [i for i in range(n) if i not in idx_train]
    infos_train = [infos[i] for i in idx_train]
    infos_valid = [infos[i] for i in idx_valid]
    return infos_train, infos_valid


def turn_level_to_word_level_sot(words):
    """Converts transcriptions with turn-level speaker tokens into word-level

    The simulation scripts output transcriptions with speaker-tokens per turn
    e.g. "»0 hello »1 hello can I help you »0 yes"
    We found that for our model, word-level speaker tokens 
    lead to better convergence, we thus reformat the transcriptions to:
    "»0 hello »1 hello »1 can »1 I »1 help »1 you »0 yes"
    """
    speaker = None
    words_converted = []
    for word in words:
        if word == "»0":
            speaker = 0
        elif word == "»1":
            speaker = 1
        else:
            words_converted.append(f"»{speaker}")
            words_converted.append(word)
    return words_converted


def get_sot_transcript(ali_path, permitted_subs):
    """Forms SOT transcription from CTMs.

    The inputs CTMs already contain speaker tokens. Here, we gather
    the words from the CTM, normalize the transcription, and convert
    turn-level SOT to word-level SOT.
    """
    end_times = []
    words = []
    speakers = []

    with open(ali_path) as f:
        for line in f:
            spk, _, st, dur, word = line.strip().split()
            end_times.append(float(st) + float(dur))
            words.append(word)
            speakers.append(int(spk))

    utt2transcription = {"_": list(zip(words, end_times, speakers))}
    words_norm = [
        w
        for w, _, _, in normalize_transcriptions(
            utt2transcription, permitted_subs
        )["_"]
    ]
    return " ".join(turn_level_to_word_level_sot(words_norm))


def prepare_simulated_data(cfg):
    infos = []
    permitted_subs = load_permitted_subs(Path(cfg.permitted_subs_path))
    for wav_path in tqdm(Path(cfg.wav_dir).glob("*.wav")):
        ali_path = Path(cfg.ali_dir) / f"{wav_path.stem}.ctm"
        sound = sf.SoundFile(wav_path)
        duration = sound.frames / sound.samplerate
        sot_transcript = get_sot_transcript(ali_path, permitted_subs)
        info = {
            "audio_filepath": str(wav_path),
            "text": sot_transcript,
            "duration": duration,
        }
        infos.append(info)

    infos_train, infos_valid = create_train_validation_split(
        infos, cfg.train_ratio
    )

    with open(cfg.output_train_manifest, "w") as fw:
        for info in infos_train:
            fw.write(json.dumps(info) + "\n")
    with open(cfg.output_valid_manifest, "w") as fw:
        for info in infos_valid:
            fw.write(json.dumps(info) + "\n")


@hydra.main(
    version_base=None, config_path="../config", config_name="main_from_scratch"
)
def main(cfg):
    prepare_simulated_data(cfg.prepare_simulated_data)


if __name__ == "__main__":
    main()
