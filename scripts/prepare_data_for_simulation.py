# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra
from pathlib import Path
from tqdm import tqdm
import logging
import random
import resampy
import json
import soundfile as sf

random.seed(111)


def prepare_alignments(alignment_dirs, prepared_alignment_dir):
    """Prepares alignments for simulation.

    Alignments from all input directories are gather into one
    `prepared_alignment_dir` directory. CTMs are modified to contain
    `SIL` words at any place with a gap between two original words.
    This is necessary so that simulation scripts correctly chunk
    the recordings.
    """
    prepared_alignment_dir.mkdir(parents=True, exist_ok=True)
    for alignment_dir in alignment_dirs:
        logging.info(f"Preparing directory {alignment_dir}")
        for ctm_file in tqdm(Path(alignment_dir).glob("*.ctm")):
            ctm_items = []
            with open(ctm_file) as f:
                prev_end = 0.0
                for line in f:
                    src_id, channel, start, dur, token = line.strip().split()[:5]
                    start, dur = float(start), float(dur)
                    if start > prev_end:
                        ctm_items.append(
                            (
                                src_id,
                                channel,
                                prev_end,
                                start - prev_end,
                                "SIL",
                            )
                        )
                    ctm_items.append((src_id, channel, start, dur, token))
                    prev_end = start + dur

            with open(prepared_alignment_dir / ctm_file.name, "w") as fw:
                for src_id, channel, start, dur, token in ctm_items:
                    fw.write(
                        f"{src_id} {channel} {start:.3f} {dur:.3f} {token}\n"
                    )


def prepare_simulation_pairs(
    simulation_pairs_dir, prepared_alignments_dir, audio_dirs, n_parts
):
    """Pairs audio files to be used as SELF and OTHER in multitalker simulation
    
    All audio files with alignments are gathered. Pairing is done randomly,
    i.e. by randomly shuffling the order of files for OTHER speaker. This can
    lead to pairing where SELF and OTHER contain the same speaker or even the
    same recording. However, due to the large amount of input data and
    speakers, this happens rarely so we do not take special care of this case.

    As we do not use distractors in our simulation, we just insert the OTHER
    recording as a placeholder for distractor. This gets ignored in the actual
    simulation. **If you plan to use distractors in your simulation,
    this needs to be changed.**
    """
    audio_files = []
    for audio_dir in audio_dirs:
        audio_files.extend(audio_dir.rglob("*.wav"))

    # filter to only the audios that have alignments
    stem2audio_path = {
        audio_file.stem: audio_file for audio_file in audio_files
    }
    alignments = {p.stem for p in Path(prepared_alignments_dir).glob("*.ctm")}
    audio_files = [
        stem2audio_path[stem] for stem in alignments if stem in stem2audio_path
    ]
    audio_files_pair = random.sample(audio_files, len(audio_files))
    pairs = list(zip(audio_files, audio_files_pair))

    n = len(pairs)
    n_per_split = n // n_parts + 1

    simulation_pairs_dir.mkdir(exist_ok=True, parents=True)
    for i in range(n_parts):
        with open(simulation_pairs_dir / f"{i}.tsv", "w") as fw:
            fw.write("self\tother\tdistractor\n")
            for self_path, other_path in pairs[
                i * n_per_split : (i + 1) * n_per_split
            ]:
                fw.write(
                    f"{str(self_path)}\t{str(other_path)}\t{str(other_path)}\n"
                )


def resample_noise(noises_dir, noises_16k_dir):
    """Resample DNS noises from 48kHz to 16kHz."""
    noises_16k_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Resampling noises to 16000Hz")
    for wavfile in tqdm(noises_dir.glob("*.wav")):
        s, fs = sf.read(wavfile)
        s_16k = resampy.resample(s, fs, 16000)
        sf.write(noises_16k_dir / wavfile.name, s_16k, 16000)


def prepare_simulation_config(
    transforms_template_path,
    transforms_path,
    noise_dir,
    rir_dir,
    noise_list,
    rir_list,
):
    """Prepares configuration file for simulation.

    Simulation scripts use a JSON configuration file, that specifies
    the transformations used when simulating the multi-speaker data.
    Among these, `SampleNoiseRemoteTransform` uses a list of noise wav-files
    and `SampleApplyRIRTransform` uses a list of room impulse responses.

    This function creates the lists of noises and RIRs and inserts paths
    of these lists into a template of the configuration file. That is,
    three files are created in this function:
    - `noise_list` with list of noises
    - `rir_list` with list of RIRs
    - `transforms_path` with the updates configuration
    """
    with open(transforms_template_path) as f:
        transform = json.load(f)

    with open(noise_list, "w") as fw:
        for wavpath in noise_dir.glob("*.wav"):
            fw.write(str(wavpath.resolve()) + "\n")
    transforms_noise = [
        t for t in transform if t[0] == "SampleNoiseRemoteTransform"
    ]
    for t in transforms_noise:
        t[1]["noise_datasets"][0][0] = str(noise_list.resolve())

    with open(rir_list, "w") as fw:
        for wavpath in rir_dir.glob("*.pkl"):
            fw.write(str(wavpath.resolve()) + "\n")
    transforms_rir = [
        t for t in transform if t[0] == "SampleApplyRIRTransform"
    ]
    for t in transforms_rir:
        t[1]["rir_datasets"][0][0] = str(rir_list.resolve())

    with open(transforms_path, "w") as fw:
        json.dump(transform, fw, indent=2)


def prepare_data_for_simulation(cfg):
    prepare_alignments(
        [Path(alignment_dir) for alignment_dir in cfg.alignment_dirs],
        Path(cfg.prepared_alignments_dir),
    )
    prepare_simulation_pairs(
        Path(cfg.simulation_pairs_dir),
        Path(cfg.prepared_alignments_dir),
        [Path(audio_dir) for audio_dir in cfg.audio_dirs],
        cfg.n_parts,
    )

    resample_noise(Path(cfg.noises_dir), Path(cfg.noises_16k_dir))
    
    prepare_simulation_config(
        Path(cfg.transforms_template_path),
        Path(cfg.transforms_path),
        Path(cfg.noises_16k_dir),
        Path(cfg.rir_dir),
        Path(cfg.noise_list),
        Path(cfg.rir_list),
    )


@hydra.main(
    version_base=None, config_path="../config", config_name="main_from_scratch"
)
def main(cfg):
    prepare_data_for_simulation(cfg.prepare_data_for_simulation)


if __name__ == "__main__":
    main()
