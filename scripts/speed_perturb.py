# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra
from pathlib import Path
import random
from torchaudio.transforms import SpeedPerturbation
from tqdm import tqdm
import soundfile as sf
import torch
import torchaudio
import logging


def perturb_alignment(factor, ctm_file, ctm_output_file):
    """Modifies timings in CTMs accoring to the speed perturbation factor."""
    with open(ctm_file) as f, open(ctm_output_file, "w") as fw:
        for line in f:
            utt_number, channel, start, dur, label = line.strip().split()[:5]
            start, dur = float(start), float(dur)
            start /= factor
            dur /= factor
            fw.write(f"{utt_number} {channel} {start:.3f} {dur:.3f} {label}\n")


def speed_perturb_one_split(
    audiofile_list,
    alignment_dir,
    output_audio_dir,
    output_alignment_dir,
    factor_range,
):
    for audiofile in tqdm(audiofile_list):
        # Skipping recordings with no alignments
        # These will not be used for training anyway.
        if not (alignment_dir / f"{audiofile.stem}.ctm").is_file():
            logging.warning(f"No alignment for {audiofile.stem}")
            continue
        s, fs = torchaudio.load(str(audiofile))

        factor = (
            random.random() * (factor_range[1] - factor_range[0])
            + factor_range[0]
        )
        speed_perturb = SpeedPerturbation(fs, [factor])
        s_perturbed, out_length = speed_perturb(s, torch.Tensor([s.shape[-1]]))

        sf.write(
            output_audio_dir / f"{audiofile.stem}_sp{factor:.3f}.wav",
            s_perturbed.numpy()[0],
            fs,
        )

        perturb_alignment(
            factor,
            alignment_dir / f"{audiofile.stem}.ctm",
            output_alignment_dir / f"{audiofile.stem}_sp{factor:.3f}.ctm",
        )


def speed_perturb(cfg):
    for (
        audio_dir,
        output_audio_dir,
        alignment_dir,
        output_alignment_dir,
    ) in zip(
        cfg.audio_dirs,
        cfg.output_audio_dirs,
        cfg.alignment_dirs,
        cfg.output_alignment_dirs,
    ):
        output_audio_dir = Path(output_audio_dir)
        output_alignment_dir = Path(output_alignment_dir)
        output_audio_dir.mkdir(exist_ok=True, parents=True)
        output_alignment_dir.mkdir(exist_ok=True, parents=True)

        audiofile_list = sorted(
            list(Path(audio_dir).rglob("*.wav"))
            + list(Path(audio_dir).rglob("*.flac"))
        )
        n_per_process = len(audiofile_list) // cfg.n_splits + 1
        speed_perturb_one_split(
            audiofile_list[
                cfg.i_split * n_per_process: (cfg.i_split + 1) * n_per_process
            ],
            Path(alignment_dir),
            output_audio_dir,
            output_alignment_dir,
            cfg.factor_range,
        )


@hydra.main(
    version_base=None, config_path="../config", config_name="main_from_scratch"
)
def main(cfg):
    speed_perturb(cfg.speed_perturb)


if __name__ == "__main__":
    main()
