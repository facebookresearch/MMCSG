# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import hydra
import soundfile as sf
from tqdm import tqdm
import random
import moviepy.editor as mpy
import numpy as np
from functools import partial


def perturb_video_after_time(get_frame, t, start_time):
    frame = get_frame(t)
    if t > start_time:
        return np.zeros_like(frame)
    else:
        return frame


def perturb_audio_after_time(get_frame, t, start_time):
    frame = get_frame(t)
    frame[t > start_time] = 1e-3 * \
        np.random.randn(*frame[t > start_time].shape)
    return frame


def prepare_perturbed_data(cfg):
    """Prepare perturbed data for testing correctness of provded timestamps.
    """
    random.seed(111)
    np.random.seed(111)
    utt2perturbation_start = {}
    with open(cfg.perturbation_list) as f:
        for line in f:
            utt, perturbation_start = line.strip().split()
            utt2perturbation_start[utt] = float(perturbation_start)

    (Path(cfg.audio_perturbed_dir) / 'perturbed').mkdir(parents=True, exist_ok=True)
    (Path(cfg.audio_perturbed_dir) / 'unperturbed').mkdir(parents=True, exist_ok=True)
    if cfg.perturb_video:
        (Path(cfg.video_perturbed_dir) /
         'perturbed').mkdir(parents=True, exist_ok=True)
        (Path(cfg.video_perturbed_dir) /
         'unperturbed').mkdir(parents=True, exist_ok=True)
    if cfg.perturb_accelerometer:
        (Path(cfg.accelerometer_perturbed_dir) /
         'perturbed').mkdir(parents=True, exist_ok=True)
        (Path(cfg.accelerometer_perturbed_dir) /
         'unperturbed').mkdir(parents=True, exist_ok=True)
    if cfg.perturb_gyroscope:
        (Path(cfg.gyroscope_perturbed_dir) /
         'perturbed').mkdir(parents=True, exist_ok=True)
        (Path(cfg.gyroscope_perturbed_dir) /
         'unperturbed').mkdir(parents=True, exist_ok=True)

    for utt in tqdm(sorted(list(utt2perturbation_start.keys()))):
        # perturb audio
        s, fs = sf.read(Path(cfg.audio_dir) / f'{utt}.wav')
        sf.write(Path(cfg.audio_perturbed_dir) /
                 'unperturbed' / f'{utt}.wav', s, fs)
        start = int(round(utt2perturbation_start[utt] * fs))
        s_perturbed = np.copy(s)
        s_perturbed[start:] = 1e-3 * \
            np.random.randn(*s_perturbed[start:].shape)
        sf.write(Path(cfg.audio_perturbed_dir) /
                 'perturbed' / f'{utt}.wav', s_perturbed, fs)

        # perturb video
        if cfg.perturb_video:
            with mpy.VideoFileClip(str(Path(cfg.video_dir) / f'{utt}.mp4')) as clip:
                clip.write_videofile(
                    str(Path(cfg.video_perturbed_dir) / 'unperturbed' / f'{utt}.mp4'))
                clip_perturbed = clip.fl(
                    partial(perturb_video_after_time,
                            start_time=utt2perturbation_start[utt]))
                clip_perturbed.audio = clip_perturbed.audio.fl(
                    partial(perturb_audio_after_time,
                            start_time=utt2perturbation_start[utt]
                            ),
                    apply_to='audio'
                )
                clip_perturbed.write_videofile(
                    str(Path(cfg.video_perturbed_dir) / 'perturbed' / f'{utt}.mp4'))

        # perturb IMU
        if cfg.perturb_accelerometer:
            acc = np.load(Path(cfg.accelerometer_dir) / f'{utt}.npy')
            np.save(Path(cfg.accelerometer_perturbed_dir) /
                    'unperturbed' / f'{utt}.npy', acc)
            acc_perturbed = np.copy(acc)
            acc_perturbed[np.arange(
                acc_perturbed[:, 0].size) / 1000 > utt2perturbation_start[utt]] = 0.0
            np.save(Path(cfg.accelerometer_perturbed_dir) /
                    'perturbed' / f'{utt}.npy', acc_perturbed)
        if cfg.perturb_gyroscope:
            gyro = np.load(Path(cfg.gyroscope_dir) / f'{utt}.npy')
            np.save(Path(cfg.gyroscope_perturbed_dir) /
                    'unperturbed' / f'{utt}.npy', gyro)
            gyro_perturbed = np.copy(gyro)
            gyro_perturbed[np.arange(
                gyro_perturbed[:, 0].size) / 1000 > utt2perturbation_start[utt]] = 0.0
            np.save(Path(cfg.gyroscope_perturbed_dir) /
                    'perturbed' / f'{utt}.npy', gyro_perturbed)


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg):
    prepare_perturbed_data(cfg.prepare_perturbed_data)


if __name__ == "__main__":
    main()
