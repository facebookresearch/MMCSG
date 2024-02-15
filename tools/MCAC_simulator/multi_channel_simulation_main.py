# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import json
import logging
import os
from pathlib import Path

import numpy as np
import soundfile as sf


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from .compose import Compose


def _get_fixed_speaker_transform(transforms_config, speaker="self"):
    config = transforms_config.copy()
    for transform in config:
        if transform[0] == "OverlapMultiTalkerTransform":
            ovl_tran_conf = transform[1]
            if speaker == "self":
                ovl_tran_conf["add_mixture_prob"] = 0
                ovl_tran_conf["single_speaker_near_field_prob"] = 1
                return Compose.setup_from_config(config)
            elif speaker == "other":
                ovl_tran_conf["add_mixture_prob"] = 0
                ovl_tran_conf["single_speaker_near_field_prob"] = 0
                return Compose.setup_from_config(config)
            elif speaker == "self+other":
                ovl_tran_conf["add_mixture_prob"] = 1
                return Compose.setup_from_config(config)
    else:
        raise ValueError(f"Unknown speaker {speaker}")


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def build_args_parser():
    parser = argparse.ArgumentParser(
        "Prepare multi-channel data for noise suppression and speech recognition tasks."
    )
    parser.add_argument("--save-meta", action="store_true")
    parser.add_argument(
        "--output-audio-handle",
        action="store_true",
        help="Whether to upload and output audio handle",
    )
    parser.add_argument(
        "--input-data-tsv",
        type=str,
        help="the input table including self, other, distractor wav file path.",
    )
    parser.add_argument(
        "--ctm-path",
        type=str,
        help="the folder that saves the processed alignment files",
    )
    parser.add_argument(
        "--output-root-path",
        type=str,
        help="the root folder where the processed data will be saved, which will contain subfolders for wav and metadata.",
    )

    # ---------------------------------------------------------------------------------
    # Config Specification group of arguments
    # ---------------------------------------------------------------------------------
    group_config = parser.add_mutually_exclusive_group(required=True)

    group_config.add_argument(
        "--transforms-config-file",
        type=str,
        help="JSON file containing the config to create the transforms to be applied",
    )

    return parser


def run_simulation() -> None:
    parser = build_args_parser()
    args = parser.parse_args()
    logger.info(args.transforms_config_file)
    # load transforms config
    with open(args.transforms_config_file, "r") as f:
        transforms_config = json.load(f)
    composed_transforms = Compose.setup_from_config(transforms_config)
    output_dir_wav = os.path.join(args.output_root_path, "simulated_wav")
    if not os.path.exists(output_dir_wav):
        os.mkdir(output_dir_wav)
    output_dir_wav_metadata = os.path.join(args.output_root_path, "simulated_metadata")
    if not os.path.exists(output_dir_wav_metadata):
        os.mkdir(output_dir_wav_metadata)

    # source path from input tsv file: self, other, distractor
    with open(args.input_data_tsv, "r") as f:
        lines = csv.reader(f, delimiter="\t")
        next(lines, None)  # skip the headers
        # start processing
        print("start processing...")
        for line in lines:
            assert (
                len(line) == 3
            ), "the input tsv format is not correct, needs to be included <self, other, distractor>"
            self_speaker_path, other_speaker_path, distractor_speaker_path = line
            # load audio files
            self_spk_audio, fs = sf.read(self_speaker_path)
            other_spk_aduio, fs = sf.read(other_speaker_path)
            distractor_audio, fs = sf.read(distractor_speaker_path)

            # load alignment files
            self_spk_ctm_name = (
                "_".join(str(Path(*Path(self_speaker_path).parts[-4:-1])).split("/"))
                + "_"
                + os.path.basename(self_speaker_path).replace(".wav", ".txt")
            )
            self_spk_ctm_path = os.path.join(args.ctm_path, self_spk_ctm_name)
            with open(self_spk_ctm_path, "rb") as f:
                self_spk_ctm = f.readline()
            other_spk_ctm_name = (
                "_".join(str(Path(*Path(other_speaker_path).parts[-4:-1])).split("/"))
                + "_"
                + os.path.basename(other_speaker_path).replace(".wav", ".txt")
            )
            other_spk_ctm_path = os.path.join(args.ctm_path, other_spk_ctm_name)
            with open(other_spk_ctm_path, "rb") as f:
                other_spk_ctm = f.readline()
            data = [
                self_spk_audio,
                other_spk_aduio,
                distractor_audio,
                self_spk_ctm,
                other_spk_ctm,
            ]
            # apply transforms
            transformed_data = composed_transforms(data)
            # save simulated wav files
            simulated_wav_name = (
                os.path.basename(self_speaker_path)[:-4]
                + "_"
                + os.path.basename(other_speaker_path)[:-4]
                + ".wav"
            )
            simulated_wav_path = os.path.join(output_dir_wav, simulated_wav_name)
            sf.write(simulated_wav_path, transformed_data["mix_mc_sig"], fs)
            # save simulated metadata
            simulated_ctm_name = (
                os.path.basename(self_speaker_path)[:-4]
                + "_"
                + os.path.basename(other_speaker_path)[:-4]
                + ".ctm"
            )
            simulated_ctm_path = os.path.join(
                output_dir_wav_metadata, simulated_ctm_name
            )
            with open(simulated_ctm_path, "w") as f:
                f.write(transformed_data["meta"]["alignment"])
            # save simulated metadata
            if args.save_meta:
                save_meta = json.dumps(transformed_data["meta"], cls=NumpyEncoder)
                meta_name = (
                    os.path.basename(self_speaker_path)[:-4]
                    + "_"
                    + os.path.basename(other_speaker_path)[:-4]
                    + ".meta"
                )
                meta_path = os.path.join(output_dir_wav_metadata, meta_name)
                with open(meta_path, "w") as f:
                    f.write(save_meta)
        print("Finished...")

if __name__ == "__main__":
    run_simulation()  # pragma: no cover
