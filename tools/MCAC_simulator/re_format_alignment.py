# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pathlib

from pathlib import Path


def convert_format(text_line):
    text_line = text_line.replace('"', "").strip().split()
    assert len(text_line) == 3
    file_name, word_labels, timesteps = text_line
    word_list = word_labels.split(",")
    timestep_list = timesteps.split(",")
    assert len(word_list) == len(timestep_list)
    new_format = []
    for index, (word, end_time) in enumerate(zip(word_list, timestep_list)):
        if index == 0:
            start_time = 0.0
        if word == "":
            word = "SIL"
        new_format.append(
            "1 A {} {} {}".format(
                start_time, str(round(float(end_time) - float(start_time), 3)), word
            )
        )
        start_time = end_time
    return "\\n".join([str(x) for x in new_format]).encode("utf-8"), file_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-original-path",
        type=str,
        default=None,
        required=True,
        help="PATH for the original alignment data",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        required=True,
        help="output path for the converted alignment data",
    )
    args = parser.parse_args()
    data_path = pathlib.Path(args.data_original_path)
    aligment_files = list(data_path.rglob("*.txt"))
    print(f"No. of Alignment files to process: {len(aligment_files)}")
    for aligment_file in aligment_files:
        with open(aligment_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                new_format, file_name = convert_format(line)
                output_prefix = "_".join(
                    str(Path(*Path(aligment_file).parts[-4:-1])).split("/")
                )
                output_file_name = output_prefix + "_" + file_name + ".txt"
                output_file = os.path.join(
                    args.output_path,
                    output_file_name,
                )
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "wb") as f:
                    f.write(new_format)


if __name__ == "__main__":
    main()  # pragma: no cover
