# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra
from pathlib import Path
import logging
from tqdm import tqdm
from re_format_alignment import convert_format


def prepare_librispeech_alignments(cfg):
    """Converts original TXT format of Librispeech alignments into CTMs.
    """
    for alignment_dir, output_alignment_dir in zip(
        cfg.alignment_dirs, cfg.output_alignment_dirs
    ):
        logging.info(f"Preparing {alignment_dir}")
        Path(output_alignment_dir).mkdir(exist_ok=True, parents=True)
        for alignment_path in tqdm(Path(alignment_dir).rglob("*.txt")):
            # some files in Librispeech alignments corpus are wrongly placed
            # we skip them here
            alignment_name = f"{alignment_path.parent.parent.stem}-{alignment_path.parent.stem}"
            if not alignment_path.stem.startswith(alignment_name):
                logging.warning(f"Skipping {alignment_path}")
                continue

            with open(alignment_path) as f:
                for line in f:
                    ctm_alignment, file_name = convert_format(line)
                    with open(
                        Path(output_alignment_dir) / f"{file_name}.ctm", "w"
                    ) as fw:
                        fw.write(ctm_alignment.decode())


@hydra.main(
    version_base=None, config_path="../config", config_name="main_from_scratch"
)
def main(cfg):
    prepare_librispeech_alignments(cfg.prepare_librispeech_alignments)


if __name__ == "__main__":
    main()
