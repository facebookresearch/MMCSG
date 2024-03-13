# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra
from pathlib import Path
from multi_channel_simulation_main import run_simulation


@hydra.main(
    version_base=None, config_path="../config", config_name="main_from_scratch"
)
def main(cfg):
    Path(cfg.simulate.simulated_data_dir).mkdir(exist_ok=True, parents=True)
    run_simulation(
        input_data_tsv=Path(cfg.simulate.simulation_pairs_dir)
        / f"{cfg.simulate.i_split}.tsv",
        ctm_path=cfg.simulate.prepared_alignment_dir,
        output_root_path=cfg.simulate.simulated_data_dir,
        transforms_config_file=cfg.simulate.transforms_config_path,
        save_meta=cfg.simulate.save_metadata,
    )


if __name__ == "__main__":
    main()
