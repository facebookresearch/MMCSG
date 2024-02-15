# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import logging

import numpy as np
import soundfile as sf


logger = logging.getLogger(__name__)



def load_column(path, col_id):
    """
    Args:
        path: path to a TSV file
        col_id: column id

    Returns:
        list of items in the col_id column
    """
    with open(path, "r") as f:
        return [line[col_id] for line in csv.reader(f, delimiter="\t")]


class SampleNoiseRemoteTransform(object):
    """
    This is a variant of SampleNoiseDatasetTransform.
    Instead of taking the noise samples from the input
    (packed clean and noise audio samples),
    the noise segments will be taken directly form lists provided at initilaization.
    In this way we don't have to run an additional flow to pack noise and clean audio samples,
    saving significantly data preperation time as well as storage space.
    """

    def __init__(
        self,
        noise_datasets,
        fs,
        noise_sources_model=(0.92, 0.07, 0.01),
        random_seed=None,
    ):
        """
        Args:
            noise_datasets: list of
                    tuple (path, col_id, id, weight)
                    or dict {'path':..., 'col_id':..., 'id':..., 'weight':...}
                where:
                    path: path to a TSV file
                    col_id: column ID of the noise handle in the TSV file
                    id: integer ID of the noise dataset. it will be used by
                        subsequent data augmentation transforms to identify
                        the source of the noise audio.
                    weight: float value of how often this dataset will be sampled.
                            Values should be positive but do not need to be summed to one.
            fs: desired sampling rate
            noise_sources_model: chances of number of noise segments
                                 to sample for each clean signal. For example
                                 (.92, .07, .01) means there is
                                    * 92% chance one noise samples
                                    *  7% chance two noise samples
                                    *  1% chance three noise samples
                                 the values represent relative weight values
                                 (meaining that they do not need to sum to one).
            random_seed: seed for the random number generator
            sample_single_channel: if True and noise signal is multi-channel,
                                   randomly select one channel from it.

        Note:
            All elements in a single dataset have the same chance being sampled
        """
        super().__init__()

        # convert to tuple form
        noise_datasets = [
            (x["path"], x["col_id"], x["id"], x["weight"]) if isinstance(x, dict) else x
            for x in noise_datasets
        ]

        # load noise datasets (lists of handles) into memory
        self.noise_lists = [
            load_column(path, col_id) for path, col_id, _, _ in noise_datasets
        ]
        # misc params
        self.dataset_ids = [i for _, _, i, _ in noise_datasets]
        # normalize
        weights = [weight for _, _, _, weight in noise_datasets]
        weight_sum = sum(weights)
        self.dataset_weights = [weight / weight_sum for weight in weights]
        self.noise_sources_model = noise_sources_model
        self.rng = np.random.default_rng(random_seed)
        self.fs = fs

        # log some settings
        logger.info("Weights for noise datasets:")
        for (path, _, _, _), weight in zip(noise_datasets, self.dataset_weights):
            logger.info(f"\t{weight:.3f}\t{path}")
        logger.info("Weights for number of noise sources:")
        for i, weight in enumerate(noise_sources_model):
            weight /= sum(noise_sources_model)
            logger.info(f"\t{weight:.3f}\t{i+1} noise")

    def _sample_and_load_noise(self):
        # choose noise dataset based on their weights
        list_id = self.rng.choice(range(len(self.noise_lists)), p=self.dataset_weights)
        # sample one segment from the dataset
        noise_handle = str(self.rng.choice(self.noise_lists[list_id]))
        # load signal
        try:
            noise_sig, fs = sf.read(noise_handle)
        except Exception as e:
            logger.error(f"Failed to load {noise_handle} from noise dataset #{list_id}")
            raise e
        return noise_sig, self.dataset_ids[list_id]

    def __call__(self, data):
        """
        Args:
            data: list of np array, only the first element is used
                  (as the clean signal)

        Returns
            a list:
                first element is the clean signal (np array)
                the rest elemtes are tuples of
                    (noise segment (np array), noise dataset id)
        """
        # sample number of noise segments
        noise_count = (
            self.rng.choice(
                range(len(self.noise_sources_model)), p=self.noise_sources_model
            )
            + 1
        )

        noise_list = [self._sample_and_load_noise() for _ in range(noise_count)]
        out_data = data + noise_list
        return out_data
