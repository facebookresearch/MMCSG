# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import hydra
import logging


def load_transcription(path):
    transcript = []
    with open(path) as f:
        for line in f:
            _, timestamp, word, speaker = line.strip().split()
            timestamp = float(timestamp)
            transcript.append((timestamp, word, speaker))
    return transcript


def test_timestamps(cfg):
    utt2perturbation_start = {}
    with open(cfg.perturbation_list) as f:
        for line in f:
            utt, perturbation_start = line.strip().split()
            utt2perturbation_start[utt] = float(perturbation_start)

    n_incorrect = 0
    for utt, start in utt2perturbation_start.items():
        hypo_orig = load_transcription(
            Path(cfg.hypotheses_original_dir) / f'{utt}')
        hypo_pert = load_transcription(
            Path(cfg.hypotheses_perturbed_dir) / f'{utt}')

        for i, hypo_item_orig in enumerate(hypo_orig):
            timestamp_orig, word_orig, speaker_orig = hypo_item_orig
            if timestamp_orig > start:
                break
            timestamp_pert, word_pert, speaker_pert = hypo_pert[i]
            if speaker_pert != speaker_orig:
                logging.warning(f'{i}th word in original hypothesis '
                                f'for utt {utt} '
                                f'{word_orig}-{speaker_orig} with timestamp '
                                f'{timestamp_orig} does not correspond to '
                                f'{i}th word in perturbed hypothesis '
                                f'{word_pert}-{speaker_pert} with timestamp '
                                f'{timestamp_pert}')
                n_incorrect += 1
                break
            if word_pert != word_orig or timestamp_pert != timestamp_orig:
                # acceptable case: this is the last word with timestamp < start
                # and in the perturbed case, it got prolonged =>
                # word_orig is prefix or word_pert
                is_last_word = i == len(hypo_orig) - \
                    1 or hypo_orig[i+1][0] > start
                is_prefix = word_pert.startswith(word_orig)
                if not is_last_word or not is_prefix:
                    logging.warning(f'{i}th word in original hypothesis '
                                    f'for utt {utt} '
                                    f'{word_orig}-{speaker_orig} with timestamp '
                                    f'{timestamp_orig} does not correspond to '
                                    f'{i}th word in perturbed hypothesis '
                                    f'{word_pert}-{speaker_pert} with timestamp '
                                    f'{timestamp_pert}')
                    n_incorrect += 1
                    break

    if n_incorrect == 0:
        logging.info('TEST PASSED SUCCESSFULLY')
    else:
        logging.info(
            f'TEST FAILED FOR {n_incorrect}/{len(utt2perturbation_start)}')


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg):
    test_timestamps(cfg.test_timestamps)


if __name__ == "__main__":
    main()
