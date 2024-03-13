# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra

# hacking cmd arguments to be able to import functions from NeMO
import sys
argv_bk = sys.argv
sys.argv = [sys.argv[0]] + ["--data_root", "", "--manifest", ""]
from process_asr_text_tokenizer import (
    __process_data,
    __build_document_from_manifests,
)
sys.argv = argv_bk


@hydra.main(
    version_base=None, config_path="../config", config_name="main_from_scratch"
)
def main(cfg):
    text_corpus_path = __build_document_from_manifests(
        cfg.prepare_tokenizer.tokenizer_dir, cfg.prepare_tokenizer.manifest
    )
    __process_data(
        text_corpus_path,
        cfg.prepare_tokenizer.tokenizer_dir,
        cfg.prepare_tokenizer.vocab_size,
        cfg.prepare_tokenizer.tokenizer,
        cfg.prepare_tokenizer.spe_type,
        lower_case=False,
        spe_character_coverage=1.0,
        spe_sample_size=-1,
        spe_train_extremely_large_corpus=False,
        spe_max_sentencepiece_length=-1,
        spe_split_by_unicode_script=True,
        spe_bos=False,
        spe_eos=False,
        spe_pad=False,
        spe_control_symbols=None,
        spe_user_defined_symbols=["»0", "»1"],
        spe_byte_fallback=False,
        spe_split_digits=False,
    )


if __name__ == "__main__":
    main()
