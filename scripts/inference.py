# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra
from src.model import MultichannelEncDecHybridRNNTCTCBPEModel
from omegaconf import open_dict, OmegaConf
from pathlib import Path
import copy
import soundfile as sf
import torch
from tqdm import tqdm
import re
import pytorch_lightning as pl
import torch.nn.functional as F
import torchaudio.functional as Fa
import numpy as np
import logging


def load_asr_model(checkpoint_path, device):
    asr_model = MultichannelEncDecHybridRNNTCTCBPEModel.load_from_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    asr_model = asr_model.to(device)
    asr_model.eval()

    # change decoding strategy
    decoding_cfg = asr_model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.strategy = "greedy"
        decoding_cfg.preserve_alignments = True
        decoding_cfg.greedy.max_symbols = 10
        decoding_cfg.fused_batch_size = -1
        asr_model.change_decoding_strategy(decoding_cfg)

    # get preprocessor (beamforming + feature extraction)
    cfg = copy.deepcopy(asr_model._cfg)
    OmegaConf.set_struct(cfg.preprocessor, False)
    cfg.preprocessor.dither = 0.0
    cfg.preprocessor.pad_to = 0
    preprocessor = asr_model.from_config_dict(cfg.preprocessor).to(device)

    return asr_model, preprocessor


def get_chunk_start_end_in_samples(
    chunk_start_frames,
    chunk_end_frames,
    window_stride,
    n_fft,
    length_samples,
):
    '''Helper function to get correct part of the input signal for each chunk.

    Values `chunk_start_frames` and `chunk_end_frames` are indices of frames
    corresponding to desired chunk *excluding* the index `chunk_end_frames`.
    This functions computed the indices of the samples corresponding to
    the chunk. This includes additional context in the beginning and end needed
    to properly compute STFT in preprocessor. This additional context will lead
    to additional frames at the output of the STFT. Number of these additional
    frames is returned as `cut_start_frames` and `cut_end_frames` so that they
    can be removed after the preprocessor.

    The additional preprended context is 2*window_stride and leads to
    two additional frames in the beginning of the chunk. The additional
    appended context is n_fft // 2 and leads to 1 additional frame
    at the end of the chunk.
    Extra care is taken at the very beginning or the very end of the signal
    where the context cannot be prepended/appended.

    Note that this function is not very general and will work well only
    for certain ratio of window_stride and n_fft, as is in the used model,
    that is 160/512.
    '''
    if chunk_start_frames < 2:
        chunk_start_samples = 0
        cut_start_frames = chunk_start_frames
    else:
        chunk_start_samples = (chunk_start_frames - 2) * window_stride
        cut_start_frames = 2

    chunk_end_samples = (chunk_end_frames-1) * window_stride + n_fft // 2 + 1
    n_appended = min(
        n_fft // 2 + 1, length_samples - chunk_end_samples + n_fft // 2 + 1
    )
    chunk_end_samples = min(chunk_end_samples, length_samples)
    cut_end_frames = n_appended // window_stride

    return (
        chunk_start_samples,
        chunk_end_samples,
        cut_start_frames,
        cut_end_frames,
    )


def transcribe_file(asr_model, preprocessor, audio_file, device):
    '''Runs streaming inference chunk-by-chunk.

    The size of the chunks is guided by the model, in particular
    `encoder.streaming_cfg` configuration dictionary.
    Each chunk has several frames of the previous chunk prepended
    (`pre_encode_cache`). A small context is also added in the beginning
    and the end of each chunk for the STFT operation in preprocessor.

    The words decoded in each chunk get the timestamp of the end of the chunk
    including the additioanl STFT context. That is, the timestamp of each word
    will correspond to the size of the input signal that has been read by any
    part of the processing at the time of decoding the word (in seconds).
    In case a partial word is decoded by the end of a chunk, and then extended
    to a full word in the following chunk, the timestamp corresponding to
    the end of the second chunk is assigned to this word.
    '''
    # initialize cache and predictions
    cache_last_channel, cache_last_time, cache_last_channel_len = (
        asr_model.encoder.get_initial_cache_state()
    )
    previous_hypotheses = None
    pred_out_stream = None

    # read configuration values
    first_chunk_size, chunk_size = asr_model.encoder.streaming_cfg.chunk_size
    first_chunk_shift, chunk_shift = asr_model.encoder.streaming_cfg.shift_size
    pre_encode_cache_size = (
        asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]
    )
    sampling_frames = asr_model.encoder.pre_encode.get_sampling_frames()[1]
    window_size = int(
        preprocessor._cfg.window_size * preprocessor._cfg.sample_rate
    )
    window_stride = int(
        preprocessor._cfg.window_stride * preprocessor._cfg.sample_rate
    )
    n_fft = preprocessor._cfg.n_fft

    # read audio
    audio, fs = sf.read(str(audio_file))
    resample_factor = int(round(fs / 16000))
    audio = torch.tensor(audio.T).float()
    audio = audio.to(device)
    audio = audio.unsqueeze(0)

    # get first chunk
    chunk_start_frames = 0
    chunk_end_frames = first_chunk_size
    (
        chunk_start_samples,
        chunk_end_samples,
        cut_start_frames,
        cut_end_frames,
    ) = get_chunk_start_end_in_samples(
        chunk_start_frames,
        chunk_end_frames,
        window_stride,
        n_fft,
        audio.shape[-1] // resample_factor,
    )
    audio_chunk = audio[..., chunk_start_samples *
                        resample_factor:chunk_end_samples*resample_factor]
    audio_chunk = Fa.resample(audio_chunk, fs, 16000)
    n_seen_samples = chunk_end_samples * resample_factor
    processed_chunk, _ = preprocessor(
        input_signal=audio_chunk,
        length=torch.Tensor([audio_chunk.shape[-1]]).to(device),
    )
    # remove additional context that was added for STFT
    processed_chunk = processed_chunk[..., cut_start_frames:]
    processed_chunk = (
        processed_chunk[..., :-cut_end_frames]
        if cut_end_frames > 0
        else processed_chunk
    )

    step = 0
    transcribed_words = []
    word_timestamps = []

    while True:  # will break when next chunk is too small
        # process the chunk by the model
        # outputs hypothesis and additional states to pass over to next step
        chunk_length = torch.Tensor([processed_chunk.shape[-1]]).to(device)
        with torch.inference_mode(), torch.no_grad():
            (
                pred_out_stream,
                transcribed_texts,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
                previous_hypotheses,
            ) = asr_model.conformer_stream_step(
                processed_signal=processed_chunk,
                processed_signal_length=chunk_length,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=False,
                previous_hypotheses=previous_hypotheses,
                previous_pred_out=pred_out_stream,
                drop_extra_pre_encoded=(
                    0
                    if step == 0
                    else asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
                ),
                return_transcription=True,
            )

        # add timestamps to the newly decoded words
        # timestamps correspond to the number of so-far read samples
        # from the input signal, `n_seen_samples`
        n_new_words = len(transcribed_texts[0].words) - len(transcribed_words)
        #  if last word in the curr hypothesis got pro-longed in this step,
        #  change its timestamp
        if (
            transcribed_words != []
            and transcribed_words[-1]
            != transcribed_texts[0].words[-n_new_words - 1]
        ):
            word_timestamps[-1] = n_seen_samples / fs
        #  all newly added words get current timestamp
        word_timestamps += [n_seen_samples / fs] * n_new_words
        transcribed_words = transcribed_texts[0].words

        # read next chunk
        chunk_start_frames += first_chunk_shift if step == 0 else chunk_shift
        chunk_end_frames = chunk_start_frames + chunk_size
        step += 1
        # remember last frames to prepend them to next chunk
        pre_encode_cache = processed_chunk[..., -pre_encode_cache_size:]
        if pre_encode_cache.shape[-1] < pre_encode_cache_size:
            pre_encode_cache = F.pad(
                pre_encode_cache,
                (0, pre_encode_cache_size - pre_encode_cache.shape[-1]),
            )
        # get the chunk itself
        (
            chunk_start_samples,
            chunk_end_samples,
            cut_start_frames,
            cut_end_frames,
        ) = get_chunk_start_end_in_samples(
            chunk_start_frames,
            chunk_end_frames,
            window_stride,
            n_fft,
            audio.shape[-1] // resample_factor,
        )
        audio_chunk = audio[..., chunk_start_samples *
                            resample_factor:chunk_end_samples*resample_factor]
        if audio_chunk.shape[-1] <= n_fft // 2 * resample_factor:
            break  # too small for STFT, we are at the end of the signal
        audio_chunk = Fa.resample(audio_chunk, fs, 16000)
        n_seen_samples = chunk_end_samples*resample_factor
        processed_chunk, _ = preprocessor(
            input_signal=audio_chunk,
            length=torch.Tensor([audio_chunk.shape[-1]]).to(device),
        )
        # remove additional context that was added for STFT
        processed_chunk = processed_chunk[..., cut_start_frames:]
        processed_chunk = (
            processed_chunk[..., :-cut_end_frames]
            if cut_end_frames > 0
            else processed_chunk
        )

        if processed_chunk.shape[-1] < sampling_frames:
            break  # to small to process by the model, we are at end of signal
        # prepend remembered cache
        processed_chunk = torch.cat(
            (pre_encode_cache, processed_chunk), dim=-1
        )

    return transcribed_words, word_timestamps


def get_attributed_words_with_timestamps(words, timestamps):
    '''Transforms SOT format into list of words with speaker labels.

    It also processes the word-timestamps accordingly.
    Example input:
        words = ['»1', 'hello', '»1', 'how', '»0', 'good']
        timestamps = [0.1, 0.2, 0.4, 0.45, 0.5, 0.6]
    Example output:
        words_ = ['hello', 'how', 'good']
        timestamps_ = [0.2, 0.45, 0.6]
        speaker_labels = [1, 1, 0]
    '''
    # if speaker attribution token is somewhere in the middle of the word
    # e.g. 'a»1bc', separate it and copy the timestamps
    # this happens rarely, but it might happen
    words_, timestamps_ = [], []
    for word, timestamp in zip(words, timestamps):
        spl = re.split("(»1|»0)", word)
        words_.extend([subword for subword in spl if subword != ""])
        timestamps_.extend([timestamp for subword in spl if subword != ""])
    words, timestamps = words_, timestamps_

    speaker = 0
    words_, timestamps_, speaker_labels = [], [], []
    for word, timestamp in zip(words, timestamps):
        if word == "»0":
            speaker = 0
        elif word == "»1":
            speaker = 1
        else:
            words_.append(word)
            timestamps_.append(timestamp)
            speaker_labels.append(speaker)
    return words_, timestamps_, speaker_labels


def write_outputs(output_file, words, timestamps, speaker_labels):
    with open(output_file, "w") as fw:
        for word, timestamp, speaker_label in zip(
            words, timestamps, speaker_labels
        ):
            fw.write(f"-\t{timestamp:.3f}\t{word}\t{speaker_label}\n")


def find_best_checkpoint(checkpoint_dir):
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    val_wers = [
        float(checkpoint.stem.split("=")[1].split("-")[0])
        for checkpoint in checkpoints
    ]
    return checkpoints[np.argmin(val_wers)]


def inference(cfg):
    '''Runs a streaming inference for recordings in input audio directory.

    The main streaming loop ran on each file is implemented in
    `transcribe_file` function.
    '''
    pl.seed_everything(111, workers=True)
    device = torch.device("cuda")
    if cfg.checkpoint_path is None:
        checkpoint_path = find_best_checkpoint(Path(cfg.checkpoint_dir))
        logging.info(f'Using checkpoint {checkpoint_path}')
    else:
        checkpoint_path = cfg.checkpoint_path
    for lookahead_size in cfg.lookahead_sizes:
        logging.info(f'Inference with lookahead size {lookahead_size}')
        asr_model, preprocessor = load_asr_model(checkpoint_path, device)
        asr_model.encoder.set_default_att_context_size([70, lookahead_size])
        asr_model.encoder.setup_streaming_params()

        (Path(cfg.output_dir) /
         f'lookahead_{lookahead_size}').mkdir(parents=True, exist_ok=True)

        for audio_file in tqdm(Path(cfg.audio_dir).glob("*.wav")):
            words, timestamps = transcribe_file(
                asr_model, preprocessor, audio_file, device
            )
            words, timestamps, speaker_labels = (
                get_attributed_words_with_timestamps(words, timestamps)
            )
            write_outputs(
                Path(cfg.output_dir) /
                f'lookahead_{lookahead_size}' / audio_file.stem.split(".")[0],
                words,
                timestamps,
                speaker_labels,
            )


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg):
    inference(cfg.inference)


if __name__ == "__main__":
    main()
