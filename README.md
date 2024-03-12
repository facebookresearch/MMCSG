# CHiME-8 Task 3 MMCSG baseline
### ASR for multimodal conversations in smart glasses
This repository contains the baseline system for CHiME-8 challenge, Task 3 MMCSG. For information on how to participate in the challenge and how to get the development and evaluation datasets, please refer to the [challenge website](https://www.chimechallenge.org/current/task3/index). 

This README is a guide on how to install and run the baseline system for the challenge. Note that we are also releasing a tool for data simulation, which can be used to extend the training dataset. You can find the tool in `tools/MCAC_simulator`. Follow the `README.md` therein to use this tool.

## Sections
1. <a href="#install">Installation</a>
2. <a href="#running">Running the baseline system</a>
3. <a href=#description>Detailed description of the system</a>
4. <a href="#evaluation">Evaluation</a>
5. <a href="#results">Results</a>
6. <a href="#troubleshooting">Troubleshooting</a>

## <a id="install">1. Installation </a>
The installation of the necessary tools is detailed in `install.sh`. We recommend to follow it step-by-step and adjust for your system if needed (e.g. for installation of PyTorch, you should set your version of CUDA following the [PyTorch installation instructions](https://pytorch.org/get-started/locally/)). The installation goes through
- Installation of necessary Python packages using `conda` and `pip`,
- Installation of the tools for evaluating multi-talker WER.

When running the system, it is necessary to activate the created conda environment and set the environment variables:

    conda activate chime_recipe
    export PYTHONPATH=$PWD/tools/multitalker_wer # necessary for evaluation step
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python # necessary for finetuning step

Finally, the fine-tuning step expects a downloaded [model from HuggingFace](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi). Please navigate to "Files and versions" tab and download the `stt_en_fastconformer_hybrid_large_streaming_multi.nemo` model. By default, the scripts expect to have the model placed in `data/stt_en_fastconformer_hybrid_large_streaming_multi.nemo`.

## <a id="running">2. Running the baseline system</a>

To run the baseline system, you need to specify the path to the downloaded CHiME data in `config/paths.yaml`, as:

    chime_data_dir: <PATH_TO_CHIME_DIRECTORY>

where `<PATH_TO_CHIME_DIRECTORY>` points to directory containing the `audio/{train,dev}` and `transcriptions/{train,dev}` subdirectories. 

Then you can run the baseline system

    python run.py

This will run data preparation, fine-tuning of the pre-trained model, inference, and evaluation. Fine-tuning and inference step are expected to be run with GPU.

### Changing configuration
The script is driven by configuration files in the `config` directory, namely `config/main.yaml` and files referenced there-in. All configuration can be modified from the command-line, e.g. 

    python run.py finetune_asr.model.optim.lr=1.0 inference.output_dir=custom_output_dir

Any of the steps can be skipped by setting their attribute `run` to `false`, e.g.

    python run.py prepare_data.run=false finetune_asr.run=false

will skip preparation of data and finetuning of the ASR system, thus will run only inference and evaluation. The individual steps can be also run separately, i.e. running `python run.py` is equivalent to running these four commands:

    python -m scripts.prepare_data
    python -m scripts.finetune_asr
    python -m scripts.inference
    python -m scripts.evaluate

The individual steps also accept additional parameters, e.g.

    python -m scripts.inference inference.output_dir=custom_output_dir

runs inference step only and changes the default output directory.

### Location of outputs
The default base directory is set in `config/main.yaml` as `exp/base` through the attributes `base_dir` and `exp_name`. Therefore, running e.g.

    python run.py exp_name=custom_experiment

will change the base directory to `exp/custom_experiment`. You can however also change the output directories for the individual steps separately (see the configuration files for the individual steps).

By default, the output transcriptions including speaker labels and per-word timestamps are put to `exp/base/inference/lookahead_*` (three sub-directories for three different settings of lookahead size). The outputs of evaluation are by default in `exp/base/results/lookahead_*`. 


## <a id="description">3. Detailed description of the system</a>

### Preparation of the data
**Inputs:** The downloaded CHiME data. In particular, these parts are used in the preparation step:
- `audio` directory
- `transcriptions` directory

**Outputs:** 
- chunked audio in `data/{train,valid}_chunks`  directories
- the manifest files in `data/{train,valid}_chunks.json`

The script `scripts/prepare_data.py` prepares the training set is then prepared for fine-tuning. This includes:
- spliting the training set into training/validation subsets
- cutting the recordings into ~20 second chunks
- preparation of SOT (serialized-output-training) style transcriptions for each chunk
- preparation of manifold files describing the training/validation data for fine-tuning of NeMO model.


This step can be configured by changing `config/prepare_data.yaml` or passing command-line arguments `prepare_data.<parameter>=<value>`.

### Fine-tuning of the model
**Inputs:**
- [pre-trained NeMO model](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi)
- train and validation manifests prepared in previous step `data/{train,valid}_chunks.json` and corresponding audios in `data/{train,valid}_chunks`
- `data/beamformer_weights.npy` coefficients of fixed beamformer

**Outputs:**
- Fine-tuned model `exp/base/finetune/checkpoints/*.ckpt` (several checkpoint are saved)

The overall system roughly follows the scheme in [(Lin et al. 2023)](https://www.isca-archive.org/interspeech_2023/lin23j_interspeech.html). For the ASR model, we use the public pre-trained cache-aware FastConformer, further described [here](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#cache-aware-streaming-conformer) and [here](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer). This is a model trained to transcribe a single speaker from single-channel audio. Without any fine-tuning, this model transcribes most of SELF speech while ignoring most of speech of OTHER speaker. Furthermore, it is not trained to do speaker attribution. In our baseline, we extend and fine-tune this model as follows:
- We prepend a fixed beamformer module before feature extraction in the model. The beamformer takes all input 7 channels and outputs 13 beams --- 12 different directions around the wearer of the glasses, plus one beam pointed towards the mouth of the wearer. 
- The input convolutional layer of the pre-trained model encoder is extended to accept all 13 beams at the input.
- The tokenizer of the pretrained model is extended to include two speaker tokens: `»0`,`»1` for SELF/OTHER speaker, i.e. the wearer of the glasses and the conversational partner. The corresponding input and output layers are extended to process these two new tokens.
- The extended model is finetuned on the chunks prepared in the previous step.


This step can be configured by changing `config/finetune_asr.yaml` or passing command-line arguments `finetune_asr.<parameter>=<value>`.

### Inference
**Inputs:**
- fine-tuned model from previous step `exp/base/finetune/checkpoints/*.ckpt`
- audios from development set `<chime_data_dir>/audio/dev`

**Outputs:**
- decoded hypotheses, including timestamps and speaker labels `exp/base/inference/*`

The script `scripts/inference.py` takes the fine-tuned model and runs inference on the development data in a streaming way. That is, the input audio-file is passed to the model chunk-by-chunk. Words decoded by the model in each chunk are assigned a timestamp, corresponding to the end of the chunk (how much of the input audio was seen by the model when emitting this word). Since the model is token-based, we consider each word to be emitted when the last token of the word is emitted. The model allows test-time configuration of lookahead size. The inference scripts runs the inference for three different lookahead sizes leading to three sets of results with different latencies.

This step can be configured by changing `config/inference.yaml` or passing command-line arguments `inference.<parameter>=<value>`.

## <a id="evaluation">4. Evaluation</a>
**Inputs:**
- hypotheses directories `exp/base/inference/*`
- list of permitted substitutions `data/permitted_substitutions.yaml`

**Outputs:**
- Overall WER `exp/base/results/lookahead_*/wer`
- WER per utterance `exp/base/results/lookahead_*/wer_per_utt`
- Latency statistics `exp/base/results/lookahead_*/latency`

The system outputs are evaluated using multitalker word-error-rate, for which we release our implementation in `tools/multitalker_wer`. This metric simultaneously compares references and hypotheses for both speakers (SELF/OTHER) and find the best alignment. Specifically:
- The hypothesis words are ordered by their timestamps.
- The reference words are ordered by timestamps separately for SELF and OTHER.
- The best alignment between the words in the hypothesis and the two references is found. The alignment consists of 6 different cases:
    - 'match': a word from the hypothesis attributed to a speaker-A is aligned to the same word from the reference of the speaker-A
    - 'insertion': a word from the hypothesis is not aligned to any word from references
    - 'deletion': a word from a reference is not aligned to any word from the hypothesis
    - 'substitution': a word from the hypothesis attributed to a speaker-A is aligned with a different word from the reference of speaker-A
    - 'speaker-attribution': a word from the hypothesis attributed to a speaker-A is aligned with the same word from the reference of speaker-B
    - 'speaker-attribution-with-substitution': a word from the hypothesis attributed to a speaker-A is aligned with a different word from the reference of speaker-B

The word error rate is then the ratio of the 5 different types of error occurances to the number of reference words. In the reported numbers, we choose to merge the 'speaker-attribution' and 'speaker-attribution-with-substitution' types of errors, and refer to their sum as speaker attribution errors.

Before the computation of the multitalker WER, both the references and the hypotheses are text-normalized. During normalization, puncuation is removed and everything is lower-cased. Words are also replaced by their normalized variants according to the input list of permitted substitutions (e.g. "c'mon" is substituted to "come on"). The word timestamps are accordingly updated if a word gets split or merged during the text normalization.

Latency is computed for each correctly recognized and correctly attributed word as a difference of the hypothesis timestamp and the reference timestamp. Mean, median, and standard deviation of these per-word latencies are then computed over the entire corpus.

Further information about the evalutation can be found at the [challenge website](https://www.chimechallenge.org/current/task3/rules#evaluation).

This step can be configured by changing `config/evaluate.yaml` or passing command-line arguments `evaluate.<parameter>=<value>`.

### Running the evaluation script on your own outputs

    python -m scrips.evaluate evaluate.hypotheses_dir=<hypotheses-dir> evaluate.result_dir=<result-dir> evaluate.hypotheses_in_subdirs=false

`<hypotheses-dir>` should contain one file per recording, of the same name as the stem of the audio filename. This file should have lines in the format:

    <start-of-word-in-seconds>\t<end-of-word-in-seconds>\t<word>\t<speaker-label>

Note that `<start-of-word-in-seconds>` is not used in any way by the evaluation script and you can fill it with any placeholder (it is present only to keep the format consistent with references). `<speaker-label>` is 0 or 1, 0 for SELF (the wearer of the glasses), 1 for OTHER (the conversational partner). Note that the two speakers should be correctly attributed, no permutations are computed during the evaluation.

`<result-dir>` is the output directory where results will be placed. You can find other parameters to the evaluation script in `config/evaluate.yaml`. If you want to run your own text normalization on the hypothesis, you can turn off the default one by `evaluate.normalize_hypotheses=false`. Note that the text normalization of the references is fixed and the same for everyone.

### Testing correctness of the timestamps
In the hypotheses, we require per-word timestamps marking how much of the input signal the system has seen when emitting the word. It is easy to overlook some aspects of the system that makes the timestamps wrong and the system non-streaming. To uncover such issues, we provide a test running on a subset of dev recordings. For each recording, the test creates a pair of perturbed and unperturbed version. The perturbed one has the signal (for all modalities) replaced with random noise or zeros from certain time onward. Running the system on such pair of recordings should yield identical hypotheses until the start time of the perturbation. We will ask you to submit the outputs of the test during the submission of the system.

To run the test, follow these steps:
1. Creating the pairs of perturbed/unperturbed recordings

        python -m scripts.prepare_perturbed_data

    In case your system does not use all modalities, you can disable their generation by e.g. `prepare_perturbed_data.perturb_video=false`. Check `config/prepare_perturbed_data.yaml` for more options. The created data are by default stored in `data/perturbation_test`.

2. Running your system on both original and perturbed version of the recordings. For the baseline system, this would be:

        python -m scripts.inference inference.audio_dir=data/perturbation_test/audio/perturbed inference.output_dir=exp/base/inference_perturbed

        python -m scripts.inference inference.audio_dir=data/perturbation_test/audio/unperturbed inference.output_dir=exp/base/inference_unperturbed

    If your inference script could be influenced by random factors, be careful to fix random seeds and process the recordings in the same order.

3. Running the test of the timestamps itself. E.g., for the outputs of the baseline for lookahead 13:

        python -m scripts.test_timestamps test_timestamps.hypotheses_original_dir=exp/base/inference_unperturbed/lookahead_13 test_timestamps.hypotheses_perturbed_dir=exp/base/inference_perturbed/lookahead_13

    The output of the test will either point out inconsistencies with the hypotheses or state passing of all tests.

## <a id="results">5. Results</a>
The results for the baseline model on dev subset are the following:
<table>
<thead>
  <tr>
    <th></th>
    <th>SELF</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th>OTHER</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Latency [s]<br>mean</td>
    <td>WER<br>[%]</td>
    <td>INS<br>[%]</td>
    <td>DEL<br>[%]</td>
    <td>SUB<br>[%]</td>
    <td>ATTR<br>[%]</td>
    <td>WER<br>[%]</td>
    <td>INS<br>[%]<br></td>
    <td>DEL<br>[%]</td>
    <td>DEL<br>[%]</td>
    <td>ATTR<br>[%]</td>
  </tr>
  <tr>
    <td>0.15</td>
    <td>17.9</td>
    <td>1.7</td>
    <td>4.2</td>
    <td>10.5</td>
    <td>1.6</td>
    <td>24.4</td>
    <td>2.6</td>
    <td>7.3</td>
    <td>12.3</td>
    <td>2.2</td>
  </tr>
  <tr>
    <td>0.34</td>
    <td>15.0</td>
    <td>1.4</td>
    <td>3.9</td>
    <td>8.4</td>
    <td>1.4</td>
    <td>21.4</td>
    <td>2.2</td>
    <td>7.2</td>
    <td>10.1</td>
    <td>1.8</td>
  </tr>
  <tr>
    <td>0.62</td>
    <td>14.3</td>
    <td>1.3</td>
    <td>3.8</td>
    <td>7.9</td>
    <td>1.3</td>
    <td>20.3</td>
    <td>2.1</td>
    <td>7.1</td>
    <td>9.6</td>
    <td>1.6</td>
  </tr>
</tbody>
</table>

## <a id="troubleshooting">5. Troubleshooting</a>

#### Compilation of multitalker WER: `glog/logging.h` `folly/Format.h` `unicode/normalizer2.h` `Eigen/Core` No such file or directory
The header files should be available after the installation of the requirements in `$CONDA_PREFIX/include` and `$CONDA_PREFIX/include/eigen3`. If you cannot locate the header files in these directories, try to rerun the conda installation as follows:

    conda install conda-forge::folly
    conda install conda-forge::glog
    conda install conda-forge::eigen
    conda install conda-forge::icu

#### CUDA or NVIDIA errors
Make sure that the version of PyTorch agrees with your CUDA version and Nvidia driver. The `install.sh` script by default assumes CUDA 12.1. If your version differs, please follow the instructions at the [PyTorch website](https://pytorch.org/get-started/locally/).

#### TypeError: SkipResumeTrainingValidationLoop._should_check_val_fx() takes 1 positional argument but 2 were given
Downgrade your PyTorch Lightning toolkit to pytorch-lightning=2.0.7 as this is the latest version supported by NeMO. 

#### Training or validation killed
While there might be various reasons for this, one potential cause is too many workers spawned by dataloader. Try to decrease the number of workers (by default 8) by passing parameters
`finetune_asr.model.train_ds.num_workers=<your-chosen-number> finetune_asr.model.validation_ds.num_workers=<your-chosen-number>`
You can also directly modify this in the configuration file `config/finetune_asr.yaml`.

#### Other problems
Please [open an issue](https://github.com/facebookresearch/MMCSG/issues/new) or contact us at [Slack](https://join.slack.com/t/chime-fey5388/shared_invite/zt-1oha0gedv-JEUr1mSztR7~iK9AxM4HOA) (channel #chime-8-mmcsg).

## License
MMCSG is CC-BY-NC licensed, as found in the LICENSE file.
