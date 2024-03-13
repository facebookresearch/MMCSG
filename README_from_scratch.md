# CHiME-8 Task 3 MMCSG baseline trained from scratch
As opposed to the main baseline scripts described in [README.md](README.md), this README described a second baseline that does not use any pre-trained model. Instead, here we train a model from scratch based on Librispeech, TEDLIUM speech corpora, DNS noise corpus, and real room impulse responses collected with the Aria glasses. These base datasets are used to simulate multi-talker conversations with MCAC simulation tools that we published. Some of the instructions are common for both baseline systems, e.g. inference and evaluation.

## Sections
1. <a href="#install">Installation</a>
2. <a href="#running">Running the baseline system</a>
3. <a href=#description>Detailed description of the system</a>
4. <a href="#evaluation">Evaluation</a>
5. <a href="#results">Results</a>
6. <a href="#troubleshooting">Troubleshooting</a>

## <a id="install">1. Installation </a>
The installation of the necessary tools is detailed in `install_from_scratch.sh`. We recommend to follow it step-by-step and adjust for your system if needed (e.g. for installation of PyTorch, you should set your version of CUDA following the [PyTorch installation instructions](https://pytorch.org/get-started/locally/)). The installation goes through
- Installation of necessary Python packages using `conda` and `pip`,
- Installation of the tools for evaluating multi-talker WER.

When running the system, it is necessary to activate the created conda environment and set the environment variables:

    conda activate chime_recipe_2
    export PYTHONPATH=$PWD/tools/multitalker_wer:$PYTHONPATH # necessary for evaluation step
    export PYTHONPATH=$PWD/tools/NeMo/scripts/tokenizers:$PYTHONPATH # necessary for preparing tokenizer
    export PYTHONPATH=$PWD/tools/MCAC_simulator:$PYTHONPATH # necessary for simulation step
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python # necessary for finetuning step

## <a id="running">2. Running the baseline system</a>
The baseline system builds on several datasets that you need to download:
- [Librispeech](https://www.openslr.org/12): in particular `train-clean-100`, `train-clean-360`, `train-other-500` subsets
- [Librispeech alignments](https://github.com/CorentinJ/librispeech-alignments): in particular, the "simple condensed format"
- [TEDLIUM Release 3](https://www.openslr.org/51/)
- [DNS challenge noises](https://github.com/microsoft/DNS-Challenge): here, follow the `download-dns-challenge-5-noise-ir.sh` script. You can comment out download of impulse responses as these will not be used.
- [CHiME challenge data](https://ai.meta.com/datasets/mmcsg-dataset/)
- [Aria room impulse responses](https://ai.meta.com/datasets/mcas-dataset/)

After downloading these, specify the paths to the data in `config/paths.yaml`, as:

    chime_data_dir: <PATH_TO_CHIME_DIRECTORY> # should contain audio, transcriptions subdirs
    librispeech_data_path: <PATH_TO_LIBRISPEECH_DIRECTORY> # should contain train-clean-* and train-other-* subdirs
    librispeech_alignments_path: <PATH_TO_LIBRISPEECH_ALI_DIRECTORY> # should contain train-clean-* and train-other-* subdirs
    tedlium_data_path: <PATH_TO_TEDLIUM_DIRECTORY> # should contain sph and stm subdirs
    dns_noise_data_path: <PATH_TO_DNS_DIRECTORY> # should contain the noise wav files (i.e. this is the `noise_fullband` dir)
    mcas_real_rir_path: <PATH_TO_MCAS_DIRECTORY> # the "CHiME_Aria_real_RIR" directory in the MCAS dataset, containing `pkl` files with RIRs

The script for running the second baseline system is `run_from_scratch.sh`. We recommend to run this script step-by-step. In particular, note that:
- Steps `prepare_tedlium_data`, `train_asr`, `finetune_multichannel_asr` and `inference` should be run on a machine with GPU. For `train_asr`, we used a machine with 8 GPUs. For the rest, single GPU is enough.
- Steps `speed_perturb` and `simulate` require parallelization in a computational cluster. Because submission to a computational cluster varies across different environments, you need to modify these steps to your setup. For both steps, we provide example bash scripts for submission to a SLURM cluster. If you use a SLURM cluster, you only need to modify the paths and partition names in the bash scripts. If you have other type of cluster, please create an analogous script.

### Changing configuration
The script is driven by configuration files in the `config` directory, namely `config/main_from_scratch.yaml` and files referenced there-in. The configuration of each step can be changed from the command-line, e.g. 

    python -m scripts.prepare_tedlium_data prepare_tedlium_data.output_tedlium_segments_dir=custom_output_dir

### Location of outputs
The default base directory is set in `config/main_from_scratch.yaml` as `exp/base_from_scratch` through the attributes `base_dir` and `exp_name`. Therefore, running e.g.

    python run.py exp_name=custom_experiment

will change the base directory to `exp/custom_experiment`. You can however also change the output directories for the individual steps separately (see the configuration files for the individual steps).

By default, the output transcriptions including speaker labels and per-word timestamps are put to `exp/base_from_scratch/inference/lookahead_*` (three sub-directories for three different settings of lookahead size). The outputs of evaluation are by default in `exp/base_from_scratch/results/lookahead_*`. 

## <a id="description">3. Detailed description of the system</a>
Here, we describe the individual steps called from `run_from_scratch.sh`.

### Preparation of TEDLIUM data
**Inputs:** 
- downloaded TEDLIUM dataset (`sph` and `stm` directories), with path specified in `config/paths.yaml` 
- list of permitted substitutions `data/permitted_substitutions.yaml`

**Outputs:** 
- segmented TEDLIUM audios `data/tedlium/segments/wav`
- word-alignments for the TEDLIUM segments `data/tedlium/segments/alignments`

The script `scripts/prepare_tedlium_data.py` prepares TEDLIUM data into a form necessary for the later simulation of multi-talker conversations. This includes:
- segmentation of the long-form recordings according to the segments specified in the corpus
- forced alignment of words in each segment using `torchaudio`

This step can be configured by changing `config/prepare_tedlium_data.yaml` or passing command-line arguments `prepare_tedlium_data.<parameter>=<value>`.

### Preparation of Librispeech alignments
**Inputs:** 
- downloaded Librispeech alignments, with path specified in `config/paths.yaml`

**Outputs:** 
- reformatted Librispeech alignments `data/librispeech/alignments/train-*`

In contrast with TEDLIUM, for Librispeech we do not use `torchaudio` to create word-alignments, but instead use publicly available pre-computed alignments. The script `scripts/prepare_librispeech_alignments.py` converts the original format of the alignments (TXT) into a format necessary for the next simulation steps (CTM)

This step can be configured by changing `config/prepare_librispeech_alignments.yaml` or passing command-line arguments `prepare_librispeech_alignments.<parameter>=<value>`.

### Speed perturbation
**Inputs:** 
- downloaded Librispeech audios, with path specified in `config/paths.yaml`
- TEDLIUM segments `data/tedlium/segments/wav`
- Librispeech alignments `data/librispeech/alignments/train-*`
- TEDLIUM alignments `data/tedlium/segments/alignments`

**Outputs:** 
- speed perturbed Librispeech audios `data/librispeech/wav_sp/train-*`
- speed perturbed TEDLIUM audios `data/tedlium/segments/wav_sp`
- speed perturbed Librispeech alignments `data/librispeech/alignments_sp/train-*`
- speed perturbed TEDLIUM alignments `data/tedlium/segments/alignments_sp`

We create one copy of all the available audio data with speed perturbation with factor in range 0.9-1.1. The script `scripts/speed_perturb.py` performs this perturbation of the audio files and also accordingly modifies the corresponsing word-alignments. 

This step needs to be run on a computational cluster. By default, we split the files into 100 parts and run the perturbation on these parts in parallel across a cluster of machines. The Python script `scripts/speed_perturb.py` accepts command-line arguments `speed_perturb.i_split` and `speed_perturb.n_splits` and will run the perturbation on the `i_split`-th partition of data split into `n_splits` parts. This Python script needs to be submitted into a computational cluster. We provide an example of how to do this in `scripts/speed_perturb.sh`, which assumes SLURM cluster. This Bash script needs to be modified with correct paths and partition names in your environment. For different computational cluster than SLURM, you need to create your own script that calls the `scripts/speed_perturb.py` with the correct `i_split` and `n_splits` arguments.

This step can be configured by changing `config/speed_perturb.yaml` or passing command-line arguments `speed_perturb.<parameter>=<value>`.

### Preparation of data for simulation
**Inputs:** 
- word-alignments of Librispeech and TEDLIUM, both original and speed-perturbed:
    - `data/librispeech/alignments/train-*`
    - `data/tedlium/segments/alignments`
    - `data/librispeech/alignments_sp/train-*`
    - `data/tedlium/segments/alignments_sp`
- audios of Librispeech and TEDLIUM, both original and speed-perturbed:
    - downloaded Librispeech audios, with path specified in `config/paths.yaml`
    - `data/tedlium/segments/wav`
    - `data/librispeech/wav_sp/train-*`
    - `data/tedlium/segments/wav_sp`
- downloaded DNS noises, with path specified in `config/paths.yaml`
- downloaded Aria RIRs, with path specified in `config/paths.yaml`
- template of simulation transformation configuration `config/simulation_transform.json`

**Outputs:** 
- prepared word-alignments `data/prepared_alignments`
- lists of audio-pairs for creation of multi-talker conversations `data/simulation_pairs`
- resampled DNS noises `data/noises`
- list of noise audios `data/noise_list.tsv`
- list of RIRs `data/rir_list.tsv`
- simulation transformation configuration `data/simulation_transform.json`

The script `scripts/prepare_data_for_simulation.py` prepares all data into formats necessary for multi-talker simulation. This includes:
- re-formatting of alignments to contain `SIL` tokens
- creating pairs of audios for SELF and OTHER speaker in the conversation; the list of pairs is split into partitions for parallelizing the simulation
- re-sampling DNS noises from 48kHz to 16kHz
- preparing a configuration file which will guide the multi-speaker simulation

This step can be configured by changing `config/prepare_data_for_simulation.yaml` or passing command-line arguments `prepare_data_for_simulation.<parameter>=<value>`.

### Simulation
**Inputs:** 
- prepared word-alignments `data/prepared_alignments`
- simulation transformation configuration `data/simulation_transform.json`
- lists of audio-pairs `data/simulation_pairs`

**Outputs:** 
- audios with simulated conversations `data/simulated/simulated_wav`
- CTM transcripts of simulated conversations `data/simulated/simulated_metadata`

The script `scripts/simulate.py` run the multi-talker simulation itself. It is a wrapper around the multi-talker simulation tools in `tools/MCAC_simulator`. 

This step needs to be run on a computational cluster. By default, we split the files into 100 parts and run the simulation on these parts in parallel across a cluster of machines. The Python script `scripts/simulate.py` accepts command-line arguments `simulate.i_split` and will run the simulation of the `data/simulation_pairs/${i_split}.tsv` list of audio pairs. This Python script needs to be submitted into a computational cluster. We provide an example of how to do this in `scripts/simulate.sh`, which assumes SLURM cluster. This Bash script needs to be modified with correct paths and partition names in your environment. For different computational cluster than SLURM, you need to create your own script that calls the `scripts/simualte.py` with the correct `i_split` argument.

This step can be configured by changing `config/simulate.yaml` or passing command-line arguments `simulate.<parameter>=<value>`.

### Preparation of simulated data
**Inputs:** 
- audios with simulated conversations `data/simulated/simulated_wav`
- CTM transcripts of simulated conversations `data/simulated/simulated_metadata`
- list of permitted substitutions `data/permitted_substitutions.yaml`

**Outputs:** 
- the manifest files in `data/{train,valid}_simulated.json`

The script `scripts/prepare_simulated_data.py` prepares manifests with lists of the simulated recordings and their transcriptions. The manifest format is used in later steps by NeMO training scripts to create the training and validation Datasets.

This step can be configured by changing `config/prepare_simulated_data.yaml` or passing command-line arguments `prepare_simulated_data.<parameter>=<value>`.

### Preparation of tokenizer
**Inputs:** 
- the training manifest file `data/train_simulated.json`

**Outputs:** 
- tokenizer `exp/base_from_scratch/tokenizer`

The script `scripts/prepare_tokenizer.py` trains a sentence-piece tokenizer on the training transcriptions.

This step can be configured by changing `config/prepare_tokenizer.yaml` or passing command-line arguments `prepare_tokenizer.<parameter>=<value>`.

### Tranining of the model
**Inputs:** 
- the manifest files `data/{train,valid}_simulated.json`
- beamforming weights `data/beamforming_weights_aria.npy`
- tokenizer `exp/base_from_scratch/tokenizer`

**Outputs:** 
- trained model `exp/base_from_scratch/train/checkpoints/*.ckpt`

The model itself is the same as in the base recipe (described in [README.md](README.md)), but trained from scratch rather than initialized with a pre-trained model. The overall system roughly follows the scheme in [(Lin et al. 2023)](https://www.isca-archive.org/interspeech_2023/lin23j_interspeech.html). For the ASR model, we use the cache-aware FastConformer architecture, further described [here](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#cache-aware-streaming-conformer) and [here](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer). This architecture is by default single-channel. In our baseline, we extend this architecture as follows:
- We prepend a fixed beamformer module before feature extraction in the model. The beamformer takes all input 7 channels and outputs 13 beams --- 12 different directions around the wearer of the glasses, plus one beam pointed towards the mouth of the wearer. 
- The input convolutional layer of the pre-trained model encoder is extended to accept all 13 beams at the input.

This step can be configured by changing `config/train_asr.yaml` or passing command-line arguments `train_asr.<parameter>=<value>`.

### Preparation of CHiME data
This step is identical to preparation of the data in the other baseline system described in [README.md](README.md).

**Inputs:** 
- downloaded CHiME data, with path specified in `config/paths.yaml`

**Outputs:** 
- chunked audio in `data/{train,valid}_chunks`  directories
- the manifest files in `data/{train,valid}_chunks.json`

The script `scripts/prepare_data.py` prepares CHiME data for finetuning of the previously trained model. This includes:
- spliting the training set into training/validation subsets
- cutting the recordings into ~20 second chunks
- preparation of SOT (serialized-output-training) style transcriptions for each chunk
- preparation of manifold files describing the training/validation data for fine-tuning.

This step can be configured by changing `config/prepare_data.yaml` or passing command-line arguments `prepare_data.<parameter>=<value>`.

### Fine-tuning of the model
**Inputs:** 
- the CHiME manifest files in `data/{train,valid}_chunks.json`
- pre-trained model `exp/base_from_scratch/train/checkpoints/*.ckpt`
- beamforming weights `data/beamforming_weights_aria.npy`
- tokenizer `exp/base_from_scratch/tokenizer`

**Outputs:** 
- fine-tuned model `exp/base_from_scratch/finetune_multichannel/checkpoints/*.ckpt`

The script `scripts/finetune_multichannel_asr.py` takes the model trained on Librispeech and TEDLIUM and fine-tunes it on the in-doman CHiME data. This step is almost the same as the fine-tuning step in the other baseline system (described in [README.md](README.md)). However, in the other baseline system, a publicly available pre-trained single-channel model is extended for multi-channel case and fine-tuned. Here, we already start from a multi-channel model, trained in the previous steps.

This step can be configured by changing `config/finetune_multichannel_asr.yaml` or passing command-line arguments `finetune_multichannel_asr.<parameter>=<value>`.

### Inference
**Inputs:**
- fine-tuned model from previous step `exp/base_from_scratch/finetune_multichannel/checkpoints/*.ckpt`
- audios from development set `<chime_data_dir>/audio/dev`

**Outputs:**
- decoded hypotheses, including timestamps and speaker labels `exp/base_from_scratch/inference/*`

The script `scripts/inference.py` takes the fine-tuned model and runs inference on the development data in a streaming way. That is, the input audio-file is passed to the model chunk-by-chunk. Words decoded by the model in each chunk are assigned a timestamp, corresponding to the end of the chunk (how much of the input audio was seen by the model when emitting this word). Since the model is token-based, we consider each word to be emitted when the last token of the word is emitted. The model allows test-time configuration of lookahead size. The inference scripts runs the inference for three different lookahead sizes leading to three sets of results with different latencies.

This step can be configured by changing `config/inference.yaml` or passing command-line arguments `inference.<parameter>=<value>`.

By default, the inference is run with the fine-tuned model. However, the same may be also done with the model before fine-tuning, i.e. model trained purely on simulated data. This can be done as follows:

    python -m scripts.inference \
        inference.checkpoint_dir=exp/base_from_scratch/train/checkpoints \
        inference.output_dir=exp/base_from_scratch/inference_wo_finetuning

## <a id="evaluation">4. Evaluation</a>
**Inputs:**
- hypotheses directories `exp/base_from_scratch/inference/*`
- list of permitted substitutions `data/permitted_substitutions.yaml`

**Outputs:**
- Overall WER `exp/base_from_scratch/results/lookahead_*/wer`
- WER per utterance `exp/base_from_scratch/results/lookahead_*/wer_per_utt`
- Latency statistics `exp/base_from_scratch/results/lookahead_*/latency`

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

By default, the outputs of the fine-tuned model are evaluate when calling the `scripts.evaluate` script. However, we might also evaluate the outputs of the model before fine-tuning, i.e. model trained purely on simulated data. This can be done as follows:

    python -m scripts.evaluate \
        evaluate.hypotheses_dir=exp/base_from_scratch/inference_wo_finetuning \
        evaluate.result_dir=exp/base_from_scratch/results_wo_finetuning

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
The results for the fine-tuned model on dev subset are the following:
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
    <td>0.08</td>
    <td>29.1</td>
    <td>3.0</td>
    <td>6.0</td>
    <td>18.4</td>
    <td>1.7</td>
    <td>37.6</td>
    <td>4.2</td>
    <td>9.1</td>
    <td>20.9</td>
    <td>3.4</td>
  </tr>
  <tr>
    <td>0.27</td>
    <td>24.9</td>
    <td>2.6</td>
    <td>5.2</td>
    <td>15.8</td>
    <td>1.3</td>
    <td>33.3</td>
    <td>3.6</td>
    <td>8.6</td>
    <td>18.4</td>
    <td>2.7</td>
  </tr>
  <tr>
    <td>0.55</td>
    <td>23.5</td>
    <td>2.5</td>
    <td>5.0</td>
    <td>14.8</td>
    <td>1.1</td>
    <td>31.7</td>
    <td>3.4</td>
    <td>8.2</td>
    <td>17.5</td>
    <td>2.5</td>
  </tr>
</tbody>
</table>

The results for the model before fine-tunining on dev subset are the following:
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
    <td>0.16</td>
    <td>29.7</td>
    <td>2.4</td>
    <td>6.2</td>
    <td>20.4</td>
    <td>0.7</td>
    <td>41.9</td>
    <td>3.8</td>
    <td>9.7</td>
    <td>25.9</td>
    <td>2.5</td>
  </tr>
  <tr>
    <td>0.35</td>
    <td>28.0</td>
    <td>2.3</td>
    <td>5.7</td>
    <td>19.3</td>
    <td>0.7</td>
    <td>40.1</td>
    <td>3.8</td>
    <td>9.4</td>
    <td>24.5</td>
    <td>2.0</td>
  </tr>
  <tr>
    <td>0.63</td>
    <td>27.4</td>
    <td>2.3</td>
    <td>5.5</td>
    <td>19.0</td>
    <td>0.7</td>
    <td>39.3</td>
    <td>3.8</td>
    <td>9.1</td>
    <td>24.1</td>
    <td>2.3</td>
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
