# CHiME-8 Task 3 multi-channel data simulator
The Multichannel Audio Conversation Simulator (MCAS) Dataset and its companion tools, the MCAC Simulator, are designed to work together to simulate two-sided conversation data for Aria glasses. These tools enable researchers to generate large-scale data required for training models such as automatic speech recognition, speaker diarization, beamforming, and more, which are compatible with Aria Glasses

## Get started 
In this part, we use Librispeech as an illustration for the multi-channel data simulation for Aria glasses.

step 1: Download [Librispeech dataset](https://www.openslr.org/12) and [alignment](https://github.com/CorentinJ/librispeech-alignments?tab=readme-ov-file)

step 2: re-organize the alignemnt format for MCAC Simulator, run

	python re_format_alignment.py  --data-original-path ~/alignments  --output-path ~/processed_alignment

step 3: run the main data simulation, for example

	python multi_channel_simulation_main.py --input-data-tsv ~/example_simu/example.tsv --ctm-path ~/processed_alignment --output-root-path ~/example_simu --transforms-config-file ~/example_simu/transform.json

input TSV files look like this:
```
# This is an example TSV file
# it includes three columns: <the audio path of self speaker, the audio path of the other speaker, the audio path of distractor>
self	other	distractor
/data/users/julincs/release_debug/librispeech_wav/train_clean_100/19/198/19-198-0012.wav	/data/users/julincs/release_debug/librispeech_wav/train_clean_100/19/198/19-198-0012.wav	/data/users/julincs/release_debug/librispeech_wav/train_clean_100/19/198/19-198-0012.wav
```

the transform json looks like this:
```
[
    [
    "SampleNoiseRemoteTransform",
    {
      "fs": 16000,
      "noise_datasets": [
        [
          "/data/users/julincs/noise_list.tsv",
          0,
          0,
          0.9
        ]
      ],
      "noise_sources_model": [
        1
      ]
    }
  ],
  [
    "OverlapMultiTalkerTransform",
    {
          "add_mixture_prob": 1,
          "prob_segment_audio": 1,
          "prob_add_distractor": 1,
          "none_overlap_frac": 0.3,
          "split_utt_threshold_min": 0.08,
          "split_utt_threshold_max": 0.3,
          "distractor_overlap_ratio": 0,
          "seed": 0
    }
  ],
  [
    "SampleApplyRIRTransform",
    {
      "fs": 16000,
      "num_noise_sources": 1,
      "rir_datasets": [
        [
          "/data/users/julincs/rir_list.tsv",
          0,
          0,
          1
        ]
      ]
    }
  ],
  [
    "SNRAdjustmentMCTransform",
    {
      "snr": [
        {
          "snr": [
            -5,
            -4,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            12,
            14,
            16,
            18,
            20,
            25,
            30
          ],
          "weight": [
            4,
            4,
            5,
            5,
            15,
            18,
            20,
            28,
            28,
            30,
            30,
            32,
            32,
            25,
            25,
            25,
            25,
            15,
            10,
            10,
            10,
            5,
            5
          ]
        }
      ],
      "fs": 16000,
      "add_noise": true,
      "add_distractor_as_bcg_noise": true,
      "use_alignment": true,
      "distractor_snr": [
        0,
        30
      ]
    }
  ]
]

```

## Run example simulation

We added a toy example to run the multi-channel simulation framework. All the needed file format can be found in `example_simu` folder. 

Please follow the above-mentioned steps to reproduce this example.

