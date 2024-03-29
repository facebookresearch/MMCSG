# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra
from collections import defaultdict
import numpy as np
from pathlib import Path
import logging
import multitalker_wer_pybind as multitalker_wer
from src.text_normalization import (
    load_permitted_subs,
    normalize_transcriptions,
)


def read_transcriptions(transcriptions_dir):
    utt2transcription = defaultdict(list)
    for path in transcriptions_dir.glob("*"):
        with open(path) as f:
            for line in f:
                _, timestamp, word, speaker = line.strip().split()
                timestamp = float(timestamp)
                speaker = int(speaker)
                utt2transcription[path.stem].append((word, timestamp, speaker))
    return utt2transcription


def get_sot_transcription(transcription):
    """Creates SOT transcription for multitalker_wer input."""
    sot_transcription = ""
    prev_speaker = None
    for word, timestamp, speaker in sorted(
        transcription, key=lambda wts: wts[1]
    ):
        if speaker != prev_speaker:
            sot_transcription += f"»{speaker} "
        sot_transcription += f"{word} "

        prev_speaker = speaker
    return sot_transcription.strip()


def compute_wer_alignments(utt2hypotheses, utt2references):
    """
    multitalker_wer tool is used to compute word error rates,
    both aggregated and for each utterance. The alignments of the references
    and hypotheses are also computed.
    """
    mwer = multitalker_wer.MultiTalkerWordErrorRate()
    options = multitalker_wer.MultiTalkerWordErrorRateOptions()
    options.caseSensitive = True

    utt2sot_hypotheses = {
        utt: get_sot_transcription(transcription)
        for utt, transcription in utt2hypotheses.items()
    }
    utt2sot_references = {
        utt: get_sot_transcription(transcription)
        for utt, transcription in utt2references.items()
    }
    utts = [u for u in utt2sot_hypotheses]

    results = mwer.computeMultiple(
        [utt2sot_references[utt] for utt in utts],
        [utt2sot_hypotheses[utt] for utt in utts],
        options,
    )
    utt2results = {utt: res for utt, res in zip(utts, results)}
    results_overall = mwer.report(results)
    return utt2results, results_overall


def compute_latency(utt2hypotheses, utt2references, utt2results):
    """
    Latency is computed as differences between reference and hypotheses
    per-word timestamps. The reference timestamps correspond to end of the
    words as computed by forced-alignment. The hypotheses timestamps correspond
    to the amount of signal that the system has seen when decoded the word.

    The latency is computed only over correct (and correctly attributed) words.
    """
    latency_per_word_all = []
    for utt in utt2hypotheses:
        timestamps_hypothesis = [t for w, t, s in utt2hypotheses[utt]]
        timestamps_references = [t for w, t, s in utt2references[utt]]

        err_types = [
            err_type.name
            for err_type in utt2results[utt].alignmentResult.codes
        ]
        # get timestamps for correct words only for ref and hyp
        err_types_wo_ins = [err for err in err_types if err != "insertion"]
        assert len(err_types_wo_ins) == len(timestamps_references)
        timestamps_references_corr = [
            t
            for t, err in zip(timestamps_references, err_types_wo_ins)
            if err == "match"
        ]
        err_types_wo_del = [err for err in err_types if err != "deletion"]
        assert len(err_types_wo_del) == len(timestamps_hypothesis)
        timestamps_hypothesis_corr = [
            t
            for t, err in zip(timestamps_hypothesis, err_types_wo_del)
            if err == "match"
        ]

        latency_per_word = [
            th - tr
            for th, tr in zip(
                timestamps_hypothesis_corr, timestamps_references_corr
            )
        ]
        latency_per_word_all.extend(latency_per_word)

    return (
        np.mean(latency_per_word_all),
        np.std(latency_per_word_all),
        np.median(latency_per_word_all),
        latency_per_word_all,
    )


def write_results(result_dir, wer_info, latency_info):
    result_dir.mkdir(exist_ok=True, parents=True)
    utt2results, results_overall = wer_info
    latency_mean, latency_std, latency_median, latency_all = latency_info

    with open(result_dir / "wer", "w") as fw:
        fw.write(
            "wer_self;wer_other;nref_self;nref_other;ins_self;ins_other;"
            "del_self;del_other;sub_self;sub_other;sa_self;sa_other\n"
        )
        fw.write(
            f"{results_overall.self.wordErrorRate:.3f};"
            f"{results_overall.other.wordErrorRate:.3f};"
            f"{results_overall.self.referenceWords};"
            f"{results_overall.other.referenceWords};"
            f"{results_overall.self.insertionRate:.3f};"
            f"{results_overall.other.insertionRate:.3f};"
            f"{results_overall.self.deletionRate:.3f};"
            f"{results_overall.other.deletionRate:.3f};"
            f"{results_overall.self.substitutionRate:.3f};"
            f"{results_overall.other.substitutionRate:.3f};"
            f"{results_overall.self.speakerAttributionErrorRate:.3f};"
            f"{results_overall.other.speakerAttributionErrorRate:.3f}\n"
        )

    with open(result_dir / "wer_per_utt", "w") as fw:
        fw.write(
            "utt;wer_self;wer_other;nref_self;nref_other;ins_self;ins_other;"
            "del_self;del_other;sub_self;sub_other;sa_self;sa_other\n"
        )
        for utt in utt2results:
            fw.write(f"{utt};")
            fw.write(
                f"{utt2results[utt].self.wordErrorRate:.3f};"
                f"{utt2results[utt].other.wordErrorRate:.3f};"
                f"{utt2results[utt].self.referenceWords};"
                f"{utt2results[utt].other.referenceWords};"
                f"{utt2results[utt].self.insertionRate:.3f};"
                f"{utt2results[utt].other.insertionRate:.3f};"
                f"{utt2results[utt].self.deletionRate:.3f};"
                f"{utt2results[utt].other.deletionRate:.3f};"
                f"{utt2results[utt].self.substitutionRate:.3f};"
                f"{utt2results[utt].other.substitutionRate:.3f};"
                f"{utt2results[utt].self.speakerAttributionErrorRate:.3f};"
                f"{utt2results[utt].other.speakerAttributionErrorRate:.3f}\n"
            )

    with open(result_dir / "latency", "w") as fw:
        fw.write("mean;std;median\n")
        fw.write(
            f"{latency_mean:.3f};{latency_std:.3f};{latency_median:.3f}\n"
        )


def write_alignments(output_dir, utt2results, words_per_line=25):
    """Writes reference/hypothesis alignments to HTML."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for utt in utt2results:
        html_strings1 = []
        html_strings2 = []
        curr_html_string1 = ""
        curr_html_string2 = ""
        ops = utt2results[utt].stringAlignment.split()
        for i, op in enumerate(ops):
            err, ref_word, hyp_word = op.split("|")
            if err == ".":
                curr_html_string1 += f'<span style="background-color:#99e699">{ref_word}</span> '
                curr_html_string2 += f'<span style="background-color:#99e699">{hyp_word}</span> '
            elif err == "D":
                curr_html_string1 += f'<span style="background-color:#ff8080">{ref_word}</span> '
                curr_html_string2 += f'<span style="background-color:#ff8080">{"-" * len(ref_word)}</span> '
            elif err == "I":
                curr_html_string1 += f'<span style="background-color:#ff8080">{"-" * len(hyp_word)}</span> '
                curr_html_string2 += f'<span style="background-color:#ff8080">{hyp_word}</span> '
            elif err in ["S", "X"]:
                maxlen = max(len(ref_word), len(hyp_word))
                w1 = ref_word.ljust(maxlen, "\xa0")
                w2 = hyp_word.ljust(maxlen, "\xa0")
                curr_html_string1 += (
                    f'<span style="background-color:#ff8080">{w1}</span> '
                )
                curr_html_string2 += (
                    f'<span style="background-color:#ff8080">{w2}</span> '
                )
            else:  # err == 'A'
                curr_html_string1 += f'<span style="background-color:#ff8080">{ref_word}</span> '
                curr_html_string2 += f'<span style="background-color:#ff8080">{hyp_word}</span> '

            if (i + 1) % words_per_line == 0:
                html_strings1.append(curr_html_string1)
                html_strings2.append(curr_html_string2)
                curr_html_string1 = ""
                curr_html_string2 = ""

        html_string = (
            '<p style="font-family:monospace;color:black">'
            + "<br />".join(
                [
                    f"{pair[0]}<br />{pair[1]}<br />"
                    for pair in zip(html_strings1, html_strings2)
                ]
            )
            + "</p>\n"
        )
        with open(output_dir / f"{utt}.html", "w") as fw:
            fw.write(html_string)


def evaluate(cfg):
    permitted_subs = load_permitted_subs(Path(cfg.permitted_subs_path))
    utt2references = normalize_transcriptions(
        read_transcriptions(Path(cfg.references_dir)), permitted_subs
    )

    if cfg.hypotheses_in_subdirs:
        subdirs = Path(cfg.hypotheses_dir).glob("*")
    else:
        subdirs = [Path(cfg.hypotheses_dir)]

    for hypotheses_subdir in subdirs:
        if not hypotheses_subdir.is_dir():
            continue
        logging.info(f'Evaluating hypotheses "{hypotheses_subdir.stem}"')

        utt2hypotheses = read_transcriptions(Path(hypotheses_subdir))
        if cfg.normalize_hypotheses:
            utt2hypotheses = normalize_transcriptions(
                utt2hypotheses, permitted_subs
            )
        assert len(utt2hypotheses) == len(utt2references)
        assert len(utt2hypotheses) == len(
            [u for u in utt2hypotheses if u in utt2references]
        )

        utt2results, results_overall = compute_wer_alignments(
            utt2hypotheses, utt2references
        )
        if cfg.write_alignments:
            write_alignments(
                Path(cfg.result_dir) / hypotheses_subdir.stem / "alignments",
                utt2results,
            )
        assert len(utt2hypotheses) == len(utt2results)
        assert len(utt2hypotheses) == len(
            [u for u in utt2hypotheses if u in utt2results]
        )
        latency_info = compute_latency(
            utt2hypotheses, utt2references, utt2results
        )

        write_results(
            Path(cfg.result_dir) / hypotheses_subdir.stem,
            (utt2results, results_overall),
            latency_info,
        )


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg):
    evaluate(cfg.evaluate)


if __name__ == "__main__":
    main()
