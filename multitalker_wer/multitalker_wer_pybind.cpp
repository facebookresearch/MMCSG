// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "MultiTalkerEditDistance.h"
#include "MultiTalkerWordErrorRate.h"

namespace py = pybind11;

PYBIND11_MODULE(multitalker_wer_pybind, m) {
  m.doc() = "MultiTalkerEditDistance";

  // MultiTalkerEditDistance
  py::enum_<facebook::chime_eval::Code>(m, "Code")
      .value("match", facebook::chime_eval::Code::match)
      .value("wrong_speaker", facebook::chime_eval::Code::wrong_speaker)
      .value("substitution", facebook::chime_eval::Code::substitution)
      .value(
          "substitution_and_wrong_speaker",
          facebook::chime_eval::Code::substitution_and_wrong_speaker)
      .value("insertion", facebook::chime_eval::Code::insertion)
      .value("deletion", facebook::chime_eval::Code::deletion)
      .export_values();

  py::class_<facebook::chime_eval::Token>(m, "Token")
      .def(py::init<>())
      .def_readwrite("label", &facebook::chime_eval::Token::label)
      .def_readwrite("speaker", &facebook::chime_eval::Token::speaker)
      .def_readwrite("start", &facebook::chime_eval::Token::start)
      .def_readwrite("end", &facebook::chime_eval::Token::end);

  py::class_<facebook::chime_eval::AlignmentResult>(m, "AlignmentResult")
      .def(py::init<>())
      .def_readwrite("ref", &facebook::chime_eval::AlignmentResult::ref)
      .def_readwrite("speaker", &facebook::chime_eval::AlignmentResult::speaker)
      .def_readwrite("hyp", &facebook::chime_eval::AlignmentResult::hyp)
      .def_readwrite("codes", &facebook::chime_eval::AlignmentResult::codes)
      .def_readwrite("score", &facebook::chime_eval::AlignmentResult::score);

  py::class_<facebook::chime_eval::MultiTalkerEditDistance>(
      m, "MultiTalkerEditDistance")
      .def(py::init<>())
      .def("compute", &facebook::chime_eval::MultiTalkerEditDistance::compute)
      .def(
          "string_to_tokens",
          &facebook::chime_eval::MultiTalkerEditDistance::string_to_tokens)
      .def(
          "alignment_to_string",
          &facebook::chime_eval::MultiTalkerEditDistance::alignment_to_string);

  m.def(
      "compute_multitalker_alignment",
      &facebook::chime_eval::compute_multitalker_alignment,
      "MultiTalkerEditDistance algorithm.",
      py::arg("ref"),
      py::arg("hyp"));

  m.def(
      "code_to_string",
      &facebook::chime_eval::codeToString,
      "Transform error code to string.",
      py::arg("code"));

  // MultiTalkerWordErrorRate
  py::class_<facebook::chime_eval::MultiTalkerWordErrorRate::SpeakerResult>(
      m, "SpeakerResult")
      .def(py::init<>())

      .def_readwrite(
          "referenceWords",
          &facebook::chime_eval::MultiTalkerWordErrorRate::SpeakerResult::
              referenceWords)
      .def_readwrite(
          "inserts",
          &facebook::chime_eval::MultiTalkerWordErrorRate::SpeakerResult::
              inserts)
      .def_readwrite(
          "deletes",
          &facebook::chime_eval::MultiTalkerWordErrorRate::SpeakerResult::
              deletes)
      .def_readwrite(
          "substitutions",
          &facebook::chime_eval::MultiTalkerWordErrorRate::SpeakerResult::
              substitutions)
      .def_readwrite(
          "wrongSpeakers",
          &facebook::chime_eval::MultiTalkerWordErrorRate::SpeakerResult::
              wrongSpeakers)
      .def_readwrite(
          "substitutionsWithWrongSpeaker",
          &facebook::chime_eval::MultiTalkerWordErrorRate::SpeakerResult::
              substitutionsWithWrongSpeaker)
      .def_readwrite(
          "wordErrorRate",
          &facebook::chime_eval::MultiTalkerWordErrorRate::SpeakerResult::
              wordErrorRate)
      .def_readwrite(
          "insertionRate",
          &facebook::chime_eval::MultiTalkerWordErrorRate::SpeakerResult::
              insertionRate)
      .def_readwrite(
          "deletionRate",
          &facebook::chime_eval::MultiTalkerWordErrorRate::SpeakerResult::
              deletionRate)
      .def_readwrite(
          "substitutionRate",
          &facebook::chime_eval::MultiTalkerWordErrorRate::SpeakerResult::
              substitutionRate)
      .def_readwrite(
          "speakerAttributionErrorRate",
          &facebook::chime_eval::MultiTalkerWordErrorRate::SpeakerResult::
              speakerAttributionErrorRate);

  py::class_<facebook::chime_eval::MultiTalkerWordErrorRate::MultiTalkerResult>(
      m, "MultiTalkerResult")
      .def(py::init<>())
      .def_readwrite(
          "self",
          &facebook::chime_eval::MultiTalkerWordErrorRate::MultiTalkerResult::
              self)
      .def_readwrite(
          "other",
          &facebook::chime_eval::MultiTalkerWordErrorRate::MultiTalkerResult::
              other)
      .def_readwrite(
          "stringAlignment",
          &facebook::chime_eval::MultiTalkerWordErrorRate::MultiTalkerResult::
              stringAlignment)
      .def_readwrite(
          "alignmentResult",
          &facebook::chime_eval::MultiTalkerWordErrorRate::MultiTalkerResult::
              alignmentResult);

  py::class_<facebook::chime_eval::MultiTalkerWordErrorRateOptions>(
      m, "MultiTalkerWordErrorRateOptions")
      .def(py::init<>())
      .def_readwrite(
          "caseSensitive",
          &facebook::chime_eval::MultiTalkerWordErrorRateOptions::caseSensitive)
      .def_readwrite(
          "stripRegex",
          &facebook::chime_eval::MultiTalkerWordErrorRateOptions::stripRegex)
      .def_readwrite(
          "punctSymbols",
          &facebook::chime_eval::MultiTalkerWordErrorRateOptions::punctSymbols)
      .def_readwrite(
          "wordSubstitutions",
          &facebook::chime_eval::MultiTalkerWordErrorRateOptions::
              wordSubstitutions);

  py::class_<facebook::chime_eval::MultiTalkerWordErrorRate>(
      m, "MultiTalkerWordErrorRate")
      .def(py::init<>())
      .def_static(
          "compute",
          &facebook::chime_eval::MultiTalkerWordErrorRate::compute,
          py::arg("reference"),
          py::arg("hypothesis"),
          py::arg("options") =
              facebook::chime_eval::MultiTalkerWordErrorRateOptions())
      .def_static(
          "computeMultiple",
          &facebook::chime_eval::MultiTalkerWordErrorRate::computeMultiple,
          py::arg("reference"),
          py::arg("hypothesis"),
          py::arg("options") =
              facebook::chime_eval::MultiTalkerWordErrorRateOptions())
      .def_static(
          "report", &facebook::chime_eval::MultiTalkerWordErrorRate::report);
}
