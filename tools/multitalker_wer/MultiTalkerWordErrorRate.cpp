// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <folly/Format.h>
#include <folly/Optional.h>
#include <folly/String.h>
#include <folly/ThreadLocal.h>
#include <folly/Unicode.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#include <folly/system/HardwareConcurrency.h>

#include <folly/logging/xlog.h>
#include <glog/logging.h>
#include <stdlib.h>
#include <unicode/unistr.h>
#include <algorithm>
#include <queue>

#include "MultiTalkerEditDistance.h"
#include "MultiTalkerWordErrorRate.h"
#include "UnicodeNorm.h"

namespace facebook {
namespace chime_eval {

// parallelForSync and getGrainSize are adapted from
// fbcode/langtech/tuna/batchdecoder/BatchDecoder.cpp
static constexpr int getGrainSize(int inputTaskCount) {
  // Assume we want to use at most 10(kWerThreads) threads from the host, and
  // each task on the thread has at least 2 tasks.
  constexpr int kMaxThreadCount = 10;
  constexpr int kMinGrainSize = 2;
  const auto grainSize =
      std::max(int(inputTaskCount) / kMaxThreadCount, kMinGrainSize);
  return grainSize;
}

static inline void parallelForSync(
    int64_t begin,
    int64_t end,
    int64_t grainSize,
    const std::shared_ptr<folly::CPUThreadPoolExecutor>& cpuPool,
    const std::function<void(int64_t, int64_t)>& f) {
  // Do nothing if there is no valid work
  if (begin >= end || end == 0) {
    return;
  }

  DCHECK_GT(grainSize, 0);
  if (begin + grainSize >= end) {
    f(begin, end);
    return;
  }
  auto sliceBegin = begin;
  while (sliceBegin < end) {
    auto sliceEnd = std::min(sliceBegin + grainSize, end);
    cpuPool->add([sliceBegin, sliceEnd, f]() { f(sliceBegin, sliceEnd); });
    sliceBegin = sliceEnd;
  }
  return;
}

void _addResultToReport(
    const MultiTalkerWordErrorRate::SpeakerResult& result,
    MultiTalkerWordErrorRate::SpeakerResult& report) {
  report.referenceWords += result.referenceWords;
  report.substitutions += result.substitutions;
  report.inserts += result.inserts;
  report.deletes += result.deletes;
  report.wrongSpeakers += result.wrongSpeakers;
  report.substitutionsWithWrongSpeaker += result.substitutionsWithWrongSpeaker;
}

void _computeRates(MultiTalkerWordErrorRate::SpeakerResult& report) {
  auto refWords = float(report.referenceWords);
  report.insertionRate = report.inserts / refWords;
  report.deletionRate = report.deletes / refWords;
  report.substitutionRate = report.substitutions / refWords;
  report.speakerAttributionErrorRate =
      (report.wrongSpeakers + report.substitutionsWithWrongSpeaker) / refWords;
  report.wordErrorRate =
      (report.substitutions + report.deletes + report.inserts +
       report.wrongSpeakers + report.substitutionsWithWrongSpeaker) /
      refWords;
}

std::string MultiTalkerWordErrorRate::normalizeString(
    const std::string& s,
    const MultiTalkerWordErrorRateOptions& options) {
  std::string text = s;

  if (!options.unicodeNorm->empty()) {
    // Needed for 'vi', where W2L and Ninja emit different NFKC/NFKD hypos,
    // and the refs match only one of these formats. So normalize this here.
    const icu::Normalizer2* normalizer;
    if (options.unicodeNorm == "NFKC") {
      normalizer = langtech::UnicodeNorm::NFKC();
    } else if (options.unicodeNorm == "NFKD") {
      normalizer = langtech::UnicodeNorm::NFKD();
    } else {
      XLOG(FATAL) << "unicodeNorm must be <empty>|NFKC|NFKD";
    }
    text = langtech::UnicodeNorm::normalize(text, normalizer);
  } // else no unicode normalization was requested

  if (!options.caseSensitive) {
    // Lowercase
    std::string lowerCased;
    auto lowercase = icu::UnicodeString::fromUTF8(text.c_str()).toLower();
    lowercase.toUTF8String(lowerCased);
    text = lowerCased;
  }

  if (options.wordSubstitutions.has_value()) {
    XLOG(DBG5) << folly::sformat("Pre-substitution: {}", text);
    std::vector<std::string> pieces;
    folly::split(' ', folly::trimWhitespace(text), pieces, true);
    std::vector<std::string> outputPieces;
    outputPieces.reserve(pieces.size());

    for (auto it = pieces.begin(); it < pieces.end(); ++it) {
      // Handle two words at once
      if (it + 2 <= pieces.end()) {
        std::vector<std::string> slice(it, it + 2);
        auto itr = options.wordSubstitutions->find(folly::join(" ", slice));
        if (itr != options.wordSubstitutions->end()) {
          XLOG(DBG5) << folly::sformat(
              "Replacing '{}' with '{}'", *it, itr->second);
          if (!itr->second.empty()) {
            outputPieces.push_back(itr->second);
          }
          it += 1; // n to m token replacement, increment by n - 1
          continue;
        }
      }
      // Handle single word
      auto itr = options.wordSubstitutions->find(*it);
      if (itr != options.wordSubstitutions->end()) {
        XLOG(DBG5) << folly::sformat(
            "Replacing '{}' with '{}'", *it, itr->second);
        if (!itr->second.empty()) {
          outputPieces.push_back(itr->second);
        }
      } else {
        outputPieces.push_back(*it);
      }
    }
    text = folly::join(" ", outputPieces);
    XLOG(DBG5) << folly::sformat("Post-substitution: {}", text);
  }
  return text;
}

MultiTalkerWordErrorRate::MultiTalkerResult MultiTalkerWordErrorRate::compute(
    const std::string& reference,
    const std::string& hypothesis,
    const MultiTalkerWordErrorRateOptions& options) {
  auto ref = MultiTalkerWordErrorRate::normalizeString(reference, options);
  auto hypo = MultiTalkerWordErrorRate::normalizeString(hypothesis, options);
  auto mwer = MultiTalkerEditDistance{};

  auto ref_t = mwer.string_to_tokens(ref);
  auto hyp_t = mwer.string_to_tokens(hypo);

  // if ref and hyp are both empty, return directly
  MultiTalkerWordErrorRate::MultiTalkerResult result{};
  if (ref_t.size() == 0 && hyp_t.size() == 0) {
    return result;
  }

  auto alignment = mwer.compute(ref_t, hyp_t);
  auto str_alignment = mwer.alignment_to_string(*alignment, ref_t, hyp_t);

  const auto alignmentSize = alignment->codes.size();
  for (int i = 0; i != alignmentSize; ++i) {
    const auto code = alignment->codes[i];
    const auto speaker = alignment->speaker[i];
    auto& spkResult = (speaker == 0) ? result.self : result.other;
    if (alignment->ref[i] >= 0) {
      // Negative index refer to insertion
      spkResult.referenceWords += 1;
    }
    switch (code) {
      case Code::match:
        // We good
        break;
      case Code::substitution:
        spkResult.substitutions += 1;
        break;
      case Code::insertion:
        spkResult.inserts += 1;
        break;
      case Code::deletion:
        spkResult.deletes += 1;
        break;
      case Code::wrong_speaker:
        spkResult.wrongSpeakers += 1;
        break;
      case Code::substitution_and_wrong_speaker:
        spkResult.substitutionsWithWrongSpeaker += 1;
        break;
      default:
        throw std::runtime_error("Code not supported.");
    }
  }

  _computeRates(result.self);
  _computeRates(result.other);
  result.stringAlignment = str_alignment;
  result.alignmentResult = *alignment;
  return result;
}

std::vector<MultiTalkerWordErrorRate::MultiTalkerResult>
MultiTalkerWordErrorRate::computeMultiple(
    const std::vector<std::string>& refs,
    const std::vector<std::string>& hypos,
    const MultiTalkerWordErrorRateOptions& options) {
  XCHECK_EQ(refs.size(), hypos.size())
      << "Expect same length of references and hypothesis in addHyposMultiple";
  auto count = refs.size();
  auto results = std::make_shared<
      std::vector<MultiTalkerWordErrorRate::MultiTalkerResult>>(count);

  auto cpuPool = std::make_shared<folly::CPUThreadPoolExecutor>(
      folly::hardware_concurrency());
  parallelForSync(
      0,
      count,
      getGrainSize(count),
      cpuPool,
      [results, &refs, &hypos, &options](size_t bBegin, size_t bEnd) mutable {
        for (int i = bBegin; i != bEnd; ++i) {
          results->at(i) = compute(refs[i], hypos[i], options);
        }
      });
  cpuPool->join();
  return *results;
}

MultiTalkerWordErrorRate::MultiTalkerResult MultiTalkerWordErrorRate::report(
    const std::vector<MultiTalkerWordErrorRate::MultiTalkerResult>& results) {
  MultiTalkerWordErrorRate::MultiTalkerResult report{};
  for (auto&& result : results) {
    _addResultToReport(result.self, report.self);
    _addResultToReport(result.other, report.other);
  }
  _computeRates(report.self);
  _computeRates(report.other);
  return report;
}

} // namespace chime_eval
} // namespace facebook
