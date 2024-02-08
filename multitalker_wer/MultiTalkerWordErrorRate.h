// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <folly/Optional.h>
#include <stdlib.h>
#include <string>

#include "MultiTalkerEditDistance.h"

namespace facebook {
namespace chime_eval {

struct MultiTalkerWordErrorRateOptions {
  std::optional<std::string> unicodeNorm = "NFKC";
  std::optional<std::string> stripRegex;
  bool caseSensitive = false;
  std::string punctSymbols = ".?!,";

  // If this unkSymbol appears in references then we'll mark hypo tokens
  // that are aligned with it as 'U' instead of the usual 'SDI.', and these
  // 'U' symbols are counted as their own unkError category. An alternative way
  // to deal with unk symbols in refs is to just delete them from refs,
  // but then words recognized over this part of the audio will be miscounted
  // as insertion errors.
  //
  // These unkSymbols in references usually represent audio for which the
  // scribe is sure it contains speech, but the scribe is unsure how to spell
  // whatever was said, usually an entity name or a technical term.
  //
  // In hive:speech_video_asr_dataset's 'transcript' column we use '%' as UNK
  // symbol, while in hive:speech_video_asr_dataset's
  // 'e2e_punct_spkr_unk_transcript' we use '\uFFFD' since the '%' is
  // confusable with post-ITN 'percent'.
  // Set this to an empty string to disable this UNK alignment logic (or
  // chose a symbol that never appears in refs).
  std::string unkSymbol = "\uFFFD";

  // Replaces one word with another before scoring.  Words are tokenized via
  // whitespace.
  std::optional<std::unordered_map<std::string, std::string>> wordSubstitutions;
};

class MultiTalkerWordErrorRate {
 public:
  struct SpeakerResult {
    int referenceWords{0};
    int inserts{0};
    int deletes{0};
    int substitutions{0};
    // Correct word but assigned to wrong speaker
    int wrongSpeakers{0};
    // Substitution of word and assignment to wrong speaker
    int substitutionsWithWrongSpeaker{0};
    float wordErrorRate{0};
    float insertionRate{0};
    float deletionRate{0};
    float substitutionRate{0};
    float speakerAttributionErrorRate{0};
  };

  struct MultiTalkerResult {
    SpeakerResult self{};
    SpeakerResult other{};
    std::string stringAlignment;
    AlignmentResult alignmentResult;
  };

  static std::string normalizeString(
      const std::string& text,
      const MultiTalkerWordErrorRateOptions& options);

  static MultiTalkerResult compute(
      const std::string& reference,
      const std::string& hypothesis,
      const MultiTalkerWordErrorRateOptions& options =
          MultiTalkerWordErrorRateOptions());

  static std::vector<MultiTalkerResult> computeMultiple(
      const std::vector<std::string>& reference,
      const std::vector<std::string>& hypothesis,
      const MultiTalkerWordErrorRateOptions& options =
          MultiTalkerWordErrorRateOptions());

  static MultiTalkerResult report(
      const std::vector<MultiTalkerResult>& results);
};
} // namespace chime_eval
} // namespace facebook
