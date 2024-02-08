// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <Eigen/Core>
#include <deque>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace facebook {
namespace chime_eval {

enum Code {
  match,
  wrong_speaker,
  substitution,
  substitution_and_wrong_speaker,
  insertion,
  deletion,
};

// Return one of ".SDIAX" like sclite pra file codes.
// . = match in word and speaker
// S = Substitution
// D = deletion
// I = Insertion
// A = Speaker attribution error (correct word assigned to the wrong speaker)
// X = S+A (substitution of word assigned to the wrong speaker)
char codeToString(Code code);

struct Token {
  std::string label;
  int speaker;
  float start;
  float end;

  Token() : label(""), speaker(0), start(0.0), end(0.0) {}
  Token(const std::string& lbl, int spk, float st = 0.0, float en = 0.0)
      : label(lbl), speaker(spk), start(st), end(en) {}
};

// Returns a sequence of 4-tuples of (ref, speaker, hyp, code).
// Except we represent these 4-tuples as 4 separate vectors of identical length.
// Here 'ref' is an integer index into a list of Tokens of a specific speaker
// indicated by 'speaker' e.g. ref=4, speaker=1 refers to the 4th token from
// speaker 1 (except for insertions where it represents the predicted speaker).
// 'hyp' are integer indices into a list of Tokens, with -1 indicating there's
// no token associated with that tuple (like for D or I errors)
struct AlignmentResult {
  std::deque<int> ref;
  std::deque<int> speaker;
  std::deque<int> hyp;
  std::deque<Code> codes;
  float score;
};

class MultiTalkerEditDistance {
 public:
  // @unkSymbol see WordErrorRateOptions.unkSymbol
  explicit MultiTalkerEditDistance(
      bool time_mediated = false,
      const std::string& unkSymbol = "\uFFFD")
      : time_mediated_(time_mediated), unkSymbol_(unkSymbol) {}

  /**
   * Given a pair of reference and hypothesis tokens, where each token has a
   * label and (optionally) start and end times, perform edit distance-based
   * text alignment similar to sclite:
   * http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm
   *
   * If time_mediated is false, perform standard alignment where:
   *    D(correct) = 0
   *    D(insertion) = 3
   *    D(deletion) = 3
   *    D(wrong_speaker) = 3
   *    D(substitution) = 4
   *    D(substitution_and_wrong_speaker) = 5
   *
   *
   * Returns a struct containing alignment result, consisting of:
   *    reference, speaker and hypothesis indices of the warping path (index -1
   *      means outside the range of the reference/hypothesis)
   *    Alignment code for each point in the warping path (can be match,
   *      wrong_speaker, substitution, substitution_and_wrong_speaker,
   *      insertion, or deletion)
   */
  std::unique_ptr<AlignmentResult> compute(
      const std::vector<Token>& refs,
      const std::vector<Token>& hyp);

  std::string alignment_to_string(
      const AlignmentResult& alignment,
      const std::vector<Token>& ref,
      const std::vector<Token>& hyp);

  std::vector<Token> string_to_tokens(const std::string& str);

 private:
  float cost(Code code, const Token* prevRef = nullptr);

  std::unique_ptr<AlignmentResult> get_result(
      const std::vector<std::vector<Token>>& refs,
      const std::vector<Token>& hyps);

  bool time_mediated_;
  const std::string unkSymbol_;
  Eigen::Tensor<float, 3, Eigen::RowMajor> scores_;
  Eigen::Tensor<long, 4, Eigen::RowMajor> backtraces_;
};

std::unique_ptr<AlignmentResult> compute_multitalker_alignment(
    std::string& ref,
    std::string& hyp);

} // namespace chime_eval
} // namespace facebook
