// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <glog/logging.h>
#include <algorithm>
#include <cmath>

#include "MultiTalkerEditDistance.h"

#include <istream>

namespace facebook {
namespace chime_eval {

std::vector<Token> get_tokens_of_speaker(
    const std::vector<Token>& ref,
    int speaker) {
  std::vector<Token> tokens_of_speaker{};
  for (auto& token : ref) {
    if (token.speaker == speaker) {
      tokens_of_speaker.push_back(token);
    }
  }
  return tokens_of_speaker;
}

char codeToString(Code code) {
  std::array<char, 6> chars{'.', 'A', 'S', 'X', 'I', 'D'};
  assert(0 <= int(code) && int(code) < chars.size());
  return chars[(int)code];
}

std::unique_ptr<AlignmentResult> MultiTalkerEditDistance::compute(
    const std::vector<Token>& ref,
    const std::vector<Token>& hyp) {
  if (ref.size() == 0 && hyp.size() == 0) {
    return nullptr;
  }

  const auto& ref0 = get_tokens_of_speaker(ref, 0);
  const auto& ref1 = get_tokens_of_speaker(ref, 1);
  const std::vector<long> null_status = {-1, -1, -1};

  std::vector<std::vector<Token>> refs{ref0, ref1};

  scores_.resize(ref0.size() + 1, ref1.size() + 1, hyp.size() + 1);
  backtraces_.resize(ref0.size() + 1, ref1.size() + 1, hyp.size() + 1, 3);

  for (long r0 = 0; r0 < scores_.dimension(0); r0++) {
    for (long r1 = 0; r1 < scores_.dimension(1); r1++) {
      for (long h = 0; h < scores_.dimension(2); h++) {
        // Initial condition: hypotesys is finished
        if (h == 0) {
          scores_(r0, r1, 0) = (r0 + r1) * this->cost(Code::deletion);
          std::vector<long> prev_status{null_status};
          if (r0 > 0 || r1 > 0) {
            prev_status = {r1 == 0 ? r0 - 1 : r0, std::max(r1 - 1, 0L), 0};
          }
          DCHECK(prev_status.size() == 3);
          backtraces_(r0, r1, h, 0) = prev_status[0];
          backtraces_(r0, r1, h, 1) = prev_status[1];
          backtraces_(r0, r1, h, 2) = prev_status[2];
          continue;
        }

        // Current step
        float best_score = std::numeric_limits<float>::infinity();
        std::vector<long> best_prev_status{null_status};
        auto& curr_spk = hyp[h - 1].speaker;
        auto& curr_hyp = hyp[h - 1];

        std::vector<long> status{r0, r1, h};
        assert(curr_spk < 2);

        if (status[curr_spk] > 0) {
          auto& curr_spk_word = refs[curr_spk][status[curr_spk] - 1];

          // Substitution or correct word on the speaker predicted by hyp
          bool is_sub = curr_spk_word.label != curr_hyp.label;
          std::vector<long> prev_status{status};
          prev_status[curr_spk]--;
          prev_status[2]--;

          float cost_sub_or_correct =
              scores_(prev_status[0], prev_status[1], prev_status[2]) +
              this->cost(is_sub ? Code::substitution : Code::match);
          if (cost_sub_or_correct < best_score) {
            best_score = cost_sub_or_correct;
            best_prev_status = {prev_status[0], prev_status[1], prev_status[2]};
          }

          // Deletion on the speaker predicted by hyp
          prev_status[2]++;
          float cost_del_curr_spk = scores_(prev_status[0], prev_status[1], h) +
              this->cost(Code::deletion);
          if (cost_del_curr_spk < best_score) {
            best_score = cost_del_curr_spk;
            best_prev_status = {prev_status[0], prev_status[1], prev_status[2]};
          }
        }

        // Insertion
        {
          std::vector<long> prev_status{status};
          prev_status[2]--;
          float cost_ins =
              scores_(prev_status[0], prev_status[1], prev_status[2]) +
              this->cost(
                  Code::insertion,
                  status[curr_spk] > 0 ? &refs[curr_spk][status[curr_spk] - 1]
                                       : nullptr);
          if (cost_ins < best_score) {
            best_score = cost_ins;
            best_prev_status = {prev_status[0], prev_status[1], prev_status[2]};
          }
        }

        auto other_spk = 1 - curr_spk;
        if (status[other_spk] > 0) {
          auto& other_spk_word = refs[other_spk][status[other_spk] - 1];
          // Wrong speaker attribution
          std::vector<long> prev_status{status};
          prev_status[other_spk]--;
          prev_status[2]--;
          bool is_sub = other_spk_word.label != curr_hyp.label;
          float cost_wrong_spk =
              scores_(prev_status[0], prev_status[1], prev_status[2]) +
              this->cost(
                  is_sub ? Code::substitution_and_wrong_speaker
                         : Code::wrong_speaker);
          if (cost_wrong_spk < best_score) {
            best_score = cost_wrong_spk;
            best_prev_status = {prev_status[0], prev_status[1], prev_status[2]};
          }

          // Deletion on the speaker not predicted by hyp
          prev_status[2]++;
          float cost_del_other_spk =
              scores_(prev_status[0], prev_status[1], prev_status[2]) +
              this->cost(Code::deletion);
          if (cost_del_other_spk < best_score) {
            best_score = cost_del_other_spk;
            best_prev_status = {prev_status[0], prev_status[1], prev_status[2]};
          }
        }

        scores_(r0, r1, h) = best_score;
        DCHECK(best_prev_status.size() == 3);
        backtraces_(r0, r1, h, 0) = best_prev_status[0];
        backtraces_(r0, r1, h, 1) = best_prev_status[1];
        backtraces_(r0, r1, h, 2) = best_prev_status[2];
      }
    }
  }
  return this->get_result(refs, hyp);
}

std::unique_ptr<AlignmentResult> MultiTalkerEditDistance::get_result(
    const std::vector<std::vector<Token>>& refs,
    const std::vector<Token>& hyp) {
  auto res = std::make_unique<AlignmentResult>();

  // Status as [index of spk1, index of spk2, index of hyp]
  std::vector<long> status{
      scores_.dimension(0) - 1,
      scores_.dimension(1) - 1,
      scores_.dimension(2) - 1};

  const std::vector<long> entry_point = {0, 0, 0};

  DCHECK(status.size() == 3);
  res->score = scores_(status[0], status[1], status[2]);
  while (status != entry_point) {
    auto prev_r0 = backtraces_(status[0], status[1], status[2], 0);
    auto prev_r1 = backtraces_(status[0], status[1], status[2], 1);
    auto prev_h = backtraces_(status[0], status[1], status[2], 2);
    std::vector<long> prev_status{prev_r0, prev_r1, prev_h};
    int curr_spk;
    int other_spk;
    bool hyp_finished;
    if (status[2] > 0) {
      curr_spk = hyp[status[2] - 1].speaker;
      other_spk = 1 - curr_spk;
      hyp_finished = false;
    } else {
      curr_spk = -2;
      other_spk = -2;
      hyp_finished = true;
    }

    if (status[0] - 1 == prev_status[0] && status[1] == prev_status[1] &&
        status[2] == prev_status[2]) {
      // Deletion in ref0
      res->codes.push_front(Code::deletion);
      res->ref.push_front(status[0] - 1);
      res->speaker.push_front(0);
      res->hyp.push_front(-1);

    } else if (
        status[0] == prev_status[0] && status[1] - 1 == prev_status[1] &&
        status[2] == prev_status[2]) {
      // Deletion in ref1
      res->codes.push_front(Code::deletion);
      res->ref.push_front(status[1] - 1);
      res->speaker.push_front(1);
      res->hyp.push_front(-1);
    } else if (
        !hyp_finished && status[0] == prev_status[0] &&
        status[1] == prev_status[1] && status[2] - 1 == prev_status[2]) {
      // Insertion
      res->codes.push_front(Code::insertion);
      res->ref.push_front(-1);
      res->speaker.push_front(curr_spk);
      res->hyp.push_front(status[2] - 1);
    } else if (
        !hyp_finished && status[curr_spk] - 1 == prev_status[curr_spk] &&
        status[other_spk] == prev_status[other_spk] &&
        status[2] - 1 == prev_status[2]) {
      // Match or Substitution in current speaker
      bool is_sub =
          (refs[curr_spk][status[curr_spk] - 1].label !=
           hyp[status[2] - 1].label);
      res->codes.push_front(is_sub ? Code::substitution : Code::match);
      res->ref.push_front(status[curr_spk] - 1);
      res->speaker.push_front(curr_spk);
      res->hyp.push_front(status[2] - 1);
    } else {
      // Match or Substitution in other speaker
      assert(
          !hyp_finished && status[curr_spk] == prev_status[curr_spk] &&
          status[other_spk] - 1 == prev_status[other_spk] &&
          status[2] - 1 == prev_status[2]);
      bool is_sub =
          (refs[other_spk][status[other_spk] - 1].label !=
           hyp[status[2] - 1].label);
      res->codes.push_front(
          is_sub ? Code::substitution_and_wrong_speaker : Code::wrong_speaker);
      res->ref.push_front(status[other_spk] - 1);
      res->speaker.push_front(other_spk);
      res->hyp.push_front(status[2] - 1);
    }
    status = prev_status;
  }
  return res;
}

float MultiTalkerEditDistance::cost(Code code, const Token* prevRef) {
  if (time_mediated_) {
    throw std::runtime_error{"Time mediated WER is not implemented"};
  } else {
    if (code == Code::match) {
      return 0;
    } else if (
        code == Code::insertion && (prevRef && prevRef->label == unkSymbol_)) {
      // Initially I used a cost of 0, with 1 we get slightly different
      // alignments around unks that caused collateral damage around them.
      // So any cost [0,1,2] should work reasonably well here.
      return 1;
    } else if (
        code == Code::insertion || code == Code::deletion ||
        code == Code::wrong_speaker) {
      return 3;
    } else if (code == Code::substitution) {
      return 4;
    } else if (code == Code::substitution_and_wrong_speaker) {
      return 5;
    } else {
      throw std::runtime_error{"Unknown code"};
    }
  }
}

std::string MultiTalkerEditDistance::alignment_to_string(
    const AlignmentResult& alignment,
    const std::vector<Token>& ref,
    const std::vector<Token>& hyp) {
  const auto& ref0 = get_tokens_of_speaker(ref, 0);
  const auto& ref1 = get_tokens_of_speaker(ref, 1);

  std::stringstream alignment_str;
  bool first = true;
  char err;
  std::string curr_ref;
  std::string curr_hyp;
  for (size_t i = 0; i < alignment.codes.size(); i++) {
    err = codeToString(alignment.codes[i]);
    DCHECK(alignment.hyp[i] >= 0 ? hyp.size() > alignment.hyp[i] : true);
    curr_hyp = alignment.hyp[i] >= 0 ? hyp[alignment.hyp[i]].label : "_";
    if (alignment.ref[i] >= 0) {
      DCHECK(
          (alignment.speaker[i] == 0 ? ref0 : ref1).size() > alignment.ref[i]);
      curr_ref =
          (alignment.speaker[i] == 0 ? ref0 : ref1)[alignment.ref[i]].label;
    } else {
      curr_ref = '_';
    }
    if (!first) {
      alignment_str << " ";
    }
    first = false;
    alignment_str << err << "|" << curr_ref << "|" << curr_hyp;
  }
  return alignment_str.str();
}

std::vector<Token> MultiTalkerEditDistance::string_to_tokens(
    const std::string& str) {
  std::vector<Token> tokens{};
  std::string w;
  int spkid = 0;
  std::stringstream ss{str};
  while (getline(ss, w, ' ')) {
    if (w.find("»") == 0) {
      spkid = std::stoi(w.substr(2, 3));
      continue;
    }
    tokens.push_back(Token{w, spkid});
  }
  return tokens;
}

std::unique_ptr<AlignmentResult> compute_multitalker_alignment(
    std::string& ref,
    std::string& hyp) {
  MultiTalkerEditDistance wer{};
  std::vector<Token> refs = wer.string_to_tokens(ref);
  std::vector<Token> hyps = wer.string_to_tokens(hyp);
  return wer.compute(refs, hyps);
}

} // namespace chime_eval
} // namespace facebook
