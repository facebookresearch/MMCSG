// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <unicode/normalizer2.h>
#include <string>

namespace facebook {
namespace langtech {

class UnicodeNorm {
 public:
  static const icu::Normalizer2* NFKC();
  static const icu::Normalizer2* NFKD();
  static std::string normalize(
      const std::string& str,
      const icu::Normalizer2* norm);
};

} // namespace langtech
} // namespace facebook
