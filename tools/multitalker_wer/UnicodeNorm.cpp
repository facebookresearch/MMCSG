// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "UnicodeNorm.h"
#include <stdexcept>

namespace facebook {
namespace langtech {

using namespace icu;

const icu::Normalizer2* UnicodeNorm::NFKC() {
  UErrorCode errorCode = U_ZERO_ERROR;
  auto normalizer = icu::Normalizer2::getNFKCInstance(errorCode);
  if (U_FAILURE(errorCode)) {
    throw std::runtime_error("Normalizer2 instance creation failed");
  }
  return normalizer;
}

const icu::Normalizer2* UnicodeNorm::NFKD() {
  UErrorCode errorCode = U_ZERO_ERROR;
  auto normalizer = icu::Normalizer2::getNFKDInstance(errorCode);
  if (U_FAILURE(errorCode)) {
    throw std::runtime_error("Normalizer2 instance creation failed");
  }
  return normalizer;
}

std::string UnicodeNorm::normalize(
    const std::string& str,
    const icu::Normalizer2* norm) {
  // convert to Unicode string (assuming utf8)
  UErrorCode errorCode = U_ZERO_ERROR;
  icu::UnicodeString input = icu::UnicodeString::fromUTF8(str);
  icu::UnicodeString output = norm->normalize(input, errorCode);
  if (U_FAILURE(errorCode)) {
    throw std::runtime_error("utf8 error");
  }
  std::string res;
  output.toUTF8String(res);
  return res;
}

} // namespace langtech
} // namespace facebook
