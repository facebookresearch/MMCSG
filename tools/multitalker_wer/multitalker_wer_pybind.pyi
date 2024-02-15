# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import auto, Enum
from typing import Any, Dict, List, Optional

class Code(Enum):
    match = auto()
    wrong_speaker = auto()
    substitution = auto()
    substitution_and_wrong_speaker = auto()
    insertion = auto()
    deletion = auto()

class Token:
    label: str
    speaker: int
    start: float
    end: float

class AlignmentResult:
    ref: List[int]
    speaker: List[int]
    hyp: List[int]
    codes: List[Code]
    score: float

class MultiTalkerEditDistance:
    def compute(self, ref: List[Token], hyp: List[Token]) -> AlignmentResult: ...
    def string_to_tokens(self, str: str) -> List[Token]: ...
    def alignment_to_string(
        self, alignment: AlignmentResult, ref: List[Token], hyp: List[Token]
    ) -> str: ...

def compute_multitalker_alignment(ref: str, hyp: str) -> AlignmentResult: ...
def code_to_string(code: Code) -> str: ...

class SpeakerResult:
    referenceWords: int
    inserts: int
    deletes: int
    substitutions: int
    wrongSpeakers: int
    substitutionsWithWrongSpeaker: int
    wordErrorRate: float
    insertionRate: float
    deletionRate: float
    substitutionRate: float
    speakerAttributionErrorRate: float

class MultiTalkerResult:
    self: SpeakerResult
    other: SpeakerResult
    stringAlignment: str
    alignmentResult: AlignmentResult

class MultiTalkerWordErrorRateOptions:
    caseSensitive: Optional[bool]
    stripRegex: Optional[str]
    punctSymbols: Optional[str]
    wordSubstitutions: Optional[Dict[str, str]]

class MultiTalkerWordErrorRate:
    @staticmethod
    def compute(
        reference: str,
        hypothesis: str,
        options: MultiTalkerWordErrorRateOptions = MultiTalkerWordErrorRateOptions(),
    ) -> MultiTalkerResult: ...
    @staticmethod
    def computeMultiple(
        reference: List[str],
        hypothesis: List[str],
        options: MultiTalkerWordErrorRateOptions = MultiTalkerWordErrorRateOptions(),
    ) -> List[MultiTalkerResult]: ...
    @staticmethod
    def report(results: List[MultiTalkerResult]) -> MultiTalkerResult: ...
