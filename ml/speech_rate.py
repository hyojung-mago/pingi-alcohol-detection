"""
speech_rate.py - 발화속도 분석
==============================

Whisper STT 결과에서 발화속도 피처를 추출합니다.

피처:
  - syllables_per_sec: 음절/초
  - words_per_sec: 단어/초
  - chars_per_sec: 글자/초
  - pause_ratio: 휴지 비율
  - articulation_rate: 실제 발화 속도 (휴지 제외)
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

from .whisper_client import WhisperResult

logger = logging.getLogger(__name__)


@dataclass
class SpeechRateFeatures:
    """발화속도 피처."""
    syllables_per_sec: float
    words_per_sec: float
    chars_per_sec: float
    pause_ratio: float
    articulation_rate: float  # 휴지 제외 음절/초
    total_duration: float
    speech_duration: float
    syllable_count: int
    word_count: int
    
    def to_dict(self) -> dict:
        return {
            "syllables_per_sec": round(self.syllables_per_sec, 2),
            "words_per_sec": round(self.words_per_sec, 2),
            "chars_per_sec": round(self.chars_per_sec, 2),
            "pause_ratio": round(self.pause_ratio, 3),
            "articulation_rate": round(self.articulation_rate, 2),
            "total_duration": round(self.total_duration, 2),
            "speech_duration": round(self.speech_duration, 2),
            "syllable_count": self.syllable_count,
            "word_count": self.word_count,
        }


def count_korean_syllables(text: str) -> int:
    """
    한국어 음절 수 계산.
    
    한글 음절 (가-힣) + 영문자 + 숫자를 카운트.
    """
    # 한글 음절 (가-힣)
    korean = len(re.findall(r'[가-힣]', text))
    # 영문자 (음절로 근사)
    english = len(re.findall(r'[a-zA-Z]+', text))
    # 숫자 (각 자리수)
    digits = len(re.findall(r'[0-9]', text))
    
    return korean + english + digits


def extract_speech_rate_features(
    whisper_result: WhisperResult,
) -> Optional[SpeechRateFeatures]:
    """
    Whisper 결과에서 발화속도 피처 추출.
    
    Args:
        whisper_result: Whisper STT 결과
        
    Returns:
        SpeechRateFeatures 또는 None
    """
    if not whisper_result or not whisper_result.text:
        logger.warning("Whisper 결과가 비어있음")
        return None
    
    text = whisper_result.text.strip()
    duration = whisper_result.duration
    
    if duration <= 0:
        logger.warning("duration이 0 이하")
        return None
    
    # 음절/단어/글자 수
    syllable_count = count_korean_syllables(text)
    word_count = len(text.split())
    char_count = len(text.replace(" ", ""))
    
    # 발화 시간 계산 (휴지 제외)
    speech_duration = 0
    if whisper_result.segments:
        for seg in whisper_result.segments:
            speech_duration += (seg.end - seg.start)
    else:
        speech_duration = duration
    
    # 휴지 비율
    pause_duration = duration - speech_duration
    pause_ratio = pause_duration / duration if duration > 0 else 0
    pause_ratio = max(0, min(1, pause_ratio))  # 0~1 클램핑
    
    # 발화속도 계산
    syllables_per_sec = syllable_count / duration if duration > 0 else 0
    words_per_sec = word_count / duration if duration > 0 else 0
    chars_per_sec = char_count / duration if duration > 0 else 0
    
    # 조음 속도 (휴지 제외)
    articulation_rate = (
        syllable_count / speech_duration 
        if speech_duration > 0 else syllables_per_sec
    )
    
    return SpeechRateFeatures(
        syllables_per_sec=syllables_per_sec,
        words_per_sec=words_per_sec,
        chars_per_sec=chars_per_sec,
        pause_ratio=pause_ratio,
        articulation_rate=articulation_rate,
        total_duration=duration,
        speech_duration=speech_duration,
        syllable_count=syllable_count,
        word_count=word_count,
    )


def calculate_speech_rate_score(
    baseline_rate: float,
    current_rate: float,
    max_decrease_ratio: float = 0.3,
) -> float:
    """
    발화속도 변화 → 0~1 점수.
    
    취함 시 발화속도 감소 → 점수 증가
    
    Args:
        baseline_rate: 베이스라인 발화속도 (syllables_per_sec)
        current_rate: 현재 발화속도
        max_decrease_ratio: 최대 감소 비율 (기본 30%)
        
    Returns:
        0.0 (변화 없음/빨라짐) ~ 1.0 (max_decrease_ratio 이상 느려짐)
    """
    if baseline_rate <= 0:
        return 0.0
    
    # 감소 비율 계산
    decrease_ratio = (baseline_rate - current_rate) / baseline_rate
    
    if decrease_ratio <= 0:
        # 오히려 빨라짐 → 취하지 않음
        return 0.0
    elif decrease_ratio >= max_decrease_ratio:
        # 최대 감소 이상 → 확실히 취함
        return 1.0
    else:
        # 선형 보간
        return decrease_ratio / max_decrease_ratio


def combine_probabilities(
    svm_proba: float,
    speech_rate_score: float,
    svm_weight: float = 0.7,
) -> float:
    """
    SVM 확률과 발화속도 점수 결합.
    
    Args:
        svm_proba: SVM 모델 확률 (0~1)
        speech_rate_score: 발화속도 점수 (0~1)
        svm_weight: SVM 가중치 (기본 0.7)
        
    Returns:
        결합된 확률 (0~1)
    """
    speech_rate_weight = 1.0 - svm_weight
    combined = svm_weight * svm_proba + speech_rate_weight * speech_rate_score
    return min(max(combined, 0.0), 1.0)
