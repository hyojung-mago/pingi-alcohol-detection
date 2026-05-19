"""
inference.py - SVM 추론
=======================

학습된 SVM 모델로 음성 취도를 예측합니다.
모델은 Pingi-demo에서 ALC 데이터로 학습되었습니다.

사용법:
  detector = DrunkDetector()
  result = detector.predict("audio.wav")
  result = detector.predict_with_baseline("test.wav", baseline_features)
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import joblib

from .config import MODEL_PATH, INFERENCE_CONFIG, WHISPER_CONFIG, FAKE_DRUNK_CONFIG
from .feature_extractor import FeatureExtractor, get_extractor
from .speech_rate import SpeechRateFeatures, calculate_speech_rate_score, combine_probabilities

logger = logging.getLogger(__name__)


@dataclass
class FakeDrunkResult:
    """취한 척(연기) 감지 결과."""
    is_fake_acting: bool
    probability: float
    message: str

    def to_dict(self) -> Dict:
        return {
            "is_fake_acting": self.is_fake_acting,
            "probability": round(self.probability, 3),
            "message": self.message,
        }


@dataclass
class PredictionResult:
    """예측 결과."""
    is_drunk: bool
    probability: float
    level: int                  # 0~5단계 (0=정상, 5=매우취함)
    level_description: str
    confidence: str             # low, medium, high
    confidence_score: float
    raw_features: Optional[Dict] = None
    baseline_comparison: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "is_drunk": self.is_drunk,
            "probability": round(self.probability, 3),
            "level": self.level,
            "level_description": self.level_description,
            "confidence": self.confidence,
            "confidence_score": round(self.confidence_score, 3),
            "raw_features": self.raw_features,
            "baseline_comparison": self.baseline_comparison,
        }


class DrunkDetector:
    """ALC 학습 기반 취도 감지기."""

    # 0~5단계 레벨 설명
    LEVEL_DESCRIPTIONS = {
        0: "정상",
        1: "약간 취함",
        2: "조금 취함",
        3: "적당히 취함",
        4: "많이 취함",
        5: "매우 취함",
    }

    def __init__(self, model_path: Union[str, Path] = None, version: str = None):
        version = version or INFERENCE_CONFIG.model_version
        if model_path is None:
            model_path = MODEL_PATH / f"drunk_detector_{version}.pkl"
        self.model_path = Path(model_path)
        self.version = version
        self._model = None
        self._extractor = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"모델 파일 없음: {self.model_path}")
        logger.info(f"모델 로드: {self.model_path}")
        raw = joblib.load(self.model_path)
        # pkl이 dict 래핑인 경우 (model, feature_weights, feature_names)
        if isinstance(raw, dict) and "model" in raw:
            self._model = raw["model"]
            self._feature_weights = raw.get("feature_weights")
            logger.info(f"  dict 래핑 모델 (keys: {list(raw.keys())})")
        else:
            self._model = raw
            self._feature_weights = None
        self._extractor = get_extractor()
        self._loaded = True
        logger.info("모델 로드 완료")

    @property
    def model(self):
        if not self._loaded:
            self.load()
        return self._model

    @property
    def extractor(self) -> FeatureExtractor:
        if not self._loaded:
            self.load()
        return self._extractor

    def extract_features(self, audio_path: Union[str, Path]) -> np.ndarray:
        features = self.extractor.extract(Path(audio_path))
        if features is None:
            raise ValueError(f"피처 추출 실패: {audio_path}")
        return features

    def _apply_weights(self, features: np.ndarray) -> np.ndarray:
        """학습 시 사용한 feature_weights 적용."""
        if self._feature_weights is not None:
            return features * self._feature_weights
        return features

    def predict(self, audio_path: Union[str, Path], return_features: bool = False) -> PredictionResult:
        """단일 오디오 취도 예측."""
        features = self.extract_features(audio_path)
        X = self._apply_weights(features).reshape(1, -1)
        proba = self.model.predict_proba(X)[0, 1]

        level = self._determine_level(proba)
        confidence, conf_score = self._calculate_confidence(proba)

        raw_features = self._extract_key_features(features) if return_features else None

        return PredictionResult(
            is_drunk=proba >= 0.5,
            probability=float(proba),
            level=level,
            level_description=self.LEVEL_DESCRIPTIONS[level],
            confidence=confidence,
            confidence_score=conf_score,
            raw_features=raw_features,
        )

    def predict_with_baseline(
        self,
        audio_path: Union[str, Path],
        baseline_features: np.ndarray,
        current_speech_rate: Optional[SpeechRateFeatures] = None,
        baseline_speech_rate: Optional[SpeechRateFeatures] = None,
    ) -> PredictionResult:
        """베이스라인 대비 취도 예측.
        
        Demo에서 75.1% 정확도 달성한 방식 + Whisper 발화속도 보정:
        - baseline_proba: 정상 상태 피처로 예측한 확률
        - current_proba: 현재 피처로 예측한 확률
        - delta = current_proba - baseline_proba
        - delta >= threshold(0.08) → drunk
        - (선택) Whisper 발화속도 분석으로 확률 보정
        
        Args:
            audio_path: 테스트 오디오 경로
            baseline_features: 베이스라인 openSMILE 피처
            current_speech_rate: 현재 Whisper 발화속도 (선택)
            baseline_speech_rate: 베이스라인 Whisper 발화속도 (선택)
        """
        current_features = self.extract_features(audio_path)
        
        # 현재 피처 확률
        X_current = self._apply_weights(current_features).reshape(1, -1)
        current_proba = self.model.predict_proba(X_current)[0, 1]
        
        # 베이스라인 피처 확률
        X_baseline = self._apply_weights(baseline_features).reshape(1, -1)
        baseline_proba = self.model.predict_proba(X_baseline)[0, 1]
        
        # Demo 방식: 단순 delta 비교
        delta = current_proba - baseline_proba
        threshold = INFERENCE_CONFIG.delta_threshold
        
        # 발화속도 보정 (Whisper 피처가 있으면)
        speech_rate_score = None
        speech_rate_info = None
        
        if (WHISPER_CONFIG.enabled and 
            current_speech_rate is not None and 
            baseline_speech_rate is not None):
            
            speech_rate_score = calculate_speech_rate_score(
                baseline_rate=baseline_speech_rate.syllables_per_sec,
                current_rate=current_speech_rate.syllables_per_sec,
                max_decrease_ratio=WHISPER_CONFIG.max_decrease_ratio,
            )
            
            # SVM 확률 + 발화속도 점수 결합
            final_proba = combine_probabilities(
                svm_proba=current_proba,
                speech_rate_score=speech_rate_score,
                svm_weight=WHISPER_CONFIG.svm_weight,
            )
            
            speech_rate_info = {
                "baseline_rate": baseline_speech_rate.syllables_per_sec,
                "current_rate": current_speech_rate.syllables_per_sec,
                "rate_change": (
                    (baseline_speech_rate.syllables_per_sec - current_speech_rate.syllables_per_sec)
                    / baseline_speech_rate.syllables_per_sec * 100
                    if baseline_speech_rate.syllables_per_sec > 0 else 0
                ),
                "speech_rate_score": speech_rate_score,
                "svm_weight": WHISPER_CONFIG.svm_weight,
            }
            
            logger.info(
                f"발화속도 보정: SVM {current_proba:.2f} + "
                f"발화속도 {speech_rate_score:.2f} = {final_proba:.2f}"
            )
        else:
            # Whisper 없으면 기존 방식
            final_proba = current_proba
        
        # 취함 판정: delta 기준 (발화속도 점수도 고려)
        if speech_rate_score is not None and speech_rate_score >= 0.5:
            # 발화속도가 크게 감소했으면 취함 판정 기준 완화
            is_drunk = delta >= (threshold * 0.5) or speech_rate_score >= 0.7
        else:
            is_drunk = delta >= threshold
        
        level = self._determine_level(final_proba)
        confidence, conf_score = self._calculate_confidence(final_proba)
        
        # 피처 변화 정보 (디버깅/분석용)
        feature_changes = self._calculate_feature_changes(baseline_features, current_features)

        baseline_comparison = {
            "baseline_proba": float(baseline_proba),
            "current_proba": float(current_proba),
            "delta": float(delta),
            "threshold": threshold,
            "feature_changes": feature_changes,
        }
        
        if speech_rate_info:
            baseline_comparison["speech_rate"] = speech_rate_info

        return PredictionResult(
            is_drunk=is_drunk,
            probability=float(final_proba),
            level=level,
            level_description=self.LEVEL_DESCRIPTIONS[level],
            confidence=confidence,
            confidence_score=conf_score,
            raw_features=self._extract_key_features(current_features),
            baseline_comparison=baseline_comparison,
        )

    def create_baseline(self, audio_path: Union[str, Path]) -> np.ndarray:
        """정상 상태 피처 추출 (베이스라인용)."""
        return self.extract_features(audio_path)

    def _determine_level(self, probability: float) -> int:
        """확률에서 0~5단계 레벨 결정."""
        thresholds = INFERENCE_CONFIG.level_thresholds
        if probability < thresholds[0]:
            return 0
        elif probability < thresholds[1]:
            return 1
        elif probability < thresholds[2]:
            return 2
        elif probability < thresholds[3]:
            return 3
        elif probability < thresholds[4]:
            return 4
        else:
            return 5

    def _calculate_confidence(self, probability: float) -> tuple:
        distance = abs(probability - 0.5) * 2
        if distance >= 0.6:
            return "high", distance
        elif distance >= 0.3:
            return "medium", distance
        else:
            return "low", distance

    def _extract_key_features(self, features: np.ndarray) -> Dict[str, float]:
        feature_names = self.extractor.feature_names
        key_patterns = [
            "F0semitoneFrom27.5Hz_sma3nz_amean",
            "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
            "jitterLocal_sma3nz_amean",
            "shimmerLocaldB_sma3nz_amean",
            "HNRdBACF_sma3nz_amean",
            "loudness_sma3_amean",
            "VoicedSegmentsPerSec",
            "MeanVoicedSegmentLengthSec",
        ]
        result = {}
        for pattern in key_patterns:
            for idx, name in enumerate(feature_names):
                if pattern in name:
                    short = name.replace("_sma3nz_amean", "").replace("_sma3nz_stddevNorm", "_std").replace("_sma3_amean", "")
                    result[short] = float(features[idx])
                    break
        return result

    def _calculate_feature_changes(self, baseline: np.ndarray, current: np.ndarray) -> Dict[str, float]:
        feature_names = self.extractor.feature_names
        key_patterns = [
            ("F0_mean", "F0semitoneFrom27.5Hz_sma3nz_amean"),
            ("F0_std", "F0semitoneFrom27.5Hz_sma3nz_stddevNorm"),
            ("jitter", "jitterLocal_sma3nz_amean"),
            ("shimmer", "shimmerLocaldB_sma3nz_amean"),
            ("HNR", "HNRdBACF_sma3nz_amean"),
            ("loudness", "loudness_sma3_amean"),
            ("voiced_per_sec", "VoicedSegmentsPerSec"),
            ("voiced_length", "MeanVoicedSegmentLengthSec"),
        ]
        changes = {}
        for short_name, pattern in key_patterns:
            for idx, name in enumerate(feature_names):
                if pattern in name:
                    base_val = baseline[idx]
                    curr_val = current[idx]
                    if abs(base_val) > 1e-6:
                        change_pct = (curr_val - base_val) / abs(base_val) * 100
                    else:
                        change_pct = 0
                    changes[short_name] = float(change_pct)
                    break
        return changes

    def _score_changes(self, changes: Dict[str, float]) -> float:
        """베이스라인 대비 변화를 0~1 스코어로 변환.

        Demo features.py와 동일한 설계 원칙:
        - 0.0 = 변화 없음 (sober)
        - 1.0 = 최대 취함 방향 변화
        - 음수 변화(오히려 좋아짐)는 0으로 클램핑
        """
        score = 0.0  # ← 핵심 수정: 0에서 시작

        # F0 변동성 증가 (가장 신뢰, Demo의 weight=0.25에 해당)
        f0_std_change = changes.get("F0_std", 0)
        if f0_std_change > 20:
            score += 0.30
        elif f0_std_change > 10:
            score += 0.15
        elif f0_std_change > 5:
            score += 0.08

        # F0 평균 증가 (Demo weight=0.08)
        f0_mean_change = changes.get("F0_mean", 0)
        if f0_mean_change > 15:
            score += 0.15
        elif f0_mean_change > 8:
            score += 0.10
        elif f0_mean_change > 3:
            score += 0.05

        # 음량 증가 (Demo weight=0.07)
        loudness_change = changes.get("loudness", 0)
        if loudness_change > 20:
            score += 0.15
        elif loudness_change > 10:
            score += 0.08

        # 발화 속도 감소 — 가장 중요한 지표 (Demo weight=0.40)
        voiced_change = changes.get("voiced_per_sec", 0)
        if voiced_change < -25:
            score += 0.40
        elif voiced_change < -15:
            score += 0.25
        elif voiced_change < -8:
            score += 0.15
        elif voiced_change < -3:
            score += 0.08

        # 발화 길이 변화 (길어지면 취함)
        voiced_len_change = changes.get("voiced_length", 0)
        if voiced_len_change > 20:
            score += 0.10
        elif voiced_len_change > 10:
            score += 0.05

        return min(max(score, 0.0), 1.0)


class FakeDrunkDetector:
    """진짜 취함(ALC) vs 취한 척 연기(Thorsten) 분류."""

    def __init__(self, model_path: Union[str, Path] = None):
        if model_path is None:
            model_path = MODEL_PATH / FAKE_DRUNK_CONFIG.model_filename
        self.model_path = Path(model_path)
        self._model = None
        self._extractor = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"취한척 모델 없음: {self.model_path}")
        logger.info(f"취한척 모델 로드: {self.model_path}")
        raw = joblib.load(self.model_path)
        self._model = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        self._extractor = get_extractor()
        self._loaded = True

    def _pick_message(self) -> str:
        messages: List[str] = list(FAKE_DRUNK_CONFIG.messages)
        return random.choice(messages)

    def predict(self, audio_path: Union[str, Path]) -> FakeDrunkResult:
        if not self._loaded:
            self.load()
        features = self._extractor.extract(Path(audio_path))
        if features is None:
            raise ValueError(f"피처 추출 실패: {audio_path}")
        proba = self._model.predict_proba(features.reshape(1, -1))[0, 1]
        is_fake = proba >= FAKE_DRUNK_CONFIG.threshold
        return FakeDrunkResult(
            is_fake_acting=is_fake,
            probability=float(proba),
            message=self._pick_message() if is_fake else "",
        )

    def is_fake(self, audio_path: Union[str, Path]) -> bool:
        return self.predict(audio_path).is_fake_acting


# 싱글톤
_detector: Optional[DrunkDetector] = None
_fake_detector: Optional[FakeDrunkDetector] = None


def get_detector() -> DrunkDetector:
    global _detector
    if _detector is None:
        _detector = DrunkDetector()
    return _detector


def get_fake_detector() -> FakeDrunkDetector:
    global _fake_detector
    if _fake_detector is None:
        _fake_detector = FakeDrunkDetector()
    return _fake_detector
