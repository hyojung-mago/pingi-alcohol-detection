"""
inference.py - SVM 추론

training_mode:
  delta_features (v2-delta, production):
    X = current_features - baseline_features
    proba = SVM(X)
    is_drunk = proba >= optimal_threshold
    level = level_thresholds(proba)  # L0~L5, train_alc fit
    change_pct = (proba - proba_at_zero_drift) × 100  # baseline 대비, threshold 대비 아님

  absolute (v2 legacy):
    proba on absolute features, delta = proba_current - proba_baseline
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import joblib

from .config import MODEL_PATH, INFERENCE_CONFIG, WHISPER_CONFIG, FAKE_DRUNK_CONFIG, SYNTHETIC_VOICE_CONFIG
from .feature_extractor import FeatureExtractor, get_extractor
from .speech_rate import SpeechRateFeatures, calculate_speech_rate_score, combine_probabilities

logger = logging.getLogger(__name__)


@dataclass
class SyntheticVoiceResult:
    """합성(TTS) 음성 감지 결과."""
    is_synthetic: bool
    probability: float
    message: str

    def to_dict(self) -> Dict:
        return {
            "is_synthetic": self.is_synthetic,
            "probability": round(self.probability, 3),
            "message": self.message,
        }


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
    change_pct: float = 0.0     # 베이스라인 대비 delta (%p), 없으면 0
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
            "change_pct": self.change_pct,
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
        self.delta_threshold = INFERENCE_CONFIG.delta_threshold
        self.level_thresholds = dict(INFERENCE_CONFIG.level_thresholds)
        self.training_mode = "absolute"  # absolute | delta_features

    def _load_weights_from_json(self):
        candidates = [MODEL_PATH / f"final_weights_{self.version}.json"]
        if self.version in ("v2", "v2-delta"):
            candidates.append(MODEL_PATH / "final_weights.json")
        for weights_path in candidates:
            if not weights_path.exists():
                continue
            try:
                data = json.loads(weights_path.read_text(encoding="utf-8"))
                if data.get("version") and data["version"] != self.version:
                    if self.version not in ("v2", "v2-delta"):
                        continue
                training = data.get("training", {})
                drunk = data.get("drunk_detection", {})
                if training.get("mode"):
                    self.training_mode = training["mode"]
                self.delta_threshold = float(
                    drunk.get("optimal_threshold", self.delta_threshold)
                )
                raw_levels = drunk.get("level_thresholds")
                if raw_levels:
                    self.level_thresholds = {
                        int(k): float(v) for k, v in raw_levels.items()
                    }
                return
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

    def load(self):
        if self._loaded:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"모델 파일 없음: {self.model_path}")
        logger.info(f"모델 로드: {self.model_path}")
        raw = joblib.load(self.model_path)
        # pkl이 dict 래핑인 경우 (model, feature_names, training_mode)
        if isinstance(raw, dict) and "model" in raw:
            self._model = raw["model"]
            self.training_mode = raw.get("training_mode", self.training_mode)
            logger.info(f"  dict 래핑 모델 (keys: {list(raw.keys())})")
        else:
            self._model = raw
        self._load_weights_from_json()
        self._extractor = get_extractor()
        self._loaded = True
        logger.info("모델 로드 완료 (mode=%s, threshold=%.3f)", self.training_mode, self.delta_threshold)

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

    def predict(self, audio_path: Union[str, Path], return_features: bool = False) -> PredictionResult:
        """단일 오디오 취도 예측."""
        features = self.extract_features(audio_path)
        X = features.reshape(1, -1)
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

    def _predict_delta_features(
        self,
        current_features: np.ndarray,
        baseline_features: np.ndarray,
        current_speech_rate: Optional[SpeechRateFeatures] = None,
        baseline_speech_rate: Optional[SpeechRateFeatures] = None,
    ) -> PredictionResult:
        """delta_features: (current - baseline) 피처 → SVM proba.

        핵심: SVM(zero_vector)의 "resting" 확률이 threshold 부근이므로,
        레벨과 취함 판정은 baseline_proba 대비 delta를 사용해야 한다.

        change_pct = (current_proba - baseline_proba) × 100
          - baseline과 동일하게 읽으면 ≈ 0 (본인 sober 대비 변화 없음)
        """
        delta_features = current_features - baseline_features
        X_delta = delta_features.reshape(1, -1)
        current_proba = float(self.model.predict_proba(X_delta)[0, 1])
        threshold = self.delta_threshold

        X_zero = np.zeros_like(delta_features).reshape(1, -1)
        baseline_proba = float(self.model.predict_proba(X_zero)[0, 1])
        delta = current_proba - baseline_proba

        logger.info(
            f"delta_features: baseline_proba={baseline_proba:.3f}, "
            f"current_proba={current_proba:.3f}, delta={delta:.3f}"
        )

        speech_rate_score = None
        speech_rate_info = None
        final_proba = current_proba

        if (
            WHISPER_CONFIG.enabled
            and current_speech_rate is not None
            and baseline_speech_rate is not None
        ):
            speech_rate_score = calculate_speech_rate_score(
                baseline_rate=baseline_speech_rate.syllables_per_sec,
                current_rate=current_speech_rate.syllables_per_sec,
                max_decrease_ratio=WHISPER_CONFIG.max_decrease_ratio,
            )
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

        # 취함 판정: baseline 대비 delta 기준 (raw proba가 아님)
        # 자연 음성 변동(같은 사람, 같은 상태)으로 delta 0.00~0.12 발생 가능
        DELTA_DRUNK_THRESHOLD = 0.12
        if speech_rate_score is not None and speech_rate_score >= 0.5:
            is_drunk = delta >= (DELTA_DRUNK_THRESHOLD * 0.5) or speech_rate_score >= 0.7
        else:
            is_drunk = delta >= DELTA_DRUNK_THRESHOLD

        # 레벨 판정: is_drunk=False이면 L0 강제
        if not is_drunk:
            level = 0
            adjusted_proba = 0.0
        else:
            # delta를 level threshold 범위에 매핑
            # DELTA_DRUNK_THRESHOLD를 넘은 만큼만 레벨에 반영
            excess = delta - DELTA_DRUNK_THRESHOLD
            anchor = self.level_thresholds[0]
            adjusted_proba = max(0.0, min(1.0, anchor + excess))
            level = self._determine_level(adjusted_proba)

        confidence, conf_score = self._calculate_confidence(final_proba)
        feature_changes = self._calculate_feature_changes(baseline_features, current_features)

        baseline_comparison = {
            "baseline_proba": baseline_proba,
            "current_proba": current_proba,
            "delta": delta,
            "adjusted_proba": adjusted_proba,
            "threshold": threshold,
            "training_mode": "delta_features",
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
            change_pct=round(delta * 100, 1),
            raw_features=self._extract_key_features(current_features),
            baseline_comparison=baseline_comparison,
        )

    # 발화 감지 최소 기준
    MIN_VOICED_SEGMENTS_PER_SEC = 0.5
    MIN_LOUDNESS = 0.05

    def _check_voice_activity(self, features: np.ndarray) -> bool:
        """녹음에 실제 발화가 포함되어 있는지 검사."""
        names = self.extractor.feature_names
        voiced_per_sec = 0.0
        loudness = 0.0
        for i, name in enumerate(names):
            if "VoicedSegmentsPerSec" in name:
                voiced_per_sec = float(features[i])
            elif name == "loudness_sma3_amean":
                loudness = float(features[i])
        if voiced_per_sec < self.MIN_VOICED_SEGMENTS_PER_SEC or loudness < self.MIN_LOUDNESS:
            logger.warning(
                f"발화 미감지: voiced/sec={voiced_per_sec:.2f}, loudness={loudness:.4f}"
            )
            return False
        return True

    def predict_with_baseline(
        self,
        audio_path: Union[str, Path],
        baseline_features: np.ndarray,
        current_speech_rate: Optional[SpeechRateFeatures] = None,
        baseline_speech_rate: Optional[SpeechRateFeatures] = None,
    ) -> PredictionResult:
        """베이스라인 대비 취도 예측.
        
        Args:
            audio_path: 테스트 오디오 경로
            baseline_features: 베이스라인 openSMILE 피처
            current_speech_rate: 현재 Whisper 발화속도 (선택)
            baseline_speech_rate: 베이스라인 Whisper 발화속도 (선택)
        
        Raises:
            ValueError: 녹음에 발화가 감지되지 않은 경우
        """
        current_features = self.extract_features(audio_path)

        if not self._check_voice_activity(current_features):
            raise ValueError("녹음에서 음성이 감지되지 않았어요. 문장을 소리 내어 읽어주세요.")

        if self.training_mode == "delta_features":
            return self._predict_delta_features(
                current_features,
                baseline_features,
                current_speech_rate,
                baseline_speech_rate,
            )

        # absolute 모드: proba delta 비교
        X_current = current_features.reshape(1, -1)
        current_proba = self.model.predict_proba(X_current)[0, 1]
        X_baseline = baseline_features.reshape(1, -1)
        baseline_proba = self.model.predict_proba(X_baseline)[0, 1]
        
        # Demo 방식: 단순 delta 비교
        delta = current_proba - baseline_proba
        threshold = self.delta_threshold
        
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

        # 레벨은 베이스라인 대비 delta로 산정한다.
        # anchor = level_thresholds[0] - delta_threshold → drunk 경계와 L0/L1 경계가 정렬됨
        # (예: 0.26 - 0.08 = 0.18). delta=0이면 L0, delta가 drunk_threshold를 넘는 순간 L1로 진입.
        level_anchor = self.level_thresholds[0] - threshold
        adjusted_proba = max(0.0, min(1.0, level_anchor + delta))
        level = self._determine_level(adjusted_proba)
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
            change_pct=round(delta * 100, 1),
            raw_features=self._extract_key_features(current_features),
            baseline_comparison=baseline_comparison,
        )

    def create_baseline(self, audio_path: Union[str, Path]) -> np.ndarray:
        """정상 상태 피처 추출 (베이스라인용)."""
        return self.extract_features(audio_path)

    def _determine_level(self, probability: float) -> int:
        """SVM proba → L0~L5. level_thresholds는 train_alc same-person fit (final_weights.json)."""
        thresholds = self.level_thresholds
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


class SyntheticVoiceDetector:
    """실제 녹음 vs 합성 음성 분류 — SVM + AASIST-L 앙상블.

    전략 (테스트 결과 기반):
      1. SVM (openSMILE eGeMAPSv02): 우리 데이터에서 완벽 분류
         - 실제 음성: 0.000~0.043, TTS: 1.000
         - 폰 녹음도 정확히 real 판정
      2. AASIST-L (raw waveform GNN): ASVspoof pretrained
         - ALC 실제: 0.018~0.273 ✅
         - 폰 녹음: 0.93~0.99 ❌ (코덱 아티팩트 오탐)
         - TTS: 대부분 >0.7 but 일부 miss (0.12, 0.16)

      → SVM 주도 + AASIST 보조:
        - SVM 확신 (>0.9): SVM 결과 채택 (AASIST 무시)
        - SVM 불확실 (0.3~0.9): 앙상블 (SVM 70% + AASIST 30%)
        - SVM < 0.3 but AASIST > 0.8: 경고 플래그 (추가 확인 권고)
    """

    SVM_WEIGHT = 0.7
    AASIST_WEIGHT = 0.3
    SVM_CONFIDENT_THRESHOLD = 0.9  # SVM만으로 확정
    SVM_REJECT_THRESHOLD = 0.3     # SVM이 "아닌 것 같다"고 확신

    def __init__(self, model_path: Union[str, Path] = None):
        if model_path is None:
            model_path = MODEL_PATH / SYNTHETIC_VOICE_CONFIG.model_filename
        self.model_path = Path(model_path)
        self._model = None
        self._feature_weights = None
        self._extractor = None
        self._aasist = None
        self._loaded = False
        self.threshold = SYNTHETIC_VOICE_CONFIG.threshold

    def _load_threshold(self):
        weights_path = MODEL_PATH / SYNTHETIC_VOICE_CONFIG.weights_filename
        if not weights_path.exists():
            return
        try:
            data = json.loads(weights_path.read_text(encoding="utf-8"))
            self.threshold = float(
                data.get("detection", {}).get("optimal_threshold", self.threshold)
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

    def load(self):
        if self._loaded:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"합성음성 모델 없음: {self.model_path}")
        logger.info(f"합성음성 SVM 모델 로드: {self.model_path}")
        raw = joblib.load(self.model_path)
        if isinstance(raw, dict) and "model" in raw:
            self._model = raw["model"]
            self._feature_weights = raw.get("feature_weights")
        else:
            self._model = raw
        self._load_threshold()
        self._extractor = get_extractor()

        # AASIST-L (optional — 없으면 SVM-only)
        try:
            from .aasist_detector import AASISTDetector
            self._aasist = AASISTDetector()
            self._aasist.load()
            logger.info("AASIST-L 앙상블 활성화")
        except (FileNotFoundError, ImportError) as e:
            self._aasist = None
            logger.info(f"AASIST-L 스킵 (SVM-only): {e}")

        self._loaded = True

    def _pick_message(self) -> str:
        return random.choice(list(SYNTHETIC_VOICE_CONFIG.messages))

    def predict(self, audio_path: Union[str, Path]) -> SyntheticVoiceResult:
        if not self._loaded:
            self.load()
        features = self._extractor.extract(Path(audio_path))
        if features is None:
            raise ValueError(f"피처 추출 실패: {audio_path}")
        if self._feature_weights is not None:
            features = features * self._feature_weights
        svm_proba = float(self._model.predict_proba(features.reshape(1, -1))[0, 1])

        # AASIST-L ensemble
        if self._aasist is not None:
            try:
                ar = self._aasist.predict(audio_path)
                aasist_proba = ar.spoof_probability
            except Exception as e:
                logger.warning(f"AASIST 추론 실패: {e}")
                aasist_proba = None
        else:
            aasist_proba = None

        final_proba = self._ensemble(svm_proba, aasist_proba)
        is_synth = final_proba >= self.threshold

        logger.info(
            "SyntheticDetector: svm=%.3f aasist=%s → final=%.3f (%s)",
            svm_proba,
            f"{aasist_proba:.3f}" if aasist_proba is not None else "N/A",
            final_proba,
            "SYNTHETIC" if is_synth else "real",
        )

        return SyntheticVoiceResult(
            is_synthetic=is_synth,
            probability=final_proba,
            message=self._pick_message() if is_synth else "",
        )

    def _ensemble(self, svm: float, aasist: Optional[float]) -> float:
        """SVM 주도 앙상블."""
        if aasist is None:
            return svm

        # SVM 확신 high → SVM 결과 신뢰
        if svm >= self.SVM_CONFIDENT_THRESHOLD:
            return svm
        # SVM 확신 low (real) → SVM 신뢰 (AASIST의 폰 녹음 오탐 방지)
        if svm <= self.SVM_REJECT_THRESHOLD:
            return svm

        # SVM 불확실 영역 → 가중 평균
        return self.SVM_WEIGHT * svm + self.AASIST_WEIGHT * aasist


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
_synthetic_detector: Optional[SyntheticVoiceDetector] = None


def get_detector() -> DrunkDetector:
    global _detector
    if _detector is None:
        _detector = DrunkDetector()
    return _detector


def get_synthetic_detector() -> SyntheticVoiceDetector:
    global _synthetic_detector
    if _synthetic_detector is None:
        _synthetic_detector = SyntheticVoiceDetector()
    return _synthetic_detector


def get_fake_detector() -> FakeDrunkDetector:
    global _fake_detector
    if _fake_detector is None:
        _fake_detector = FakeDrunkDetector()
    return _fake_detector
