"""
config.py - ML 설정

기본 production 모델: v2-delta (ALC delta-feature SVM)
학습: ml/train_alc.py | 추론: ml/inference.py
임계값은 models/final_weights.json (train_alc fit 결과) 우선 로드.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict
import opensmile


# ============================================================
# 경로
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models"


# ============================================================
# 피처 설정
# ============================================================

@dataclass
class FeatureConfig:
    """openSMILE eGeMAPSv02 피처 추출 설정."""
    opensmile_feature_set: str = "eGeMAPSv02"
    opensmile_feature_level: str = "Functionals"
    target_sample_rate: int = 16000


# ============================================================
# 추론 설정
# ============================================================

@dataclass
class InferenceConfig:
    """추론 설정. level_thresholds / delta_threshold는 final_weights.json이 우선."""

    # production 기본. drunk_detector_v2-delta.pkl
    model_version: str = "v2-delta"

    # 아래 level_thresholds / delta_threshold는 JSON 없을 때 fallback.
    # v2-delta fit 값은 final_weights.json 참고 (train_alc --mode delta).
    #
    # L0/L1: optimal_threshold (drunk proba, accuracy fit)
    # L1~L5: drunk≥threshold proba 20% 분위 (train_alc --mode delta)
    level_thresholds: Dict[int, float] = field(default_factory=lambda: {
        0: 0.625,
        1: 0.711,
        2: 0.783,
        3: 0.835,
        4: 0.892,
        5: 1.0,
    })
    
    level_descriptions: Dict[int, str] = field(default_factory=lambda: {
        0: "정상",
        1: "약간 취함",
        2: "조금 취함",
        3: "적당히 취함",
        4: "많이 취함",
        5: "매우 취함",
    })

    # v2-delta: proba(current-baseline) >= threshold → drunk (기본 0.625)
    # v2 absolute legacy: (proba_current - proba_baseline) >= threshold
    delta_threshold: float = 0.625


# ============================================================
# Whisper 설정
# ============================================================

@dataclass
class WhisperConfig:
    """Whisper STT API 설정."""
    
    # API 엔드포인트
    base_url: str = "https://op1-api.magovoice.com/whisper"
    
    # 요청 설정
    timeout: float = 30.0
    max_retries: int = 2
    language: str = "ko"
    
    # 발화속도 분석 설정 (ALC 데이터 기반 최적화)
    # 테스트 결과: SVM만 사용이 최적 (UAR 72.9%)
    # 발화속도 추가 시 성능 저하 (openSMILE이 이미 음향 기반 발화속도 포함)
    # 한국어 데이터로 추가 검증 후 가중치 조정 필요
    svm_weight: float = 1.0          # SVM 모델 가중치 (현재 100%)
    speech_rate_weight: float = 0.0  # 발화속도 점수 가중치 (현재 0%)
    max_decrease_ratio: float = 0.3  # 취함 판정 최대 감소율 (30%)
    
    # 활성화 여부 (한국어 데이터 검증 전까지 비활성화 권장)
    enabled: bool = False


# ============================================================
# 취한 척 감지 (Thorsten Emotional + ALC)
# ============================================================

@dataclass
class FakeDrunkConfig:
    """연기/가짜 취함 감지 (fake_drunk_detector_v1.pkl)."""

    model_filename: str = "fake_drunk_detector_v1.pkl"
    # proba[:, 1] >= threshold → 취한 척 (legacy, production 미사용)
    threshold: float = 0.5
    enabled: bool = False

    messages: tuple = (
        "취한 척 하신 거 같은데… 흠 ~ 🎭",
        "에헤이~ 연기 잘하시네요! 취한 척이에요 🎭",
        "술 취한 게 아니라 연기 아닌가요? 흠흠 ~",
        "핑이 눈엔 취한 척이에요! 🎭",
    )


# ============================================================
# 합성 음성 감지 (OpenAI TTS vs 실제 녹음)
# ============================================================

@dataclass
class SyntheticVoiceConfig:
    """TTS/합성 음성 감지 — SVM + AASIST-L 앙상블.

    v1: openSMILE SVM only
    v2: SVM 주도 + AASIST-L 보조 앙상블
    """

    model_filename: str = "synthetic_voice_detector_v1.pkl"
    weights_filename: str = "final_weights_synthetic.json"
    aasist_weights: str = "AASIST-L.pth"
    # proba[:, 1] >= threshold → synthetic (class 1)
    threshold: float = 0.5
    enabled: bool = True
    aasist_enabled: bool = True  # AASIST 앙상블 활성화

    messages: tuple = (
        "합성 음성이에요! 🎙️ 실제 목소리로 다시 녹음해 주세요.",
        "AI로 만든 목소리 같아요 🎙️ 마이크로 직접 녹음해 주세요.",
        "핑이는 합성 음성은 분석하지 않아요 🎙️",
    )


# ============================================================
# 인스턴스
# ============================================================

FEATURE_CONFIG = FeatureConfig()
INFERENCE_CONFIG = InferenceConfig()
WHISPER_CONFIG = WhisperConfig()
FAKE_DRUNK_CONFIG = FakeDrunkConfig()
SYNTHETIC_VOICE_CONFIG = SyntheticVoiceConfig()


def get_opensmile():
    """openSMILE 인스턴스 생성."""
    feature_set = getattr(opensmile.FeatureSet, FEATURE_CONFIG.opensmile_feature_set)
    feature_level = getattr(opensmile.FeatureLevel, FEATURE_CONFIG.opensmile_feature_level)
    return opensmile.Smile(feature_set=feature_set, feature_level=feature_level)
