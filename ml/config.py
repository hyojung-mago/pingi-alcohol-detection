"""
config.py - ML 설정
===================

모델 경로, 피처 설정, 추론 임계값을 중앙 관리합니다.
학습은 Pingi-demo에서 수행하고, 여기서는 추론만 합니다.
"""

import os
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
    """추론 임계값 및 가중치."""

    model_version: str = "v2"

    # 취도 레벨 임계값 (0~5단계, ALC 데이터 6분위 기반)
    # 각 단계별 Drunk 비율 증가: +12~18%p (통계적 유의미)
    # 0단계: S:86% D:14% → 5단계: S:13% D:87%
    level_thresholds: Dict[int, float] = field(default_factory=lambda: {
        0: 0.26,   # 0단계 상한: Sober 86%, Drunk 14%
        1: 0.37,   # 1단계 상한: Sober 74%, Drunk 26%
        2: 0.49,   # 2단계 상한: Sober 58%, Drunk 42%
        3: 0.62,   # 3단계 상한: Sober 44%, Drunk 56%
        4: 0.74,   # 4단계 상한: Sober 25%, Drunk 75%
        5: 1.0,    # 5단계: Sober 13%, Drunk 87%
    })
    
    level_descriptions: Dict[int, str] = field(default_factory=lambda: {
        0: "정상",
        1: "약간 취함",
        2: "조금 취함",
        3: "적당히 취함",
        4: "많이 취함",
        5: "매우 취함",
    })

    # 베이스라인 비교 방식 (Demo와 동일)
    # delta = current_proba - baseline_proba >= threshold → drunk
    # Demo full_pipeline.py에서 최적화된 값: 0.08 (75.1% 정확도 달성)
    delta_threshold: float = 0.08


# ============================================================
# Whisper 설정
# ============================================================

@dataclass
class WhisperConfig:
    """Whisper STT API 설정."""
    
    base_url: str = os.getenv("WHISPER_BASE_URL", "https://op1-api.magovoice.com/whisper")
    
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
    enabled: bool = os.getenv("WHISPER_ENABLED", "false").lower() == "true"


# ============================================================
# 취한 척 감지 (Thorsten Emotional + ALC)
# ============================================================

@dataclass
class FakeDrunkConfig:
    """연기/가짜 취함 감지 (fake_drunk_detector_v1.pkl)."""

    model_filename: str = "fake_drunk_detector_v1.pkl"
    # proba[:, 1] >= threshold → 취한 척 (학습: 0=진짜 취함, 1=연기)
    threshold: float = 0.5
    enabled: bool = True

    messages: tuple = (
        "취한 척 하신 거 같은데… 흠 ~ 🎭",
        "에헤이~ 연기 잘하시네요! 취한 척이에요 🎭",
        "술 취한 게 아니라 연기 아닌가요? 흠흠 ~",
        "핑이 눈엔 취한 척이에요! 🎭",
    )


# ============================================================
# 인스턴스
# ============================================================

FEATURE_CONFIG = FeatureConfig()
INFERENCE_CONFIG = InferenceConfig()
WHISPER_CONFIG = WhisperConfig()
FAKE_DRUNK_CONFIG = FakeDrunkConfig()


def get_opensmile():
    """openSMILE 인스턴스 생성."""
    feature_set = getattr(opensmile.FeatureSet, FEATURE_CONFIG.opensmile_feature_set)
    feature_level = getattr(opensmile.FeatureLevel, FEATURE_CONFIG.opensmile_feature_level)
    return opensmile.Smile(feature_set=feature_set, feature_level=feature_level)
