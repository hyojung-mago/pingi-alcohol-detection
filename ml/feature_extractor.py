"""
feature_extractor.py - openSMILE 피처 추출
==========================================

eGeMAPSv02 (88개 피처) 추출기.
학습/추론 모두 동일한 추출기를 사용합니다.
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Optional

from .config import get_opensmile

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """음성 피처 추출기."""

    def __init__(self):
        self._smile = None
        self._feature_names: List[str] = None

    @property
    def smile(self):
        if self._smile is None:
            logger.info("openSMILE 초기화 중...")
            self._smile = get_opensmile()
            self._feature_names = self._smile.feature_names
            logger.info(f"  피처 수: {len(self._feature_names)}")
        return self._smile

    @property
    def feature_names(self) -> List[str]:
        if self._feature_names is None:
            _ = self.smile
        return self._feature_names

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def extract(self, audio_path: Path) -> Optional[np.ndarray]:
        """WAV 파일 → (88,) numpy array. 실패 시 None."""
        try:
            df = self.smile.process_file(str(audio_path))
            features = df.values.flatten()
            if np.any(np.isnan(features)):
                logger.warning(f"NaN 피처 발견: {audio_path.name}")
                features = np.nan_to_num(features, nan=0.0)
            return features
        except Exception as e:
            logger.error(f"피처 추출 실패: {audio_path} - {e}")
            return None


_extractor: Optional[FeatureExtractor] = None


def get_extractor() -> FeatureExtractor:
    """싱글톤."""
    global _extractor
    if _extractor is None:
        _extractor = FeatureExtractor()
    return _extractor
