"""핑이 AI — ML 추론 모듈."""

from .config import MODEL_PATH, FEATURE_CONFIG, INFERENCE_CONFIG
from .feature_extractor import FeatureExtractor, get_extractor
from .inference import DrunkDetector, get_detector, PredictionResult
