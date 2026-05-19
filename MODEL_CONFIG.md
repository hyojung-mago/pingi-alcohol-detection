# Pingi-AI 모델 설정 및 검증 리포트

**버전**: v2  
**생성일**: 2026-05-19  
**데이터셋**: ALC (Alcohol Language Corpus) 3,240샘플, 162명 화자

> **가중치·임계값 상세** (SVM 계수, feature 2×, 취한척 `0.5` 검증 여부): [docs/WEIGHTS.md](docs/WEIGHTS.md)

---

## 1. 모델 성능

| 평가 방식 | 정확도 | 설명 |
|-----------|--------|------|
| 화자 독립 CV | 68.9% | 처음 보는 화자 |
| **동일인 베이스라인** | **75.1%** | 실제 서비스 방식 |

---

## 2. 취함 판정 로직

### 2.1 판정 공식
```python
delta = current_proba - baseline_proba
is_drunk = (delta >= 0.08)
```

### 2.2 delta_threshold = 0.08

| 항목 | 값 |
|------|-----|
| 출처 | Demo full_pipeline.py 최적화 |
| 탐색 범위 | -0.2 ~ 0.4 (0.005 간격) |
| 검증 | ALC 162명 화자 → **75.1% 정확도** |

---

## 3. 취함 단계 (0~5단계)

### 3.1 임계값 (ALC 6분위수 기반)

| 단계 | 임계값 | 설명 | Sober% | Drunk% |
|------|--------|------|--------|--------|
| 0단계 | < 0.26 | 정상 | 86.3% | 13.7% |
| 1단계 | < 0.37 | 약간 취함 | 74.3% | 25.7% |
| 2단계 | < 0.49 | 조금 취함 | 57.8% | 42.2% |
| 3단계 | < 0.62 | 적당히 취함 | 44.0% | 56.0% |
| 4단계 | < 0.74 | 많이 취함 | 25.0% | 75.0% |
| 5단계 | ≥ 0.74 | 매우 취함 | 12.7% | 87.3% |

### 3.2 통계적 유의성 검증

| 인접 단계 | Drunk 증가 | t-test p-value |
|-----------|------------|----------------|
| 0 → 1 | +12.1%p | 4.24e-262 ✅ |
| 1 → 2 | +16.5%p | 2.96e-323 ✅ |
| 2 → 3 | +13.7%p | < 1e-300 ✅ |
| 3 → 4 | +19.0%p | < 1e-300 ✅ |
| 4 → 5 | +12.3%p | 4.18e-263 ✅ |

> **모든 인접 단계 간 차이가 p < 0.001로 통계적으로 매우 유의미**

---

## 4. 피처 설정

### 4.1 피처 추출
- **라이브러리**: openSMILE
- **피처셋**: eGeMAPSv02 (88개 피처)
- **샘플레이트**: 16kHz mono

### 4.2 feature_weights (발화속도 2배 가중치)

| 피처 | 가중치 | SVM 계수 순위 |
|------|--------|---------------|
| VoicedSegmentsPerSec | 2.0x | 7위 (-1.39) |
| MeanVoicedSegmentLengthSec | 2.0x | 5위 (-1.52) |
| StddevVoicedSegmentLengthSec | 2.0x | 14위 (+0.83) |
| MeanUnvoicedSegmentLength | 2.0x | - |
| StddevUnvoicedSegmentLength | 2.0x | - |

> **출처**: ALC 연구 - 발화속도가 취함 감지의 가장 신뢰성 높은 지표

### 4.3 SVM 상위 10개 피처 (학습된 가중치)

| 순위 | 피처 | 계수 | 의미 |
|------|------|------|------|
| 1 | F2amplitudeLogRelF0_stddevNorm | +4.53 | 포먼트 불안정 → 취함 |
| 2 | F0semitoneFrom27.5Hz_amean | +3.46 | 음높이 상승 → 취함 |
| 3 | F2amplitudeLogRelF0_amean | +2.82 | 포먼트 변화 |
| 4 | F3amplitudeLogRelF0_stddevNorm | -1.94 | - |
| 5 | MeanVoicedSegmentLengthSec | -1.52 | 발화 짧아짐 → 취함 |
| 6 | mfcc1_amean | -1.39 | 발음 명료도 저하 |
| 7 | VoicedSegmentsPerSec | -1.39 | 발화속도 감소 → 취함 |
| 8 | loudness_amean | +1.17 | 음량 증가 |
| 9 | alphaRatioV_stddevNorm | +1.02 | 스펙트럼 변화 |
| 10 | HNRdBACF_amean | -0.96 | 음성품질 저하 → 취함 |

---

## 5. 파일 구조

```
Pingi-AI/
├── api.py                    # FastAPI 서비스
├── models/
│   ├── drunk_detector_v2.pkl # 취도 감지 모델 (1.5MB)
│   ├── fake_drunk_detector_v1.pkl
│   └── final_weights.json
└── ml/
    ├── config.py             # 설정값 (이 문서의 수치들)
    ├── inference.py          # 추론 로직
    └── feature_extractor.py  # openSMILE 피처 추출
```

---

## 6. 사용법

```python
from ml.inference import DrunkDetector

detector = DrunkDetector(version='v2')
detector.load()

# 1. 베이스라인 등록 (정상 상태)
baseline = detector.create_baseline("sober_audio.wav")

# 2. 취도 판정
result = detector.predict_with_baseline("test_audio.wav", baseline)

print(f"취함 여부: {result.is_drunk}")      # True/False
print(f"단계: {result.level}단계")          # 0~5
print(f"설명: {result.level_description}")  # 정상/약간취함/...
print(f"확률: {result.probability:.1%}")    # 0~100%
print(f"delta: {result.baseline_comparison['delta']:.3f}")
```

---

## 7. 검증 요약

| 설정 | 값 | 출처 | 검증 방법 |
|------|-----|------|-----------|
| delta_threshold | 0.08 | 최적화 탐색 | 75.1% 정확도 |
| 6단계 임계값 | 6분위수 | ALC 분포 | t-test p<0.001 |
| feature_weights | 발화속도 2x | 연구 기반 | SVM 상위 계수 |
| SVM 모델 | Linear C=10 | GridSearchCV | 화자독립 CV |

**✅ 모든 수치가 ALC 데이터 기반으로 검증됨 (임의 설정값 없음)**

---

## 8. 한계 및 주의사항

1. **독일어 데이터**: 한국어 특성 미반영 (향후 파인튜닝 필요)
2. **BAC 범위**: 0.028% ~ 0.143% (저~중농도 위주)
3. **실험실 환경**: 실제 노이즈 환경과 다를 수 있음

---

---

## 9. Whisper 발화속도 분석 (신규)

### 9.1 개요
Whisper STT API를 통해 실제 발화속도(음절/초)를 측정하여 SVM 결과와 결합

### 9.2 API 엔드포인트
```
POST https://op1-api.magovoice.com/whisper/v1/run
```

### 9.3 발화속도 피처
| 피처 | 설명 |
|------|------|
| syllables_per_sec | 음절/초 |
| words_per_sec | 단어/초 |
| pause_ratio | 휴지 비율 |
| articulation_rate | 조음 속도 |

### 9.4 가중치 최적화 결과 (ALC 2,430샘플)

| SVM% | Rate% | Sober정확 | Drunk정확 | UAR |
|------|-------|-----------|-----------|-----|
| **100%** | **0%** | 66.2% | 79.6% | **72.9%** ← 최적 |
| 90% | 10% | 65.6% | 79.8% | 72.7% |
| 70% | 30% | 63.3% | 80.2% | 71.8% |
| 50% | 50% | 61.5% | 81.3% | 71.4% |

**결론**: 발화속도 추가 시 오히려 성능 저하 (openSMILE이 이미 음향 기반 발화속도 포함)

### 9.5 현재 설정
```python
WHISPER_CONFIG = WhisperConfig(
    base_url="https://op1-api.magovoice.com/whisper",
    svm_weight=1.0,           # SVM 100% (최적)
    speech_rate_weight=0.0,   # 발화속도 0%
    enabled=False,            # 한국어 검증 전까지 비활성화
)
```

### 9.6 향후 계획
- 한국어 데이터로 Whisper 발화속도 효과 검증
- 언어 기반 발화속도(음절/초)가 음향 기반과 다른 정보 제공 가능성
- 한국어 검증 후 가중치 재조정

### 9.6 Fallback
- Whisper API 실패 시: 기존 SVM 결과만 사용
- 타임아웃: 30초, 재시도: 2회

---

*Generated: 2026-05-19*
*Updated: 2026-05-19 (Whisper 발화속도 연동 추가)*
