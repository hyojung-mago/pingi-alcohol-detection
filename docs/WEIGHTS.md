# Pingi 가중치·임계값 상세 정리

**버전**: v2 (`drunk_detector_v2.pkl`, `fake_drunk_detector_v1.pkl`)  
**설정 파일**: `ml/config.py`, `models/final_weights.json`  
**학습 출처**: Pingi-demo `scripts/ml/full_pipeline.py` (ALC + Thorsten Emotional)

> 취도(delta)는 **ALC에서 임계값 탐색으로 검증**됨.  
> 취한 척(0.5)은 **학습 시 hold-out 정확도는 있으나, 0.5 자체는 ROC 최적화가 아님** → §6 참고.

---

## 목차

1. [전체 구조](#1-전체-구조)
2. [학습 전 피처 가중치 (feature_weights)](#2-학습-전-피처-가중치-feature_weights)
3. [SVM 취도 모델 학습 가중치 (상위 20)](#3-svm-취도-모델-학습-가중치-상위-20)
4. [서비스 판정 임계값 (취도)](#4-서비스-판정-임계값-취도)
5. [Whisper 결합 가중치](#5-whisper-결합-가중치)
6. [취한 척 감지 (fake_drunk)](#6-취한-척-감지-fake_drunk)
7. [베이스라인 변화량 스코어 규칙 (참고)](#7-베이스라인-변화량-스코어-규칙-참고)
8. [한국어 ad-hoc 검증 (참고)](#8-한국어-ad-hoc-검증-참고)
9. [설정 변경 방법](#9-설정-변경-방법)

---

## 1. 전체 구조

```
WAV (16kHz mono)
    → openSMILE eGeMAPSv02 (88 features)
    → × feature_weights (발화·세그먼트 5종 2배)
    → StandardScaler → SVM

취도:  predict_with_baseline → delta = current_proba - baseline_proba
       is_drunk ⇔ delta ≥ 0.08

취한척: FakeDrunkDetector → proba(class=연기) ≥ 0.5
       (취도 판정과 독립, 먼저 검사)
```

| 구분 | 모델 | 커널 | 핵심 임계값 | 검증 수준 |
|------|------|------|-------------|-----------|
| 취도 | `drunk_detector_v2` | Linear C=10 | delta ≥ **0.08** | ALC 동일인 75.1% + 한국어 2명 |
| 취한 척 | `fake_drunk_detector_v1` | RBF C=1 | proba ≥ **0.5** | ALC/Thorsten hold-out 98.9% (§6) |

---

## 2. 학습 전 피처 가중치 (feature_weights)

추론 시: `weighted = raw_features * feature_weights` 후 SVM 입력.

**적용 피처 (2.0×)** — ALC 연구: 발화속도·세그먼트가 취함 감지에 가장 신뢰도 높음.

| # | 피처명 (eGeMAPSv02) | 가중치 |
|---|---------------------|--------|
| 1 | `VoicedSegmentsPerSec` | **2.0** |
| 2 | `MeanVoicedSegmentLengthSec` | **2.0** |
| 3 | `StddevVoicedSegmentLengthSec` | **2.0** |
| 4 | `MeanUnvoicedSegmentLength` | **2.0** |
| 5 | `StddevUnvoicedSegmentLength` | **2.0** |

**나머지 83개 피처**: **1.0** (가중 없음)

**코드**: `ml/inference.py` → `DrunkDetector._apply_weights()`  
**저장**: `drunk_detector_v2.pkl` dict 키 `feature_weights`

---

## 3. SVM 취도 모델 학습 가중치 (상위 20)

**계산**: `importance = SVM.coef × feature_weights` (Linear SVM, class=취함 방향)

양수(+) → 값이 클수록 **취함** 쪽, 음수(-) → **sober** 쪽.

| 순위 | 계수 | 방향 | 2× | 피처 (요약) |
|------|------|------|-----|-------------|
| 1 | +4.531 | 취함↑ | | F2 진폭 변동 (stddevNorm) |
| 2 | +3.456 | 취함↑ | | F0 평균 (semitone) |
| 3 | +2.816 | 취함↑ | | F2 진폭 평균 |
| 4 | −1.936 | 취함↓ | | F3 진폭 변동 |
| 5 | −1.520 | 취함↓ | ✓ | 평균 유성음 구간 길이 |
| 6 | −1.393 | 취함↓ | | MFCC1 평균 |
| 7 | −1.392 | 취함↓ | ✓ | **초당 유성음 구간 수 (발화속도)** |
| 8 | +1.166 | 취함↑ | | loudness 평균 |
| 9 | +1.018 | 취함↑ | | alphaRatio (voiced) 변동 |
| 10 | −0.963 | 취함↓ | | HNR (음성 품질) |
| 11 | −0.946 | 취함↓ | | F0 80 백분위 |
| 12 | +0.907 | 취함↑ | | spectralFlux (voiced) 변동 |
| 13 | −0.898 | 취함↓ | | spectralFlux 변동 |
| 14 | +0.833 | 취함↑ | ✓ | 유성음 구간 길이 표준편차 |
| 15 | −0.822 | 취함↓ | | F0 20 백분위 |
| 16 | +0.755 | 취함↑ | | F1 진폭 평균 |
| 17 | +0.727 | 취함↑ | | loudness 변동 |
| 18 | +0.722 | 취함↑ | | hammarbergIndex (voiced) |
| 19 | +0.672 | 취함↑ | | alphaRatio (voiced) 평균 |
| 20 | −0.665 | 취함↓ | | loudness 50 백분위 |

**해석 요약**

- **취함 신호**: F0·포먼트 불안정↑, loudness↑, **말 빠르기↓** (VoicedSegmentsPerSec↓), HNR↓  
- **2× 가중** 피처가 상위권에 다수 포함 → 설계 의도와 일치

전체 88개 목록·재현: `models/final_weights.json` → `drunk_detection.feature_importance_top20`

---

## 4. 서비스 판정 임계값 (취도)

### 4.1 delta (메인 취함 여부)

| 항목 | 값 | 검증 |
|------|-----|------|
| **delta_threshold** | **0.08** | ALC 동일인 베이스라인 그리드 탐색 |
| 판정 | `current_proba - baseline_proba ≥ 0.08` | UAR/정확도 **75.1%** (`final_weights.json`) |
| F1 (동일인) | **0.810** | |

**절대 확률 0.5**는 서비스 취함 판정에 **사용하지 않음** (레벨 표시용 `current_proba`만 사용).

### 4.2 레벨 0~5 (절대 확률 구간)

ALC 전체 분포 **6분위** 기반 (`ml/config.py` → `level_thresholds`).

| 레벨 | 확률 상한 | ALC 참고 (Sober / Drunk) |
|------|-----------|---------------------------|
| 0 정상 | < 0.26 | 86% / 14% |
| 1 | < 0.37 | 74% / 26% |
| 2 | < 0.49 | 58% / 42% |
| 3 | < 0.62 | 44% / 56% |
| 4 | < 0.74 | 25% / 75% |
| 5 매우 취함 | ≤ 1.0 | 13% / 87% |

개인 **delta**로 취함 여부를 정하고, **레벨**은 같은 녹음의 `current_proba`로 매핑.

---

## 5. Whisper 결합 가중치

ALC 2,430샘플, SVM 확률 + Whisper 음절/초 점수 결합 스윕.

| SVM 가중치 | 발화속도 가중치 | UAR | 비고 |
|------------|-----------------|-----|------|
| **1.0** | **0.0** | **72.9%** | **현재 채택** |
| 0.9 | 0.1 | 72.7% | |
| 0.7 | 0.3 | 71.8% | |
| 0.5 | 0.5 | 71.4% | |

**현재 설정** (`ml/config.py`):

```python
WHISPER_CONFIG.svm_weight = 1.0
WHISPER_CONFIG.speech_rate_weight = 0.0
WHISPER_CONFIG.enabled = False
WHISPER_CONFIG.max_decrease_ratio = 0.3  # 발화속도 보조 켤 때만
```

**이유**: openSMILE에 이미 음향 기반 발화속도(`VoicedSegmentsPerSec` 등) 포함 → STT 중복.  
한국어에서 Whisper 보조 채널은 **미검증** (`test_whisper_korean_ablation.py`).

**결합 공식** (활성화 시):

```
combined = svm_weight × svm_proba + speech_rate_weight × speech_rate_score
(speech_rate_weight = 1 - svm_weight)
```

---

## 6. 취한 척 감지 (fake_drunk)

### 6.1 역할

- **진짜 취함**(ALC drunk) vs **성우 연기 취함**(Thorsten Emotional `drunk` 300 WAV)
- API `/analyze`에서 **취도 분석 전** 선행 검사
- 연기로 판정 시: `level=0`, `status=fake_acting`, 고정 멘트

### 6.2 모델

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced'))
])
```

| 라벨 | 의미 | 데이터 |
|------|------|--------|
| 0 | 진짜 취함 | ALC drunk **200** 샘플 |
| 1 | 취한 척 연기 | Thorsten drunk **100** 샘플 |

**feature_weights 없음** (88차원 raw → scaler → RBF).

### 6.3 임계값 `proba ≥ 0.5` — 검증 여부 (중요)

| 질문 | 답 |
|------|-----|
| **0.5를 ROC/PR로 따로 찾았나?** | **아니오.** sklearn `predict_proba[:,1] ≥ 0.5` 기본 결정 경계. |
| **모델 성능 수치는?** | hold-out **30%** (random split, seed=42): **정확도 98.9%**, **F1 98.1%** |
| **데이터 규모** | 학습 300 (real 200 + fake 100), 테스트 약 90 |
| **한국어 검증?** | **공식 벤치마크 없음** (§8 ad-hoc만) |
| **한국어 진짜 취함 오탐?** | A·B 취함 영상 **p≈0.15~0.17** → 연기로 안 잡힘 (양호) |

**정리**

- **98.9%**는 **독일어 ALC 진짜 취함 vs 독일어 성우 연기** 구분 성능이지,  
  **“한국어에서 0.5가 최적”**이라는 뜻은 **아님**.
- **0.5**는 관례적 기본값이며, 취도 `delta=0.08`처럼 **별도 임계값 튜닝 리포트는 없음**.
- Precision/Recall 98%/99% (`FINAL_REPORT.md`)도 **동일 hold-out, threshold=0.5 가정** 하의 수치.

### 6.4 운영 설정

```python
# ml/config.py
FAKE_DRUNK_CONFIG.threshold = 0.5
FAKE_DRUNK_CONFIG.enabled = True
```

판정: `predict_proba(X)[0, 1] >= 0.5` → `is_fake_acting=True`  
(`ml/inference.py` → `FakeDrunkDetector.predict`)

### 6.5 향후 검증 권장

1. Thorsten + 한국어 연기 샘플(있으면)으로 **ROC 곡선** → 최적 threshold  
2. 한국어 **진짜 취함** n≥20에서 **FPR** (연기로 오판 비율) 측정  
3. threshold **0.5 → 0.6~0.7** 올리면 연기 recall↓, 진짜 취함 오탐↓ trade-off 확인

---

## 7. 베이스라인 변화량 스코어 규칙 (참고)

`DrunkDetector._score_changes()` — **현재 API 취함 판정에는 미사용** (delta만 사용).  
디버깅·향후 UI용 규칙 가중치.

| 변화 지표 | 조건 (sober 대비 %) | 가산 점수 |
|-----------|---------------------|-----------|
| F0_std | >20 / >10 / >5 | +0.30 / +0.15 / +0.08 |
| F0_mean | >15 / >8 / >3 | +0.15 / +0.10 / +0.05 |
| loudness | >20 / >10 | +0.15 / +0.08 |
| **voiced_per_sec** | **<-25 / <-15 / <-8 / <-3** | **+0.40 / +0.25 / +0.15 / +0.08** |
| voiced_length | >20 / >10 | +0.10 / +0.05 |

최종 `score`는 0~1 클램프. Demo `features.py` 설계와 동일 계열.

---

## 8. 한국어 ad-hoc 검증 (참고)

`docs/validation_person_a.md` — 라벨 확인된 A사람만 문서화.

| 대상 | 취도 delta | 취한척 proba (class=연기) |
|------|------------|---------------------------|
| A sober | — | 0.112 |
| A drunk | **+0.447** ✓ | 0.166 |
| B sober (8702) | — | (미기록) |
| B drunk (2425) | **+0.582** ✓ | ~0.15대 |

→ 취도 **delta 방식** 한국어 2명 성공.  
→ 취한척 **진짜 취함을 연기로 오판하지 않음** (0.5 미만).

---

## 9. 설정 변경 방법

| 바꾸고 싶은 것 | 파일 | 키 |
|----------------|------|-----|
| delta 임계값 | `ml/config.py` | `INFERENCE_CONFIG.delta_threshold` |
| 레벨 구간 | `ml/config.py` | `INFERENCE_CONFIG.level_thresholds` |
| 취한척 임계값 | `ml/config.py` | `FAKE_DRUNK_CONFIG.threshold` |
| Whisper 비율 | `ml/config.py` | `WHISPER_CONFIG.svm_weight` 등 |
| 학습 피처 2× | Pingi-demo 재학습 → pkl 교체 | `feature_weights` |

**주의**: `feature_weights`·SVM 계수는 **재학습 없이** config만으로 변경 불가 (pkl 내부).

---

## 관련 문서

- [MODEL_CONFIG.md](../MODEL_CONFIG.md) — 요약·사용법
- [validation_person_a.md](./validation_person_a.md) — 한국어 A사람 검증
- [../models/final_weights.json](../models/final_weights.json) — 수치 스냅샷
- Pingi-demo [models/FINAL_REPORT.md](../../Pingi-demo/models/FINAL_REPORT.md) — 학습 리포트
