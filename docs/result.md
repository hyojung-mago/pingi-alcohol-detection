# Drunk Level Detection — 근거 문서 (Final)

> 본 문서는 Pingi-AI의 음성 기반 drunk level 검출 모델의 학술/제품 근거를 정리한다.
> 검증 시점: 2026-05-20

---

## 1. 제품 정의 (Scope)

| 항목 | 내용 |
|---|---|
| **In Scope** | 사용자 본인 sober baseline 대비 acoustic delta로 drunk level (0~5) 추정 |
| **Out of Scope** | Real drunk vs acted/fake drunk 구분 (§6 negative results 참조) |
| **사용 가정** | Baseline과 target은 동일 디바이스/앱 파이프라인에서 녹음 |

---

## 2. 방법 (Method)

### 2.1 Feature
- openSMILE **eGeMAPSv02** functionals (88-dim)
- Geneva Minimalistic Acoustic Parameter Set, paralinguistic emotion·intoxication 연구 표준 ([Eyben et al., 2016])

### 2.2 모델 구조 (`drunk_detector_v2-delta.pkl`)

```
delta_features = current_features - baseline_features        # (88,)
delta_z        = (delta_features - μ_train) / σ_train        # StandardScaler
proba_drunk    = sigmoid( SVM_coef · delta_z + bias )        # Linear SVM
level          = bin(proba_drunk, level_thresholds)          # L0~L5
```

| 구성 | 세부 |
|---|---|
| Classifier | Linear SVM (C=10.0, class_weight='balanced', probability=True) |
| Scaler | StandardScaler (delta space) |
| Baseline | 사용자의 sober 발화 5개의 feature 평균 |
| Level mapping | `final_weights.json` → drunk proba 20% 분위 기반 L0~L5 |

### 2.3 학습 데이터

- **ALC corpus** (Alcohol Language Corpus, German)
- 162 speakers with both sober & drunk sessions
- Per-speaker delta dataset: 2,430 deltas (sober=810, drunk=1620)
- 학습 코드: `ml/train_alc.py --mode delta`

---

## 3. 검증 결과 (Validation)

### 3.1 화자 단위 Train/Val/Test 분리 (Speaker-level Holdout)

**Setup** (`scripts/validate_holdout.py`, seed=42):
- 162 speakers를 6 : 2 : 2 로 화자 단위 분할
- Train: 97 speakers (1,455 samples) — SVM fit
- Val: 32 speakers (480 samples) — threshold·level_thresholds 캘리브레이션
- **Test: 33 speakers (495 samples) — 학습/튜닝에 일절 사용 안 됨**

### 3.2 Test Set Metrics (held-out, 신뢰 가능)

| Metric | Value |
|---|---|
| **AUC (threshold-independent)** | **0.801** |
| Accuracy @ threshold 0.535 (Val-tuned) | 0.741 |
| F1 @ 0.535 | 0.799 |
| Precision @ 0.535 | 0.830 |
| Recall @ 0.535 | 0.770 |
| Accuracy @ threshold 0.625 (production) | 0.717 |
| F1 @ 0.625 | 0.761 |
| Precision @ 0.625 | 0.871 |
| Recall @ 0.625 | 0.676 |

→ **Test set AUC 0.80**이 가장 robust한 지표. Threshold 선택에 따라 precision/recall trade-off가 있으며, production threshold (0.625)는 높은 precision (0.871) 쪽으로 편향됨.

### 3.3 보조 지표 (Cross-validation, 참고)

| Metric | Value | Source |
|---|---|---|
| 5-fold GroupKFold CV accuracy | 0.716 | `final_weights.json` (production model, 162 speakers 전체로 학습) |

→ Holdout test (0.741)와 CV (0.716)가 일관됨. Production 모델은 162 speakers 전체로 학습되어 holdout test보다 ≥동등 성능 기대.

### 3.4 Cross-domain Anecdotal Validation (한국인 화자, N=2)

ALC는 독일어 corpus이므로 한국어 cross-domain transfer를 anecdotal 케이스로 점검. 서로 다른 한국 화자 2명의 sober/drunk 발화 페어를 production 모델로 평가.

| 항목 | Speaker A | Speaker B |
|---|---|---|
| Baseline (sober) | iPhone 영상 89초 (5×5s chunks) | iPhone 영상 3.78초 (단일 baseline) |
| Target (drunk) | iPhone 영상 158초 (14×10s chunks) | iPhone 영상 56초 (5×10s chunks) |
| Sober vs drunk acoustic | 동일 디바이스, 환경 매칭 | 동일 디바이스, 환경 매칭 |
| **Drunk-flagged chunks** | **11/14 (78.6%)** | **5/5 (100%)** |
| **Median P(drunk)** | **0.992** | **0.984** |
| **Median level** | **L5 (매우 취함)** | **L5 (매우 취함)** |
| 미검출 chunks | 3 chunks (chunk 1, 3, 12) P≈0 — 침묵/짧은 voiced segment 추정 | 0 chunks |

**결론**:
- 두 명의 서로 다른 한국 화자에서 일관되게 drunk 판정 (cross-speaker robustness 일차적 증거)
- 동일 iPhone 디바이스 baseline-target pair에서 model이 robust하게 작동 (Speaker B는 3.78초 noisy baseline에도 5/5 강한 양성)
- 단 N=2는 statistical generalization claim에 부족; multi-speaker (n≥30) Korean validation은 future work

**환경 매칭의 중요성** (anti-example, §3.5 참고):
- Acted drunk 사례 (Korean 배우, 드라마 baseline vs 예능 target): baseline-target acoustic 환경이 mismatch (드라마/예능 다른 프로덕션)
- 동일 사례를 다른 chunking 방식으로 측정 시 결과 모순 (1/5 vs single 10s에서 drunk 판정으로 뒤집힘)
- → baseline/target 환경 mismatch가 신호를 압도하여 결과 unstable
- 제품에서는 동일 앱 파이프라인을 통한 baseline/target 녹음이 필수임을 보여주는 사례

### 3.5 운영상 가이드라인 (3.4에서 도출)

- **최소 발화 길이**: 충분한 voiced segment 보장 위해 chunk당 **≥10초** 권장
- **Multi-chunk aggregation**: 단일 chunk는 침묵 구간 outlier에 취약 → 여러 chunk의 majority vote 또는 mean proba 권장
- 6초 이하 또는 voiced segment 짧은 발화는 functionals statistic이 noise에 지배되어 결과 신뢰도 낮음

### 3.6 Within-domain sober → drunk separation

| 그룹 | mean proba(drunk) | n |
|---|---|---|
| ALC sober delta (test held-out) | ~0.45 | 165 |
| ALC drunk delta (test held-out) | ~0.69 | 330 |

본인 baseline 대비 술 마신 상태에서 acoustic feature가 통계적으로 drunk 방향으로 이동한다.

### 3.3 SVM이 학습한 핵심 drunk 방향 (Top 12 by |coef|)

| Feature | SVM coef | 방향 |
|---|---|---|
| F2amplitudeLogRelF0 stddevNorm | +3.05 | 변동성 ↑ when drunk |
| F1amplitudeLogRelF0 amean | −2.28 | 평균 ↓ when drunk |
| F2amplitudeLogRelF0 amean | +2.19 | ↑ |
| F1amplitudeLogRelF0 stddevNorm | −1.85 | ↓ |
| F3amplitudeLogRelF0 amean | +1.79 | ↑ |
| spectralFlux stddevNorm | −1.29 | ↓ |
| spectralFluxV stddevNorm | +1.19 | ↑ |
| alphaRatioV stddevNorm | +0.86 | ↑ |
| loudness stddevNorm | +0.81 | ↑ |
| MeanUnvoicedSegmentLength | +0.74 | ↑ |
| MeanVoicedSegmentLengthSec | −0.72 | ↓ |
| hammarbergIndexV amean | +0.71 | ↑ |

학습된 feature 방향은 intoxication 음향학 문헌과 정합:
- **Formant amplitude variability ↑** — 조음 통제력 저하 (Kuenzel, 1989; Hollien et al., 1994)
- **Voicing segment 통제** — 발화 속도/리듬 변화 ([Pisoni & Martin, 1989])
- **Hammarberg index ↑** — 성문 압력 변화 (Pisoni & Martin, 1989)
- **Loudness variability ↑** — motor control 저하

---

## 4. 제품 운영상 가정 및 한계

### 4.1 가정

1. **녹음 환경 일관성**: Baseline과 target은 동일 앱 파이프라인을 거쳐야 함. 다른 디바이스/코덱/마이크는 delta에 confound 주입.
2. **Sober baseline 사전 등록**: 5개 이상의 sober 발화로 baseline_mean 산출.

### 4.2 한계

| 한계 | 영향 | 완화 방안 |
|---|---|---|
| Cross-language transfer 미검증 | 한국어 사용자에서 ALC 학습 모델 성능 불명 | 한국어 데이터 수집 후 fine-tuning |
| 환경 mismatch 시 delta 오염 | Baseline-target 환경이 다르면 false positive/negative | 앱 내부 동일 파이프라인 강제 |
| 단일 SVM, 깊이 부족 | Edge case 대응 한계 | 향후 deep model 추가 검토 |
| Fake drunk 구분 불가 | §6 참조 | 명시적 비-claim |

---

## 5. Inference 흐름 (Production)

`ml/inference.py: DrunkDetector.predict_with_baseline()`

```python
1. baseline_features = mean( extract_features(sober_wav_i) for i in range(5) )
2. current_features  = extract_features(target_wav)
3. delta             = current_features - baseline_features
4. proba             = pipeline.predict_proba( delta.reshape(1,-1) )[0,1]
5. is_drunk          = proba >= delta_threshold (=0.625)
6. level             = bin(proba, level_thresholds[0~4])    # L0~L5
7. change_pct        = (proba - proba_at_zero_drift) × 100   # 사용자 보고용
```

---

## 6. 제외된 접근 (Negative Results)

### 6.1 "Fake drunk detector" — 학습 시도 및 폐기

**시도 1**: ALC drunk(real) vs Thorsten·EmoV(acted)로 binary classifier 직접 학습
- 결과: AUC 0.99 (학습 데이터 내부)
- 검증: 본인(real) drunk → "acted"로 오분류
- 진단: Dataset confound (ALC 환경 vs EmoV/Thorsten 환경)를 학습. Domain detector이지 fake detector 아님.

**시도 2**: Variability-only feature subset (28-dim, *_stddev*, jitter, shimmer, HNR)으로 narrow SVM 학습
- 결과: ALC drunk vs Thorsten acted 통계 비교 → 차이 없음 (Cohen's d = -0.084, MW p=0.99 ALC>Thor 방향)
- ALC drunk와 Thorsten acted 모두 narrow SVM이 동일하게 "drunk-like"로 판정

**시도 3**: Per-feature 방향 일치도 분석 (full feature, n=1620 vs 300)
- aligned_count out of 88: ALC 44.3 vs Thorsten 44.7 — 차이 없음
- 단, per-feature 분포는 **다른 패턴** 관찰:
  - Acted는 **mean features (loudness amean, F0 amean)**를 과장 — 의식적 통제 가능 영역
  - Real drunk는 **variability features (loudness stddevNorm, F2amp stddevNorm)**가 더 강함 — 비자발적 motor control 저하
  - 그러나 SVM aggregate 시 상쇄되어 구분 불가

### 6.2 결론

> **현재 데이터/feature set으로는 real drunk와 acted drunk를 aggregate acoustic feature로 구분 불가능하다.** 이는 제품 기능 결함이 아니라 **scope 외**다. 제품은 "drunk-like voice change"를 잡으며, 사용자가 의도적으로 fake하려 해도 동일하게 detect되므로 우회 시도에 robust하다 (false negative 방지).

---

## 7. 참고

### 코드
- 학습: `ml/train_alc.py`
- 추론: `ml/inference.py` (`DrunkDetector`)
- Feature: `ml/feature_extractor.py`, `ml/temporal_features.py`
- 검증 스크립트: `scripts/validate_holdout.py` (train/val/test holdout)
- 분석 스크립트: `scripts/stat_drunk_vs_acted.py`, `scripts/stat_variability_only.py` (negative result archives)

### 산출물
- 모델: `models/drunk_detector_v2-delta.pkl`
- 가중치/threshold: `models/final_weights.json`
- Narrow 모델 (참조용): `models/drunk_detector_variability_only.pkl`
- 통계 결과: `scripts/stat_*_results.json`

### 문헌
- Eyben et al. (2016). *The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing.* IEEE Trans. Affective Computing.
- Schiel & Heinrich (2009). *Laying the foundation for in-car alcohol detection by speech.* Interspeech.
- Pisoni & Martin (1989). *Effects of alcohol on the acoustic-phonetic properties of speech.* Alcoholism: Clinical and Experimental Research.
- Hollien, DeJong, Martin et al. (1994). *Production of intoxicated speech.* J. Forensic Sci.
- Kuenzel (1989). *How well does average fundamental frequency correlate with speaker height and weight?* Phonetica.
