# A사람 한국어 검증 (sober ↔ 취함)

**검증일**: 2026-05-19  
**상태**: 화자 본인 확인 — sober / 취함 라벨 신뢰 가능  
**모델**: `drunk_detector_v2.pkl` (ALC 학습), delta 임계값 `0.08`

---

## 1. 사용 파일

| 구분 | 원본 | 추출 WAV |
|------|------|----------|
| **정상 (sober)** | `IMG_7261.mov` (~90초) | `data/ad_hoc/person_a_sober.wav` |
| **취함 (drunk)** | `_talkv_..._talkv_high.MP4` (~51초) | `data/ad_hoc/talkv_drunk.wav` |

- 오디오: ffmpeg → mono 16 kHz PCM  
- 분석 스크립트: `analyze_one_video.py`, `predict_with_baseline()` (`ml/inference.py`)

---

## 2. 절대 점수 (베이스라인 없음)

| 녹음 | SVM 확률 | 레벨 | 절대 0.5 기준 | 신뢰도 |
|------|----------|------|---------------|--------|
| sober | **0.528** | L3 | 취함 쪽 (경계) | low |
| drunk | **0.975** | L5 | 취함 | high |

**sober 구간별 (15초)** — 전체 90초, 6구간

| 구간 | 확률 | 레벨 |
|------|------|------|
| min ~ max | 0.231 ~ 0.771 | L0 ~ L5 |
| 평균 | 0.485 | — |

**drunk 구간별 (15초)** — 4구간

| 구간 | 확률 | 레벨 |
|------|------|------|
| min ~ max | 0.876 ~ 0.991 | L5 |
| 평균 | 0.952 | — |

**해석**: sober도 절대값만 보면 0.5 근처·L3로 “살짝 취함”처럼 나올 수 있음 → **절대 점수만으로는 부족**, 개인 베이스라인 필요.

---

## 3. 앱 방식 (Pingi 핵심: sober → baseline, drunk → test)

| 항목 | 값 |
|------|-----|
| `baseline_proba` (sober) | 0.528 |
| `current_proba` (drunk) | 0.975 |
| **delta** | **+0.447** |
| 임계값 | 0.08 |
| **취함 판정 (`delta ≥ 0.08`)** | **예** |
| 표시 레벨 (current 확률 기준) | L5 (매우 취함) |
| 신뢰도 | high |

**역방향 검증** (drunk를 baseline, sober를 test): delta **-0.447** → 취함 **아니오** (기대와 일치).

---

## 4. 음향 변화 (sober → drunk, %)

| 피처 | 변화 |
|------|------|
| F0 변동 (std) | **+123%** |
| loudness | **+91%** |
| HNR | **-50%** |
| jitter | **+47%** |
| voiced segment length | -34% |
| F0 mean | -28% |

취함 시 흔히 기대되는 방향(변동↑, 선명도↓, 떨림↑)과 대체로 일치.

---

## 5. 취한 척 (fake drunk)

| 녹음 | 연기 확률 | 판정 |
|------|-----------|------|
| sober | 0.112 | 정상 |
| drunk | 0.166 | 정상 (연기 아님) |

---

## 6. Whisper (참고, drunk 전체만)

- 전사: 정상 동작 (한국어)
- 음절/초 ≈ **5.0** (서비스 기본: Whisper OFF, SVM+delta만 사용)

---

## 7. 결론 (A사람만)

1. **같은 사람 sober ↔ drunk**에서 delta **+0.447**로 취함 구분 **성공**.
2. **절대 SVM만** 쓰면 sober **0.53**으로 오탐 위험 → **level 0 베이스라인 + delta** 설계가 타당함.
3. 한 건의 한국어 실측이지만, **라벨이 확실한 A사람** 기준으로 Pingi 파이프라인 **방향성 검증 통과**.

---

## 8. 재현 명령

```bash
cd Pingi-AI

# 절대 + 구간별
python analyze_one_video.py data/ad_hoc/person_a_sober.wav
python analyze_one_video.py data/ad_hoc/talkv_drunk.wav

# delta (Python)
python3 -c "
from pathlib import Path
from ml.inference import DrunkDetector
d = DrunkDetector(version='v2'); d.load()
b = d.create_baseline('data/ad_hoc/person_a_sober.wav')
r = d.predict_with_baseline('data/ad_hoc/talkv_drunk.wav', b)
bc = r.baseline_comparison
print('delta', bc['delta'], 'is_drunk', r.is_drunk, 'level', r.level)
"
```

---

*B사람(IMG_8859 sober / IMG_2425 drunk)은 baseline_proba가 sober부터 ~0.94로 높아 delta 검증 실패 — 본 문서 범위 외.*
