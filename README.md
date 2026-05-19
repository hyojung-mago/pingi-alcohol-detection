# 핑이 (Pingi) — AI 서버

> openSMILE + SVM 기반 음성 취도 분석 API (FastAPI)

## 아키텍처

```
백엔드 (8000) → AI 서버 (8001) → openSMILE 피처 추출 → SVM 예측
                    ↕
                PostgreSQL (베이스라인 피처 저장)
```

- **openSMILE**: 음성에서 eGeMAPS 피처 (88차원) 추출
- **SVM**: 베이스라인 대비 취도 변화 예측
- **ffmpeg**: WebM/기타 → WAV 변환 (브라우저 녹음 호환)

## Quick Start

```bash
# Python 3.9+
pip install -r requirements.txt

# ffmpeg 설치 (macOS)
brew install ffmpeg

# 환경 변수
cp .env.example .env
# DATABASE_URL, UPLOAD_BASE 설정

# 서버 실행 (포트 8001)
DATABASE_URL="postgresql://user@localhost:5432/pingi" \
  python3 -m uvicorn api:app --host 0.0.0.0 --port 8001 --reload
```

## 환경 변수 (.env)

| 변수 | 설명 |
|------|------|
| `DATABASE_URL` | PostgreSQL 연결 문자열 (백엔드와 동일 DB) |
| `UPLOAD_BASE` | 백엔드 업로드 폴더 경로 (`../pingi-backend/uploads`) |

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 헬스체크 (모델 로드 상태) |
| POST | `/analyze` | 핑이타임 녹음 취도 분석 |
| POST | `/analyze-baseline` | 베이스라인 녹음 피처 저장 |

### POST /analyze

```json
{
  "recording_id": "rec_xxx",
  "member_id": "m_xxx",
  "audio_url": "uploads/rec_xxx.wav",
  "previous_level": 0
}
```

응답: `{ score, level, level_description, change_rate, is_drunk, confidence, probability, is_fake_acting, status }`

### POST /analyze-baseline

```json
{
  "member_id": "m_xxx",
  "audio_urls": ["uploads/baseline_1.wav", "uploads/baseline_2.wav", "uploads/baseline_3.wav"]
}
```

응답: `{ success, member_id, n_features, message }`

## 프로젝트 구조

```
Pingi-AI/
├── api.py                  # FastAPI 서버 (엔드포인트, 오디오 변환)
├── ml/
│   ├── config.py           # 모델/피처 설정
│   ├── feature_extractor.py # openSMILE 피처 추출
│   └── inference.py        # DrunkDetector, FakeDrunkDetector
├── models/                 # 학습된 모델 파일 (.pkl) ← gitignore
├── data/                   # 학습 데이터 ← gitignore
├── docs/                   # 모델 문서
├── requirements.txt
└── .env.example
```

## 오디오 처리

브라우저 MediaRecorder가 WebM으로 녹음하지만 `.wav` 확장자로 저장되는 경우가 있어, 파일 헤더(RIFF 매직 바이트)를 확인하여 실제 WAV가 아니면 ffmpeg로 16kHz mono WAV로 변환합니다.

## 연관 서비스

| 서비스 | 경로 | 포트 |
|--------|------|------|
| 백엔드 | `../pingi-backend` | 8000 |
| 프론트엔드 | `../Pingi-Front` | 5173 |
