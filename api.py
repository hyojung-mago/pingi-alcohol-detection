"""
api.py - 핑이 AI 분석 서비스 (FastAPI)

=============================================================================
학습된 SVM 모델로 음성 취도를 분석하고 결과를 PostgreSQL에 기록합니다.

흐름:
  프론트(녹음) → 백엔드(Express) → 이 API → DB 기록 → 백엔드가 읽기

엔드포인트:
  POST /analyze           — 녹음 취도 분석 + Recording/Member 업데이트
  POST /analyze-baseline  — 베이스라인 피처 추출 + Baseline 업데이트
  GET  /health            — 헬스 체크

실행:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
=============================================================================
"""

import os
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ml.inference import DrunkDetector, FakeDrunkDetector
from ml.config import FAKE_DRUNK_CONFIG
from ml.config import WHISPER_CONFIG
from ml.whisper_client import WhisperClient, get_whisper_client
from ml.speech_rate import extract_speech_rate_features, SpeechRateFeatures

# ============================================================
# 로깅
# ============================================================

logging.basicConfig(level=logging.INFO, format="[핑이AI] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# 설정
# ============================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/pingi",
)

UPLOAD_BASE = Path(os.getenv(
    "UPLOAD_BASE",
    str(Path(__file__).parent.parent / "pingi-backend" / "uploads"),
))

# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="핑이 AI 분석 서비스",
    version="1.0.0",
    description="ALC 연구 기반 음성 취도 분석 API",
)

detector: Optional[DrunkDetector] = None
fake_detector: Optional[FakeDrunkDetector] = None


@app.on_event("startup")
async def load_model():
    global detector, fake_detector
    logger.info("SVM 모델 로딩 중...")
    detector = DrunkDetector(version="v2")
    detector.load()
    if FAKE_DRUNK_CONFIG.enabled:
        try:
            fake_detector = FakeDrunkDetector()
            fake_detector.load()
            logger.info("취한척 감지 모델 로드 완료 🎭")
        except FileNotFoundError as e:
            fake_detector = None
            logger.warning(f"취한척 모델 스킵: {e}")
    logger.info("모델 로드 완료 ✅")


# ============================================================
# DB 헬퍼
# ============================================================

def get_db():
    return psycopg2.connect(DATABASE_URL)


def update_recording(recording_id: str, score: float, level: int, change_rate: float):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE recordings SET score=%s, level=%s, change_rate=%s WHERE id=%s",
            (score, level, change_rate, recording_id),
        )
        conn.commit()
        logger.info(f"Recording {recording_id}: L{level} ({change_rate:+.1f}%)")
    finally:
        conn.close()


def update_member_level(member_id: str, level: int):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE members SET current_level=%s WHERE id=%s", (level, member_id))
        conn.commit()
    finally:
        conn.close()


def save_baseline_features(member_id: str, features_bytes: bytes, speech_rate_json: Optional[str] = None):
    conn = get_db()
    try:
        cur = conn.cursor()
        if speech_rate_json:
            cur.execute(
                "UPDATE baselines SET features=%s, speech_rate=%s WHERE member_id=%s",
                (psycopg2.Binary(features_bytes), speech_rate_json, member_id),
            )
        else:
            cur.execute(
                "UPDATE baselines SET features=%s WHERE member_id=%s",
                (psycopg2.Binary(features_bytes), member_id),
            )
        conn.commit()
        logger.info(f"베이스라인 피처 저장: {member_id}")
    finally:
        conn.close()


def load_baseline_speech_rate(member_id: str) -> Optional[SpeechRateFeatures]:
    """베이스라인 발화속도 로드."""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT speech_rate FROM baselines WHERE member_id=%s", (member_id,))
        row = cur.fetchone()
        if not row or row[0] is None:
            return None
        data = json.loads(row[0])
        return SpeechRateFeatures(**data)
    except Exception as e:
        logger.warning(f"발화속도 로드 실패: {e}")
        return None
    finally:
        conn.close()


def load_baseline_features(member_id: str) -> Optional[np.ndarray]:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT features FROM baselines WHERE member_id=%s", (member_id,))
        row = cur.fetchone()
        if not row or row[0] is None:
            return None
        return np.frombuffer(bytes(row[0]), dtype=np.float32)
    finally:
        conn.close()


# ============================================================
# 오디오 변환
# ============================================================

def resolve_audio_path(audio_url: str) -> Path:
    """DB 상대 경로 → 절대 경로."""
    clean = audio_url.replace("uploads/", "").lstrip("/")
    return UPLOAD_BASE / clean


def _is_real_wav(path: Path) -> bool:
    """파일 헤더가 실제 RIFF/WAV인지 확인."""
    try:
        with open(path, "rb") as f:
            header = f.read(4)
            return header == b"RIFF"
    except Exception:
        return False


def convert_to_wav(input_path: Path) -> Path:
    """모든 오디오 → 16kHz mono .wav (openSMILE용). 확장자와 무관하게 항상 확인."""
    if input_path.suffix.lower() == ".wav" and _is_real_wav(input_path):
        return input_path

    output_path = Path(tempfile.mktemp(suffix=".wav"))
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(input_path), "-ar", "16000", "-ac", "1", "-f", "wav", str(output_path)],
            capture_output=True, check=True, timeout=30,
        )
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"ffmpeg 변환 실패: {e}")
        raise HTTPException(500, f"오디오 변환 실패: {input_path.name}")


# ============================================================
# 요청/응답 모델
# ============================================================

class AnalyzeRequest(BaseModel):
    recording_id: str
    member_id: str
    audio_url: str
    previous_level: int = 0


class AnalyzeResponse(BaseModel):
    score: float
    level: int
    level_description: str
    change_rate: float
    is_drunk: bool
    confidence: str
    probability: float
    feature_changes: Optional[dict] = None
    is_fake_acting: bool = False
    status: str = "normal"  # normal | fake_acting | drunk
    fake_probability: Optional[float] = None


class BaselineRequest(BaseModel):
    member_id: str
    audio_urls: list[str]


class BaselineResponse(BaseModel):
    success: bool
    member_id: str
    n_features: int
    message: str


# ============================================================
# 엔드포인트
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": detector is not None and detector._loaded,
        "model_version": detector.version if detector else None,
        "fake_detector_loaded": fake_detector is not None and fake_detector._loaded,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """
    녹음 취도 분석.

    1. .webm → .wav 변환
    2. 베이스라인 피처 로드
    3. SVM 예측 + Whisper 발화속도 분석
    4. Recording + Member 테이블 업데이트
    """
    if not detector or not detector._loaded:
        raise HTTPException(503, "모델 미로드")

    audio_path = resolve_audio_path(req.audio_url)
    if not audio_path.exists():
        raise HTTPException(404, f"오디오 파일 없음: {audio_path}")

    wav_path = convert_to_wav(audio_path)
    temp_created = wav_path != audio_path
    whisper_client = get_whisper_client() if WHISPER_CONFIG.enabled else None

    try:
        # 1) 취한 척(연기) 감지 — 걸리면 취도 분석 생략
        if fake_detector and fake_detector._loaded and FAKE_DRUNK_CONFIG.enabled:
            fake_result = fake_detector.predict(str(wav_path))
            if fake_result.is_fake_acting:
                logger.info(
                    f"취한척 감지: proba={fake_result.probability:.2f} — {fake_result.message}"
                )
                update_recording(req.recording_id, 0.0, 0, 0.0)
                update_member_level(req.member_id, 0)
                return AnalyzeResponse(
                    score=0.0,
                    level=0,
                    level_description=fake_result.message,
                    change_rate=0.0,
                    is_drunk=False,
                    confidence="high",
                    probability=round(fake_result.probability, 3),
                    feature_changes=None,
                    is_fake_acting=True,
                    status="fake_acting",
                    fake_probability=round(fake_result.probability, 3),
                )

        baseline_features = load_baseline_features(req.member_id)

        # Whisper 발화속도 분석 (선택적)
        current_speech_rate = None
        baseline_speech_rate = None
        
        if whisper_client and WHISPER_CONFIG.enabled and baseline_features is not None:
            try:
                # 현재 오디오 발화속도
                whisper_result = await whisper_client.transcribe(
                    str(wav_path),
                    lang=WHISPER_CONFIG.language,
                )
                if whisper_result:
                    current_speech_rate = extract_speech_rate_features(whisper_result)
                    logger.info(f"현재 발화속도: {current_speech_rate.syllables_per_sec:.1f} 음절/초")
                
                # 베이스라인 발화속도 로드
                baseline_speech_rate = load_baseline_speech_rate(req.member_id)
                if baseline_speech_rate:
                    logger.info(f"베이스라인 발화속도: {baseline_speech_rate.syllables_per_sec:.1f} 음절/초")
            except Exception as e:
                logger.warning(f"Whisper 분석 실패 (SVM만 사용): {e}")

        if baseline_features is not None:
            result = detector.predict_with_baseline(
                str(wav_path),
                baseline_features,
                current_speech_rate=current_speech_rate,
                baseline_speech_rate=baseline_speech_rate,
            )
            feature_changes = (
                result.baseline_comparison.get("feature_changes")
                if result.baseline_comparison
                else None
            )
        else:
            result = detector.predict(str(wav_path), return_features=True)
            feature_changes = None

        # inference.py가 직접 0~5 숫자 레벨 반환
        level = result.level
        change_rate = round((result.probability - 0.5) * 200, 1)
        score = round(result.probability, 3)

        update_recording(req.recording_id, score, level, change_rate)
        update_member_level(req.member_id, level)

        return AnalyzeResponse(
            score=score,
            level=level,
            level_description=result.level_description,
            change_rate=change_rate,
            is_drunk=result.is_drunk,
            confidence=result.confidence,
            probability=round(result.probability, 3),
            feature_changes=feature_changes,
            is_fake_acting=False,
            status="drunk" if result.is_drunk else "normal",
        )
    finally:
        if temp_created and wav_path.exists():
            wav_path.unlink()


@app.post("/analyze-baseline", response_model=BaselineResponse)
async def analyze_baseline(req: BaselineRequest):
    """
    베이스라인 오디오 3개 → 피처 평균 + 발화속도 평균 → DB 저장.
    """
    if not detector or not detector._loaded:
        raise HTTPException(503, "모델 미로드")

    all_features = []
    all_speech_rates = []
    whisper_client = get_whisper_client() if WHISPER_CONFIG.enabled else None

    for audio_url in req.audio_urls:
        audio_path = resolve_audio_path(audio_url)
        if not audio_path.exists():
            raise HTTPException(404, f"베이스라인 오디오 없음: {audio_path}")

        wav_path = convert_to_wav(audio_path)
        temp_created = wav_path != audio_path

        try:
            # openSMILE 피처 추출
            features = detector.extract_features(str(wav_path))
            all_features.append(features)
            
            # Whisper 발화속도 분석 (선택적)
            if whisper_client and WHISPER_CONFIG.enabled:
                try:
                    whisper_result = await whisper_client.transcribe(
                        str(wav_path),
                        lang=WHISPER_CONFIG.language,
                    )
                    if whisper_result:
                        speech_rate = extract_speech_rate_features(whisper_result)
                        if speech_rate:
                            all_speech_rates.append(speech_rate)
                            logger.info(f"발화속도: {speech_rate.syllables_per_sec:.1f} 음절/초")
                except Exception as e:
                    logger.warning(f"Whisper 분석 실패 (무시): {e}")
        finally:
            if temp_created and wav_path.exists():
                wav_path.unlink()

    avg_features = np.mean(all_features, axis=0)
    
    # 발화속도 평균 계산
    speech_rate_json = None
    if all_speech_rates:
        avg_speech_rate = SpeechRateFeatures(
            syllables_per_sec=np.mean([sr.syllables_per_sec for sr in all_speech_rates]),
            words_per_sec=np.mean([sr.words_per_sec for sr in all_speech_rates]),
            chars_per_sec=np.mean([sr.chars_per_sec for sr in all_speech_rates]),
            pause_ratio=np.mean([sr.pause_ratio for sr in all_speech_rates]),
            articulation_rate=np.mean([sr.articulation_rate for sr in all_speech_rates]),
            total_duration=np.mean([sr.total_duration for sr in all_speech_rates]),
            speech_duration=np.mean([sr.speech_duration for sr in all_speech_rates]),
            syllable_count=int(np.mean([sr.syllable_count for sr in all_speech_rates])),
            word_count=int(np.mean([sr.word_count for sr in all_speech_rates])),
        )
        speech_rate_json = json.dumps(avg_speech_rate.to_dict())
        logger.info(f"평균 발화속도: {avg_speech_rate.syllables_per_sec:.1f} 음절/초")
    
    save_baseline_features(req.member_id, avg_features.tobytes(), speech_rate_json)

    message = f"베이스라인 피처 저장 완료 ({len(avg_features)}차원)"
    if speech_rate_json:
        message += f" + 발화속도"

    return BaselineResponse(
        success=True,
        member_id=req.member_id,
        n_features=len(avg_features),
        message=message,
    )


# ============================================================
# 직접 실행
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
