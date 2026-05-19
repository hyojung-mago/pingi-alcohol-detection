"""
whisper_client.py - Whisper STT API 클라이언트
================================================

Mago Whisper API를 호출하여 음성을 텍스트로 변환합니다.

API: https://op1-api.magovoice.com/whisper
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import httpx

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """STT 세그먼트."""
    start: float
    end: float
    text: str


@dataclass
class WhisperResult:
    """Whisper STT 결과."""
    text: str
    segments: List[Segment]
    language: str
    duration: float
    
    @property
    def word_count(self) -> int:
        """단어 수."""
        return len(self.text.split())
    
    @property
    def char_count(self) -> int:
        """글자 수 (공백 제외)."""
        return len(self.text.replace(" ", ""))


class WhisperClient:
    """Mago Whisper API 클라이언트."""
    
    DEFAULT_BASE_URL = "https://op1-api.magovoice.com/whisper"
    
    def __init__(
        self,
        base_url: str = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.timeout = timeout
        self.max_retries = max_retries
    
    async def transcribe(
        self,
        audio_path: Union[str, Path],
        lang: str = "ko",
    ) -> Optional[WhisperResult]:
        """
        오디오 파일을 텍스트로 변환.
        
        Args:
            audio_path: 오디오 파일 경로
            lang: 언어 코드 (기본 "ko")
            
        Returns:
            WhisperResult 또는 실패 시 None
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.error(f"오디오 파일 없음: {audio_path}")
            return None
        
        url = f"{self.base_url}/v1/run"
        
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    with open(audio_path, "rb") as f:
                        files = {"file": (audio_path.name, f, "audio/wav")}
                        params = {"lang": lang, "task": "transcribe"}
                        
                        response = await client.post(
                            url,
                            files=files,
                            params=params,
                        )
                
                if response.status_code == 200:
                    return self._parse_response(response.json())
                else:
                    logger.warning(
                        f"Whisper API 오류 (시도 {attempt+1}): "
                        f"{response.status_code} - {response.text}"
                    )
                    
            except httpx.TimeoutException:
                logger.warning(f"Whisper API 타임아웃 (시도 {attempt+1})")
            except Exception as e:
                logger.error(f"Whisper API 예외 (시도 {attempt+1}): {e}")
        
        logger.error(f"Whisper API 호출 실패: {audio_path}")
        return None
    
    def transcribe_sync(
        self,
        audio_path: Union[str, Path],
        lang: str = "ko",
    ) -> Optional[WhisperResult]:
        """동기 버전의 transcribe."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.error(f"오디오 파일 없음: {audio_path}")
            return None
        
        url = f"{self.base_url}/v1/run"
        
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    with open(audio_path, "rb") as f:
                        files = {"file": (audio_path.name, f, "audio/wav")}
                        params = {"lang": lang, "task": "transcribe"}
                        
                        response = client.post(
                            url,
                            files=files,
                            params=params,
                        )
                
                if response.status_code == 200:
                    return self._parse_response(response.json())
                else:
                    logger.warning(
                        f"Whisper API 오류 (시도 {attempt+1}): "
                        f"{response.status_code} - {response.text}"
                    )
                    
            except httpx.TimeoutException:
                logger.warning(f"Whisper API 타임아웃 (시도 {attempt+1})")
            except Exception as e:
                logger.error(f"Whisper API 예외 (시도 {attempt+1}): {e}")
        
        logger.error(f"Whisper API 호출 실패: {audio_path}")
        return None
    
    def _parse_response(self, data: dict) -> WhisperResult:
        """API 응답 파싱.
        
        Mago Whisper API 응답 형식:
        {
          "code": 700,
          "message": "Success",
          "content": {
            "result": {
              "audio_info": {"duration": ...},
              "ko": {"text": "...", "language": "ko"},
              "decoding_time": 0.8
            }
          }
        }
        """
        # content.result 추출
        content = data.get("content", {})
        result = content.get("result", {})
        audio_info = result.get("audio_info", {})
        
        # 언어별 결과 찾기 (ko, de, en 등)
        text = ""
        language = "ko"
        for key, value in result.items():
            if isinstance(value, dict) and "text" in value:
                text = value.get("text", "")
                language = value.get("language", key)
                break
        
        # duration: audio_info에서 또는 텍스트 기반 추정
        duration = audio_info.get("duration", 0)
        if duration == 0 and text:
            # 대략적인 추정: 평균 5음절/초
            from .speech_rate import count_korean_syllables
            syllables = count_korean_syllables(text) if language == "ko" else len(text.split())
            duration = max(syllables / 5.0, 1.0)
        
        # decoding_time으로 대체
        if duration == 0:
            duration = result.get("decoding_time", 1.0) * 10  # 대략적 추정
        
        return WhisperResult(
            text=text,
            segments=[],  # 이 API는 segments를 직접 제공하지 않음
            language=language,
            duration=duration,
        )
    
    async def health_check(self) -> bool:
        """API 헬스 체크."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False


# 싱글톤
_client: Optional[WhisperClient] = None


def get_whisper_client() -> WhisperClient:
    """WhisperClient 싱글톤."""
    global _client
    if _client is None:
        _client = WhisperClient()
    return _client
