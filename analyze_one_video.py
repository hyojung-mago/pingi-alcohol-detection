#!/usr/bin/env python3
"""단일 오디오/동영상(추출 wav) ad-hoc 분석."""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

import numpy as np

from ml.config import INFERENCE_CONFIG
from ml.inference import DrunkDetector, FakeDrunkDetector
from ml.speech_rate import extract_speech_rate_features
from ml.whisper_client import get_whisper_client


def segment_wav(wav: Path, out_dir: Path, seg_sec: float = 15.0) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(wav),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    duration = float(probe.stdout.strip())
    paths = []
    t = 0.0
    i = 0
    while t < duration - 1.0:
        out = out_dir / f"seg_{i:02d}.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", str(t), "-i", str(wav),
                "-t", str(seg_sec), "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", str(out),
            ],
            capture_output=True,
            check=True,
        )
        paths.append(out)
        t += seg_sec
        i += 1
    return paths


def print_result(label: str, r, fake=None):
    print(f"\n--- {label} ---")
    print(f"  SVM 확률 (취함=1): {r.probability:.3f}")
    print(f"  레벨: {r.level} ({r.level_description})")
    print(f"  is_drunk (절대 0.5 기준): {r.is_drunk}")
    print(f"  신뢰도: {r.confidence} ({r.confidence_score:.3f})")
    if r.raw_features:
        print("  주요 음향:")
        for k, v in r.raw_features.items():
            print(f"    {k}: {v:.4f}")
    if fake:
        print(f"  취한척: {fake.is_fake_acting} (p={fake.probability:.3f})")
        if fake.is_fake_acting:
            print(f"    → {fake.message}")


async def whisper_rate(path: Path) -> float | None:
    client = get_whisper_client()
    res = await client.transcribe(str(path), lang="ko")
    if not res:
        return None
    sr = extract_speech_rate_features(res)
    if not sr:
        return None
    print(f"  Whisper: \"{res.text[:80]}{'...' if len(res.text)>80 else ''}\"")
    print(f"    음절/초={sr.syllables_per_sec:.2f}, 단어/초={sr.words_per_sec:.2f}, 휴지={sr.pause_ratio:.2f}")
    return sr.syllables_per_sec


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path)
    parser.add_argument("--whisper", action="store_true")
    parser.add_argument("--segment-sec", type=float, default=15.0)
    args = parser.parse_args()

    wav = args.wav.resolve()
    if not wav.exists():
        print(f"파일 없음: {wav}", file=sys.stderr)
        sys.exit(1)

    detector = DrunkDetector(version="v2")
    detector.load()
    fake_det = FakeDrunkDetector()
    fake_det.load()

    print("=" * 60)
    print("  Pingi ad-hoc 음성 분석")
    print("=" * 60)
    print(f"  파일: {wav.name}")
    print(f"  참고: 개인 sober 베이스라인 없으면 delta 방식 불가 → 절대 확률만")

    r = detector.predict(wav, return_features=True)
    fake = fake_det.predict(wav)
    print_result("전체 (~51초)", r, fake)

    if args.whisper:
        print("\n[Whisper STT - 전체]")
        await whisper_rate(wav)

    seg_dir = wav.parent / f"{wav.stem}_segments"
    segs = segment_wav(wav, seg_dir, args.segment_sec)
    print(f"\n[구간별 분석] {len(segs)}개 × {args.segment_sec}s")
    probs = []
    for p in segs:
        rr = detector.predict(p, return_features=True)
        ff = fake_det.predict(p)
        probs.append(rr.probability)
        print_result(p.name, rr, ff)

    print("\n--- 요약 ---")
    print(f"  구간 확률: min={min(probs):.3f}, max={max(probs):.3f}, mean={np.mean(probs):.3f}")
    print(f"  ALC 레벨 임계값 참고: L0<{INFERENCE_CONFIG.level_thresholds[0]}, "
          f"L1<{INFERENCE_CONFIG.level_thresholds[1]}, ... L5=1.0")
    print("\n  ⚠ 이 모델은 ALC(영어)로 학습됨. 한국어+개인 베이스라인 없이는 참고용입니다.")
    print("  서비스 정확도는 Pingi 앱에서 녹음한 sober 베이스라인 + delta로 측정합니다.")


if __name__ == "__main__":
    asyncio.run(main())
