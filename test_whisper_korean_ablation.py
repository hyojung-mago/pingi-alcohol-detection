#!/usr/bin/env python3
"""
test_whisper_korean_ablation.py — 한국어에서 Whisper가 음향과 "다른 신호"인지 검증

목적: Whisper를 보조 채널로 켤지 말지 데이터로 결정.

준비 (mini-set, 화자당):
  data/korean_eval/
    spk01/sober/*.wav   (베이스라인, 2~3개)
    spk01/drunk/*.wav   (음주 후, 2~3개)
    spk02/...

실행:
  export KAGGLE_API_TOKEN=...   # Whisper API만 필요
  python test_whisper_korean_ablation.py
  python test_whisper_korean_ablation.py --root data/korean_eval --no-whisper  # 음향만 빠르게

판단 기준 (요약):
  1) 상관: Whisper 음절/초 vs openSMILE VoicedSegmentsPerSec  |r| < 0.85 → 중복 아님 가능
  2) UAR: 채널 B(Whisper만) 또는 C(SVM+Whisper)가 A(SVM delta)보다 +3%p 이상 → 보조 가치
  3) 둘 다 아니면 Whisper OFF 유지
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ml.config import INFERENCE_CONFIG
from ml.inference import DrunkDetector
from ml.speech_rate import (
    calculate_speech_rate_score,
    combine_probabilities,
    extract_speech_rate_features,
)
from ml.whisper_client import get_whisper_client


@dataclass
class Sample:
    path: Path
    speaker: str
    label: bool  # True=drunk


def load_samples(root: Path) -> List[Sample]:
    samples: List[Sample] = []
    for spk_dir in sorted(root.iterdir()):
        if not spk_dir.is_dir():
            continue
        for label_name, label in [("sober", False), ("drunk", True)]:
            folder = spk_dir / label_name
            if not folder.is_dir():
                continue
            for wav in sorted(folder.glob("*.wav")):
                samples.append(Sample(wav, spk_dir.name, label))
    return samples


def uar(y_true: List[bool], y_pred: List[bool]) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    sr = tn / (tn + fp) if (tn + fp) else 0.0
    dr = tp / (tp + fn) if (tp + fn) else 0.0
    return (sr + dr) / 2


def voiced_rate_from_features(detector: DrunkDetector, features: np.ndarray) -> float:
    names = detector.extractor.feature_names
    for i, name in enumerate(names):
        if "VoicedSegmentsPerSec" in name:
            return float(features[i])
    return 0.0


async def run_whisper_rates(paths: List[Path]) -> Dict[Path, float]:
    client = get_whisper_client()
    out: Dict[Path, float] = {}
    for p in paths:
        result = await client.transcribe(str(p), lang="ko")
        if not result:
            continue
        sr = extract_speech_rate_features(result)
        if sr:
            out[p] = sr.syllables_per_sec
    return out


def evaluate_channel_a(
    detector: DrunkDetector,
    by_speaker: Dict[str, List[Sample]],
    threshold: float,
) -> Tuple[float, List[bool], List[bool]]:
    y_true, y_pred = [], []
    for spk, items in by_speaker.items():
        sober = [s for s in items if not s.label]
        if not sober:
            continue
        baseline_feats = [detector.extract_features(str(s.path)) for s in sober[:3]]
        baseline = np.mean(baseline_feats, axis=0)
        Xb = detector._apply_weights(baseline).reshape(1, -1)
        baseline_proba = detector.model.predict_proba(Xb)[0, 1]
        for s in items:
            feat = detector.extract_features(str(s.path))
            X = detector._apply_weights(feat).reshape(1, -1)
            proba = detector.model.predict_proba(X)[0, 1]
            delta = proba - baseline_proba
            y_true.append(s.label)
            y_pred.append(delta >= threshold)
    return uar(y_true, y_pred), y_true, y_pred


def evaluate_channel_b(
    by_speaker: Dict[str, List[Sample]],
    whisper_rates: Dict[Path, float],
    max_decrease: float,
    rate_threshold: float = 0.5,
) -> Tuple[float, List[bool], List[bool]]:
    """Whisper 발화속도 감소만으로 취함 판정."""
    y_true, y_pred = [], []
    for spk, items in by_speaker.items():
        sober_rates = [
            whisper_rates[s.path]
            for s in items
            if not s.label and s.path in whisper_rates
        ]
        if not sober_rates:
            continue
        baseline_rate = float(np.mean(sober_rates))
        for s in items:
            if s.path not in whisper_rates:
                continue
            score = calculate_speech_rate_score(
                baseline_rate, whisper_rates[s.path], max_decrease
            )
            y_true.append(s.label)
            y_pred.append(score >= rate_threshold)
    return uar(y_true, y_pred), y_true, y_pred


def evaluate_channel_c(
    detector: DrunkDetector,
    by_speaker: Dict[str, List[Sample]],
    whisper_rates: Dict[Path, float],
    delta_threshold: float,
    svm_weight: float,
    max_decrease: float,
    combine_threshold: float = 0.5,
) -> Tuple[float, List[bool], List[bool]]:
    y_true, y_pred = [], []
    for spk, items in by_speaker.items():
        sober = [s for s in items if not s.label]
        if not sober:
            continue
        baseline_feats = [detector.extract_features(str(s.path)) for s in sober[:3]]
        baseline = np.mean(baseline_feats, axis=0)
        Xb = detector._apply_weights(baseline).reshape(1, -1)
        baseline_proba = detector.model.predict_proba(Xb)[0, 1]

        sober_rates = [
            whisper_rates[s.path]
            for s in sober
            if s.path in whisper_rates
        ]
        baseline_rate = float(np.mean(sober_rates)) if sober_rates else 0.0

        for s in items:
            feat = detector.extract_features(str(s.path))
            X = detector._apply_weights(feat).reshape(1, -1)
            proba = detector.model.predict_proba(X)[0, 1]
            delta = proba - baseline_proba
            acoustic_drunk = delta >= delta_threshold

            if s.path in whisper_rates and baseline_rate > 0:
                sr_score = calculate_speech_rate_score(
                    baseline_rate, whisper_rates[s.path], max_decrease
                )
                combined = combine_probabilities(proba, sr_score, svm_weight)
                pred = combined >= combine_threshold or (
                    acoustic_drunk and sr_score >= 0.35
                )
            else:
                pred = acoustic_drunk

            y_true.append(s.label)
            y_pred.append(pred)
    return uar(y_true, y_pred), y_true, y_pred


def correlation_acoustic_vs_whisper(
    detector: DrunkDetector,
    samples: List[Sample],
    whisper_rates: Dict[Path, float],
) -> Optional[float]:
    xs, ys = [], []
    for s in samples:
        if s.path not in whisper_rates:
            continue
        feat = detector.extract_features(str(s.path))
        xs.append(voiced_rate_from_features(detector, feat))
        ys.append(whisper_rates[s.path])
    if len(xs) < 4:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


async def main_async(args):
    root = Path(args.root)
    if not root.exists():
        print(f"❌ 폴더 없음: {root}")
        print(__doc__)
        return

    samples = load_samples(root)
    if not samples:
        print("❌ wav 없음. spkXX/sober, spkXX/drunk 구조 확인")
        return

    by_speaker: Dict[str, List[Sample]] = {}
    for s in samples:
        by_speaker.setdefault(s.speaker, []).append(s)

    detector = DrunkDetector(version="v2")
    detector.load()
    threshold = INFERENCE_CONFIG.delta_threshold

    print("=" * 60)
    print("  한국어 Whisper vs 음향 ablation")
    print("=" * 60)
    print(f"  샘플: {len(samples)} ({len(by_speaker)}명)")
    print(f"  sober: {sum(1 for s in samples if not s.label)}, drunk: {sum(1 for s in samples if s.label)}")

    whisper_rates: Dict[Path, float] = {}
    if not args.no_whisper:
        print("\n[1] Whisper STT (음절/초)...")
        whisper_rates = await run_whisper_rates([s.path for s in samples])
        print(f"  성공: {len(whisper_rates)}/{len(samples)}")
    else:
        print("\n[1] Whisper 스킵 (--no-whisper)")

    print("\n[2] 채널별 UAR (화자별 베이스라인)")
    uar_a, _, _ = evaluate_channel_a(detector, by_speaker, threshold)
    print(f"  A) SVM delta only (현재 서비스):     UAR {uar_a*100:.1f}%")

    if whisper_rates:
        r = correlation_acoustic_vs_whisper(detector, samples, whisper_rates)
        print(f"\n[3] 중복 검사: VoicedSegmentsPerSec vs Whisper 음절/초")
        if r is not None:
            print(f"  Pearson r = {r:.3f}")
            if abs(r) >= 0.85:
                print("  → 높은 상관: 같은 축(속도)일 가능성 큼, Whisper 보조 가치 낮음")
            else:
                print("  → 상관 낮음: Whisper가 다른 정보일 수 있음 → C 채널 검토")

        uar_b, _, _ = evaluate_channel_b(by_speaker, whisper_rates, args.max_decrease)
        print(f"\n  B) Whisper 발화속도 only:          UAR {uar_b*100:.1f}%")

        best_w, best_uar = 0.7, -1.0
        for w in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            uar_c, _, _ = evaluate_channel_c(
                detector, by_speaker, whisper_rates, threshold, w, args.max_decrease
            )
            if uar_c > best_uar:
                best_uar, best_w = uar_c, w
        print(f"  C) SVM + Whisper (최적 가중치):     UAR {best_uar*100:.1f}%  (svm_weight={best_w})")

        print("\n[4] 결론")
        gain = (best_uar - uar_a) * 100
        if gain >= 3.0 and (r is None or abs(r) < 0.85):
            print(f"  ✅ Whisper 보조 채널 검토 가치 있음 (A 대비 +{gain:.1f}%p)")
            print(f"     config: enabled=True, svm_weight={best_w}, speech_rate_weight={1-best_w:.1f}")
        elif gain >= 3.0:
            print(f"  ⚠ UAR은 오르지만 r={r:.2f}로 중복 가능 → 가중치 낮게만 시험")
        else:
            print(f"  ❌ Whisper OFF 유지 권장 (A 대비 {gain:+.1f}%p, 유의미 개선 없음)")
    else:
        print("\n  Whisper 결과 없음 — API 확인 후 재실행")


def main():
    parser = argparse.ArgumentParser(description="한국어 Whisper ablation")
    parser.add_argument("--root", default="data/korean_eval")
    parser.add_argument("--no-whisper", action="store_true", help="음향만 빠르게")
    parser.add_argument("--max-decrease", type=float, default=0.3)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
