#!/usr/bin/env python3
"""
test_inference.py - ALC 데이터셋으로 Pingi-AI 모델 추론 테스트

화자별 sober/drunk 세션을 찾아서:
  1. sober 파일 → 베이스라인 생성
  2. drunk 파일 → predict_with_baseline
  3. sober 파일 → predict_with_baseline (정상 확인)
  4. 정확도, 혼동 행렬 출력
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

from ml.inference import DrunkDetector

ALC_PATH = Path(__file__).parent.parent / "Pingi-demo" / "data" / "alc" / "ALC"


def parse_session_metadata(session_dir: Path) -> dict:
    """세션 메타데이터(annot.json)에서 화자, BAC, drunk 여부 파싱."""
    annot_files = list(session_dir.glob("*_annot.json"))
    if not annot_files:
        return {}

    with open(annot_files[0]) as f:
        data = json.load(f)

    try:
        labels = data["levels"][0]["items"][0]["labels"]
        label_dict = {l["name"]: l["value"] for l in labels}
        return {
            "spn": label_dict.get("spn", ""),
            "is_drunk": label_dict.get("alc", "") == "a",
            "bac": float(label_dict.get("aak", 0) or 0),
        }
    except (KeyError, IndexError):
        return {}


def find_speaker_pairs(alc_path: Path, max_speakers: int = 20):
    """화자별 sober/drunk 세션 페어 찾기."""
    speakers = defaultdict(lambda: {"sober": None, "drunk": None})

    for session_dir in sorted(alc_path.iterdir()):
        if not session_dir.is_dir() or session_dir.name.startswith("."):
            continue

        meta = parse_session_metadata(session_dir)
        if not meta or not meta["spn"]:
            continue

        spn = meta["spn"]
        wav_files = sorted(session_dir.glob("*.wav"))
        if not wav_files:
            continue

        if meta["is_drunk"]:
            speakers[spn]["drunk"] = {
                "dir": session_dir,
                "files": wav_files,
                "bac": meta["bac"],
            }
        else:
            speakers[spn]["sober"] = {
                "dir": session_dir,
                "files": wav_files,
                "bac": 0.0,
            }

    # sober+drunk 둘 다 있는 화자만
    pairs = {}
    for spn, data in speakers.items():
        if data["sober"] and data["drunk"]:
            pairs[spn] = data
            if len(pairs) >= max_speakers:
                break

    return pairs


def main():
    print("=" * 60)
    print("  핑이 AI — ALC 데이터셋 추론 테스트")
    print("=" * 60)

    if not ALC_PATH.exists():
        print(f"❌ ALC 데이터 없음: {ALC_PATH}")
        sys.exit(1)

    # 모델 로드
    print("\n[1] 모델 로드...")
    detector = DrunkDetector(version="v2")
    detector.load()
    print(f"  ✅ {detector.model_path.name}")

    # 화자 페어 찾기
    print("\n[2] 화자 페어 탐색...")
    pairs = find_speaker_pairs(ALC_PATH, max_speakers=15)
    print(f"  ✅ {len(pairs)}명 화자 (sober+drunk)")

    # 추론 테스트
    print("\n[3] 추론 테스트...")
    print(f"  {'화자':<8} {'BAC':>6} {'Sober→':>8} {'Drunk→':>8} {'정확':>4}")
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*4}")

    results = []
    files_per_speaker = 5  # 화자당 최대 5개 파일

    for spn, data in pairs.items():
        sober_files = data["sober"]["files"][:files_per_speaker]
        drunk_files = data["drunk"]["files"][:files_per_speaker]
        bac = data["drunk"]["bac"]

        # 베이스라인: sober 파일 3개 평균
        try:
            baseline_features = []
            for f in sober_files[:3]:
                feat = detector.extract_features(str(f))
                baseline_features.append(feat)
            baseline = np.mean(baseline_features, axis=0)
        except Exception as e:
            print(f"  {spn:<8} 베이스라인 실패: {e}")
            continue

        # Sober 추론
        sober_correct = 0
        sober_total = 0
        sober_probs = []
        for f in sober_files:
            try:
                result = detector.predict_with_baseline(str(f), baseline)
                sober_probs.append(result.probability)
                if not result.is_drunk:
                    sober_correct += 1
                sober_total += 1
            except Exception:
                pass

        # Drunk 추론
        drunk_correct = 0
        drunk_total = 0
        drunk_probs = []
        for f in drunk_files:
            try:
                result = detector.predict_with_baseline(str(f), baseline)
                drunk_probs.append(result.probability)
                if result.is_drunk:
                    drunk_correct += 1
                drunk_total += 1
            except Exception:
                pass

        if sober_total == 0 or drunk_total == 0:
            continue

        sober_acc = sober_correct / sober_total * 100
        drunk_acc = drunk_correct / drunk_total * 100
        avg_sober_prob = np.mean(sober_probs)
        avg_drunk_prob = np.mean(drunk_probs)

        correct_mark = "✓" if sober_acc >= 50 and drunk_acc >= 50 else "✗"
        print(
            f"  {spn:<8} {bac:>5.3f} "
            f"{avg_sober_prob:>5.1%}({sober_correct}/{sober_total}) "
            f"{avg_drunk_prob:>5.1%}({drunk_correct}/{drunk_total}) "
            f"{correct_mark}"
        )

        results.append({
            "spn": spn,
            "bac": bac,
            "sober_acc": sober_acc,
            "drunk_acc": drunk_acc,
            "sober_prob_avg": float(avg_sober_prob),
            "drunk_prob_avg": float(avg_drunk_prob),
        })

    # 종합 결과
    if results:
        print(f"\n{'='*60}")
        print("  종합 결과")
        print(f"{'='*60}")

        avg_sober_acc = np.mean([r["sober_acc"] for r in results])
        avg_drunk_acc = np.mean([r["drunk_acc"] for r in results])
        avg_sober_prob = np.mean([r["sober_prob_avg"] for r in results])
        avg_drunk_prob = np.mean([r["drunk_prob_avg"] for r in results])
        overall_acc = (avg_sober_acc + avg_drunk_acc) / 2

        print(f"  화자 수: {len(results)}명")
        print(f"  Sober 정확도: {avg_sober_acc:.1f}% (평균 확률: {avg_sober_prob:.1%})")
        print(f"  Drunk 정확도: {avg_drunk_acc:.1f}% (평균 확률: {avg_drunk_prob:.1%})")
        print(f"  전체 정확도 (UAR): {overall_acc:.1f}%")
        print(f"  확률 분리도: {avg_drunk_prob - avg_sober_prob:.1%}")
        print()

        both_correct = sum(1 for r in results if r["sober_acc"] >= 50 and r["drunk_acc"] >= 50)
        print(f"  양쪽 모두 맞춘 화자: {both_correct}/{len(results)}")
    else:
        print("\n❌ 추론 결과 없음")


if __name__ == "__main__":
    main()
