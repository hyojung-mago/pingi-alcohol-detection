#!/usr/bin/env python3
"""
test_gradus.py — GradusSpeech-v1 외부 검증 (ALC 외 주취 음성)

목적: 한국어/서비스 성능 평가가 아니라, ALC에만 과적합됐는지 sanity check.
성공 기준 (엄격한 UAR 아님):
  - intoxicated 샘플 평균 확률 > sober 평균 확률
  - 확률 분리도(separation) > 5%p 정도면 "신호 있음"으로 해석

사용:
  kaggle datasets download -d octoded/gradusspeech-v1 -p data/gradus --unzip
  python test_gradus.py
  python test_gradus.py --root data/gradus --min-bac 1.5
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.inference import DrunkDetector

GRADUS_ROOT = Path(__file__).parent / "data" / "gradus"

# CSV 컬럼 후보 (Kaggle 메타 구조가 다를 수 있어 유연하게)
FILENAME_COLS = ("filename", "file", "file_name", "path", "audio", "wav")
SPEAKER_COLS = ("speaker", "speaker_id", "spk", "subject", "participant")
LABEL_COLS = ("label", "state", "condition", "class", "intoxication", "status")
BAC_COLS = ("bac", "BAC", "bac_permille", "alcohol", "brac")


def _pick_column(columns: List[str], candidates: Tuple[str, ...]) -> Optional[str]:
    lower = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _normalize_label(value) -> Optional[bool]:
    """True=drunk, False=sober, None=unknown."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "drunk", "intoxicated", "alcohol", "positive", "yes"}:
        return True
    if s in {"0", "false", "sober", "normal", "negative", "no"}:
        return False
    if "drunk" in s or "intox" in s or "alcohol" in s:
        return True
    if "sober" in s or "normal" in s:
        return False
    return None


def _bac_from_row(row: pd.Series, bac_col: Optional[str]) -> Optional[float]:
    if bac_col and bac_col in row.index:
        try:
            v = float(row[bac_col])
            if not np.isnan(v):
                return v
        except (TypeError, ValueError):
            pass
    # state 문자열에서 BAC 추출 시도 (예: "bac_0.7_1.5")
    for val in row.astype(str):
        m = re.search(r"(\d+(?:\.\d+)?)\s*‰", val)
        if m:
            return float(m.group(1))
    return None


def discover_metadata(root: Path) -> Tuple[pd.DataFrame, Path]:
    """CSV 메타데이터와 오디오 루트 탐색."""
    csv_files = sorted(root.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"CSV 없음: {root}")

    # 가장 행이 많은 CSV를 메타로 사용
    best_df = None
    best_path = None
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if best_df is None or len(df) > len(best_df):
            best_df = df
            best_path = csv_path

    if best_df is None or best_path is None:
        raise FileNotFoundError("읽을 수 있는 CSV가 없습니다.")

    return best_df, best_path


def build_samples(
    root: Path,
    min_bac: float = 0.0,
    drunk_bac_threshold: float = 1.5,
) -> List[dict]:
    """
    샘플 목록 생성. 라벨 우선순위:
      1) boolean label 컬럼
      2) BAC >= drunk_bac_threshold → drunk
      3) 파일명/경로 휴리스틱 (sober/drunk)
    """
    wav_index: Dict[str, Path] = {}
    for wav in root.rglob("*.wav"):
        wav_index[wav.name] = wav
        wav_index[str(wav.relative_to(root))] = wav

    try:
        meta_df, meta_path = discover_metadata(root)
        print(f"  메타 CSV: {meta_path.relative_to(root)} ({len(meta_df)}행)")
    except FileNotFoundError:
        print("  메타 CSV 없음 → 파일명 휴리스틱만 사용")
        meta_df = None
        meta_path = None

    samples: List[dict] = []

    if meta_df is not None:
        fn_col = _pick_column(list(meta_df.columns), FILENAME_COLS)
        spk_col = _pick_column(list(meta_df.columns), SPEAKER_COLS)
        label_col = _pick_column(list(meta_df.columns), LABEL_COLS)
        bac_col = _pick_column(list(meta_df.columns), BAC_COLS)

        if fn_col is None:
            print(f"  ⚠ filename 컬럼 없음. columns={list(meta_df.columns)}")
        else:
            for _, row in meta_df.iterrows():
                raw_name = str(row[fn_col])
                wav_path = wav_index.get(raw_name) or wav_index.get(Path(raw_name).name)
                if wav_path is None:
                    continue

                bac = _bac_from_row(row, bac_col)
                label = _normalize_label(row[label_col]) if label_col else None

                if label is None and bac is not None:
                    if bac < min_bac:
                        continue
                    label = bac >= drunk_bac_threshold
                if label is None:
                    label = _guess_from_name(raw_name)

                if label is None:
                    continue

                samples.append({
                    "path": wav_path,
                    "label": label,
                    "bac": bac,
                    "speaker": str(row[spk_col]) if spk_col else _speaker_from_name(raw_name),
                })

    if not samples:
        print("  CSV 매칭 실패 → 경로/파일명 휴리스틱 (GradusSpeech 폴더 구조 지원)")
        root_label = _root_folder_label(root)
        for wav in sorted(root.rglob("*.wav")):
            rel = str(wav.relative_to(root))
            label = root_label if root_label is not None else _guess_from_name(str(wav))
            bac = _bac_from_path(wav)
            if label is None and bac is not None:
                if bac < min_bac:
                    continue
                label = bac >= drunk_bac_threshold
            if label is None:
                continue
            # 화자: Sp3, Sp12 등
            speaker = "unknown"
            for part in wav.parts:
                if part.lower().startswith("sp") and part[2:].isdigit():
                    speaker = part
                    break
            if speaker == "unknown":
                speaker = _speaker_from_name(wav.name)
            samples.append({
                "path": wav,
                "label": label,
                "bac": bac,
                "speaker": speaker,
            })

    return samples


def _guess_from_name(name: str) -> Optional[bool]:
    n = name.lower().replace("\\", "/")
    if "sober" in n or "/normal/" in n or "control" in n:
        return False
    if "drunk" in n or "intox" in n or "alcohol" in n:
        return True
    # GradusSpeech-v1 folder layout: .../Sober/... or .../BAC_0.7/...
    if "/sober" in n or n.endswith("/sober") or "/sober/" in n:
        return False
    if "bac_" in n:
        return True
    return None


def _root_folder_label(root: Path) -> Optional[bool]:
    """리프 폴더명만 라벨로 (예: Sober, BAC_1.5+)."""
    name = root.name.lower()
    if name == "sober":
        return False
    if name.startswith("bac_"):
        return True
    return None


def _bac_from_path(path: Path) -> Optional[float]:
    """경로에서 BAC 추정 (예: BAC_1.5+, BAC_0.7)."""
    parts = [p.lower() for p in path.parts]
    for part in parts:
        if part.startswith("bac_"):
            tag = part.replace("bac_", "").replace("+", "").strip()
            try:
                return float(tag)
            except ValueError:
                pass
    return None


def _speaker_from_name(name: str) -> str:
    stem = Path(name).stem
    for part in re.split(r"[_\-\s]+", stem):
        if part.isdigit():
            return part
    return stem.split("_")[0] if "_" in stem else stem


def evaluate(
    detector: DrunkDetector,
    samples: List[dict],
    use_baseline: bool = True,
    max_per_speaker: int = 10,
) -> dict:
    """화자별 sober 베이스라인 + delta 추론 또는 절대 추론."""
    by_speaker: Dict[str, Dict[str, List[dict]]] = {}
    for s in samples:
        spk = s["speaker"]
        by_speaker.setdefault(spk, {"sober": [], "drunk": []})
        key = "drunk" if s["label"] else "sober"
        by_speaker[spk][key].append(s)

    all_probs: List[float] = []
    all_labels: List[bool] = []
    sober_probs: List[float] = []
    drunk_probs: List[float] = []

    for spk, groups in by_speaker.items():
        sober_list = groups["sober"][:max_per_speaker]
        drunk_list = groups["drunk"][:max_per_speaker]
        if not sober_list and not drunk_list:
            continue

        baseline = None
        if use_baseline and sober_list:
            feats = []
            for item in sober_list[:3]:
                try:
                    feats.append(detector.extract_features(str(item["path"])))
                except Exception:
                    pass
            if feats:
                baseline = np.mean(feats, axis=0)

        def infer(item: dict) -> Optional[float]:
            try:
                if baseline is not None:
                    r = detector.predict_with_baseline(str(item["path"]), baseline)
                else:
                    r = detector.predict(str(item["path"]))
                return r.probability
            except Exception:
                return None

        for item in sober_list:
            p = infer(item)
            if p is None:
                continue
            all_probs.append(p)
            all_labels.append(False)
            sober_probs.append(p)

        for item in drunk_list:
            p = infer(item)
            if p is None:
                continue
            all_probs.append(p)
            all_labels.append(True)
            drunk_probs.append(p)

    if not all_probs:
        return {"error": "no predictions"}

    preds = [p >= 0.5 for p in all_probs]
    tp = sum(1 for p, y in zip(preds, all_labels) if y and p)
    tn = sum(1 for p, y in zip(preds, all_labels) if not y and not p)
    fp = sum(1 for p, y in zip(preds, all_labels) if not y and p)
    fn = sum(1 for p, y in zip(preds, all_labels) if y and not p)

    sober_recall = tn / (tn + fp) if (tn + fp) else 0.0
    drunk_recall = tp / (tp + fn) if (tp + fn) else 0.0
    uar = (sober_recall + drunk_recall) / 2

    avg_sober = float(np.mean(sober_probs)) if sober_probs else 0.0
    avg_drunk = float(np.mean(drunk_probs)) if drunk_probs else 0.0
    separation = avg_drunk - avg_sober

    return {
        "n_total": len(all_probs),
        "n_sober": len(sober_probs),
        "n_drunk": len(drunk_probs),
        "uar": uar * 100,
        "sober_recall": sober_recall * 100,
        "drunk_recall": drunk_recall * 100,
        "avg_sober_prob": avg_sober,
        "avg_drunk_prob": avg_drunk,
        "separation": separation,
    }


def main():
    parser = argparse.ArgumentParser(description="GradusSpeech-v1 외부 검증")
    parser.add_argument("--root", type=Path, default=GRADUS_ROOT)
    parser.add_argument("--min-bac", type=float, default=0.0, help="이 값 미만 BAC 샘플 제외")
    parser.add_argument(
        "--drunk-bac",
        type=float,
        default=1.5,
        help="BAC(‰) 이상을 drunk로 (논문 이진 기준)",
    )
    parser.add_argument("--no-baseline", action="store_true", help="화자별 sober 베이스라인 미사용")
    parser.add_argument("--version", default="v2")
    parser.add_argument(
        "--subset",
        default="low-noise measured BrAC",
        help="GradusSpeech 하위 폴더 (기본: BrAC 측정 구간)",
    )
    parser.add_argument("--max-files", type=int, default=200, help="최대 추론 파일 수")
    args = parser.parse_args()

    print("=" * 60)
    print("  핑이 AI — GradusSpeech-v1 외부 검증")
    print("=" * 60)
    print("  ※ 러시아어 꼬리말 — ALC(독일어) 외 일반화 sanity check")

    root = args.root.resolve()
    subset = root / args.subset if args.subset else root
    if subset.exists():
        root = subset
        print(f"  subset: {args.subset}")

    if not root.exists():
        print(f"\n❌ 데이터 없음: {root}")
        print("\n다운로드:")
        print("  pip install kaggle")
        print("  # ~/.kaggle/kaggle.json 설정 후")
        print("  kaggle datasets download -d octoded/gradusspeech-v1 -p data/gradus --unzip")
        sys.exit(1)

    wav_count = len(list(root.rglob("*.wav")))
    print(f"\n[1] 데이터: {root}")
    print(f"  WAV 파일: {wav_count}개")
    if wav_count == 0:
        print("  ❌ WAV 없음 — unzip 확인")
        sys.exit(1)

    print("\n[2] 샘플 구성...")
    samples = build_samples(root, min_bac=args.min_bac, drunk_bac_threshold=args.drunk_bac)
    n_sober = sum(1 for s in samples if not s["label"])
    n_drunk = sum(1 for s in samples if s["label"])
    print(f"  sober: {n_sober}, drunk: {n_drunk}, total: {len(samples)}")
    if len(samples) == 0:
        print("  ❌ 라벨 매칭 실패 — CSV 컬럼/파일명 확인")
        sys.exit(1)

    if len(samples) > args.max_files:
        # 클래스 균형 유지하며 샘플링
        import random
        random.seed(42)
        sober = [s for s in samples if not s["label"]]
        drunk = [s for s in samples if s["label"]]
        half = args.max_files // 2
        samples = random.sample(sober, min(half, len(sober))) + random.sample(
            drunk, min(half, len(drunk))
        )
        print(f"  (--max-files) {args.max_files}개로 축소")

    print("\n[3] 모델 로드...")
    detector = DrunkDetector(version=args.version)
    detector.load()

    print("\n[4] 추론 (baseline=" + ("off" if args.no_baseline else "per-speaker sober") + ")...")
    stats = evaluate(detector, samples, use_baseline=not args.no_baseline)
    if "error" in stats:
        print(f"  ❌ {stats['error']}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  결과 (참고용 — ALC 75%와 직접 비교 금지)")
    print(f"{'='*60}")
    print(f"  샘플: sober {stats['n_sober']}, drunk {stats['n_drunk']}")
    print(f"  UAR (0.5 threshold): {stats['uar']:.1f}%")
    print(f"  Sober recall: {stats['sober_recall']:.1f}%  |  Drunk recall: {stats['drunk_recall']:.1f}%")
    print(f"  평균 확률 sober: {stats['avg_sober_prob']:.1%}  drunk: {stats['avg_drunk_prob']:.1%}")
    print(f"  확률 분리도: {stats['separation']:.1%}")

    if stats["separation"] > 0.05:
        print("\n  ✅ drunk > sober 방향 — ALC 외부에서도 신호 있음 (과적합만은 아닐 가능성)")
    elif stats["separation"] > 0:
        print("\n  ⚠ 약한 분리 — 방향은 맞으나 신뢰 낮음")
    else:
        print("\n  ❌ 분리 없음/역전 — 언어·과제 차이 또는 모델 한계 가능")

    print("\n  해석 가이드:")
    print("  - UAR 50% 근처도 정상 (러시아어+꼬리말 vs 독일어 ALC)")
    print("  - 핵심은 분리도: drunk 평균 확률이 sober보다 충분히 높은지")
    print("  - 한국어는 별도 mini-set 또는 Tang OSF 오디오 확인 후")


if __name__ == "__main__":
    main()
