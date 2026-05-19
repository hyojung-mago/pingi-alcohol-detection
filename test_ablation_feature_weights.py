#!/usr/bin/env python3
"""feature_weights 2× vs 1× 빠른 ablation (ALC 캐시, 동일인 delta)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CACHE = Path(__file__).parent.parent / "Pingi-demo" / "data" / "cache" / "alc_features.npz"
DELTA_FIXED = 0.08

SPEECH_KEYWORDS = ("voiced", "segment", "pause")


def build_feature_weights(feature_names: list[str], use_2x: bool) -> np.ndarray:
    w = np.ones(len(feature_names))
    if not use_2x:
        return w
    for i, name in enumerate(feature_names):
        if any(kw in name.lower() for kw in SPEECH_KEYWORDS):
            w[i] = 2.0
    return w


def train_model(X: np.ndarray, y: np.ndarray) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", C=10.0, probability=True, class_weight="balanced")),
        ]
    )
    model.fit(X, y)
    return model


def evaluate_baseline_delta(
    model,
    X: np.ndarray,
    y: np.ndarray,
    speakers: np.ndarray,
    threshold: float,
) -> dict:
    results = []
    for spn in np.unique(speakers):
        mask = speakers == spn
        X_spk, y_spk = X[mask], y[mask]
        sober_idx = np.where(y_spk == 0)[0]
        drunk_idx = np.where(y_spk == 1)[0]
        if len(sober_idx) < 6 or len(drunk_idx) == 0:
            continue
        baseline_proba = model.predict_proba(X_spk[sober_idx[:5]])[:, 1].mean()
        for idx in sober_idx[5:]:
            p = model.predict_proba(X_spk[idx : idx + 1])[0, 1]
            results.append((0, p - baseline_proba))
        for idx in drunk_idx:
            p = model.predict_proba(X_spk[idx : idx + 1])[0, 1]
            results.append((1, p - baseline_proba))

    y_true = np.array([r[0] for r in results])
    delta = np.array([r[1] for r in results])
    pred = (delta >= threshold).astype(int)
    sr = accuracy_score(y_true[y_true == 0], pred[y_true == 0]) if (y_true == 0).any() else 0
    dr = accuracy_score(y_true[y_true == 1], pred[y_true == 1]) if (y_true == 1).any() else 0
    return {
        "accuracy": accuracy_score(y_true, pred),
        "uar": (sr + dr) / 2,
        "f1": f1_score(y_true, pred),
        "sober_acc": sr,
        "drunk_acc": dr,
        "n": len(y_true),
        "threshold": threshold,
    }


def main():
    if not CACHE.exists():
        print(f"캐시 없음: {CACHE}")
        sys.exit(1)

    data = np.load(CACHE, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    speakers = data["speaker_ids"]
    names = list(data["feature_names"])

    w2 = build_feature_weights(names, use_2x=True)
    w1 = build_feature_weights(names, use_2x=False)
    n2x = int((w2 != 1.0).sum())

    print("=" * 60)
    print("  feature_weights ablation (ALC 캐시)")
    print("=" * 60)
    print(f"  샘플: {len(X)}, 화자: {len(np.unique(speakers))}, 2× 피처: {n2x}개")

    rows = []
    for label, w in [("2× (현재 v2)", w2), ("1× (ablation)", w1)]:
        print(f"\n학습 중: {label}...")
        Xw = X * w
        model = train_model(Xw, y)
        fixed = evaluate_baseline_delta(model, Xw, y, speakers, DELTA_FIXED)
        rows.append((label, fixed))
        print(f"--- {label} ---")
        print(f"  delta={DELTA_FIXED}: acc {fixed['accuracy']*100:.1f}%  UAR {fixed['uar']*100:.1f}%  F1 {fixed['f1']*100:.1f}%")
        print(f"    sober {fixed['sober_acc']*100:.1f}%  drunk {fixed['drunk_acc']*100:.1f}%  (n={fixed['n']})")

    print("\n" + "=" * 60)
    print("  결론 (ALC 동일인, delta=0.08 고정)")
    print("=" * 60)
    uar_diff = (rows[0][1]["uar"] - rows[1][1]["uar"]) * 100
    print(f"  UAR: 2× − 1× = {uar_diff:+.1f}%p  (문서 v2 기준 75.1% ≈ 2× 파이프라인)")
    if uar_diff > 0.5:
        print("  → 2× 유지 권장")
    elif uar_diff < -0.5:
        print("  → 1×로 재학습 검토")
    else:
        print("  → 차이 미미, 2× 없이 재학습해도 됨 (한국어 재검증 필요)")


if __name__ == "__main__":
    main()
