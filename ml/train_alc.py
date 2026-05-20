#!/usr/bin/env python3
"""
train_alc.py — ALC SVM 학습

모드:
  delta (권장, v2-delta):
    학습 X = current_features - baseline_mean (same-person sober 5개)
    SVM binary → proba
    optimal_threshold: same-person 2430샘플 accuracy fit (L0/L1)
    level_thresholds L1~L5: 동일 proba, drunk 샘플 20/40/60/80% 분위

  absolute (legacy v2):
    학습 X = absolute features (sober vs drunk)
    추론 시 proba delta 캘리브레이션

출력: models/drunk_detector_{version}.pkl, models/final_weights.json
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.config import MODEL_PATH, INFERENCE_CONFIG

logging.basicConfig(level=logging.INFO, format="[train_alc] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
ALC_CACHE = PROJECT_ROOT.parent / "Pingi-demo" / "data" / "cache" / "alc_features.npz"
DEFAULT_BASELINE_N = 5


def load_alc_data() -> dict:
    if not ALC_CACHE.exists():
        raise FileNotFoundError(
            f"ALC 캐시 없음: {ALC_CACHE}\n"
            "  cd Pingi-demo && python -m scripts.ml.prepare_dataset"
        )
    data = np.load(ALC_CACHE, allow_pickle=True)
    return {
        "X": data["X"],
        "y": data["y"],
        "speakers": data["speaker_ids"],
        "feature_names": list(data["feature_names"]),
    }


def build_feature_weights(feature_names: list[str]) -> np.ndarray:
    """Deprecated no-op kept for backward import compatibility (train_synthetic/train_korean)."""
    return np.ones(len(feature_names))


def train_svm(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple:
    gkf = GroupKFold(n_splits=5)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="linear", C=10.0, probability=True, class_weight="balanced")),
    ])
    cv_scores = cross_val_score(pipeline, X, y, cv=gkf, groups=groups, scoring="accuracy")
    logger.info("화자 독립 CV: %.1f%% ± %.1f%%", cv_scores.mean() * 100, cv_scores.std() * 100)
    pipeline.fit(X, y)
    return pipeline, float(cv_scores.mean())


def collect_delta_feature_samples(
    X: np.ndarray,
    y: np.ndarray,
    speakers: np.ndarray,
    baseline_n: int = DEFAULT_BASELINE_N,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """화자별 baseline mean 피처 대비 delta 피처 행렬."""
    X_list, y_list, spk_list = [], [], []

    for spk in np.unique(speakers):
        mask = speakers == spk
        X_spk, y_spk = X[mask], y[mask]
        sober_idx = np.where(y_spk == 0)[0]
        drunk_idx = np.where(y_spk == 1)[0]

        if len(sober_idx) < baseline_n + 1 or len(drunk_idx) == 0:
            continue

        baseline = X_spk[sober_idx[:baseline_n]].mean(axis=0)
        for idx in list(sober_idx[baseline_n:]) + list(drunk_idx):
            X_list.append(X_spk[idx] - baseline)
            y_list.append(int(y_spk[idx]))
            spk_list.append(str(spk))

    return np.array(X_list), np.array(y_list), np.array(spk_list)


def collect_delta_samples(
    model,
    X: np.ndarray,
    y: np.ndarray,
    speakers: np.ndarray,
    baseline_n: int = DEFAULT_BASELINE_N,
) -> list[dict]:
    """absolute 모드: proba delta 평가 샘플."""
    results: list[dict] = []

    for spk in np.unique(speakers):
        mask = speakers == spk
        X_spk, y_spk = X[mask], y[mask]
        sober_idx = np.where(y_spk == 0)[0]
        drunk_idx = np.where(y_spk == 1)[0]

        if len(sober_idx) < baseline_n + 1 or len(drunk_idx) == 0:
            continue

        baseline_proba = model.predict_proba(X_spk[sober_idx[:baseline_n]])[:, 1].mean()

        for idx in list(sober_idx[baseline_n:]) + list(drunk_idx):
            proba = float(model.predict_proba(X_spk[idx : idx + 1])[0, 1])
            results.append({
                "speaker": str(spk),
                "true": int(y_spk[idx]),
                "delta": proba - baseline_proba,
                "proba": proba,
                "baseline_proba": float(baseline_proba),
            })

    return results


def predict_proba_samples(model, X: np.ndarray, y: np.ndarray) -> list[dict]:
    probas = model.predict_proba(X)[:, 1]
    return [{"true": int(label), "proba": float(p)} for label, p in zip(y, probas)]


def find_optimal_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    score_range: tuple[float, float, float] | None = None,
) -> dict:
    if score_range is None:
        lo, hi, step = -0.2, 0.4, 0.005
    else:
        lo, hi, step = score_range

    thresholds = np.arange(lo, hi, step)
    best_acc, best_thresh = 0.0, INFERENCE_CONFIG.delta_threshold

    for t in thresholds:
        pred = (scores >= t).astype(int)
        acc = accuracy_score(labels, pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = float(t)

    pred = (scores >= best_thresh).astype(int)
    return {
        "threshold": best_thresh,
        "accuracy": float(best_acc),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "n_samples": int(len(labels)),
    }


def calibrate_level_thresholds_absolute(
    delta_results: list[dict],
    drunk_threshold: float,
) -> dict[int, float]:
    drunk_deltas = np.array([r["delta"] for r in delta_results if r["true"] == 1])
    if len(drunk_deltas) < 10:
        return dict(INFERENCE_CONFIG.level_thresholds)

    level_anchor = INFERENCE_CONFIG.level_thresholds[0] - drunk_threshold
    adjusted = level_anchor + drunk_deltas
    q20, q40, q60, q80 = np.quantile(adjusted, [0.2, 0.4, 0.6, 0.8])

    return {
        0: float(level_anchor + drunk_threshold),
        1: float(max(q20, level_anchor + drunk_threshold + 0.01)),
        2: float(max(q40, q20 + 0.01)),
        3: float(max(q60, q40 + 0.01)),
        4: float(max(q80, q60 + 0.01)),
        5: 1.0,
    }


def calibrate_level_thresholds_delta(
    proba_results: list[dict],
    drunk_threshold: float,
) -> dict[int, float]:
    """L0/L1=drunk_threshold. L1~L5=threshold 이상 drunk proba 20/40/60/80% 분위."""
    drunk_probas = np.array([r["proba"] for r in proba_results if r["true"] == 1])
    drunk_above = drunk_probas[drunk_probas >= drunk_threshold]
    if len(drunk_above) < 10:
        return dict(INFERENCE_CONFIG.level_thresholds)

    q20, q40, q60, q80 = np.quantile(drunk_above, [0.2, 0.4, 0.6, 0.8])
    return {
        0: float(drunk_threshold),
        1: float(max(q20, drunk_threshold + 0.001)),
        2: float(max(q40, q20 + 0.001)),
        3: float(max(q60, q40 + 0.001)),
        4: float(max(q80, q60 + 0.001)),
        5: 1.0,
    }


def log_level_band_distribution(
    proba_results: list[dict],
    drunk_threshold: float,
    level_thresholds: dict[int, float],
) -> None:
    """drunk 샘플 L1~L5 구간별 비율 (캘리브레이션 검증용)."""
    drunk_probas = np.array([r["proba"] for r in proba_results if r["true"] == 1])
    drunk_above = drunk_probas[drunk_probas >= drunk_threshold]
    if len(drunk_above) == 0:
        return

    bounds = [level_thresholds[i] for i in range(6)]
    logger.info("drunk L1~L5 band (above threshold, n=%d):", len(drunk_above))
    for lv in range(1, 6):
        lo, hi = bounds[lv - 1], bounds[lv]
        if lv < 5:
            cnt = int(np.sum((drunk_above >= lo) & (drunk_above < hi)))
        else:
            cnt = int(np.sum(drunk_above >= lo))
        logger.info("  L%d [%.3f, %.3f): %d (%.1f%%)", lv, lo, hi, cnt, 100 * cnt / len(drunk_above))


def evaluate_levels_absolute(
    delta_results: list[dict],
    drunk_threshold: float,
    level_thresholds: dict[int, float],
) -> dict:
    level_anchor = level_thresholds[0] - drunk_threshold
    correct = 0

    for r in delta_results:
        adjusted = max(0.0, min(1.0, level_anchor + r["delta"]))
        level = 5
        for lv in range(5):
            if adjusted < level_thresholds[lv]:
                level = lv
                break
        ok = (level == 0) if r["true"] == 0 else (level >= 1)
        correct += int(ok)

    n = len(delta_results)
    return {"level_accuracy": float(correct / n) if n else 0.0, "n_samples": n}


def evaluate_levels_delta(
    proba_results: list[dict],
    drunk_threshold: float,
) -> dict:
    correct = 0
    for r in proba_results:
        if r["true"] == 0:
            ok = r["proba"] < drunk_threshold
        else:
            ok = r["proba"] >= drunk_threshold
        correct += int(ok)

    n = len(proba_results)
    return {"level_accuracy": float(correct / n) if n else 0.0, "n_samples": n}


def extract_importance(model, feature_names: list[str]) -> dict[str, float]:
    coef = model.named_steps["clf"].coef_[0]
    importance = {name: float(c) for name, c in zip(feature_names, coef)}
    return dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True))


def save_artifacts(
    *,
    model,
    feature_names: list[str],
    training_mode: str,
    version: str,
    baseline_n: int,
    cv_mean: float,
    metrics: dict,
    level_thresholds: dict[int, float],
    importance: dict,
    usage: dict,
) -> Path:
    model_bundle = {
        "model": model,
        "feature_names": feature_names,
        "training_mode": training_mode,
    }
    out_path = MODEL_PATH / f"drunk_detector_{version}.pkl"
    joblib.dump(model_bundle, out_path)
    logger.info("모델 저장: %s", out_path)

    weights_json = {
        "version": version,
        "created_at": datetime.now().isoformat(),
        "training": {
            "mode": training_mode,
            "baseline_n": baseline_n,
            "alc_cache": str(ALC_CACHE),
        },
        "drunk_detection": {
            "model": f"SVM Linear (ALC, {training_mode})",
            "optimal_threshold": metrics["threshold"],
            "level_thresholds": {str(k): v for k, v in level_thresholds.items()},
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "cv_accuracy": cv_mean,
            "level_accuracy": metrics.get("level_accuracy", metrics["accuracy"]),
            "n_eval_samples": metrics["n_samples"],
            "feature_importance_top20": dict(list(importance.items())[:20]),
        },
        "usage": usage,
    }

    weights_path = MODEL_PATH / "final_weights.json"
    weights_path.write_text(json.dumps(weights_json, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("가중치 저장: %s", weights_path)
    return out_path


def train_absolute(baseline_n: int, version: str) -> Path:
    data = load_alc_data()
    X, y, speakers = data["X"], data["y"], data["speakers"]
    feature_names = data["feature_names"]

    logger.info("[absolute] ALC %d samples, %d speakers", len(y), len(set(speakers)))
    model, cv_mean = train_svm(X, y, speakers)

    delta_results = collect_delta_samples(model, X, y, speakers, baseline_n)
    deltas = np.array([r["delta"] for r in delta_results])
    labels = np.array([r["true"] for r in delta_results])
    metrics = find_optimal_threshold(deltas, labels)
    level_thresholds = calibrate_level_thresholds_absolute(delta_results, metrics["threshold"])
    level_metrics = evaluate_levels_absolute(delta_results, metrics["threshold"], level_thresholds)
    metrics["level_accuracy"] = level_metrics["level_accuracy"]

    logger.info("[absolute] delta 정확도: %.1f%% (threshold=%.3f)",
                metrics["accuracy"] * 100, metrics["threshold"])

    return save_artifacts(
        model=model,
        feature_names=feature_names,
        training_mode="absolute",
        version=version,
        baseline_n=baseline_n,
        cv_mean=cv_mean,
        metrics=metrics,
        level_thresholds=level_thresholds,
        importance=extract_importance(model, feature_names),
        usage={
            "baseline_comparison": f"proba_delta >= {metrics['threshold']:.3f} → drunk",
            "level_mapping": "adjusted_proba = (level_thresholds[0] - optimal_threshold) + delta",
            "model_file": f"drunk_detector_{version}.pkl",
        },
    )


def train_delta(baseline_n: int, version: str) -> Path:
    data = load_alc_data()
    X, y, speakers = data["X"], data["y"], data["speakers"]
    feature_names = data["feature_names"]

    X_delta, y_delta, spk_delta = collect_delta_feature_samples(X, y, speakers, baseline_n)
    if len(y_delta) == 0:
        raise RuntimeError("delta 피처 샘플 없음")

    logger.info("[delta] 학습 샘플: %d (sober=%d, drunk=%d, speakers=%d)",
                len(y_delta), int((y_delta == 0).sum()), int((y_delta == 1).sum()), len(set(spk_delta)))

    model, cv_mean = train_svm(X_delta, y_delta, spk_delta)

    proba_results = predict_proba_samples(model, X_delta, y_delta)
    probas = np.array([r["proba"] for r in proba_results])
    labels = np.array([r["true"] for r in proba_results])
    metrics = find_optimal_threshold(probas, labels, score_range=(0.05, 0.95, 0.005))
    level_thresholds = calibrate_level_thresholds_delta(proba_results, metrics["threshold"])
    log_level_band_distribution(proba_results, metrics["threshold"], level_thresholds)
    level_metrics = evaluate_levels_delta(proba_results, metrics["threshold"])
    metrics["level_accuracy"] = level_metrics["level_accuracy"]

    logger.info("[delta] proba 정확도: %.1f%% (threshold=%.3f, CV=%.1f%%)",
                metrics["accuracy"] * 100, metrics["threshold"], cv_mean * 100)
    logger.info("[delta] level_thresholds: %s",
                {k: round(v, 3) for k, v in level_thresholds.items()})

    return save_artifacts(
        model=model,
        feature_names=feature_names,
        training_mode="delta_features",
        version=version,
        baseline_n=baseline_n,
        cv_mean=cv_mean,
        metrics=metrics,
        level_thresholds=level_thresholds,
        importance=extract_importance(model, feature_names),
        usage={
            "baseline_comparison": f"proba(current-baseline) >= {metrics['threshold']:.3f} → drunk",
            "level_mapping": "level = f(proba) with level_thresholds on delta-feature model output",
            "model_file": f"drunk_detector_{version}.pkl",
        },
    )


def train(mode: str = "absolute", baseline_n: int = DEFAULT_BASELINE_N, version: str = "v2") -> Path:
    if mode == "delta":
        return train_delta(baseline_n, version)
    if mode == "absolute":
        return train_absolute(baseline_n, version)
    raise ValueError(f"unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description="ALC SVM 학습")
    parser.add_argument("--mode", choices=["absolute", "delta"], default="delta",
                        help="absolute: 절대분류+delta캘리브 / delta: delta피처 직접학습")
    parser.add_argument("--baseline-n", type=int, default=DEFAULT_BASELINE_N)
    parser.add_argument("--version", default=None,
                        help="미지정 시 v2 (absolute) / v2-delta (delta)")
    args = parser.parse_args()

    version = args.version or ("v2-delta" if args.mode == "delta" else "v2")
    train(mode=args.mode, baseline_n=args.baseline_n, version=version)


if __name__ == "__main__":
    main()
