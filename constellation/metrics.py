"""Evaluator based on the supplied brief, not official Kaggle scorer code.

Absent-class F1 and present-class F1 are averaged. An F1 with a zero
denominator is set to zero. Geometry pairs are greedily matched by ascending
distance across ALL predicted present points, independently of patch IDs/m.
Geometry/localization return zero when no corresponding true points exist.
"""
from __future__ import annotations

import numpy as np

from .data import row_points


def reward(distance: np.ndarray | float) -> np.ndarray:
    return np.clip((36.0 - np.asarray(distance)) / 24.0, 0.0, 1.0)


def score_scene(truth: dict[str, str], predicted: dict[str, str]) -> dict[str, float]:
    if truth["Id"] != predicted["Id"] or int(truth["n_patches"]) != int(predicted["n_patches"]):
        raise ValueError("Scene ID or patch count does not match")
    true = row_points(truth)
    pred = row_points(predicted)
    tp = sum(a is not None and b is not None for a, b in zip(true, pred))
    tn = sum(a is None and b is None for a, b in zip(true, pred))
    fp = sum(a is None and b is not None for a, b in zip(true, pred))
    fn = sum(a is not None and b is None for a, b in zip(true, pred))
    f1_pos = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1_neg = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    localization = [float(reward(np.linalg.norm(np.array(a[:2]) - b[:2]))) if b else 0.0
                    for a, b in zip(true, pred) if a]
    figures = np.array([p[:2] for p in true if p and p[2]], dtype=float).reshape(-1, 2)
    points = np.array([p[:2] for p in pred if p], dtype=float).reshape(-1, 2)
    geo_sum = 0.0
    if len(figures) and len(points):
        distances = np.linalg.norm(figures[:, None] - points[None], axis=2)
        used_a, used_b = set(), set()
        for flat in np.argsort(distances.ravel(), kind="stable"):
            a, b = np.unravel_index(flat, distances.shape)
            if distances[a, b] >= 36:
                break
            if a not in used_a and b not in used_b:
                geo_sum += float(reward(distances[a, b]))
                used_a.add(a)
                used_b.add(b)
    parts = {
        "presence": (f1_pos + f1_neg) / 2,
        "localization": float(np.mean(localization)) if localization else 0.0,
        "geometric_recovery": geo_sum / len(figures) if len(figures) else 0.0,
        "identification": float(truth["constellation"] == predicted["constellation"]),
    }
    parts["score"] = sum(parts[key] * weight for key, weight in zip(parts, [.25, .20, .25, .30]))
    parts.update(present_f1=f1_pos, absent_f1=f1_neg, tp=tp, tn=tn, fp=fp, fn=fn)
    return parts
