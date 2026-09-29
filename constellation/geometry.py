"""Geometric transforms and RANSAC for point correspondences.

Families used at inference:
- similarity (translation, rotation, uniform scale) for patch-to-sky alignment
- affine (adds non-uniform scale and shear) for constellation identification

Homography is implemented for tests but not used: a 32x32 star crop does not
need a projective model, and schematic templates are not perspective views.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def transform_points(fit: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 3x2 least-squares fit: [x' y'] = [x y 1] @ fit."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    return np.column_stack([points, np.ones(len(points))]) @ fit


def residuals(fit: np.ndarray, source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    return np.linalg.norm(transform_points(fit, source) - destination, axis=1)


def similarity_from_points(source: np.ndarray, destination: np.ndarray) -> np.ndarray | None:
    """Umeyama similarity: translation, rotation, uniform scale. Needs ≥2 points."""
    source = np.asarray(source, dtype=float).reshape(-1, 2)
    destination = np.asarray(destination, dtype=float).reshape(-1, 2)
    if len(source) < 2 or len(source) != len(destination):
        return None
    src_mean = source.mean(axis=0)
    dst_mean = destination.mean(axis=0)
    src = source - src_mean
    dst = destination - dst_mean
    variance = float(np.square(src).sum() / len(src))
    if variance < 1e-12:
        return None
    covariance = dst.T @ src / len(src)
    u, singular, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    scale = float(singular.sum() / variance)
    if not 0.35 < scale < 2.5:
        return None
    translation = dst_mean - scale * rotation @ src_mean
    linear = scale * rotation.T
    return np.vstack([linear, translation])


def affine_from_points(source: np.ndarray, destination: np.ndarray) -> np.ndarray | None:
    """Unconstrained affine (3x2). Needs ≥3 non-collinear points."""
    source = np.asarray(source, dtype=float).reshape(-1, 2)
    destination = np.asarray(destination, dtype=float).reshape(-1, 2)
    if len(source) < 3 or len(source) != len(destination):
        return None
    design = np.column_stack([source, np.ones(len(source))])
    if np.linalg.matrix_rank(design) < 3:
        return None
    fit, *_ = np.linalg.lstsq(design, destination, rcond=None)
    linear = fit[:2].T
    singular = np.linalg.svd(linear, compute_uv=False)
    if singular[-1] < 0.05 or singular[0] / singular[-1] > 8:
        return None
    return fit


def homography_from_points(source: np.ndarray, destination: np.ndarray) -> np.ndarray | None:
    """DLT projective transform (3x3). Not used by inference."""
    source = np.asarray(source, dtype=float).reshape(-1, 2)
    destination = np.asarray(destination, dtype=float).reshape(-1, 2)
    if len(source) < 4:
        return None
    matrix = []
    for (x, y), (u, v) in zip(source, destination):
        matrix.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        matrix.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(matrix, dtype=float))
    homography = vt[-1].reshape(3, 3)
    if abs(homography[2, 2]) < 1e-12:
        return None
    return homography / homography[2, 2]


@dataclass
class RansacResult:
    model: np.ndarray
    inliers: np.ndarray
    n_inliers: int


def ransac(source: np.ndarray, destination: np.ndarray, *, estimate, min_samples: int,
           threshold: float, trials: int, rng: np.random.Generator) -> RansacResult | None:
    """Minimal-subset RANSAC, then least-squares refit on inliers."""
    source = np.asarray(source, dtype=float).reshape(-1, 2)
    destination = np.asarray(destination, dtype=float).reshape(-1, 2)
    count = len(source)
    if count < min_samples or count != len(destination) or trials < 1:
        return None
    best_inliers = None
    best_count = min_samples - 1
    for _ in range(trials):
        chosen = rng.choice(count, min_samples, replace=False)
        model = estimate(source[chosen], destination[chosen])
        if model is None:
            continue
        inliers = residuals(model, source, destination) < threshold
        support = int(inliers.sum())
        if support > best_count:
            best_count = support
            best_inliers = inliers
    if best_inliers is None:
        return None
    model = estimate(source[best_inliers], destination[best_inliers])
    if model is None:
        return None
    inliers = residuals(model, source, destination) < threshold
    if int(inliers.sum()) < min_samples:
        return None
    return RansacResult(model, inliers, int(inliers.sum()))


def ransac_similarity(source, destination, *, threshold: float, trials: int,
                      rng: np.random.Generator) -> RansacResult | None:
    return ransac(source, destination, estimate=similarity_from_points, min_samples=2,
                  threshold=threshold, trials=trials, rng=rng)


def ransac_affine(source, destination, *, threshold: float, trials: int,
                  rng: np.random.Generator) -> RansacResult | None:
    return ransac(source, destination, estimate=affine_from_points, min_samples=3,
                  threshold=threshold, trials=trials, rng=rng)


def similarity_params(fit: np.ndarray) -> tuple[float, float, float, float]:
    """Return (tx, ty, scale, angle) for a similarity stored as a 3x2 fit."""
    scale = float(np.hypot(fit[0, 0], fit[0, 1]))
    angle = float(np.arctan2(fit[0, 1], fit[0, 0]))
    origin = transform_points(fit, np.zeros((1, 2)))[0]
    return float(origin[0]), float(origin[1]), scale, angle
