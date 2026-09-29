"""Experimental full-resolution, rotation/scale masked-NCC candidate stream.

This module is deliberately separate from inference. It takes only the supplied
image/patch pixels; synthetic inputs are used only by its regression tests.
"""
from __future__ import annotations

import cv2
import numpy as np


def relative_gap(candidates: list[dict]) -> float:
    """Best-vs-runner-up score gap, normalized by the best score's headroom."""
    if len(candidates) < 2:
        return 0.0
    first, second = candidates[0]['score'], candidates[1]['score']
    return float((first - second) / max(1.0 - first, .01))


def _template(patch: np.ndarray, scale: float, degrees: float) -> tuple[np.ndarray, np.ndarray]:
    """Warp a patch onto a fixed 31-pixel disc in sky coordinates."""
    offsets = np.arange(-15, 16, dtype=np.float32)
    xx, yy = np.meshgrid(offsets, offsets)
    theta = np.deg2rad(degrees)
    co, si = float(np.cos(theta)), float(np.sin(theta))
    src_x = (15.5 + (co * xx + si * yy) / scale).astype(np.float32)
    src_y = (15.5 + (-si * xx + co * yy) / scale).astype(np.float32)
    radius = min(15.0, 15.0 * scale)
    mask = ((xx * xx + yy * yy) <= radius * radius).astype(np.float32)
    warped = cv2.remap(patch, src_x, src_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)
    return warped, mask


def masked_ncc_candidates(
    sky: np.ndarray,
    patch: np.ndarray,
    *,
    scales: tuple[float, ...] = (.93, 1.0, 1.08),
    angle_step: int = 10,
    keep: int = 30,
    separation: int = 13,
) -> list[dict]:
    """Return distinct full-sky peaks using a disc-normalized NCC response.

    The numerator is one ordinary correlation per angle. The masked local
    mean/variance denominator is shared across all angles at a given scale.
    This avoids OpenCV's much slower masked ``TM_CCOEFF_NORMED`` path.
    Images should already be filtered identically, if filtering is desired.
    """
    if sky.ndim != 2 or patch.shape != (32, 32):
        raise ValueError("Expected a grayscale sky and a 32x32 grayscale patch")
    if not 0 < angle_step <= 180 or 360 % angle_step:
        raise ValueError("angle_step must divide 360")
    if keep < 1 or separation < 1 or not scales or any(s <= 0 for s in scales):
        raise ValueError("Invalid scale, keep, or separation")
    image = np.ascontiguousarray(sky, dtype=np.float32)
    query = np.ascontiguousarray(patch, dtype=np.float32)
    if min(image.shape) < 31:
        return []
    image_sq = image * image
    response_shape = (image.shape[0] - 30, image.shape[1] - 30)
    best = np.full(response_shape, -np.inf, dtype=np.float32)
    best_scale = np.zeros(response_shape, dtype=np.float32)
    best_angle = np.zeros(response_shape, dtype=np.int16)
    for scale in scales:
        _, mask = _template(query, float(scale), 0.0)
        n = float(mask.sum())
        local_sum = cv2.matchTemplate(image, mask, cv2.TM_CCORR)
        local_sum_sq = cv2.matchTemplate(image_sq, mask, cv2.TM_CCORR)
        local_variance = np.maximum(local_sum_sq - local_sum * local_sum / n, 0)
        # A floor prevents tiny-variance regions from scoring implausibly high.
        local_sd = np.sqrt(local_variance / n)
        floor = .5 * float(np.median(local_sd))
        denominator = np.sqrt(np.maximum(local_variance, n * floor * floor))
        for degrees in range(0, 360, angle_step):
            warped, _ = _template(query, float(scale), float(degrees))
            centred = (warped - float(np.sum(warped * mask) / n)) * mask
            norm = float(np.linalg.norm(centred))
            if norm < 1e-6:
                continue
            numerator = cv2.matchTemplate(image, centred, cv2.TM_CCORR)
            response = numerator / np.maximum(denominator * norm, 1e-6)
            improved = response > best
            best[improved] = response[improved]
            best_scale[improved] = scale
            best_angle[improved] = degrees
    if not np.isfinite(best).any():
        return []
    # One maximum over all transforms, then spatial non-maximum suppression.
    radius = max(1, separation // 2)
    maxima = best == cv2.dilate(best, np.ones((2 * radius + 1,) * 2, np.uint8))
    yy, xx = np.nonzero(maxima)
    if not len(xx):
        return []
    order = np.argsort(best[yy, xx])[::-1]
    result: list[dict] = []
    for index in order:
        x, y = int(xx[index] + 15), int(yy[index] + 15)
        if any((x - item['x']) ** 2 + (y - item['y']) ** 2 < separation ** 2
               for item in result):
            continue
        result.append(dict(x=x, y=y, score=float(best[yy[index], xx[index]]),
                           scale=float(best_scale[yy[index], xx[index]]),
                           degrees=int(best_angle[yy[index], xx[index]])))
        if len(result) >= keep:
            break
    return result
