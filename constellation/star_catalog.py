"""Broad, locally contrastive star centres from a supplied sky image."""
from __future__ import annotations

import cv2
import numpy as np


def broad_star_centres(sky: np.ndarray, *, count: int = 300,
                       fine_sigma: float = 4., broad_sigma: float = 12.,
                       nms_size: int = 11) -> np.ndarray:
    """Return the strongest broad DoG peaks, without labels or reference names."""
    if count < 1 or fine_sigma <= 0 or broad_sigma <= fine_sigma or nms_size < 1:
        raise ValueError('Invalid broad-star detector parameters')
    array = sky.astype(np.float32)
    fine = cv2.GaussianBlur(array, (0, 0), fine_sigma)
    broad = cv2.GaussianBlur(array, (0, 0), broad_sigma)
    response = fine - broad
    maxima = ((response >= cv2.dilate(response,
                                     np.ones((nms_size, nms_size), np.uint8))) &
              (response > 0))
    margin = max(22, nms_size)
    maxima[:margin] = maxima[-margin:] = False
    maxima[:, :margin] = maxima[:, -margin:] = False
    y, x = np.nonzero(maxima)
    if len(x) > count:
        chosen = np.argpartition(response[y, x], -count)[-count:]
        x, y = x[chosen], y[chosen]
    # Stable rank order is required by likelihood-ratio star evidence.
    order = np.argsort(response[y, x])[::-1]
    x, y = x[order], y[order]
    return np.column_stack((x, y)).astype(float)
