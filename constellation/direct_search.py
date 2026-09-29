"""Coarse full-sky rotated-template candidates from supplied scene pixels."""
from __future__ import annotations

import cv2
import numpy as np


def template_variants(patch: np.ndarray, inner_radius: float = 0):
    yy, xx = np.mgrid[:32, :32]
    radius2 = (xx - 15.5) ** 2 + (yy - 15.5) ** 2
    disk = ((radius2 <= 14 ** 2) & (radius2 >= inner_radius ** 2)).astype(np.uint8)
    for scale in (.75, 1.0, 1.25):
        size = int(2 * np.ceil(16 * scale) + 1)
        center = (size - 1) / 2
        for degrees in range(0, 360, 30):
            matrix = cv2.getRotationMatrix2D((15.5, 15.5), degrees, scale)
            matrix[:, 2] += center - 15.5
            image = cv2.warpAffine(patch, matrix, (size, size),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            mask = cv2.warpAffine(disk, matrix, (size, size),
                                  flags=cv2.INTER_NEAREST).astype(np.float32)
            yield scale, degrees, image, mask


def find_candidates(sky_downsampled: np.ndarray, patch_filtered: np.ndarray,
                    downsample: float = .375, keep: int = 30) -> list[dict]:
    """Find spatially distinct full-sky matches; no labels or filenames used."""
    candidates = []
    for scale, degrees, template, mask in template_variants(patch_filtered):
        if downsample < 1:
            template = cv2.resize(template, None, fx=downsample, fy=downsample,
                                  interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, template.shape[::-1], interpolation=cv2.INTER_AREA)
        response = cv2.matchTemplate(sky_downsampled, template,
                                     cv2.TM_CCORR_NORMED, mask=mask)
        response = np.nan_to_num(response, nan=-1, posinf=-1, neginf=-1)
        count = min(10, response.size)
        top = np.argpartition(response.ravel(), -count)[-count:]
        for index in top:
            y, x = np.unravel_index(index, response.shape)
            candidates.append(dict(x=int(round((x + (template.shape[1] - 1) / 2)
                                               / downsample)),
                                   y=int(round((y + (template.shape[0] - 1) / 2)
                                               / downsample)),
                                   score=float(response[y, x]), scale=scale,
                                   degrees=degrees))
    candidates.sort(key=lambda item: -item['score'])
    distinct = []
    for item in candidates:
        if all(np.hypot(item['x'] - other['x'], item['y'] - other['y']) >= 8
               for other in distinct):
            distinct.append(item)
            if len(distinct) >= keep:
                break
    return distinct
