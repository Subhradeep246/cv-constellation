"""Pixel-only verification of cached full-resolution patch candidate sites."""
from __future__ import annotations

import numpy as np

from .fullres_ncc import _template


def local_ncc(sky: np.ndarray, patch: np.ndarray, candidate: dict) -> float:
    """NCC of a transformed query and the sky at one candidate centre."""
    x, y = int(candidate['x']), int(candidate['y'])
    if x < 15 or y < 15 or x + 16 > sky.shape[1] or y + 16 > sky.shape[0]:
        return -1.
    source = sky[y-15:y+16, x-15:x+16].astype(np.float32)
    query, mask = _template(patch.astype(np.float32),
                            float(candidate['scale']), float(candidate['degrees']))
    count = float(mask.sum())
    source = (source - float((source * mask).sum() / count)) * mask
    query = (query - float((query * mask).sum() / count)) * mask
    denominator = float(np.linalg.norm(source) * np.linalg.norm(query))
    return float((source * query).sum() / denominator) if denominator > 1e-6 else 0.


def rank_candidates(sky: np.ndarray, patch: np.ndarray,
                    candidates: list[dict], raw_weight: float,
                    top_k: int = 30) -> list[dict]:
    """Rerank cached DoG sites with independent raw-image NCC evidence."""
    if raw_weight < 0 or top_k < 1:
        raise ValueError('raw_weight must be nonnegative and top_k positive')
    scored = []
    for candidate in candidates[:top_k]:
        item = dict(candidate)
        item['raw'] = local_ncc(sky, patch, item)
        item['combined'] = float(item['score'] + raw_weight * item['raw'])
        scored.append(item)
    return sorted(scored, key=lambda item: item['combined'], reverse=True)


def select_candidate(ranked: list[dict], raw_weight: float,
                     gap_threshold: float) -> dict | None:
    """Accept a distinct best site with the same normalized-gap convention."""
    if len(ranked) < 2:
        return None
    first, second = ranked[0]['combined'], ranked[1]['combined']
    gap = (first - second) / max(1. + raw_weight - first, .01)
    return ranked[0] if gap >= gap_threshold else None
