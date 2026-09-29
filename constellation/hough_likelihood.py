"""Optional node-normalized evidence score for global figure hypotheses.

Weights are approximate log-likelihood ratios motivated by measured chance
star coverage and patch issue rates; this remains experimental until tested
on complete scenes. No validation names or labels are used.
"""
from __future__ import annotations

import numpy as np


def likelihood_score(pose: dict, node_count: int,
                     patch_gaps: list[float], *,
                     clear_gap: float = .16,
                     budget_prior: float = 0.) -> float:
    """Score supported and unsupported nodes instead of counting inliers."""
    in_frame = int(pose['in_frame_nodes'])
    chosen = pose['chosen']
    total = 0.0
    for item in chosen:
        gap = patch_gaps[item['patch'] - 1]
        weight = 2.0 if gap >= clear_gap else 1.0
        total += weight - .5 * (item['distance'] / 10.0) ** 2
    total -= .9 * max(0, in_frame - len(chosen))
    for distance, rank in zip(pose['star_distances'], pose['star_ranks']):
        if distance > 20:
            total -= 3.0
        elif rank <= 40:
            total += 3.9
        elif rank <= 120:
            total += 3.7
        else:
            total += 2.0
    total += 4.0 * pose['copy_nodes']
    total -= max(0, node_count - in_frame)
    # The patch-count prior proved less stable than node evidence on unseen
    # scenes, so it is attenuated rather than dictating the name.
    total += .25 * budget_prior
    return float(total)
