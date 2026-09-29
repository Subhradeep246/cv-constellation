"""Training-only check of whether figure stars are unusually bright in the sky."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .data import DEFAULT_DATA, discover_scenes, read_gray, read_rows, row_points
from .localize import Localizer, LocalizerConfig


def main() -> None:
    _, rows = read_rows(DEFAULT_DATA / 'train_ground_truth.csv')
    scenes = {s.scene_id: s for s in discover_scenes(DEFAULT_DATA / 'train')}
    loc = Localizer(LocalizerConfig(max_sources=100000))
    for row in rows:
        sky = read_gray(scenes[row['Id']].image_path)
        sources = loc.detect(sky)
        tree = cKDTree(sources)
        details = []
        for i, gt in enumerate(row_points(row), 1):
            if gt is None or not gt[2]:
                continue
            d, index = tree.query(gt[:2])
            rank = len(sources) - int(index) if d < 4 else None
            details.append((i, round(float(d), 1), rank))
        print(row['Id'], 'figure source distance/rank:', details, flush=True)


if __name__ == '__main__':
    cv2.setNumThreads(8)
    main()
