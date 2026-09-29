"""Training-only oracle probe for seeing whether sky PSF adaptation helps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .data import DEFAULT_DATA, discover_scenes, read_gray, read_rows, row_points
from .localize import Localizer, LocalizerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=Path('outputs/blur-oracle.json'))
    args = parser.parse_args()
    cv2.setNumThreads(8)
    loc = Localizer(LocalizerConfig())
    _, rows = read_rows(args.data / 'train_ground_truth.csv')
    scenes = {scene.scene_id: scene for scene in discover_scenes(args.data / 'train')}
    records = []
    for row in rows:
        scene = scenes[row['Id']]
        sky = read_gray(scene.image_path)
        patches = [(path, gt) for path, gt in zip(scene.patches, row_points(row)) if gt]
        for sigma in (0, 1, 2, 3, 4):
            source = (sky if sigma == 0 else cv2.GaussianBlur(sky, (0, 0), sigma))
            filtered = loc.filtered(source)
            for path, gt in patches:
                proposals = np.array([[gt[0], gt[1], scale] for scale in loc.config.scales],
                                     dtype=np.float32)
                match = loc.verify(filtered, read_gray(path), proposals)
                records.append(dict(scene=scene.scene_id, patch=path.stem, sigma=sigma,
                                    score=match.score,
                                    error_px=float(np.hypot(match.x - gt[0], match.y - gt[1]))))
        print(scene.scene_id, 'complete', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
