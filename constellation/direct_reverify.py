"""Training-only probe: score direct candidates with the existing verifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import DEFAULT_DATA, discover_scenes, read_gray
from .localize import Localizer, LocalizerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate_file', type=Path)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    record = json.loads(args.candidate_file.read_text(encoding='utf-8'))
    scene = next(s for s in discover_scenes(args.data / 'train')
                 if s.scene_id == record['scene'])
    loc = Localizer(LocalizerConfig(refine_candidates=30))
    sky = loc.filtered(read_gray(scene.image_path))
    patch = read_gray(scene.patches[record['patch'] - 1])
    proposals = np.asarray([[item['x'], item['y'], item['scale']]
                            for item in record['candidates']], dtype=np.float32)
    match = loc.verify(sky, patch, proposals)
    gt = record['truth']
    rank = next((i + 1 for i, item in enumerate(match.alternatives)
                 if gt and np.hypot(item['x'] - gt[0], item['y'] - gt[1]) < 12), None)
    print(json.dumps(dict(scene=record['scene'], patch=record['patch'],
                          original_candidate_count=len(record['candidates']),
                          reverified_count=len(match.alternatives), true_rank=rank,
                          best_score=match.score,
                          top=match.alternatives[:5]), indent=2))


if __name__ == '__main__':
    main()
