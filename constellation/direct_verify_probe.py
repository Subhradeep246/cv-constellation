"""Training-only audit of full-resolution verification for saved direct hits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import DEFAULT_DATA, discover_scenes, read_gray, read_rows, row_points
from .localize import Localizer, LocalizerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--outputs', type=Path, default=Path('outputs'))
    parser.add_argument('--scene', choices=('pisces', 'scorpius', 'taurus'))
    parser.add_argument('--top', type=int, default=30)
    args = parser.parse_args()
    scenes = {item.scene_id: item for item in discover_scenes(args.data / 'train')}
    _, truth = read_rows(args.data / 'train_ground_truth.csv')
    truth = {row['Id']: row_points(row) for row in truth}
    diagnostic = json.loads((args.outputs / 'hough-final-train/evaluate-patches.json').read_text(encoding='utf-8'))
    original = {(r['scene'], r['patch']): r for r in diagnostic}
    localizer = Localizer(LocalizerConfig(refine_candidates=args.top))
    for name, scene in scenes.items():
        if args.scene and name != args.scene:
            continue
        sky = localizer.filtered(read_gray(scene.image_path))
        results = []
        for i, path in enumerate(scene.patches, 1):
            candidates = json.loads((args.outputs / f'direct-{name}-all' /
                                     f'patch_{i:02d}.json').read_text(encoding='utf-8'))['candidates'][:args.top]
            proposals = np.array([[c['x'], c['y'], c['scale']] for c in candidates],
                                 dtype=np.float32)
            match = localizer.verify(sky, read_gray(path), proposals)
            gt = truth[name][i-1]
            before = original[(name, f'patch_{i:02d}')]
            record = dict(patch=i, truth_present=gt is not None,
                          direct_score=round(match.score, 3),
                          direct_present=match.present,
                          baseline_score=round(before['score'], 3),
                          baseline_correct=bool(gt is not None and before['present'] and
                                                before['error_px'] <= 12),
                          direct_correct=bool(gt is not None and match.present and
                                              np.hypot(match.x-gt[0], match.y-gt[1]) <= 12))
            results.append(record)
        print(json.dumps(dict(scene=name, present_truth=sum(r['truth_present'] for r in results),
                              baseline_correct=sum(r['baseline_correct'] for r in results),
                              direct_correct=sum(r['direct_correct'] for r in results),
                              direct_false_positive=sum(r['direct_present'] and not r['truth_present']
                                                        for r in results),
                              changed=[r for r in results
                                       if r['baseline_correct'] != r['direct_correct']]), indent=2),
              flush=True)


if __name__ == '__main__':
    main()
