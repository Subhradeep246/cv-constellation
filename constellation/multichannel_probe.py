"""Training-only rerank of cached NCC sites using independent image channels.

Ground truth is loaded only after each candidate's image scores are computed.
This diagnostic never contributes labelled data to submission inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import DEFAULT_DATA, discover_scenes, read_gray, read_rows, row_points
from .localize import Localizer
from .multichannel import local_ncc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('outputs/multichannel-probe/report.json'))
    args = parser.parse_args()
    report = json.loads(Path('outputs/fullres-ncc-audit/hp20-train.json').read_text())
    audit = {(e['scene'], e['patch']): e for e in report['entries']}
    _, truth_rows = read_rows(DEFAULT_DATA / 'train_ground_truth.csv')
    scenes = {s.scene_id: s for s in discover_scenes(DEFAULT_DATA / 'train')}
    entries = []
    for row in truth_rows:
        scene = scenes[row['Id']]
        sky = read_gray(scene.image_path)
        medium_sky = Localizer.filtered(sky, 12., 1.5)
        targets = row_points(row)
        for i, patch_path in enumerate(scene.patches, 1):
            cached = audit[(scene.scene_id, i)]
            patch = read_gray(patch_path)
            medium_patch = Localizer.filtered(patch, 12., 1.5)
            candidates = []
            for candidate in cached['candidates']:
                item = dict(candidate)
                item['medium'] = local_ncc(medium_sky, medium_patch, candidate)
                item['raw'] = local_ncc(sky, patch, candidate)
                candidates.append(item)
            target = targets[i - 1]
            entries.append(dict(scene=scene.scene_id, patch=i,
                                present=target is not None,
                                truth_xy=list(target[:2]) if target else None,
                                candidates=candidates))
        print(f'{scene.scene_id}: {len(scene.patches)} patches rescored', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(entries), encoding='utf-8')
    for medium_weight, raw_weight in ((0,0),(.25,0),(.5,0),(1,0),
                                      (0,.25),(0,.5),(0,1),(.5,.5)):
        by_scene = {}
        for entry in entries:
            if not entry['present'] or not entry['candidates']:
                continue
            ranked = sorted(entry['candidates'],
                            key=lambda c: c['score'] + medium_weight*c['medium']
                            + raw_weight*c['raw'], reverse=True)
            x,y = entry['truth_xy']
            hit = np.hypot(ranked[0]['x']-x,ranked[0]['y']-y) <= 12
            count, total = by_scene.get(entry['scene'], (0,0))
            by_scene[entry['scene']] = (count+int(hit), total+1)
        print(json.dumps(dict(medium_weight=medium_weight, raw_weight=raw_weight,
                              top1=by_scene)), flush=True)
    print('Results:', args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
