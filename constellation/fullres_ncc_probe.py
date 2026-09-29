"""Training-only diagnostic for the independent masked-NCC candidate stream."""
from __future__ import annotations

import argparse
import json
import time

from .data import DEFAULT_DATA, discover_scenes, read_gray, read_rows, row_points
from .fullres_ncc import masked_ncc_candidates
from .localize import Localizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--patch', type=int, required=True)
    parser.add_argument('--angle-step', type=int, default=10)
    parser.add_argument('--keep', type=int, default=30)
    parser.add_argument('--channel', choices=('hp', 'bp', 'raw'), default='hp')
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    scenes = {s.scene_id: s for s in discover_scenes(DEFAULT_DATA / 'train')}
    if args.scene not in scenes:
        parser.error('Unknown training scene')
    scene = scenes[args.scene]
    if not 1 <= args.patch <= len(scene.patches):
        parser.error('Patch index is out of range')
    _, truth_rows = read_rows(DEFAULT_DATA / 'train_ground_truth.csv')
    truth = next(r for r in truth_rows if r['Id'] == args.scene)
    target = row_points(truth)[args.patch - 1]
    sky = read_gray(scene.image_path)
    patch = read_gray(scene.patches[args.patch - 1])
    if args.channel == 'raw':
        filtered_sky, filtered_patch = sky, patch
    else:
        background = 3.0 if args.channel == 'hp' else 8.0
        foreground = .7 if args.channel == 'hp' else 1.0
        filtered_sky = Localizer.filtered(sky, background, foreground)
        filtered_patch = Localizer.filtered(patch, background, foreground)
    started = time.perf_counter()
    candidates = masked_ncc_candidates(filtered_sky, filtered_patch,
                                       angle_step=args.angle_step, keep=args.keep)
    result = {'scene': args.scene, 'patch': args.patch, 'channel': args.channel,
              'seconds': time.perf_counter() - started,
              'truth_present': target is not None,
              'truth_xy': list(target[:2]) if target is not None else None,
              'nearest_rank_12px': next((i + 1 for i, c in enumerate(candidates)
                                         if target is not None and
                                         (c['x'] - target[0]) ** 2 +
                                         (c['y'] - target[1]) ** 2 <= 12 ** 2), None),
              'candidates': candidates[:3] if args.summary else candidates}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
