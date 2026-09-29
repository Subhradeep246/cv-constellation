"""Build a label-free multichannel candidate from a frozen Hough submission.

This post-Hough experiment reuses content-checked full-resolution NCC caches.
It never loads training images or labels and leaves the scored CSV untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .candidate_cache import get_fullres_candidates
from .data import (DEFAULT_DATA, discover_scenes, parse_cell, read_gray,
                   read_rows, validate_submission, write_rows)
from .identify import load_patterns
from .multichannel import rank_candidates, select_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--baseline', type=Path,
                        default=Path('outputs/hough-broadstar-validation/submission.csv'))
    parser.add_argument('--details', type=Path,
                        default=Path('outputs/hough-broadstar-validation/predict-patches.json'))
    parser.add_argument('--cache-dir', type=Path,
                        default=Path('outputs/fullres-cache-shared'))
    parser.add_argument('--raw-weight', type=float, default=.5)
    parser.add_argument('--gap', type=float, default=.16)
    parser.add_argument('--output', type=Path,
                        default=Path('outputs/hough-fullres-multichannel-validation'))
    args = parser.parse_args()
    if args.raw_weight < 0 or args.gap < 0:
        parser.error('Weights and gap must be nonnegative')
    fields, sample = read_rows(args.data / 'sample_submission.csv')
    base_fields, baseline = read_rows(args.baseline)
    if fields != base_fields:
        raise ValueError('Baseline submission header differs from sample')
    details = json.loads(args.details.read_text(encoding='utf-8'))
    detail_map = {(d['scene'], d['patch']): d for d in details}
    scenes = {s.scene_id: s for s in discover_scenes(args.data / 'validation')}
    base_map = {row['Id']: row for row in baseline}
    predictions = []
    image_sizes = {}
    report = []
    for original in sample:
        scene_id = original['Id']
        scene = scenes[scene_id]
        sky = read_gray(scene.image_path)
        patches = [read_gray(p) for p in scene.patches]
        candidate_groups = get_fullres_candidates(
            sky, patches, args.cache_dir / f'validation-{scene_id}.json',
            log=lambda _: None)
        row = dict(base_map[scene_id])
        accepted = changed = promoted = 0
        for i, (path, patch, candidates) in enumerate(
                zip(scene.patches, patches, candidate_groups)):
            field = path.stem
            if detail_map[(scene_id, field)]['hough_assigned']:
                continue
            ranked = rank_candidates(sky, patch, candidates, args.raw_weight)
            chosen = select_candidate(ranked, args.raw_weight, args.gap)
            if chosen is None:
                continue
            previous = parse_cell(row[field])
            member = previous[2] if previous else 0
            proposed = f"({chosen['x']:.2f}, {chosen['y']:.2f}, {member})"
            if row[field] != proposed:
                changed += 1
                promoted += previous is None
            row[field] = proposed
            accepted += 1
        predictions.append(row)
        image_sizes[scene_id] = (sky.shape[1], sky.shape[0])
        report.append(dict(scene=scene_id, accepted=accepted,
                           changed=changed, promoted=promoted))
        print(f'{scene_id}: accepted {accepted}, changed {changed}, '
              f'promoted {promoted}', flush=True)
    patterns = load_patterns(args.data / 'patterns')
    validate_submission(predictions, sample, {p.name for p in patterns}, image_sizes)
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / 'submission.csv', fields, predictions)
    (args.output / 'report.json').write_text(json.dumps({
        'baseline': str(args.baseline), 'raw_weight': args.raw_weight,
        'gap': args.gap, 'scenes': report}, indent=2), encoding='utf-8')
    print('Saved', (args.output / 'submission.csv').resolve(), flush=True)


if __name__ == '__main__':
    main()
