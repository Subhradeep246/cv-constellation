"""Checkpointed labelled-data audit of masked-NCC candidate recall.

The labels are used exclusively here to measure candidate quality. This
module is not imported by submission inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .data import DEFAULT_DATA, discover_scenes, read_gray, read_rows, row_points
from .fullres_ncc import masked_ncc_candidates
from .localize import Localizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', help='One labelled scene; default is all')
    parser.add_argument('--angle-step', type=int, default=20)
    parser.add_argument('--keep', type=int, default=30)
    parser.add_argument('--output', type=Path,
                        default=Path('outputs/fullres-ncc-audit/hp20-train.json'))
    args = parser.parse_args()
    scenes = {s.scene_id: s for s in discover_scenes(DEFAULT_DATA / 'train')}
    _, truth_rows = read_rows(DEFAULT_DATA / 'train_ground_truth.csv')
    if args.scene and args.scene not in scenes:
        parser.error('Unknown labelled scene')
    if args.output.exists():
        report = json.loads(args.output.read_text(encoding='utf-8'))
        if report['angle_step'] != args.angle_step or report['keep'] != args.keep:
            parser.error('Output contains results for another configuration')
    else:
        report = {'angle_step': args.angle_step, 'keep': args.keep,
                  'channel': 'DoG(0.7,3)', 'entries': []}
    done = {(entry['scene'], entry['patch']) for entry in report['entries']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for truth in truth_rows:
        scene_id = truth['Id']
        if args.scene and args.scene != scene_id:
            continue
        scene = scenes[scene_id]
        targets = row_points(truth)
        if all((scene_id, i) in done for i in range(1, len(scene.patches) + 1)):
            continue
        sky = read_gray(scene.image_path)
        filtered_sky = Localizer.filtered(sky, 3.0, .7)
        for i, patch_path in enumerate(scene.patches, 1):
            if (scene_id, i) in done:
                continue
            patch = Localizer.filtered(read_gray(patch_path), 3.0, .7)
            started = time.perf_counter()
            candidates = masked_ncc_candidates(filtered_sky, patch,
                                               angle_step=args.angle_step,
                                               keep=args.keep)
            target = targets[i - 1]
            rank = next((j + 1 for j, candidate in enumerate(candidates)
                         if target is not None and
                         np.hypot(candidate['x'] - target[0],
                                  candidate['y'] - target[1]) <= 12), None)
            first = candidates[0]['score'] if candidates else 0.0
            second = candidates[1]['score'] if len(candidates) > 1 else 0.0
            entry = {
                'scene': scene_id, 'patch': i, 'truth_present': target is not None,
                'truth_figure': bool(target[2]) if target is not None else False,
                'truth_xy': list(target[:2]) if target is not None else None,
                'nearest_rank_12px': rank,
                'top1_error_px': float(np.hypot(candidates[0]['x'] - target[0],
                                                 candidates[0]['y'] - target[1]))
                                 if target is not None and candidates else None,
                'score1': first, 'score2': second,
                'relative_gap': (first - second) / max(1.0 - first, .01),
                'seconds': time.perf_counter() - started,
                'candidates': candidates,
            }
            report['entries'].append(entry)
            args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(f'{scene_id} {i:02d}/{len(scene.patches)} '
                  f'present={target is not None} rank={rank} '
                  f'gap={entry["relative_gap"]:.3f} '
                  f'{entry["seconds"]:.1f}s', flush=True)
    entries = report['entries']
    present = [e for e in entries if e['truth_present']]
    absent = [e for e in entries if not e['truth_present']]
    summary = {
        'patches': len(entries), 'present': len(present), 'absent': len(absent),
        'present_top1': sum(e['nearest_rank_12px'] == 1 for e in present),
        'present_top30': sum(e['nearest_rank_12px'] is not None for e in present),
        'figure_top1': sum(e['nearest_rank_12px'] == 1 for e in present if e['truth_figure']),
        'figure_total': sum(e['truth_figure'] for e in present),
        'median_gap_present': float(np.median([e['relative_gap'] for e in present])) if present else None,
        'median_gap_absent': float(np.median([e['relative_gap'] for e in absent])) if absent else None,
        'seconds_total': sum(e['seconds'] for e in entries),
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
