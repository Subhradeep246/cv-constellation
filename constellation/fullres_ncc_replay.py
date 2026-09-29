"""Training-only replay of full-resolution candidates over frozen predictions.

This diagnostic never uses labels to produce inference predictions. Labels are
read only after each candidate decision to score the controlled ablation.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from .data import DEFAULT_DATA, read_rows
from .metrics import score_scene


def replay(base: dict, details: dict, evidence: dict, threshold: float,
           *, include_assigned: bool = False) -> tuple[dict, int]:
    row = copy.deepcopy(base)
    changes = 0
    scene = row['Id']
    for i in range(1, int(row['n_patches']) + 1):
        key = (scene, i)
        item = evidence.get(key)
        prior = details.get(key)
        if not item or not item['candidates'] or not prior:
            continue
        if prior['hough_assigned'] and not include_assigned:
            continue
        if item['relative_gap'] < threshold:
            continue
        top = item['candidates'][0]
        field = f'patch_{i:02d}'
        member = 1 if prior['hough_assigned'] else 0
        proposed = f"({top['x']:.2f}, {top['y']:.2f}, {member})"
        if row[field] != proposed:
            row[field] = proposed
            changes += 1
    return row, changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path,
                        default=Path('outputs/fullres-ncc-audit/hp20-train.json'))
    parser.add_argument('--baseline', type=Path,
                        default=Path('outputs/hough-broadstar-train/evaluate-predictions.csv'))
    parser.add_argument('--details', type=Path,
                        default=Path('outputs/hough-broadstar-train/evaluate-patches.json'))
    args = parser.parse_args()
    report = json.loads(args.evidence.read_text(encoding='utf-8'))
    evidence = {(e['scene'], e['patch']): e for e in report['entries']}
    details = json.loads(args.details.read_text(encoding='utf-8'))
    details = {(d['scene'], int(d['patch'][-2:])): d for d in details}
    _, baseline = read_rows(args.baseline)
    _, truth = read_rows(DEFAULT_DATA / 'train_ground_truth.csv')
    truth = {r['Id']: r for r in truth}
    thresholds = (.05, .1, .16, .25, .4, .6)
    per_threshold: dict[float, dict[str, float]] = {}
    for threshold in thresholds:
        rows = []
        for base in baseline:
            if base['Id'] not in truth:
                continue
            after, changes = replay(base, details, evidence, threshold)
            before_score = score_scene(truth[base['Id']], base)['score']
            after_score = score_scene(truth[base['Id']], after)['score']
            rows.append((base['Id'], before_score, after_score, changes))
        per_threshold[threshold] = {r[0]: r[2] for r in rows}
        print(json.dumps({
            'threshold': threshold,
            'mean_before': float(np.mean([r[1] for r in rows])),
            'mean_after': float(np.mean([r[2] for r in rows])),
            'scenes': [{'scene': r[0], 'before': r[1], 'after': r[2],
                        'changed_cells': r[3]} for r in rows]
        }), flush=True)
    held_out = []
    for scene in sorted(truth):
        others = [s for s in truth if s != scene]
        chosen = max(thresholds,
                     key=lambda t: (np.mean([per_threshold[t][s] for s in others]), t))
        held_out.append(dict(scene=scene, chosen_threshold=chosen,
                             score=per_threshold[chosen][scene]))
    print(json.dumps({'leave_one_scene_out': held_out,
                      'mean': float(np.mean([e['score'] for e in held_out]))}),
          flush=True)


if __name__ == '__main__':
    main()
