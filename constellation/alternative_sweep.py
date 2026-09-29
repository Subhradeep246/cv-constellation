"""Retrospective score sweep over photometric-vs-geometry alternative cost."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import DEFAULT_DATA, read_rows, row_points
from .identify import load_patterns, rank_patterns_with_alternatives
from .metrics import score_scene


def run(data: Path, diagnostics_path: Path, penalties: tuple[float, ...]) -> dict:
    _, truth = read_rows(data / 'train_ground_truth.csv')
    truth_by_id = {row['Id']: row for row in truth}
    patches = json.loads(diagnostics_path.read_text(encoding='utf-8'))
    patterns = load_patterns(data / 'patterns')
    scenes = sorted({p['scene'] for p in patches})
    all_by_scene = {scene: [p for p in patches if p['scene'] == scene] for scene in scenes}
    by_scene = {scene: sorted([p for p in patches if p['scene'] == scene and p['present']],
                              key=lambda p: p['score'], reverse=True) for scene in scenes}
    results = []
    for scene in scenes:
        for penalty in penalties:
            group = by_scene[scene]
            candidates = [p['alternatives'] or [dict(x=p['x'], y=p['y'], score=p['score'])]
                          for p in group]
            ranking = rank_patterns_with_alternatives(patterns, candidates,
                                                       alternative_penalty=penalty)
            gt = truth_by_id[scene]
            pred = {key: '-1' for key in gt}
            pred.update(Id=scene, n_patches=gt['n_patches'],
                        constellation=ranking[0]['name'] if ranking else 'unknown')
            assigned = {}
            if ranking:
                chosen = ranking[0]
                for index, value in chosen.get('point_locations', {}).items():
                    assigned[int(index)] = value
            for i, item in enumerate(group):
                xy = assigned.get(i, {'x': item['x'], 'y': item['y']})
                pred[item['patch']] = str((xy['x'], xy['y'], 0))
            for item in all_by_scene[scene]:
                if not item['present']:
                    pred[item['patch']] = '-1'
            metric = score_scene(gt, pred)
            results.append(dict(scene=scene, penalty=penalty, predicted=pred['constellation'],
                                correct=pred['constellation'] == gt['constellation'],
                                score=metric['score'], localization=metric['localization'],
                                geometry=metric['geometric_recovery'],
                                presence=metric['presence'],
                                matched_nodes=ranking[0]['matched_nodes'] if ranking else 0))
    summary = {}
    for penalty in penalties:
        values = [r for r in results if r['penalty'] == penalty]
        summary[str(penalty)] = dict(
            mean_score=float(np.mean([r['score'] for r in values])),
            name_accuracy=float(np.mean([r['correct'] for r in values])),
            names={r['scene']: r['predicted'] for r in values},
            per_scene={r['scene']: r['score'] for r in values})
    return dict(note='Retrospective hyperparameter sweep on all three development scenes; not held out.',
                penalties=list(penalties), summary=summary, per_scene=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostics', type=Path)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=Path('outputs/alternative-sweep.json'))
    parser.add_argument('--penalties', type=float, nargs='+', default=[0, 2, 5, 10, 20, 40])
    args = parser.parse_args()
    report = run(args.data, args.diagnostics, tuple(args.penalties))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
