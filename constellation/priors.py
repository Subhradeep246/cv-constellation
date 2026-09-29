"""Offline test of a bounded patch-count / in-frame-figure-size prior."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import DEFAULT_DATA, discover_scenes, read_gray, read_rows, row_points
from .identify import load_patterns, rank_patterns


def _in_frame_nodes(record: dict, nodes: np.ndarray, width: int, height: int) -> int:
    matrix = np.asarray(record['matrix'], dtype=float)
    translation = np.asarray(record['translation'], dtype=float)
    projected = nodes @ matrix.T + translation
    return int(((projected[:, 0] >= 0) & (projected[:, 0] < width) &
                (projected[:, 1] >= 0) & (projected[:, 1] < height)).sum())


def analyze(data: Path, predictions: Path) -> dict:
    _, truth = read_rows(data / 'train_ground_truth.csv')
    _, pred = read_rows(predictions)
    pred = {row['Id']: row for row in pred}
    scenes = {scene.scene_id: scene for scene in discover_scenes(data / 'train')}
    patterns = load_patterns(data / 'patterns')
    pattern_map = {pattern.name: pattern for pattern in patterns}
    rows = []
    for gt in truth:
        row = pred[gt['Id']]
        points = np.array([p[:2] for p in row_points(row) if p], dtype=float).reshape(-1, 2)
        ranking = rank_patterns(patterns, points)
        hypotheses = []
        for item in ranking:
            inside = _in_frame_nodes(item, pattern_map[item['name']].nodes,
                                     *read_gray(scenes[gt['Id']].image_path).shape[::-1])
            if not inside:
                prior = -20.0
            else:
                ratio = int(gt['n_patches']) / inside
                prior = max(-20.0, -.5 * (np.log(ratio / 2.97) / .35) ** 2)
            hypotheses.append({**item, 'in_frame_nodes': inside,
                               'patches_per_node': None if not inside else int(gt['n_patches']) / inside,
                               'count_log_prior': prior})
        for beta in (0, .05, .1, .2, .35, .5, 1.0, 2.0):
            for width in (.35, .7, 1.05):
                # Recompute prior width; beta 0 is the unchanged geometry rank.
                def adjusted(h):
                    raw = -20.0 if not h['in_frame_nodes'] else max(
                        -20.0, -.5 * (np.log(h['patches_per_node'] / 2.97) / width) ** 2)
                    return h['score'] + beta * raw
                top = max(hypotheses, key=adjusted) if hypotheses else None
                rows.append(dict(scene=gt['Id'], true_name=gt['constellation'],
                                 beta=beta, width=width,
                                 predicted_name=top['name'] if top else 'unknown',
                                 correct=bool(top and top['name'] == gt['constellation']),
                                 top_score=None if not top else top['score'],
                                 prior=None if not top else adjusted(top)-top['score'],
                                 top3=[dict(name=h['name'], score=h['score'],
                                            in_frame_nodes=h['in_frame_nodes'],
                                            patches_per_node=h['patches_per_node'])
                                       for h in sorted(hypotheses,key=adjusted,reverse=True)[:3]]))
    settings = {}
    for beta in (0, .05, .1, .2, .35, .5, 1.0, 2.0):
        for width in (.35, .7, 1.05):
            subset = [r for r in rows if r['beta'] == beta and r['width'] == width]
            settings[f'beta={beta},width={width}'] = dict(
                accuracy=float(np.mean([r['correct'] for r in subset])),
                predictions={r['scene']:r['predicted_name'] for r in subset})
    return dict(note='Retrospective three-scene diagnostic; coordinates and labels were used during development. '
                     'A train-selected prior is not a generalization guarantee. '
                     'Geometry scores, extent, and all predictions are held fixed.',
                settings=settings, per_scene=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=Path('outputs/patch-budget-prior.json'))
    args = parser.parse_args()
    report = analyze(args.data, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['settings'], indent=2))


if __name__ == '__main__':
    main()
