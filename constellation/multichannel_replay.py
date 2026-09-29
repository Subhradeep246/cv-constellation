"""Training-only score replay for masked-NCC candidate reranking."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .data import DEFAULT_DATA, read_rows
from .metrics import score_scene


def rerank(candidates: list[dict], raw_weight: float, top_k: int) -> list[dict]:
    return sorted(candidates[:top_k],
                  key=lambda c: c['score'] + raw_weight * c['raw'], reverse=True)


def decision(candidates: list[dict], raw_weight: float,
             threshold: float, top_k: int) -> dict | None:
    ranked = rerank(candidates, raw_weight, top_k)
    if len(ranked) < 2:
        return None
    first = ranked[0]['score'] + raw_weight * ranked[0]['raw']
    second = ranked[1]['score'] + raw_weight * ranked[1]['raw']
    gap = (first - second) / max(1. + raw_weight - first, .01)
    return ranked[0] if gap >= threshold else None


def main() -> None:
    entries = json.loads(Path('outputs/multichannel-probe/report.json').read_text())
    evidence = {(e['scene'], e['patch']): e for e in entries}
    details = json.loads(Path('outputs/hough-broadstar-train/evaluate-patches.json').read_text())
    details = {(d['scene'], int(d['patch'][-2:])): d for d in details}
    _, baseline = read_rows(Path('outputs/hough-broadstar-train/evaluate-predictions.csv'))
    _, truth = read_rows(DEFAULT_DATA / 'train_ground_truth.csv')
    truth = {r['Id']: r for r in truth}
    grid = []
    for raw_weight in (0., .25, .5, 1., 2.):
        for top_k in (5, 30):
            for threshold in (.05, .1, .16, .25, .4, .6):
                scores = {}
                for original in baseline:
                    row = copy.deepcopy(original)
                    for i in range(1, int(row['n_patches']) + 1):
                        key = row['Id'], i
                        if details[key]['hough_assigned']:
                            continue
                        chosen = decision(evidence[key]['candidates'], raw_weight,
                                          threshold, top_k)
                        if chosen is None:
                            continue
                        row[f'patch_{i:02d}'] = f"({chosen['x']:.2f}, {chosen['y']:.2f}, 0)"
                    scores[row['Id']] = score_scene(truth[row['Id']], row)['score']
                grid.append(dict(weight=raw_weight, top_k=top_k,
                                 threshold=threshold, scores=scores,
                                 mean=float(np.mean(list(scores.values())))))
    for item in sorted(grid, key=lambda x: -x['mean'])[:20]:
        print(json.dumps(item), flush=True)
    print('BASELINE NCC', [x for x in grid if x['weight'] == 0 and
                          x['top_k'] == 30 and x['threshold'] == .16][0], flush=True)
    for held_out in truth:
        training = [s for s in truth if s != held_out]
        chosen = max(grid, key=lambda g: (np.mean([g['scores'][s] for s in training]),
                                          g['threshold'], -g['weight']))
        print('LOO', held_out, 'choice', chosen['weight'], chosen['top_k'],
              chosen['threshold'], 'score', chosen['scores'][held_out], flush=True)


if __name__ == '__main__':
    main()
