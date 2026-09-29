"""Development-only replay of presence gates on saved patch diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import DEFAULT_DATA, read_rows
from .metrics import score_scene


def replay(truth: list[dict], predictions: list[dict], details: list[dict],
           *, threshold: float | None = None, fraction: float | None = None) -> dict:
    reference = {row['Id']: row for row in truth}
    patches: dict[str, list[dict]] = {}
    for detail in details:
        patches.setdefault(detail['scene'], []).append(detail)
    scores = {}
    for original in predictions:
        scene = original['Id']
        row = dict(original)
        items = patches[scene]
        if fraction is None:
            allowed = {item['patch'] for item in items if item['score'] >= threshold}
        else:
            limit = round(len(items) * fraction)
            allowed = {item['patch'] for item in sorted(items, key=lambda x: -x['score'])[:limit]}
        for item in items:
            patch = item['patch']
            if item['hough_assigned'] or patch in allowed:
                # Reuse the saved coordinates. This tests only the presence gate.
                row[patch] = f"({item['x']:.2f}, {item['y']:.2f}, 0)"
            else:
                row[patch] = '-1'
        scores[scene] = score_scene(reference[scene], row)
    return dict(mean={key: float(np.mean([s[key] for s in scores.values()]))
                      for key in ('score', 'presence', 'localization',
                                  'geometric_recovery', 'identification')},
                scenes={name: round(value['score'], 4) for name, value in scores.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=Path('outputs/hough-final-train'))
    args = parser.parse_args()
    _, truth = read_rows(args.data / 'train_ground_truth.csv')
    _, predictions = read_rows(args.output / 'evaluate-predictions.csv')
    details = json.loads((args.output / 'evaluate-patches.json').read_text(encoding='utf-8'))
    experiments = []
    for threshold in (.5, .55, .6, .65, .7, .75, .8, .85):
        experiments.append(dict(gate='threshold', value=threshold,
                                **replay(truth, predictions, details, threshold=threshold)))
    for fraction in (.45, .5, .55, .6, .65, .7):
        experiments.append(dict(gate='fraction', value=fraction,
                                **replay(truth, predictions, details, fraction=fraction)))
    print(json.dumps(experiments, indent=2))


if __name__ == '__main__':
    main()
