"""Training-only replay of saved Hough identification and assignment variants."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import DEFAULT_DATA, read_rows
from .metrics import score_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--outputs', type=Path, default=Path('outputs'))
    parser.add_argument('--margin', type=float, default=1.2)
    args = parser.parse_args()
    _, truth = read_rows(args.data / 'train_ground_truth.csv')
    _, baseline = read_rows(args.outputs / 'baseline-fast/evaluate-predictions.csv')
    baseline = {row['Id']: row for row in baseline}
    results = []
    for row in truth:
        scene = row['Id']
        result = json.loads((args.outputs / f'hough-{scene}-refined.json').read_text(encoding='utf-8'))
        ranking = result['per_pattern']
        gap = ranking[0]['combined'] - ranking[1]['combined']
        use = gap >= args.margin
        record = dict(scene=scene, hough_name=ranking[0]['pattern'],
                      hough_margin=gap, used_hough=use, policies={})
        for policy in ('name_only', 'assign_existing', 'assign_all'):
            pred = dict(baseline[scene])
            if use:
                pred['constellation'] = ranking[0]['pattern']
                if policy != 'name_only':
                    for item in ranking[0]['chosen']:
                        patch = f"patch_{item['patch']:02d}"
                        if policy == 'assign_existing' and pred[patch] == '-1':
                            continue
                        pred[patch] = f"({item['x']}, {item['y']}, 1)"
            record['policies'][policy] = score_scene(row, pred)
        results.append(record)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
