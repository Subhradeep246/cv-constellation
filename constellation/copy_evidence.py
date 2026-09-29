"""Training-only audit of displacement-consistent copy-move evidence."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .data import DEFAULT_DATA, read_rows, row_points


def clustered_pairs(pairs: list[dict], *, min_ncc: float = .9,
                    bin_size: int = 4, min_support: int = 5):
    buckets = Counter()
    included = []
    for item in pairs:
        if item['annulus_ncc'] < min_ncc:
            continue
        a, b = np.asarray(item['a']), np.asarray(item['b'])
        delta = b - a
        if delta[0] < 0 or (delta[0] == 0 and delta[1] < 0):
            a, b = b, a
            delta = -delta
        key = (int(round(delta[0] / bin_size)),
               int(round(delta[1] / bin_size)))
        buckets[key] += 1
        included.append((key, a, b, item['annulus_ncc']))
    good = {key for key, value in buckets.items() if value >= min_support}
    selected = [dict(shift=key, a=a.tolist(), b=b.tolist(), ncc=float(ncc))
                for key, a, b, ncc in included if key in good]
    return buckets, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    records = json.loads(args.report.read_text(encoding='utf-8'))
    _, truth = read_rows(args.data / 'train_ground_truth.csv')
    true_rows = {row['Id']: row for row in truth}
    for record in records:
        if record['scene'] not in true_rows:
            continue
        buckets, selected = clustered_pairs(record['pairs'])
        anchors = np.asarray([point for pair in selected for point in (pair['a'], pair['b'])],
                             dtype=float).reshape(-1, 2)
        figure, off = [], []
        for i, gt in enumerate(row_points(true_rows[record['scene']]), 1):
            if gt is None:
                continue
            distance = float(np.linalg.norm(anchors - gt[:2], axis=1).min()) if len(anchors) else None
            (figure if gt[2] else off).append(dict(patch=i, distance=distance))
        print(json.dumps(dict(scene=record['scene'], selected_pairs=len(selected),
                              shifts=[dict(dx=key[0]*4, dy=key[1]*4, count=count)
                                      for key, count in buckets.most_common(15) if count >= 5],
                              figure_close10=sum(item['distance'] is not None and item['distance'] < 10
                                                 for item in figure),
                              off_close10=sum(item['distance'] is not None and item['distance'] < 10
                                              for item in off),
                              figure=figure, off=off), indent=2), flush=True)


if __name__ == '__main__':
    main()
