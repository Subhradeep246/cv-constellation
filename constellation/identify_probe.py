"""Replay identification variants from saved, label-free patch diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import DEFAULT_DATA
from .identify import load_patterns, rank_patterns, rank_patterns_similarity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostics', type=Path)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--no-similarity', action='store_true')
    args = parser.parse_args()
    patches = json.loads(args.diagnostics.read_text(encoding='utf-8'))
    patterns = load_patterns(args.data / 'patterns')
    for scene in sorted({p['scene'] for p in patches}):
        selected = sorted((p for p in patches if p['scene'] == scene and p['present']),
                          key=lambda p: -p['score'])
        points = np.array([[p['x'], p['y']] for p in selected])
        affine = rank_patterns(patterns, points)
        similarity = [] if args.no_similarity else rank_patterns_similarity(patterns, points)
        print(scene, 'affine:', [(r['name'], round(r['score'], 2)) for r in affine[:3]],
              'similarity:', [(r['name'], round(r['score'], 2)) for r in similarity[:3]], flush=True)
        for weight in (0, .25, .5, 1, 2, 4):
            def value(item):
                singular = np.linalg.svd(np.asarray(item['matrix']), compute_uv=False)
                return item['score'] - weight * np.log(singular[0] / singular[1])
            ranked = sorted(affine, key=value, reverse=True)
            selected = next((i + 1 for i, item in enumerate(ranked)
                             if item['name'] == scene), None)
            print('  aspect penalty', weight, 'top', ranked[0]['name'],
                  'true rank', selected, flush=True)


if __name__ == '__main__':
    main()
