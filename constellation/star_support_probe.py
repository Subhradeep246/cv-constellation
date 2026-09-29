"""Development-only replay of bright-star evidence on saved Hough poses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .data import DEFAULT_DATA, read_gray
from .identify import load_patterns


def strong_stars(sky: np.ndarray, count: int = 300,
                 fine_sigma: float = .9,
                 broad_sigma: float = 3.2) -> np.ndarray:
    array = sky.astype(np.float32)
    fine = cv2.GaussianBlur(array, (0, 0), fine_sigma)
    response = fine - cv2.GaussianBlur(array, (0, 0), broad_sigma)
    peaks = response >= cv2.dilate(response, np.ones((5, 5), np.uint8))
    peaks[:22] = peaks[-22:] = False
    peaks[:, :22] = peaks[:, -22:] = False
    y, x = np.nonzero(peaks)
    if len(x) > count:
        take = np.argpartition(response[y, x], -count)[-count:]
        x, y = x[take], y[take]
    return np.column_stack((x, y)).astype(float)


def star_evidence(entry: dict, nodes: np.ndarray, tree: cKDTree,
                  image_shape: tuple[int, int], tolerance: float) -> dict:
    projected = nodes @ np.asarray(entry['matrix']).T + np.asarray(entry['translation'])
    in_frame = ((projected[:, 0] >= 0) & (projected[:, 0] < image_shape[1]) &
                (projected[:, 1] >= 0) & (projected[:, 1] < image_shape[0]))
    distances = tree.query(projected[in_frame])[0]
    return dict(star_hits=int(np.sum(distances <= tolerance)),
                star_score=float(np.exp(-.5 * (distances / tolerance) ** 2).sum()),
                in_frame=int(in_frame.sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--outputs', type=Path, default=Path('outputs'))
    parser.add_argument('--stars', type=int, default=300)
    parser.add_argument('--tolerance', type=float, default=20)
    parser.add_argument('--weight', type=float, default=1)
    parser.add_argument('--fine', type=float, default=.9)
    parser.add_argument('--broad', type=float, default=3.2)
    args = parser.parse_args()
    patterns = {p.name: p.nodes for p in load_patterns(args.data / 'patterns')}
    for scene in ('pisces', 'scorpius', 'taurus'):
        image = read_gray(args.data / 'train' / scene / f'{scene}_image.png')
        tree = cKDTree(strong_stars(image, args.stars, args.fine, args.broad))
        report = json.loads((args.outputs / f'hough-{scene}-cap200.json').read_text(encoding='utf-8'))
        candidates = []
        for entry in report['per_pattern']:
            evidence = star_evidence(entry, patterns[entry['pattern']], tree,
                                     image.shape, args.tolerance)
            candidates.append(dict(name=entry['pattern'],
                                   base=float(entry['combined']), **evidence))
        for weight in (0, args.weight):
            ranked = sorted(candidates,
                            key=lambda item: item['base'] + weight * item['star_score'],
                            reverse=True)
            winner = ranked[0]
            print(json.dumps(dict(scene=scene, weight=weight,
                                  true_rank=next((i + 1 for i, item in enumerate(ranked)
                                                  if item['name'] == scene), None),
                                  winner=winner['name'],
                                  margin=round((ranked[0]['base'] + weight * ranked[0]['star_score']) -
                                               (ranked[1]['base'] + weight * ranked[1]['star_score']), 3),
                                  top=ranked[:5])))


if __name__ == '__main__':
    main()
