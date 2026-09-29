"""Training-only replay of photometric signals on saved candidate locations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .data import DEFAULT_DATA, discover_scenes, read_gray, read_rows, row_points


def correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    x = a[mask].astype(np.float32)
    y = b[mask].astype(np.float32)
    x -= x.mean()
    y -= y.mean()
    return float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-8))


def warped_crop(sky: np.ndarray, patch: np.ndarray, item: dict) -> np.ndarray:
    yy, xx = np.mgrid[:patch.shape[0], :patch.shape[1]].astype(np.float32)
    u, v = xx - (patch.shape[1] - 1) / 2, yy - (patch.shape[0] - 1) / 2
    angle, scale = float(item['angle']), float(item['scale'])
    co, si = np.cos(angle), np.sin(angle)
    mx = item['x'] + scale * (co * u - si * v)
    my = item['y'] + scale * (si * u + co * v)
    return cv2.remap(sky, mx.astype(np.float32), my.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def photometric_features(sky: np.ndarray, patch: np.ndarray, item: dict) -> dict:
    crop = warped_crop(sky, patch, item)
    yy, xx = np.mgrid[:patch.shape[0], :patch.shape[1]]
    radius = np.hypot(xx - 15.5, yy - 15.5)
    disk = radius <= 15
    annulus = (radius >= 5) & disk
    a, b = patch.astype(np.float32), crop.astype(np.float32)
    low_a = cv2.GaussianBlur(np.log1p(a), (0, 0), 1.6)
    low_b = cv2.GaussianBlur(np.log1p(b), (0, 0), 1.6)
    high_a = a - cv2.GaussianBlur(a, (0, 0), 2.5)
    high_b = b - cv2.GaussianBlur(b, (0, 0), 2.5)
    return dict(log_blur=correlation(low_a, low_b, disk),
                residual=correlation(high_a, high_b, disk),
                outer=correlation(high_a, high_b, annulus))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=Path('outputs/hough-final-train'))
    args = parser.parse_args()
    details = json.loads((args.output / 'evaluate-patches.json').read_text(encoding='utf-8'))
    _, truth = read_rows(args.data / 'train_ground_truth.csv')
    truth = {row['Id']: row_points(row) for row in truth}
    scenes = {scene.scene_id: scene for scene in discover_scenes(args.data / 'train')}
    evaluations = []
    for scene_id, scene in scenes.items():
        sky = read_gray(scene.image_path)
        rows = [row for row in details if row['scene'] == scene_id]
        for row in rows:
            patch_number = int(row['patch'].removeprefix('patch_')) - 1
            gt = truth[scene_id][patch_number]
            if gt is None or not row['alternatives']:
                continue
            patch = read_gray(scene.patches[patch_number])
            candidates = []
            for item in row['alternatives']:
                values = photometric_features(sky, patch, item)
                values.update(original=float(item['score']),
                              true=bool(np.hypot(item['x'] - gt[0],
                                                 item['y'] - gt[1]) <= 12))
                candidates.append(values)
            if not any(item['true'] for item in candidates):
                continue
            entry = dict(scene=scene_id, patch=row['patch'], reachable=True)
            for key in ('original', 'log_blur', 'residual', 'outer'):
                winner = max(candidates, key=lambda item: item[key])
                entry[key] = winner['true']
            # A fixed combination to test complementary channels without
            # fitting any coefficient to individual scenes.
            winner = max(candidates, key=lambda item: item['log_blur'] + item['residual'])
            entry['combined'] = winner['true']
            evaluations.append(entry)
    summary = {}
    for scene in scenes:
        rows = [row for row in evaluations if row['scene'] == scene]
        summary[scene] = dict(reachable=len(rows),
                              **{key: sum(row[key] for row in rows)
                                 for key in ('original', 'log_blur', 'residual',
                                             'outer', 'combined')})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
