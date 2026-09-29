"""Training-only coarse constellation Hough search over direct alternatives."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .data import DEFAULT_DATA
from .identify import load_patterns


def heatmap(folder: Path, step: int, top: int, baseline: Path | None):
    rows = [json.loads(path.read_text(encoding='utf-8'))
            for path in sorted(folder.glob('patch_*.json'))]
    if not rows:
        raise ValueError('No candidate files found')
    size = int(np.ceil(3000 / step))
    image = np.zeros((size, size), dtype=np.float32)
    candidates = []
    for row in rows:
        for item in row['candidates'][:top]:
            candidates.append(dict(item, patch=row['patch']))
    if baseline:
        patches = json.loads(baseline.read_text(encoding='utf-8'))
        scene = rows[0]['scene']
        for item in patches:
            if item['scene'] == scene and item['present']:
                candidates.append(dict(x=item['x'], y=item['y'], score=item['score'],
                                       patch=int(item['patch'].removeprefix('patch_'))))
    for item in candidates:
        x = int(round(item['x'] / step))
        y = int(round(item['y'] / step))
        if 0 <= x < size and 0 <= y < size:
            image[y, x] = 1
    image = cv2.dilate(image, np.ones((3, 3), np.uint8))
    return rows[0]['scene'], image, candidates


def pose_scores(image: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    height, width = image.shape
    scores = np.zeros_like(image)
    for dx, dy in offsets:
        dx, dy = int(dx), int(dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        y0, y1 = max(0, -dy), min(height, height - dy)
        if x0 < x1 and y0 < y1:
            scores[y0:y1, x0:x1] += image[y0+dy:y1+dy, x0+dx:x1+dx]
    return scores


def search_pattern(image: np.ndarray, nodes: np.ndarray, step: int,
                   scales: np.ndarray, angles: np.ndarray,
                   min_support: float = 0,
                   include_reflection: bool = False,
                   max_hits_per_transform: int = 0):
    centered = nodes - nodes.mean(axis=0)
    best = []
    hits = []
    for reflected in ((False, True) if include_reflection else (False,)):
        target = centered * ([-1, 1] if reflected else [1, 1])
        for scale in scales:
            for degrees in angles:
                angle = np.deg2rad(degrees)
                rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                     [np.sin(angle), np.cos(angle)]])
                offsets = np.rint(target @ rotation.T * (scale / step)).astype(int)
                scores = pose_scores(image, offsets)
                _, value, _, (x, y) = cv2.minMaxLoc(scores)
                best.append(dict(support=value, scale=float(scale), degrees=float(degrees),
                                 reflected=reflected, x=float(x * step), y=float(y * step)))
                if min_support:
                    candidate_mask = scores >= min_support
                    if max_hits_per_transform:
                        candidate_mask &= scores >= cv2.dilate(scores,
                                                                np.ones((5, 5), np.uint8))
                    ys, xs = np.nonzero(candidate_mask)
                    if max_hits_per_transform and len(xs) > max_hits_per_transform:
                        chosen = np.argpartition(scores[ys, xs],
                                                 -max_hits_per_transform)[-max_hits_per_transform:]
                        ys, xs = ys[chosen], xs[chosen]
                    hits.extend(dict(support=float(scores[yy, xx]), scale=float(scale),
                                     degrees=float(degrees), reflected=reflected,
                                     x=float(xx * step), y=float(yy * step))
                                for yy, xx in zip(ys, xs))
    return sorted(best, key=lambda item: -item['support'])[:10], hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--pattern')
    parser.add_argument('--step', type=int, default=8)
    parser.add_argument('--top', type=int, default=12)
    parser.add_argument('--min-scale', type=float, default=3.0)
    parser.add_argument('--max-scale', type=float, default=8.0)
    parser.add_argument('--scale-step', type=float, default=.25)
    parser.add_argument('--angle-step', type=float, default=5.0)
    parser.add_argument('--min-support', type=float, default=0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    scene, image, candidates = heatmap(args.folder, args.step, args.top,
                                       args.baseline)
    patterns = load_patterns(args.data / 'patterns')
    if args.pattern:
        patterns = [p for p in patterns if p.name == args.pattern]
        if not patterns:
            raise ValueError(f'Unknown pattern: {args.pattern}')
    scales = np.arange(args.min_scale, args.max_scale + args.scale_step / 2,
                       args.scale_step)
    angles = np.arange(0, 360, args.angle_step)
    collected = []
    for pattern in patterns:
        best, hits = search_pattern(image, pattern.nodes, args.step, scales, angles,
                                    args.min_support)
        collected.append(dict(pattern=pattern.name, hits=hits))
        print(json.dumps(dict(scene=scene, pattern=pattern.name,
                              candidates=len(candidates), occupied=int(image.sum()),
                              seconds=round(time.perf_counter() - started, 2),
                              hit_count=len(hits), best_support=best[0]['support'])), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(scene=scene, patterns=collected)), encoding='utf-8')


if __name__ == '__main__':
    cv2.setNumThreads(8)
    main()
