"""Whole-sky approximate copied-neighborhood search around detected sources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .data import DEFAULT_DATA, discover_scenes, read_gray
from .localize import Localizer, LocalizerConfig


RADII = np.arange(18, 50, 5, dtype=np.float32)
ANGLES = np.arange(32, dtype=np.float32) * (2 * np.pi / 32)
OFFSETS = np.stack([RADII[:, None] * np.cos(ANGLES),
                    RADII[:, None] * np.sin(ANGLES)], axis=-1).reshape(-1, 2)


def sample_annuli(image: np.ndarray, centers: np.ndarray) -> np.ndarray:
    out = np.empty((len(centers), len(OFFSETS)), dtype=np.float32)
    for start in range(0, len(centers), 256):
        stop = min(start + 256, len(centers))
        xy = centers[start:stop, None] + OFFSETS[None]
        values = cv2.remap(image, xy[..., 0], xy[..., 1], cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT101)
        out[start:stop] = values
    out -= out.mean(axis=1, keepdims=True)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)
    return out


def find_copies(image: np.ndarray, max_sources: int = 35000,
                neighbors: int = 20, threshold: float = .80) -> dict:
    cv2.setRNGSeed(17)
    loc = Localizer(LocalizerConfig(max_sources=max_sources))
    centers = loc.detect(image)
    inside = ((centers[:, 0] >= 55) & (centers[:, 0] < image.shape[1]-55) &
              (centers[:, 1] >= 55) & (centers[:, 1] < image.shape[0]-55))
    centers = centers[inside]
    if len(centers) < 2:
        return dict(source_count=len(centers), neighbor_pairs=0,
                    threshold=threshold, pairs=[])
    smooth = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.5)
    descriptors = sample_annuli(smooth, centers)
    rng = np.random.default_rng(17)
    projector = rng.normal(size=(descriptors.shape[1], 48)).astype(np.float32)
    projected = np.ascontiguousarray(descriptors @ projector)
    projected /= np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-8)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=6), dict(checks=128))
    matcher.add([projected])
    matcher.train()
    pairs = set()
    for matches in matcher.knnMatch(projected, k=min(neighbors, len(centers))):
        for match in matches:
            i, j = sorted((match.queryIdx, match.trainIdx))
            if i == j or np.linalg.norm(centers[i]-centers[j]) < 100:
                continue
            pairs.add((i, j))
    results = []
    for i, j in pairs:
        score = float(descriptors[i] @ descriptors[j])
        if score >= threshold:
            results.append(dict(a=centers[i].tolist(), b=centers[j].tolist(),
                                annulus_ncc=score))
    results.sort(key=lambda item: -item['annulus_ncc'])
    return dict(source_count=len(centers), neighbor_pairs=len(pairs),
                threshold=threshold, pairs=results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--split', choices=('train', 'validation'), default='train')
    parser.add_argument('--scene')
    parser.add_argument('--sources', type=int, default=35000)
    parser.add_argument('--output', type=Path, default=Path('outputs/copy-annulus.json'))
    args = parser.parse_args()
    cv2.setNumThreads(8)
    records = []
    for scene in discover_scenes(args.data / args.split):
        if args.scene and scene.scene_id != args.scene:
            continue
        result = find_copies(read_gray(scene.image_path), max_sources=args.sources)
        records.append(dict(scene=scene.scene_id, **result))
        print(json.dumps(dict(scene=scene.scene_id, source_count=result['source_count'],
                              neighbor_pairs=result['neighbor_pairs'],
                              matches=len(result['pairs']), top=result['pairs'][:5])), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records), encoding='utf-8')


if __name__ == '__main__':
    main()
