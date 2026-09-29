"""Whole-image exact-window copy-move probe for supplied sky images.

The hash is only a candidate screen. Every reported pair is verified by exact
window pixels, and each displacement must be supported by many separate
windows. This deliberately does not use query labels or scene filenames.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .data import DEFAULT_DATA, discover_scenes, read_gray


OFFSETS = ((0, 0), (0, 15), (15, 0), (15, 15),
           (4, 9), (9, 4), (7, 12), (12, 7))


def exact_copy_windows(image: np.ndarray, *, window: int = 16,
                       min_range: int = 20, min_separation: int = 80,
                       max_group: int = 8) -> dict:
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError('Expected a grayscale uint8 sky image')
    height, width = image.shape
    valid_h, valid_w = height - window + 1, width - window + 1
    if valid_h < 1 or valid_w < 1:
        raise ValueError('Image smaller than copy window')
    if window != 16:
        raise ValueError('The eight hash offsets are defined for 16x16 windows')
    hashes = np.zeros((valid_h, valid_w), dtype=np.uint64)
    low = np.full((valid_h, valid_w), 255, dtype=np.uint8)
    high = np.zeros((valid_h, valid_w), dtype=np.uint8)
    for bit, (dy, dx) in enumerate(OFFSETS):
        view = image[dy:dy+valid_h, dx:dx+valid_w]
        hashes |= view.astype(np.uint64) << np.uint64(8 * bit)
        np.minimum(low, view, out=low)
        np.maximum(high, view, out=high)
    valid = (high.astype(np.int16) - low.astype(np.int16)) >= min_range
    flat_indices = np.flatnonzero(valid)
    keys = hashes.ravel()[flat_indices]
    order = np.argsort(keys, kind='stable')
    flat_indices = flat_indices[order]
    keys = keys[order]
    changes = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1, len(keys)]
    counts = Counter()
    examples = defaultdict(list)
    verified_pairs = 0
    for begin, end in zip(changes[:-1], changes[1:]):
        if not 2 <= end - begin <= max_group:
            continue
        positions = flat_indices[begin:end]
        for a in range(len(positions)):
            y1, x1 = divmod(int(positions[a]), valid_w)
            for b in range(a + 1, len(positions)):
                y2, x2 = divmod(int(positions[b]), valid_w)
                if max(abs(x2 - x1), abs(y2 - y1)) < min_separation:
                    continue
                if not np.array_equal(image[y1:y1+window, x1:x1+window],
                                      image[y2:y2+window, x2:x2+window]):
                    continue
                dx, dy = x2-x1, y2-y1
                sx, sy, tx, ty = x1, y1, x2, y2
                # The sorted hash order is unrelated to spatial order.
                if dx < 0 or (dx == 0 and dy < 0):
                    dx, dy = -dx, -dy
                    sx, sy, tx, ty = x2, y2, x1, y1
                shift = (dx, dy)
                counts[shift] += 1
                verified_pairs += 1
                if len(examples[shift]) < 500:
                    examples[shift].append((sx + window / 2, sy + window / 2,
                                            tx + window / 2, ty + window / 2))
    return dict(valid_windows=int(valid.sum()), verified_pairs=verified_pairs,
                shifts=[dict(dx=dx, dy=dy, windows=count,
                             example_centers=examples[(dx, dy)])
                        for (dx, dy), count in counts.most_common(100)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--split', choices=('train', 'validation'), default='train')
    parser.add_argument('--scene')
    parser.add_argument('--output', type=Path, default=Path('outputs/copy-move.json'))
    args = parser.parse_args()
    records = []
    for scene in discover_scenes(args.data / args.split):
        if args.scene and scene.scene_id != args.scene:
            continue
        result = exact_copy_windows(read_gray(scene.image_path))
        records.append(dict(scene=scene.scene_id, **result))
        print(json.dumps(dict(scene=scene.scene_id,
                              valid_windows=result['valid_windows'],
                              verified_pairs=result['verified_pairs'],
                              top_shifts=[{k:v for k,v in item.items() if k != 'example_centers'}
                                          for item in result['shifts'][:10]])), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records), encoding='utf-8')


if __name__ == '__main__':
    cv2.setNumThreads(8)
    main()
