"""Training-only visual comparison of labelled query patches and sky crops."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .data import DEFAULT_DATA, discover_scenes, read_gray, read_rows, row_points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene')
    parser.add_argument('patches', nargs='+', type=int)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=Path('outputs/visual-diagnostics.png'))
    args = parser.parse_args()
    _, rows = read_rows(args.data / 'train_ground_truth.csv')
    row = next(item for item in rows if item['Id'] == args.scene)
    scene = next(item for item in discover_scenes(args.data / 'train')
                 if item.scene_id == args.scene)
    sky = read_gray(scene.image_path)
    truth = row_points(row)
    tiles = []
    for number in args.patches:
        gt = truth[number - 1]
        if gt is None:
            continue
        query = read_gray(scene.patches[number - 1])
        x, y = (int(round(gt[0])), int(round(gt[1])))
        crop = cv2.getRectSubPix(sky, (32, 32), (float(x), float(y)))
        context = cv2.getRectSubPix(sky, (96, 96), (float(x), float(y)))
        context = cv2.resize(context, (256, 256), interpolation=cv2.INTER_NEAREST)
        query = cv2.resize(query, (256, 256), interpolation=cv2.INTER_NEAREST)
        crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_NEAREST)
        tile = np.concatenate([query, crop, context], axis=1)
        cv2.putText(tile, f'{args.scene} {number:02d}: query | sky | context',
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .48, 255, 1)
        tiles.append(tile)
    if not tiles:
        raise ValueError('No present patches selected')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), np.concatenate(tiles, axis=0))
    print(args.output.resolve())


if __name__ == '__main__':
    main()
