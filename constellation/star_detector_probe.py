"""Training-only comparison of multiscale bright-star peak detectors."""
from __future__ import annotations

import json

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .data import DEFAULT_DATA, read_gray, read_rows, row_points


def peaks(sky: np.ndarray, fine_sigma: float, broad_sigma: float,
          count: int, nms_size: int = 11) -> np.ndarray:
    image = sky.astype(np.float32)
    fine = cv2.GaussianBlur(image, (0, 0), fine_sigma)
    broad = cv2.GaussianBlur(image, (0, 0), broad_sigma)
    response = fine - broad
    maxima = (response >= cv2.dilate(response,
                                    np.ones((nms_size, nms_size), np.uint8))) & (response > 0)
    margin = max(22, nms_size)
    maxima[:margin] = maxima[-margin:] = False
    maxima[:, :margin] = maxima[:, -margin:] = False
    yy, xx = np.nonzero(maxima)
    if len(xx) > count:
        chosen = np.argpartition(response[yy, xx], -count)[-count:]
        xx, yy = xx[chosen], yy[chosen]
    return np.column_stack((xx, yy)).astype(float)


def main() -> None:
    _, rows = read_rows(DEFAULT_DATA / 'train_ground_truth.csv')
    output = {}
    rng = np.random.default_rng(17)
    for row in rows:
        name = row['Id']
        sky = read_gray(DEFAULT_DATA / 'train' / name / f'{name}_image.png')
        figure = np.array([p[:2] for p in row_points(row) if p is not None and p[2]])
        random_points = rng.uniform([0, 0], [sky.shape[1], sky.shape[0]], (1000, 2))
        checks = []
        for fine, broad in ((.9, 3.2), (1.5, 5), (2, 7), (3, 10),
                            (4, 12), (5, 16), (8, 24)):
            for count in (300, 1000, 3000):
                centers = peaks(sky, fine, broad, count)
                tree = cKDTree(centers)
                checks.append(dict(fine=fine, broad=broad, count=count,
                                   figure_hits=int((tree.query(figure)[0] <= 20).sum()),
                                   random_hits=int((tree.query(random_points)[0] <= 20).sum()),
                                   detected=len(centers)))
        output[name] = checks
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
