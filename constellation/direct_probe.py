"""Training-only rotated-template retrieval probe; does not affect inference."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

from .data import DEFAULT_DATA, discover_scenes, read_gray, read_rows, row_points
from .direct_search import template_variants
from .localize import Localizer


def refine_candidates(sky: np.ndarray, patch: np.ndarray,
                      candidates: list[dict]) -> list[dict]:
    yy, xx = np.mgrid[:32, :32].astype(np.float32)
    mask = (xx - 15.5) ** 2 + (yy - 15.5) ** 2 <= 14 ** 2
    u, v = xx[mask] - 15.5, yy[mask] - 15.5
    query = patch[mask].astype(np.float32)
    query /= max(float(np.linalg.norm(query)), 1e-8)
    results = []
    for item in candidates:
        angle = -np.deg2rad(item['degrees'])
        initial = [item['x'], item['y'], item['scale'], angle]

        def objective(params):
            x, y, scale, theta = params
            co, si = np.cos(theta), np.sin(theta)
            mx = (x + scale * (co * u - si * v))[None].astype(np.float32)
            my = (y + scale * (si * u + co * v))[None].astype(np.float32)
            sample = cv2.remap(sky, mx, my, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT101)[0]
            return -float(np.dot(query, sample) / max(np.linalg.norm(sample), 1e-8))

        bounds = [(max(0, item['x'] - 3), min(sky.shape[1]-1, item['x'] + 3)),
                  (max(0, item['y'] - 3), min(sky.shape[0]-1, item['y'] + 3)),
                  (max(.45, item['scale'] * .8), min(1.9, item['scale'] * 1.2)),
                  (angle - .35, angle + .35)]
        if any(low >= high for low, high in bounds):
            continue
        opt = minimize(objective, initial, method='Powell', bounds=bounds,
                       options=dict(maxiter=25, xtol=.02, ftol=.0001))
        initial_score = -objective(initial)
        score, values = (-float(opt.fun), opt.x) if -opt.fun > initial_score else (initial_score, initial)
        results.append(dict(item, refined_score=score,
                            refined_x=float(values[0]), refined_y=float(values[1]),
                            refined_scale=float(values[2]), refined_angle=float(values[3])))
    return sorted(results, key=lambda item: -item['refined_score'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene')
    parser.add_argument('patch', type=int)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--downsample', type=float, default=1.0)
    parser.add_argument('--inner-radius', type=float, default=0)
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('outputs/direct-probe.json'))
    args = parser.parse_args()
    if not 0 < args.downsample <= 1:
        parser.error('--downsample must be in (0, 1]')
    if not 0 <= args.inner_radius < 14:
        parser.error('--inner-radius must be in [0, 14)')
    cv2.setNumThreads(8)
    _, rows = read_rows(args.data / 'train_ground_truth.csv')
    row = next(item for item in rows if item['Id'] == args.scene)
    scene = next(item for item in discover_scenes(args.data / 'train')
                 if item.scene_id == args.scene)
    gt = row_points(row)[args.patch - 1]
    full_sky = Localizer.filtered(read_gray(scene.image_path))
    sky = full_sky
    patch = Localizer.filtered(read_gray(scene.patches[args.patch - 1]))
    if args.downsample < 1:
        sky = cv2.resize(sky, None, fx=args.downsample, fy=args.downsample,
                         interpolation=cv2.INTER_AREA)
    candidates = []
    truth_scores = []
    started = time.perf_counter()
    for scale, degrees, template, mask in template_variants(patch, args.inner_radius):
        if args.downsample < 1:
            template = cv2.resize(template, None, fx=args.downsample,
                                  fy=args.downsample, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, template.shape[::-1], interpolation=cv2.INTER_AREA)
        response = cv2.matchTemplate(sky, template, cv2.TM_CCORR_NORMED, mask=mask)
        response = np.nan_to_num(response, nan=-1, posinf=-1, neginf=-1)
        count = 10
        top = np.argpartition(response.ravel(), -count)[-count:]
        for index in top:
            y, x = np.unravel_index(index, response.shape)
            candidates.append(dict(x=int(round((x + (template.shape[1] - 1) / 2)
                                               / args.downsample)),
                                   y=int(round((y + (template.shape[0] - 1) / 2)
                                               / args.downsample)),
                                   score=float(response[y, x]), scale=scale, degrees=degrees))
        if gt:
            x = int(round(gt[0] * args.downsample - (template.shape[1] - 1) / 2))
            y = int(round(gt[1] * args.downsample - (template.shape[0] - 1) / 2))
            if 0 <= y < response.shape[0] and 0 <= x < response.shape[1]:
                truth_scores.append(dict(score=float(response[y, x]),
                                         scale=scale, degrees=degrees))
    candidates.sort(key=lambda item: -item['score'])
    distinct = []
    for item in candidates:
        if all(np.hypot(item['x'] - other['x'], item['y'] - other['y']) >= 8
               for other in distinct):
            distinct.append(item)
    result = dict(scene=args.scene, patch=args.patch, truth=gt,
                  seconds=time.perf_counter() - started,
                  truth_best=max(truth_scores, key=lambda item: item['score']) if truth_scores else None,
                  candidates=distinct[:30])
    if args.refine:
        result['refined'] = refine_candidates(full_sky, patch, distinct[:30])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    best_error = (min(float(np.hypot(item['x'] - gt[0], item['y'] - gt[1]))
                      for item in distinct[:30]) if gt else None)
    print(json.dumps(dict(scene=args.scene, patch=args.patch,
                          seconds=round(result['seconds'], 2),
                          best_score=distinct[0]['score'] if distinct else None,
                          best_error=best_error,
                          refined_best_error=(float(np.hypot(result['refined'][0]['refined_x'] - gt[0],
                                                             result['refined'][0]['refined_y'] - gt[1]))
                                              if args.refine and gt and result['refined'] else None),
                          truth_best=result['truth_best']), indent=2))


if __name__ == '__main__':
    main()
