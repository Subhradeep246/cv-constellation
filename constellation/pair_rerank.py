"""Scene-held-out probe of a locally trained direct-match candidate reranker.

Only supplied training labels are used for fitting. No hidden-scene labels are
read. This is an experiment, not part of the default submission pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

from .data import DEFAULT_DATA, discover_scenes, read_gray, read_rows
from .identify import load_patterns, rank_patterns
from .localize import Localizer
from .metrics import score_scene


FEATURES = ('direct_score', 'baseline_score', 'baseline_flag', 'full',
            'outer', 'core', 'lowpass', 'highpass', 'gradient', 'zncc')


def cosine(a, b):
    return float(np.dot(a.ravel(), b.ravel()) /
                 max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8))


def feature_vector(sky: np.ndarray, patch: np.ndarray, item: dict) -> np.ndarray:
    yy, xx = np.mgrid[:32, :32].astype(np.float32)
    u, v = xx - 15.5, yy - 15.5
    angle = -np.deg2rad(item['degrees'])
    co, si = np.cos(angle), np.sin(angle)
    scale = item['scale']
    mx = (item['x'] + scale * (co * u - si * v)).astype(np.float32)
    my = (item['y'] + scale * (si * u + co * v)).astype(np.float32)
    sample = cv2.remap(sky, mx, my, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT101)
    radius2 = u * u + v * v
    disk = radius2 <= 14 ** 2
    outer = disk & (radius2 >= 6 ** 2)
    core = disk & (radius2 < 6 ** 2)
    qlow = cv2.GaussianBlur(patch, (0, 0), 1.5)
    slow = cv2.GaussianBlur(sample, (0, 0), 1.5)
    qhigh, shigh = patch - qlow, sample - slow
    qgrad = cv2.Laplacian(patch, cv2.CV_32F)
    sgrad = cv2.Laplacian(sample, cv2.CV_32F)
    qzero, szero = patch[disk] - patch[disk].mean(), sample[disk] - sample[disk].mean()
    return np.array([
        item['score'] if item['source'] == 'direct' else 0,
        item['score'] if item['source'] == 'baseline' else 0,
        float(item['source'] == 'baseline'),
        cosine(patch[disk], sample[disk]),
        cosine(patch[outer], sample[outer]),
        cosine(patch[core], sample[core]),
        cosine(qlow[disk], slow[disk]),
        cosine(qhigh[disk], shigh[disk]),
        cosine(qgrad[disk], sgrad[disk]),
        cosine(qzero, szero),
    ], dtype=np.float32)


def build_records(data: Path, outputs: Path, baseline_path: Path) -> list[dict]:
    scenes = {s.scene_id: s for s in discover_scenes(data / 'train')}
    baseline = {(p['scene'], p['patch']): p
                for p in json.loads(baseline_path.read_text(encoding='utf-8'))}
    records = []
    for name in ('pisces', 'scorpius', 'taurus'):
        folder = outputs / f'direct-{name}-all'
        sky = Localizer.filtered(read_gray(scenes[name].image_path))
        for path in sorted(folder.glob('patch_*.json')):
            row = json.loads(path.read_text(encoding='utf-8'))
            number = row['patch']
            patch = Localizer.filtered(read_gray(scenes[name].patches[number-1]))
            items = [dict(item, source='direct') for item in row['candidates']]
            b = baseline[(name, f'patch_{number:02d}')]
            items.append(dict(x=b['x'], y=b['y'], scale=b['scale'],
                              degrees=-np.rad2deg(b['angle']), score=b['score'],
                              source='baseline'))
            for rank, item in enumerate(items):
                if not 4 <= item['x'] < sky.shape[1]-4 or not 4 <= item['y'] < sky.shape[0]-4:
                    continue
                x = feature_vector(sky, patch, item)
                gt = row['truth']
                positive = bool(gt and np.hypot(item['x']-gt[0], item['y']-gt[1]) < 12)
                records.append(dict(scene=name, patch=number, rank=rank,
                                    xy=(item['x'], item['y']), positive=positive,
                                    truth_present=bool(gt), source=item['source'], features=x))
        print(name, 'features complete', flush=True)
    return records


def fit_logistic(x: np.ndarray, y: np.ndarray, regularization: float = 3.0):
    center = x.mean(axis=0)
    spread = np.maximum(x.std(axis=0), .05)
    x = (x - center) / spread
    design = np.column_stack([x, np.ones(len(x))])
    positives = max(1, int(y.sum()))
    negatives = max(1, len(y) - positives)
    weights = np.where(y, len(y)/(2*positives), len(y)/(2*negatives))

    def objective(coefficients):
        logits = design @ coefficients
        loss = np.sum(weights * (np.logaddexp(0, logits) - y * logits))
        loss += regularization * np.dot(coefficients[:-1], coefficients[:-1])
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
        gradient = design.T @ (weights * (probabilities - y))
        gradient[:-1] += 2 * regularization * coefficients[:-1]
        return loss, gradient

    result = minimize(objective, np.zeros(design.shape[1]), jac=True,
                      method='L-BFGS-B')
    if not result.success:
        raise RuntimeError(result.message)
    return center, spread, result.x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--outputs', type=Path, default=Path('outputs'))
    parser.add_argument('--baseline', type=Path,
                        default=Path('outputs/baseline-fast/evaluate-patches.json'))
    args = parser.parse_args()
    cv2.setNumThreads(8)
    records = build_records(args.data, args.outputs, args.baseline)
    _, truth = read_rows(args.data / 'train_ground_truth.csv')
    truth_by_scene = {row['Id']: row for row in truth}
    patterns = load_patterns(args.data / 'patterns')
    baseline_rows = {(p['scene'], p['patch']): p
                     for p in json.loads(args.baseline.read_text(encoding='utf-8'))}
    report = []
    for held_out in ('pisces', 'scorpius', 'taurus'):
        train = [r for r in records if r['scene'] != held_out]
        test = [r for r in records if r['scene'] == held_out]
        x = np.stack([r['features'] for r in train])
        y = np.array([r['positive'] for r in train], dtype=float)
        center, spread, coefficients = fit_logistic(x, y)
        groups = {}
        for row in records:
            row['rerank_score'] = float((row['features']-center)/spread @ coefficients[:-1]
                                          + coefficients[-1])
            groups.setdefault((row['scene'], row['patch']), []).append(row)
        train_bests = [max(group, key=lambda item: item['rerank_score'])
                       for (scene, _), group in groups.items() if scene != held_out]
        thresholds = np.linspace(-6, 6, 97)

        def macro_presence_f1(best_rows, threshold):
            values = []
            for scene in sorted({item['scene'] for item in best_rows}):
                selected = [item for item in best_rows if item['scene'] == scene]
                tp = sum(item['truth_present'] and item['rerank_score'] >= threshold for item in selected)
                tn = sum(not item['truth_present'] and item['rerank_score'] < threshold for item in selected)
                fp = sum(not item['truth_present'] and item['rerank_score'] >= threshold for item in selected)
                fn = sum(item['truth_present'] and item['rerank_score'] < threshold for item in selected)
                pos = 2 * tp / max(2 * tp + fp + fn, 1)
                neg = 2 * tn / max(2 * tn + fp + fn, 1)
                values.append((pos + neg) / 2)
            return float(np.mean(values))

        threshold = max(thresholds, key=lambda t: macro_presence_f1(train_bests, t))
        test_groups = {patch: group for (scene, patch), group in groups.items()
                       if scene == held_out}
        true_count = sum(group[0]['truth_present'] for group in test_groups.values())
        covered = sum(any(item['positive'] for item in group) for group in test_groups.values())
        reranked = sum(max(group, key=lambda item: item['rerank_score'])['positive']
                       for group in test_groups.values())
        direct = sum(max((item for item in group if item['source'] == 'direct'),
                         key=lambda item: item['features'][0])['positive']
                     for group in test_groups.values())
        baseline = sum(next((item['positive'] for item in group
                             if item['source'] == 'baseline'), False)
                       for group in test_groups.values())
        policies = {}
        for policy in ('baseline_presence', 'trained_threshold'):
            selected = []
            pred = {key: '-1' for key in truth_by_scene[held_out]}
            pred.update(Id=held_out, n_patches=truth_by_scene[held_out]['n_patches'])
            for number, group in test_groups.items():
                best = max(group, key=lambda item: item['rerank_score'])
                patch_name = f'patch_{number:02d}'
                keep = (baseline_rows[(held_out, patch_name)]['present']
                        if policy == 'baseline_presence' else best['rerank_score'] >= threshold)
                if keep:
                    x, y = best['xy']
                    pred[patch_name] = str((x, y, 0))
                    selected.append((best['rerank_score'], x, y))
            selected.sort(reverse=True)
            ranked = rank_patterns(patterns, np.array([[x, y] for _, x, y in selected]))
            pred['constellation'] = ranked[0]['name'] if ranked else 'unknown'
            metric = score_scene(truth_by_scene[held_out], pred)
            policies[policy] = dict(score=metric['score'], name=pred['constellation'],
                                    presence=metric['presence'], localization=metric['localization'],
                                    geometry=metric['geometric_recovery'],
                                    identification=metric['identification'])
        report.append(dict(held_out=held_out, present=true_count, covered=covered,
                           baseline_top=baseline, direct_top=direct,
                           reranked_top=reranked, threshold=float(threshold),
                           policies=policies,
                           coefficients=dict(zip(FEATURES, coefficients[:-1]))))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
