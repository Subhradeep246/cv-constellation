"""Non-destructive image preparation and conservative copy-neighborhood audit.

Quality flags are review hints, never grounds to drop a query. The exported
float views are diagnostics, not restored ground truth or inference inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import cv2
import numpy as np

from .data import DEFAULT_DATA, discover_scenes, read_gray

MODES = ('none', 'gamma', 'bilateral', 'clahe', 'starlet-light', 'starlet')


def starlet_denoise(image: np.ndarray, threshold_sigma: float = 2.5,
                    levels: int = 2) -> np.ndarray:
    """Denoise fine scales with an undecimated B3-spline (a trous) transform.

    Each detail band's noise scale is estimated by MAD. The smooth component is
    always retained, so broad nebulosity and background are not removed here.
    This is deterministic and uses no training data or external image priors.
    """
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError('Expected a grayscale uint8 image')
    if threshold_sigma < 0 or levels < 1:
        raise ValueError('Expected a nonnegative threshold and positive level count')
    smooth = image.astype(np.float32)
    retained = np.zeros_like(smooth)
    taps = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16
    for level in range(levels):
        step = 1 << level
        kernel = np.zeros(4 * step + 1, dtype=np.float32)
        kernel[::step] = taps
        next_smooth = cv2.sepFilter2D(smooth, cv2.CV_32F, kernel, kernel,
                                      borderType=cv2.BORDER_REFLECT_101)
        detail = smooth - next_smooth
        median = np.median(detail)
        sigma = np.median(np.abs(detail - median)) / .67448975
        retained += np.where(np.abs(detail - median) >= threshold_sigma * sigma,
                             detail, 0)
        smooth = next_smooth
    return np.clip(np.rint(smooth + retained), 0, 255).astype(np.uint8)


def prepare_image(image: np.ndarray, mode: str = 'none') -> np.ndarray:
    """Identical recipe for query and sky; no labels or split-dependent choices."""
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError('Expected a grayscale uint8 image')
    if mode == 'none':
        return image
    if mode == 'gamma':
        lut = np.rint(255 * (np.arange(256) / 255) ** .6).astype(np.uint8)
        return cv2.LUT(image, lut)
    if mode == 'bilateral':
        return cv2.bilateralFilter(image, 5, 25, 25)
    if mode == 'clahe':
        # Keep tile scale approximately constant in pixels across sky and patch.
        grid = tuple(max(1, int(np.ceil(size / 32))) for size in image.shape[::-1])
        return cv2.createCLAHE(clipLimit=2, tileGridSize=grid).apply(image)
    if mode == 'starlet-light':
        return starlet_denoise(image, threshold_sigma=1.5)
    if mode == 'starlet':
        return starlet_denoise(image)
    raise ValueError(f'Unknown preprocessing mode: {mode}')


def normalized_view(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    residual = cv2.GaussianBlur(arr, (0, 0), .6) - cv2.GaussianBlur(arr, (0, 0), 7)
    # A floor avoids magnifying numerical noise in blank patches. No clipping.
    return residual / max(float(np.sqrt(np.mean(residual ** 2))), 1.)


def unique_patches(patches: list[np.ndarray]) -> tuple[list[np.ndarray], list[int]]:
    """Deduplicate computation within ONE scene, then expand every query ID."""
    buckets, unique, inverse = {}, [], []
    for patch in patches:
        key = (patch.shape, patch.dtype.str, patch.tobytes())
        if key not in buckets:
            buckets[key] = len(unique)
            unique.append(patch)
        inverse.append(buckets[key])
    return unique, inverse


def quality(image: np.ndarray) -> dict:
    arr = image.astype(np.float32)
    hh = (arr[::2, ::2] - arr[1::2, ::2] - arr[::2, 1::2] + arr[1::2, 1::2]) / 2
    noise = float(np.median(np.abs(hh - np.median(hh))) / .67449)
    p = np.percentile(arr, [1, 50, 99])
    flags = []
    if arr.std() < 1:
        flags.append('almost_constant')
    if p[2] - p[0] < 15:
        flags.append('low_dynamic_range')
    if arr.mean() < 15:
        flags.append('dim_mean')
    if (arr == 255).mean() >= .01:
        flags.append('high_endpoint_at_least_1pct')
    return dict(mean=float(arr.mean()), std=float(arr.std()), p1=float(p[0]),
                median=float(p[1]), p99=float(p[2]), zero_fraction=float((arr == 0).mean()),
                high_endpoint_fraction=float((arr == 255).mean()),
                haar_noise_proxy=noise, flags=flags)


def _unit(rows: np.ndarray) -> np.ndarray:
    rows = rows - rows.mean(axis=-1, keepdims=True)
    return rows / np.maximum(np.linalg.norm(rows, axis=-1, keepdims=True), 1e-8)


def copy_neighborhoods(sky: np.ndarray, max_centers: int = 300) -> dict:
    """Screen bright peaks, excluding the generic central star from correlation.

    Not exhaustive copy-move detection: no rotation, scaling or large center
    offsets are searched. A candidate needs both raw and high-pass agreement.
    """
    arr = sky.astype(np.float32)
    response = cv2.GaussianBlur(arr, (0, 0), 1) - cv2.GaussianBlur(arr, (0, 0), 4)
    maxima = (response == cv2.dilate(response, np.ones((9, 9), np.uint8))) & (response > 3)
    maxima[:54] = maxima[-54:] = False
    maxima[:, :54] = maxima[:, -54:] = False
    ys, xs = np.nonzero(maxima)
    chosen = np.argsort(response[ys, xs])[::-1][:max_centers]
    xy = np.column_stack([xs[chosen], ys[chosen]])
    if len(xy) < 2:
        return dict(centers_tested=len(xy), pairs=[])
    yy, xx = np.mgrid[-50:51, -50:51]
    mask = (xx ** 2 + yy ** 2 >= 20 ** 2) & (xx ** 2 + yy ** 2 <= 48 ** 2)
    crops = [arr[y-50:y+51, x-50:x+51] for x, y in xy]
    vec = _unit(np.stack([c[mask] for c in crops]))
    corr = vec @ vec.T
    ii, jj = np.nonzero(np.triu(corr >= .70, 1))
    high = arr - cv2.GaussianBlur(arr, (0, 0), 3)
    pairs = []
    for i, j in zip(ii, jj):
        if np.linalg.norm(xy[i] - xy[j]) < 120:
            continue
        x, y = xy[i]
        q = _unit(high[y-50:y+51, x-50:x+51][mask][None])[0]
        x, y = xy[j]
        best = None
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                b = arr[y+dy-50:y+dy+51, x+dx-50:x+dx+51]
                raw = float(np.clip(vec[i] @ _unit(b[mask][None])[0], -1, 1))
                if raw < .85:
                    continue
                hp = high[y+dy-50:y+dy+51, x+dx-50:x+dx+51]
                fine = float(np.clip(q @ _unit(hp[mask][None])[0], -1, 1))
                if fine >= .85 and (best is None or fine > best['highpass_ncc']):
                    best = dict(a=xy[i].tolist(), b=[int(x+dx), int(y+dy)],
                                annulus_ncc=raw, highpass_ncc=fine)
        if best:
            pairs.append(best)
    return dict(centers_tested=len(xy), pairs=sorted(pairs, key=lambda p: -p['highpass_ncc']))


def clean_dataset(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError('Output must be separate from and not contain the original dataset')
    output.mkdir(parents=True, exist_ok=True)
    records, skies, errors, hashes = [], [], [], {}
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(root.rglob('*')) if p.is_file()}
    for split in ('train', 'validation'):
        if not (root / split).exists():
            continue
        for scene in discover_scenes(root / split):
            print(f'Clean/audit {split}/{scene.scene_id}', flush=True)
            sky = read_gray(scene.image_path)
            skies.append(dict(split=split, scene=scene.scene_id, shape=list(sky.shape),
                              copy_candidates=copy_neighborhoods(sky)))
            clean, ids = [], []
            for path in scene.patches:
                rel = path.relative_to(root).as_posix()
                try:
                    image = read_gray(path)
                    if image.shape != (32, 32):
                        raise ValueError(f'Expected 32x32, got {image.shape}')
                except ValueError as exc:
                    errors.append(dict(path=rel, error=str(exc)))
                    continue
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                hashes.setdefault(digest, []).append(rel)
                records.append(dict(path=rel, pixel_sha256=digest, **quality(image)))
                ids.append(path.stem)
                clean.append(normalized_view(image))
            if clean:
                dest = output / 'normalized' / split
                dest.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest / f'{scene.scene_id}.npz',
                                    patch_ids=np.array(ids), normalized=np.stack(clean))
    after = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob('*')) if p.is_file()}
    if after != source_hashes:
        raise RuntimeError('Source dataset changed during audit; integrity comparison failed')
    report = dict(policy='No deletion, relabeling, recentering, or original-pixel changes. '
                         'Normalized float32 views are diagnostics, not inference inputs. '
                         'Flags are heuristic; clipping and blur cannot be inferred from brightness alone.',
                  copy_search_limits='Top 300 bright peaks, translation +/-2px, annulus radii 20..48; '
                                     'candidate pairs, not an exhaustive or semantic duplicate classification.',
                  source_files_verified=len(source_hashes), source_hashes=source_hashes,
                  patches=records, scenes=skies, errors=errors,
                  exact_pixel_groups=[v for v in hashes.values() if len(v) > 1])
    report['summary'] = dict(patches=len(records), scenes=len(skies), errors=len(errors),
                            exact_duplicate_groups=len(report['exact_pixel_groups']),
                            copy_candidate_pairs=sum(len(s['copy_candidates']['pairs']) for s in skies),
                            flags={f: sum(f in r['flags'] for r in records)
                                   for f in sorted({f for r in records for f in r['flags']})})
    (output / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=Path('outputs/cleaned'))
    args = parser.parse_args()
    cv2.setNumThreads(8)
    result = clean_dataset(args.data, args.output)
    print(json.dumps(result['summary'], indent=2))
    if result['errors']:
        raise SystemExit('Invalid images found; see manifest. No files deleted.')
