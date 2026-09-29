"""Content-checked, label-free cache for expensive masked-NCC candidates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .fullres_ncc import masked_ncc_candidates
from .localize import Localizer


def _content_digest(sky: np.ndarray, patches: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for image in [sky, *patches]:
        array = np.ascontiguousarray(image)
        digest.update(str(array.shape).encode('ascii'))
        digest.update(array.dtype.str.encode('ascii'))
        digest.update(memoryview(array))
    return digest.hexdigest()


def get_fullres_candidates(sky: np.ndarray, patches: list[np.ndarray],
                           cache_path: Path, *, log=print) -> list[list[dict]]:
    """Load matching pixels' candidates or compute/checkpoint them per query."""
    implementation = hashlib.sha256()
    for filename in ('fullres_ncc.py', 'localize.py'):
        implementation.update(Path(__file__).with_name(filename).read_bytes())
    spec = {
        'input_sha256': _content_digest(sky, patches),
        'algorithm_sha256': implementation.hexdigest(),
        'channel': 'DoG(0.7,3)', 'angle_step': 20,
        'scales': [.93, 1.0, 1.08], 'keep': 30,
    }
    cached = None
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            cached = None
    if cached is None or cached.get('spec') != spec or not isinstance(cached.get('candidates'), list):
        cached = {'spec': spec, 'candidates': []}
    groups = cached['candidates']
    if len(groups) > len(patches) or any(not isinstance(g, list) for g in groups):
        groups = []
        cached['candidates'] = groups
    if len(groups) == len(patches):
        log(f'  loaded {len(groups)} full-resolution NCC candidate sets from cache')
        return groups
    filtered_sky = Localizer.filtered(sky, 3.0, .7)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for j in range(len(groups), len(patches)):
        query = Localizer.filtered(patches[j], 3.0, .7)
        groups.append(masked_ncc_candidates(filtered_sky, query,
                                            angle_step=20, keep=30))
        temporary = cache_path.with_suffix(cache_path.suffix + '.tmp')
        temporary.write_text(json.dumps(cached), encoding='utf-8')
        temporary.replace(cache_path)
        if (j + 1) % 10 == 0 or j + 1 == len(patches):
            log(f'  full-resolution NCC candidates {j+1}/{len(patches)}')
    return groups
