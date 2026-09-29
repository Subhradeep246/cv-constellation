"""Label-free audit of confident NCC sites in uncertain validation scenes.

This writes diagnostics only, never a Kaggle submission or name override.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .candidate_cache import get_fullres_candidates
from .data import DEFAULT_DATA, discover_scenes, read_gray
from .hough_infer import choose_hough_name
from .identify import load_patterns
from .localize import Match


def original_match(detail: dict) -> Match:
    alternative = detail['alternatives'][0] if detail['alternatives'] else detail
    return Match(x=float(alternative['x']), y=float(alternative['y']),
                 score=float(detail['score']),
                 second_score=float(detail['second_score']),
                 margin=float(detail['margin']), scale=float(detail['scale']),
                 angle=float(detail['angle']),
                 present=bool(detail['baseline_present']),
                 method=detail['method'], alternatives=detail['alternatives'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', help='Audit one validation scene')
    parser.add_argument('--gap', type=float, default=.16)
    parser.add_argument('--output', type=Path,
                        default=Path('outputs/confident-hough-validation/report.json'))
    args = parser.parse_args()
    base = Path('outputs/hough-broadstar-validation')
    report = json.loads((base / 'predict-report.json').read_text())
    details = json.loads((base / 'predict-patches.json').read_text())
    scenes = {s.scene_id: s for s in discover_scenes(DEFAULT_DATA / 'validation')}
    patterns = load_patterns(DEFAULT_DATA / 'patterns')
    results = []
    for original in report['scenes']:
        scene_id = original['scene']
        if args.scene and scene_id != args.scene:
            continue
        if not args.scene and original['hough_identification']['used']:
            continue
        scene = scenes[scene_id]
        ordered = sorted((d for d in details if d['scene'] == scene_id),
                         key=lambda d: d['patch'])
        matches = [original_match(d) for d in ordered]
        sky = read_gray(scene.image_path)
        patches = [read_gray(p) for p in scene.patches]
        candidates = get_fullres_candidates(
            sky, patches,
            Path('outputs/fullres-cache-shared') / f'validation-{scene_id}.json',
            log=lambda _: None)
        revised = choose_hough_name(
            patterns, sky, patches, matches,
            original['hough_identification']['baseline_name'],
            bright_star_weight=1., extra_candidates=candidates,
            confident_extra_gap=args.gap,
            log=lambda _: None)
        result = dict(scene=scene_id, old_name=original['constellation'],
                      old_margin=original['hough_identification']['margin'],
                      new_name=revised['name'], new_hough_name=revised['hough_name'],
                      new_margin=revised['margin'], new_used=revised['used'],
                      assigned=len(revised['assigned']), ranked=revised['ranked'])
        results.append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)
    print('Saved', args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
