"""Training-only replay of confidence-gated NCC sites in the Hough stage.

Candidate sites come from the pixel-only NCC audit. Ground truth is used only
after inference to score the experiment, never to select candidates or poses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import DEFAULT_DATA, discover_scenes, read_gray, read_rows
from .fullres_ncc import relative_gap
from .hough_infer import choose_hough_name
from .identify import load_patterns
from .localize import Match
from .metrics import score_scene


def original_match(detail: dict) -> Match:
    """Undo saved Hough coordinate overrides in frozen patch diagnostics."""
    alternative = detail['alternatives'][0] if detail['alternatives'] else detail
    return Match(x=float(alternative['x']), y=float(alternative['y']),
                 score=float(detail['score']),
                 second_score=float(detail['second_score']),
                 margin=float(detail['margin']), scale=float(detail['scale']),
                 angle=float(detail['angle']),
                 present=bool(detail['baseline_present']),
                 method=detail['method'], alternatives=detail['alternatives'])


def replay_row(scene_id: str, matches: list[Match], candidates: list[list[dict]],
               hough: dict, gap_threshold: float) -> dict[str, str]:
    assigned = {int(i) - 1: value for i, value in hough['assigned'].items()}
    row = {'Id': scene_id, 'n_patches': str(len(matches)),
           'constellation': hough['name']}
    for i, match in enumerate(matches):
        location = assigned.get(i)
        x, y = ((float(location['x']), float(location['y'])) if location else
                (match.x, match.y))
        present = match.present or location is not None
        if location is None and relative_gap(candidates[i]) >= gap_threshold:
            x = float(candidates[i][0]['x'])
            y = float(candidates[i][0]['y'])
            present = True
        row[f'patch_{i+1:02d}'] = (
            f'({x:.2f}, {y:.2f}, {int(location is not None)})' if present else '-1')
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', help='Optional labelled training scene')
    parser.add_argument('--gap', type=float, default=.16)
    parser.add_argument('--output', type=Path,
                        default=Path('outputs/confident-hough-audit/report.json'))
    args = parser.parse_args()
    audit = json.loads(Path('outputs/fullres-ncc-audit/hp20-train.json').read_text())
    details = json.loads(Path('outputs/hough-broadstar-train/evaluate-patches.json').read_text())
    baseline = json.loads(Path('outputs/hough-broadstar-train/evaluate-report.json').read_text())
    _, truth_rows = read_rows(DEFAULT_DATA / 'train_ground_truth.csv')
    scenes = {s.scene_id: s for s in discover_scenes(DEFAULT_DATA / 'train')}
    truth = {r['Id']: r for r in truth_rows}
    details_by_scene = {name: [d for d in details if d['scene'] == name]
                        for name in scenes}
    audit_by_scene = {name: [e for e in audit['entries'] if e['scene'] == name]
                      for name in scenes}
    baseline_by_scene = {r['scene']: r for r in baseline['scenes']}
    patterns = load_patterns(DEFAULT_DATA / 'patterns')
    results = []
    for scene_id, scene in scenes.items():
        if args.scene and scene_id != args.scene:
            continue
        ordered_details = sorted(details_by_scene[scene_id], key=lambda d: d['patch'])
        ordered_audit = sorted(audit_by_scene[scene_id], key=lambda e: e['patch'])
        if len(ordered_details) != len(scene.patches) or len(ordered_audit) != len(scene.patches):
            raise ValueError(f'Missing cached evidence for {scene_id}')
        matches = [original_match(d) for d in ordered_details]
        candidates = [e['candidates'] for e in ordered_audit]
        sky = read_gray(scene.image_path)
        patches = [read_gray(p) for p in scene.patches]
        old = baseline_by_scene[scene_id]
        hough = choose_hough_name(
            patterns, sky, patches, matches,
            old['hough_identification']['baseline_name'],
            bright_star_weight=1., extra_candidates=candidates,
            confident_extra_gap=args.gap)
        row = replay_row(scene_id, matches, candidates, hough, args.gap)
        metric = score_scene(truth[scene_id], row)
        result = dict(scene=scene_id, baseline_score=old['metrics']['score'],
                      score=metric['score'], metrics=metric,
                      true_name=truth[scene_id]['constellation'],
                      predicted_name=row['constellation'], hough=hough)
        results.append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(json.dumps({key: result[key] for key in
                          ('scene', 'baseline_score', 'score', 'true_name',
                           'predicted_name')}), flush=True)
    print('Results:', args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
