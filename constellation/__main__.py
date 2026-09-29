from __future__ import annotations

# Bound numerical threading BEFORE importing NumPy/OpenCV on a laptop.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import argparse
import csv
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .audit import audit_dataset
from .candidate_cache import get_fullres_candidates
from .cleaning import MODES, prepare_image, unique_patches
from .data import DEFAULT_DATA, discover_scenes, read_gray, read_rows, row_points, write_rows, validate_submission
from .fullres_ncc import relative_gap
from .identify import load_patterns, rank_patterns, rank_patterns_with_alternatives
from .hough_infer import choose_hough_name
from .localize import Localizer, LocalizerConfig
from .metrics import score_scene


def main():
    parser = argparse.ArgumentParser(description="Local constellation detection pipeline")
    parser.add_argument("command", choices=["audit", "evaluate", "predict", "oracle-identify"])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--scene", help="Select one scene for development; never used by the algorithm")
    parser.add_argument("--limit-patches", type=int, help="Smoke test only; produces a clearly labelled partial evaluation")
    parser.add_argument("--sources", type=int, default=35000)
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--refine-candidates", type=int, default=8,
                        help="Number of distinct polar candidates to verify per patch; 8 is baseline")
    parser.add_argument("--threshold", type=float, default=.80)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--grid-step", type=int, default=0, help="Optional dense position fallback; 4 is thorough, 0 is fast")
    parser.add_argument("--no-sift", action="store_true")
    parser.add_argument("--preprocess", choices=MODES, default="none",
                        help="Experimental image preparation, applied to both sky and queries")
    parser.add_argument("--background-sigma", type=float, default=7.0,
                        help="Gaussian background scale in the difference-of-Gaussians matcher; 7 is baseline")
    parser.add_argument("--foreground-sigma", type=float, default=.6,
                        help="Fine Gaussian scale in the difference-of-Gaussians matcher; .6 is baseline")
    parser.add_argument("--descriptor", choices=('magnitude', 'crossphase'), default='magnitude',
                        help="Rotation-invariant retrieval descriptor; magnitude is baseline")
    parser.add_argument("--descriptor-whitening", action="store_true",
                        help="Experimental per-scene diagonal whitening of retrieval descriptors")
    parser.add_argument("--no-harris", action="store_true",
                        help="Disable Harris corners; DoG blob peaks remain")
    parser.add_argument("--skip-identification", action="store_true")
    parser.add_argument("--joint-alternatives", action="store_true",
                        help="Let the ranked constellation hypothesis choose among verified patch locations")
    parser.add_argument("--hough-identification", action="store_true",
                        help="Experimental, slower copy-aware global pose search and patch assignment")
    parser.add_argument("--mirror-hough", action="store_true",
                        help="Experimental reflection hypotheses in the copy-aware pose search")
    parser.add_argument("--small-hough", action="store_true",
                        help="Experimental four- and five-node global hypotheses")
    parser.add_argument("--bright-star-weight", type=float, default=0.,
                        help="Experimental support from broad DoG sky stars; 0 disables it")
    parser.add_argument("--star-seeded-hough", action="store_true",
                        help="Experimental sky-star evidence in coarse Hough generation")
    parser.add_argument("--fullres-ncc", action="store_true",
                        help="Experimental full-resolution masked-NCC patch corrections (slow)")
    parser.add_argument("--fullres-hough", action="store_true",
                        help="Also feed top five masked-NCC sites into global pose search")
    parser.add_argument("--confident-fullres-hough", action="store_true",
                        help="Feed only distinct top NCC sites to Hough, using --fullres-gap")
    parser.add_argument("--likelihood-hough", action="store_true",
                        help="Experimental supported/unsupported-node likelihood score")
    parser.add_argument("--fullres-gap", type=float, default=.16,
                        help="Relative gap required to accept a masked-NCC patch correction")
    parser.add_argument("--fullres-cache-dir", type=Path,
                        help="Optional shared cache directory for content-checked NCC candidates")
    parser.add_argument("--no-preprocess-probe", action="store_true",
                        help="Skip the optional training-only preprocess comparison during audit")
    args = parser.parse_args()
    if (args.sources < 1 or args.top_k < 1 or args.refine_candidates < 1 or
            args.grid_step < 0 or args.threads < 1):
        parser.error("Sources, top-k, refine-candidates and threads must be positive; grid-step must be nonnegative")
    if not 2 <= args.background_sigma <= 64:
        parser.error("Background sigma must be in [2,64]")
    if not 0 < args.foreground_sigma < args.background_sigma:
        parser.error("Foreground sigma must be positive and smaller than background sigma")
    if not 0 <= args.threshold <= 1 or not 0 <= args.margin <= 2:
        parser.error("Threshold must be in [0,1], margin in [0,2]")
    if args.limit_patches is not None and (args.limit_patches < 1 or args.command == "predict"):
        parser.error("--limit-patches must be positive and is disallowed for submission prediction")
    if args.hough_identification and args.skip_identification:
        parser.error("--hough-identification is incompatible with --skip-identification")
    if args.mirror_hough and not args.hough_identification:
        parser.error("--mirror-hough requires --hough-identification")
    if args.small_hough and not args.hough_identification:
        parser.error("--small-hough requires --hough-identification")
    if args.bright_star_weight < 0 or args.bright_star_weight > 3:
        parser.error("--bright-star-weight must be in [0,3]")
    if args.bright_star_weight and not args.hough_identification:
        parser.error("--bright-star-weight requires --hough-identification")
    if args.star_seeded_hough and (not args.hough_identification or
                                   not args.bright_star_weight):
        parser.error("--star-seeded-hough requires --hough-identification and --bright-star-weight")
    if not 0 <= args.fullres_gap <= 5:
        parser.error("--fullres-gap must be in [0,5]")
    if args.fullres_hough and (not args.fullres_ncc or not args.hough_identification):
        parser.error("--fullres-hough requires --fullres-ncc and --hough-identification")
    if args.confident_fullres_hough and not args.fullres_hough:
        parser.error("--confident-fullres-hough requires --fullres-hough")
    if args.likelihood_hough and not (args.fullres_hough and args.bright_star_weight):
        parser.error("--likelihood-hough requires --fullres-hough and --bright-star-weight")
    cv2.setNumThreads(max(1, args.threads))
    cv2.setRNGSeed(17)
    patterns = load_patterns(args.data / "patterns")
    fields, sample = read_rows(args.data / "sample_submission.csv")
    truth = []
    if args.command != 'predict':
        _, truth = read_rows(args.data / "train_ground_truth.csv")
    if args.command == "audit":
        environment = {
            "python": platform.python_version(), "logical_cpus": os.cpu_count(),
            "opencv": cv2.__version__, "numpy": np.__version__, "data": str(args.data),
            "patterns": [{"name": p.name, "nodes": len(p.nodes)} for p in patterns],
            "training": [{"id": r["Id"], "patches": int(r["n_patches"]),
                          "present": sum(p is not None for p in row_points(r)),
                          "figure": sum(p is not None and p[2] for p in row_points(r))} for r in truth],
            "validation_scenes": len(sample),
            "validation_patches": sum(int(r["n_patches"]) for r in sample),
        }
        report = audit_dataset(args.data, include_preprocess_probe=not args.no_preprocess_probe)
        report["environment"] = environment
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "patch-audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"environment": environment, "policy": report["policy"],
                          "summary": report["summary"],
                          "preprocess_probe": {k: report["preprocess_probe"].get(k)
                                               for k in ("available", "patches_probed", "summary",
                                                         "methods_beating_raw_mean_score",
                                                         "enabled_in_matcher", "note")
                                               if k in report["preprocess_probe"]},
                          "report": str((args.output / "patch-audit.json").resolve())}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "oracle-identify":
        result = []
        for row in truth:
            if args.scene and row['Id'] != args.scene:
                continue
            for mode in ("figure_only", "all_present"):
                pts = np.array([p[:2] for p in row_points(row) if p and (mode == 'all_present' or p[2])])
                start = time.perf_counter()
                ranked = rank_patterns(patterns, pts)
                entry = dict(scene=row['Id'], mode=mode, seconds=time.perf_counter()-start,
                             true_rank=next((i+1 for i,r in enumerate(ranked) if r['name']==row['constellation']), None),
                             ranking=ranked)
                result.append(entry)
                print(f"{row['Id']} {mode}: true rank {entry['true_rank']}; top {ranked[0]['name'] if ranked else 'unknown'}", flush=True)
                (args.output / 'oracle-identification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return
    split = 'train' if args.command == 'evaluate' else 'validation'
    schema = truth if args.command == 'evaluate' else sample
    scenes = {s.scene_id:s for s in discover_scenes(args.data / split)}
    if args.scene and args.scene not in scenes:
        parser.error(f"Unknown scene: {args.scene}")
    if args.command == 'predict' and args.scene:
        parser.error("--scene is disallowed for full submission prediction")
    config = LocalizerConfig(max_sources=args.sources, candidates_per_scale=args.top_k,
                             refine_candidates=args.refine_candidates,
                             threshold=args.threshold, minimum_margin=args.margin, sift=not args.no_sift,
                             harris=not args.no_harris, grid_step=args.grid_step,
                             background_sigma=args.background_sigma,
                             foreground_sigma=args.foreground_sigma,
                             descriptor_mode=args.descriptor,
                             descriptor_whitening=args.descriptor_whitening)
    localizer = Localizer(config)
    predictions, reports, diagnostics = [], [], []
    image_sizes = {}
    start_all = time.perf_counter()
    for original in schema:
        if args.scene and original['Id'] != args.scene:
            continue
        scene = scenes.get(original['Id'])
        if scene is None or len(scene.patches) != int(original['n_patches']):
            raise ValueError(f"Missing scene or patch count mismatch: {original['Id']}")
        if {p.stem for p in scene.patches} != {f'patch_{i:02d}' for i in range(1,int(original['n_patches'])+1)}:
            raise ValueError(f"Patch filenames do not match the CSV columns: {original['Id']}")
        paths = scene.patches[:args.limit_patches] if args.limit_patches else scene.patches
        print(f"Scene {scene.scene_id}: {len(paths)} patches", flush=True)
        sky = read_gray(scene.image_path)
        image_sizes[scene.scene_id] = (sky.shape[1],sky.shape[0])
        prepared_sky = prepare_image(sky, args.preprocess)
        patches = [prepare_image(read_gray(p), args.preprocess) for p in paths]
        unique, inverse = unique_patches(patches)
        unique_matches, stats = localizer.match_scene(prepared_sky, unique,
                                                     log=lambda s:print(s, flush=True))
        matches = [unique_matches[i] for i in inverse]
        fullres_candidates = None
        if args.fullres_ncc:
            cache_root = args.fullres_cache_dir or args.output / 'fullres_cache'
            fullres_candidates = get_fullres_candidates(
                prepared_sky, patches, cache_root / f'{split}-{scene.scene_id}.json',
                log=lambda s: print(s, flush=True))
        stats['unique_queries'] = len(unique)
        stats['issued_queries'] = len(patches)
        stats['proposals_per_patch'] = [stats['proposals_per_patch'][i] for i in inverse]
        indices = sorted([i for i,m in enumerate(matches) if m.present], key=lambda i:matches[i].score, reverse=True)
        candidate_groups = [
            matches[i].alternatives or [dict(x=matches[i].x, y=matches[i].y, score=matches[i].score,
                                             scale=matches[i].scale, angle=matches[i].angle)]
            for i in indices
        ]
        if args.skip_identification:
            ranked = []
        elif args.joint_alternatives:
            ranked = rank_patterns_with_alternatives(patterns, candidate_groups)
        else:
            ranked = rank_patterns(patterns, np.array([[matches[i].x,matches[i].y] for i in indices]))
        name = ranked[0]['name'] if ranked else 'unknown'
        members = {indices[i] for i in ranked[0]['point_indices']} if ranked else set()
        assigned_locations = ({indices[int(i)]: (float(value['x']), float(value['y']))
                               for i, value in ranked[0].get('point_locations', {}).items()}
                              if ranked else {})
        hough_result = None
        if args.hough_identification:
            hough_result = choose_hough_name(
                patterns, prepared_sky, patches, matches, name,
                background_sigma=args.background_sigma,
                foreground_sigma=args.foreground_sigma,
                include_reflection=args.mirror_hough,
                include_small_figures=args.small_hough,
                bright_star_weight=args.bright_star_weight,
                star_seeded=args.star_seeded_hough,
                extra_candidates=fullres_candidates if args.fullres_hough else None,
                confident_extra_gap=args.fullres_gap if args.confident_fullres_hough else None,
                likelihood=args.likelihood_hough,
                log=lambda s: print(s, flush=True))
            if hough_result['used']:
                name = hough_result['name']
                assigned_locations.update({int(i)-1: (float(value['x']), float(value['y']))
                                           for i, value in hough_result['assigned'].items()})
                members = {int(i)-1 for i in hough_result['assigned']}
        hough_assigned = ({int(i)-1 for i in hough_result['assigned']}
                          if hough_result and hough_result['used'] else set())
        fullres_promoted = 0
        row = {key:'-1' for key in fields}
        row.update(Id=scene.scene_id, n_patches=str(len(paths)), constellation=name)
        for i, (path, match) in enumerate(zip(paths, matches)):
            x, y = assigned_locations.get(i, (match.x, match.y))
            present = match.present or i in hough_assigned
            fullres_detail = None
            if fullres_candidates is not None and i not in hough_assigned:
                candidates = fullres_candidates[i]
                gap = relative_gap(candidates)
                accepted = bool(candidates and gap >= args.fullres_gap)
                if accepted:
                    x, y = candidates[0]['x'], candidates[0]['y']
                    present = True
                    fullres_promoted += 1
                fullres_detail = dict(gap=gap, accepted=accepted,
                                      best=candidates[0] if candidates else None,
                                      runner_up=candidates[1] if len(candidates) > 1 else None)
            if present:
                row[path.stem] = f"({x:.2f}, {y:.2f}, {int(i in members)})"
            detail = dict(scene=scene.scene_id, patch=path.stem, **match.to_dict())
            detail['x'], detail['y'] = x, y
            detail['baseline_present'] = match.present
            detail['present'] = present
            detail['hough_assigned'] = i in hough_assigned
            if fullres_detail is not None:
                detail['fullres_ncc'] = fullres_detail
            if args.command == 'evaluate':
                gt = row_points(original)[i]
                detail.update(truth_present=gt is not None, truth_member=gt[2] if gt else 0,
                              error_px=float(np.linalg.norm(np.array([x,y])-gt[:2])) if gt else None)
            diagnostics.append(detail)
        predictions.append(row)
        report = dict(scene=scene.scene_id, **stats, constellation=name, ranking=ranked[:5],
                      joint_alternatives=args.joint_alternatives,
                      hough_identification=hough_result,
                      fullres_ncc=dict(enabled=args.fullres_ncc,
                                       hough_candidates=args.fullres_hough,
                                       gap_threshold=args.fullres_gap,
                                       accepted=fullres_promoted))
        if args.command == 'evaluate':
            partial_truth = dict(original, n_patches=str(len(paths)))
            report['metrics'] = score_scene(partial_truth, row)
            print(json.dumps(report['metrics'], indent=2), flush=True)
        reports.append(report)
        # Checkpoint results after each completed scene.
        tag = 'partial' if args.limit_patches else args.command
        write_rows(args.output / f'{tag}-predictions.csv', fields, predictions)
        (args.output / f'{tag}-patches.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
        summary = dict(config=asdict(config), preprocess=args.preprocess, partial=bool(args.limit_patches), seconds=time.perf_counter()-start_all, scenes=reports)
        if args.command == 'evaluate':
            summary['mean_metrics'] = {k:float(np.mean([r['metrics'][k] for r in reports])) for k in ['presence','localization','geometric_recovery','identification','score']}
        (args.output / f'{tag}-report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if args.command == 'predict':
        # Only produce the final submission name after the entire requested set completes.
        validate_submission(predictions,sample,{p.name for p in patterns},image_sizes)
        write_rows(args.output / 'submission.csv', fields, predictions)
    print(f"Saved results in {args.output.resolve()} ({time.perf_counter()-start_all:.1f}s)", flush=True)


if __name__ == '__main__':
    main()
