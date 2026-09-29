"""Optional label-free constellation search using candidate and copy geometry.

The coarse search and confidence gate are deliberately separate from the
default localizer. It only uses the current sky, queries and supplied patterns.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .copy_annulus import find_copies
from .copy_evidence import clustered_pairs
from .direct_search import find_candidates
from .fullres_ncc import relative_gap
from .hough_probe import search_pattern
from .hough_likelihood import likelihood_score
from .hough_refine import prepare_groups, score_pose, select_hits
from .localize import Localizer, LocalizerConfig, Match
from .star_catalog import broad_star_centres


def merge_extra_candidates(coarse: list[dict], extra: list[dict],
                           *, limit: int = 5, separation: float = 8) -> list[dict]:
    """Append a bounded independent stream without duplicate sites."""
    merged = list(coarse)
    for item in extra[:limit]:
        if all(np.hypot(item['x'] - c['x'], item['y'] - c['y']) >= separation
               for c in merged):
            merged.append(item)
    return merged


def confident_extra_candidates(extra: list[dict], gap_threshold: float) -> list[dict]:
    """Expose only a distinct top NCC site to the global geometry search."""
    return extra[:1] if relative_gap(extra) >= gap_threshold else []


def choose_hough_name(patterns, sky: np.ndarray, patches: list[np.ndarray],
                      matches: list[Match], baseline_name: str,
                      *, background_sigma: float = 7.0,
                      foreground_sigma: float = .6,
                      margin_threshold: float = 1.2,
                      include_reflection: bool = False,
                      include_small_figures: bool = False,
                      bright_star_weight: float = 0.,
                      star_seeded: bool = False,
                      extra_candidates: list[list[dict]] | None = None,
                      confident_extra_gap: float | None = None,
                      likelihood: bool = False,
                      log=print) -> dict:
    """Return an optional new name and node-assigned patch locations."""
    started = time.perf_counter()
    filtered = Localizer.filtered(sky, background_sigma, foreground_sigma)
    downsample = .375
    coarse_sky = cv2.resize(filtered, None, fx=downsample, fy=downsample,
                            interpolation=cv2.INTER_AREA)
    groups = []
    for i, patch in enumerate(patches):
        query = Localizer.filtered(patch, background_sigma, foreground_sigma)
        candidates = find_candidates(coarse_sky, query, downsample=downsample)[:12]
        if extra_candidates is not None:
            # Keep this independent candidate stream bounded; adding every
            # peak would make chance constellations too easy to assemble.
            extra = extra_candidates[i]
            if confident_extra_gap is not None:
                extra = confident_extra_candidates(extra, confident_extra_gap)
            candidates = merge_extra_candidates(candidates, extra)
        if matches[i].present:
            candidates.append(dict(x=matches[i].x, y=matches[i].y,
                                   scale=matches[i].scale,
                                   degrees=-np.rad2deg(matches[i].angle),
                                   score=matches[i].score))
        groups.append(candidates)
        if (i + 1) % 10 == 0 or i + 1 == len(patches):
            log(f'  direct candidates {i+1}/{len(patches)}')
    direct_seconds = time.perf_counter() - started
    groups = prepare_groups(groups)
    grid_step = 8
    height = int(np.ceil(sky.shape[0] / grid_step))
    width = int(np.ceil(sky.shape[1] / grid_step))
    evidence = np.zeros((height, width), dtype=np.float32)
    for group in groups:
        for item in group:
            x = int(round(item['x'] / grid_step))
            y = int(round(item['y'] / grid_step))
            if 0 <= x < width and 0 <= y < height:
                evidence[y, x] = 1
    evidence = cv2.dilate(evidence, np.ones((3, 3), np.uint8))
    stars = broad_star_centres(sky) if bright_star_weight > 0 or star_seeded else np.empty((0, 2))
    if star_seeded:
        star_evidence = np.zeros_like(evidence)
        for sx, sy in stars:
            x, y = int(round(sx / grid_step)), int(round(sy / grid_step))
            if 0 <= x < width and 0 <= y < height:
                star_evidence[y, x] = 1
        star_evidence = cv2.dilate(star_evidence, np.ones((3, 3), np.uint8))
        evidence = np.maximum(evidence, star_evidence)
    copies = find_copies(sky)
    _, selected = clustered_pairs(copies['pairs'])
    anchors = np.asarray([xy for item in selected for xy in (item['a'], item['b'])],
                         dtype=float).reshape(-1, 2)
    copy_tree = cKDTree(anchors) if len(anchors) else None
    star_tree = cKDTree(stars) if bright_star_weight > 0 and len(stars) else None
    patch_gaps = [((group[0]['score'] - group[1]['score']) /
                   max(1 - group[0]['score'], .01))
                  if len(group) > 1 else 0.
                  for group in (extra_candidates or [[] for _ in patches])]
    log(f'  copy evidence {len(selected)} paired neighborhoods')
    copy_seconds = time.perf_counter() - started - direct_seconds
    scales = np.arange(3., 8.25, .25)
    angles = np.arange(0., 360., 5.)
    ranked = []
    for pattern_index, pattern in enumerate(patterns):
        if include_small_figures and len(pattern.nodes) < 4:
            # Two or three points admit too many accidental sky alignments;
            # they need a different source of evidence than this Hough grid.
            continue
        min_support = min(6, len(pattern.nodes)) if include_small_figures else 6
        _, hits = search_pattern(evidence, pattern.nodes, grid_step, scales,
                                 angles, min_support=min_support,
                                 include_reflection=include_reflection,
                                 max_hits_per_transform=5 if star_seeded else 0)
        best = None
        for hit in select_hits(hits, 200):
            value = score_pose(hit, pattern.nodes, groups, copy_tree=copy_tree,
                               star_tree=star_tree,
                               image_shape=sky.shape)
            count = max(value['in_frame_nodes'], 1)
            budget_prior = max(-.5 * (np.log(len(patches) / count / 2.97) / .35) ** 2,
                               -20.)
            value['name'] = pattern.name
            if likelihood:
                value['combined'] = likelihood_score(
                    value, len(pattern.nodes), patch_gaps,
                    budget_prior=budget_prior)
            else:
                value['combined'] = (value['weighted']
                                     - .3 * max(0, count - value['matched'])
                                     + value['copy_nodes'] + budget_prior
                                     + bright_star_weight * value['star_score'])
            if best is None or value['combined'] > best['combined']:
                best = value
        if best is not None:
            ranked.append(best)
        if (pattern_index + 1) % 12 == 0 or pattern_index + 1 == len(patterns):
            log(f'  global patterns {pattern_index+1}/{len(patterns)}')
    geometry_seconds = time.perf_counter() - started - direct_seconds - copy_seconds
    ranked.sort(key=lambda item: -item['combined'])
    gap = (ranked[0]['combined'] - ranked[1]['combined']) if len(ranked) > 1 else 0.
    use = bool(len(ranked) > 1 and gap >= margin_threshold)
    chosen = ranked[0] if use else None
    assignments = ({item['patch']: dict(x=item['x'], y=item['y'])
                    for item in chosen['chosen']} if chosen else {})
    result = dict(used=use, baseline_name=baseline_name,
                  include_reflection=include_reflection,
                  include_small_figures=include_small_figures,
                  bright_star_weight=bright_star_weight,
                  star_seeded=star_seeded,
                  extra_candidates=extra_candidates is not None,
                  confident_extra_gap=confident_extra_gap,
                  likelihood=likelihood,
                  name=chosen['name'] if chosen else baseline_name,
                  hough_name=ranked[0]['name'] if ranked else None,
                  margin=float(gap), assigned=assignments,
                  copy_pairs=len(selected),
                  stage_seconds=dict(direct=direct_seconds, copy=copy_seconds,
                                     geometry=geometry_seconds),
                  ranked=[dict(name=item['name'], combined=float(item['combined']),
                               matched=item['matched'], in_frame_nodes=item['in_frame_nodes'],
                               copy_nodes=item['copy_nodes'],
                               star_hits=item['star_hits'],
                               star_score=item['star_score']) for item in ranked[:5]],
                  seconds=time.perf_counter()-started)
    log(f"  Hough name {result['hough_name']} margin {gap:.2f}; "
        f"{'accepted' if use else 'kept baseline'}")
    return result
