"""Partial point-set alignment against supplied schematic star nodes.

The initial search varies reflection, aspect ratio and similarity pose.
Promising hypotheses receive one-to-one assignment and affine refinement.
This is a bounded baseline, not a claim of affine invariance at retrieval.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .geometry import affine_from_points, ransac_affine


@dataclass
class Pattern:
    name: str
    nodes: np.ndarray


def load_patterns(folder: Path) -> list[Pattern]:
    patterns = []
    for path in sorted(folder.glob("*_pattern.png")):
        rgba = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3:
            raise ValueError(f"Expected an RGB/RGBA template: {path}")
        # Supplied artwork has white star disks and colored connecting lines.
        white = np.min(rgba[:, :, :3], axis=2) >= 185
        if rgba.shape[2] == 4:
            white &= rgba[:, :, 3] >= 128
        count, _, stats, centroids = cv2.connectedComponentsWithStats(white.astype(np.uint8), 8)
        nodes = np.array([centroids[i] for i in range(1, count) if 4 <= stats[i, cv2.CC_STAT_AREA] <= 400], dtype=float)
        if len(nodes) < 2:
            raise ValueError(f"Fewer than two star nodes extracted from {path}")
        patterns.append(Pattern(path.stem.removesuffix("_pattern"), nodes))
    if not patterns:
        raise ValueError(f"No reference patterns: {folder}")
    return patterns


def assignment(projected, points, tolerance):
    distances = np.linalg.norm(projected[:, None] - points[None], axis=2)
    # Dummy columns allow every template star to be unmatched.
    cost = np.concatenate([distances, np.full((len(projected), len(projected)), tolerance)], axis=1)
    a, b = linear_sum_assignment(cost)
    keep = b < len(points)
    a, b = a[keep], b[keep]
    keep = distances[a, b] < tolerance
    return a[keep], b[keep], distances[a[keep], b[keep]]


def rank_patterns_similarity(patterns: list[Pattern], points: np.ndarray, *, max_points=32) -> list[dict]:
    points = np.asarray(points, dtype=float).reshape(-1, 2)[:max_points]
    if len(points) < 4:
        return []
    span = float(np.linalg.norm(np.ptp(points, axis=0)))
    tolerance = max(20.0, span * .025)
    tree = cKDTree(points)
    pairs = np.array(list(combinations(range(len(points)), 2)))
    pairs = np.concatenate([pairs, pairs[:, ::-1]])
    delta = points[pairs[:, 1]] - points[pairs[:, 0]]
    lengths = np.linalg.norm(delta, axis=1)
    valid = lengths > span * .12
    pairs, delta, lengths = pairs[valid], delta[valid], lengths[valid]
    if not len(pairs):
        return []
    target_angle = np.arctan2(delta[:, 1], delta[:, 0])
    ranking = []
    for pattern in patterns:
        nodes = pattern.nodes
        anchors = np.array(list(combinations(range(len(nodes)), 2)))
        lens = np.linalg.norm(nodes[anchors[:, 1]] - nodes[anchors[:, 0]], axis=1)
        # Long noncoincident anchors reduce positional uncertainty.
        anchors = anchors[np.argsort(lens)[-min(24, len(anchors)):]]
        best = []
        for aspect in (.5, .75, 1.0, 1.33, 2.0):
            for handedness in (-1, 1):
                stretch = np.diag([aspect * handedness, 1.0])
                scaled = nodes @ stretch.T
                for anchor in anchors:
                    v = scaled[anchor[1]]-scaled[anchor[0]]
                    if np.linalg.norm(v) < 1:
                        continue
                    scale = lengths / np.linalg.norm(v)
                    angle = target_angle-np.arctan2(v[1], v[0])
                    co, si = scale*np.cos(angle), scale*np.sin(angle)
                    matrices = np.stack([co, -si, si, co], axis=1).reshape(-1, 2, 2) @ stretch
                    translation = points[pairs[:, 0]] - np.einsum('bij,j->bi', matrices, nodes[anchor[0]])
                    projected = np.einsum('bij,nj->bni', matrices, nodes) + translation[:, None]
                    distances, ids = tree.query(projected)
                    support = np.maximum(0, 1-(distances/tolerance)**2)
                    # Penalize collapsed many-to-one support even at coarse search.
                    for k in range(len(nodes)):
                        if k:
                            duplicate = np.any((ids[:, k, None] == ids[:, :k]) & (distances[:, :k] < tolerance), axis=1)
                            support[duplicate, k] = 0
                    score = support.sum(axis=1) / np.sqrt(len(nodes))
                    top = np.argsort(score)[-2:]
                    best.extend((float(score[i]), matrices[i], translation[i]) for i in top)
        best.sort(key=lambda t: t[0], reverse=True)
        refined = []
        for _, matrix, translation in best[:12]:
            for _ in range(4):
                projected = nodes @ matrix.T + translation
                a, b, distances = assignment(projected, points, tolerance)
                if len(a) < 4:
                    break
                design = np.column_stack([nodes[a], np.ones(len(a))])
                if np.linalg.matrix_rank(design) < 3:
                    break
                fit = np.linalg.lstsq(design, points[b], rcond=None)[0]
                updated = fit[:2].T
                singular = np.linalg.svd(updated, compute_uv=False)
                if singular[-1] < .01 or singular[0]/singular[-1] > 5:
                    break
                matrix, translation = updated, fit[2]
            a, b, distances = assignment(nodes @ matrix.T+translation, points, tolerance)
            support = np.maximum(0, 1-(distances/tolerance)**2).sum()
            value = float(support/np.sqrt(len(nodes)))
            refined.append(dict(name=pattern.name, score=value, matched_nodes=len(a), total_nodes=len(nodes),
                                residual_px=float(np.mean(distances)) if len(distances) else None,
                                point_indices=b.tolist(), tolerance_px=tolerance,
                                matrix=matrix.tolist(), translation=translation.tolist()))
        if refined:
            ranking.append(max(refined, key=lambda r:r['score']))
    return sorted(ranking, key=lambda r:r['score'], reverse=True)


def quad_index(points: np.ndarray):
    """Canonical four-point descriptor from opposite triangle area ratios.

    Under an affine transformation all triangle areas scale by |det(A)|.
    Sorting by opposite area yields a vertex ordering except in symmetric
    configurations. Nearly collinear quads are deliberately excluded.
    """
    quads = np.array(list(combinations(range(len(points)), 4)), dtype=int).reshape(-1, 4)
    if not len(quads):
        return np.empty((0, 3)), quads
    p = points[quads]
    areas = []
    for omit in range(4):
        tri = p[:, [i for i in range(4) if i != omit]]
        a, b = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
        areas.append(np.abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]))
    areas = np.stack(areas, axis=1)
    total = areas.sum(axis=1)
    spread = np.max(np.sum((p-p.mean(axis=1,keepdims=True))**2, axis=2), axis=1)
    valid = (total > spread*.1) & (areas.min(axis=1) > total*.015)
    areas, quads, total = areas[valid], quads[valid], total[valid]
    order = areas.argsort(axis=1)
    quads = np.take_along_axis(quads, order, axis=1)
    code = np.take_along_axis(areas, order, axis=1)/total[:, None]
    return code[:, :3], quads


def rank_patterns(patterns: list[Pattern], points: np.ndarray, *, max_points=40) -> list[dict]:
    """Affine quad hypotheses, RANSAC inlier selection, and least-squares refit.

    Feature matches (quad correspondences) can include outliers. RANSAC keeps
    the affine with the most inliers, then least squares refits those inliers.
    At least five consistent nodes are needed for an informative accepted
    hypothesis: four points alone can produce accidental affine agreements.
    Small patterns remain ambiguous under an unconstrained affine model.
    """
    points = np.asarray(points, dtype=float).reshape(-1,2)[:max_points]
    if len(points) < 4:
        return []
    code, target_quads = quad_index(points)
    if not len(code):
        return []
    hash_tree, point_tree = cKDTree(code), cKDTree(points)
    span = np.linalg.norm(np.ptp(points, axis=0))
    tolerance = max(12., float(span*.008))
    rng = np.random.default_rng(17)
    ranking = []
    for pattern in patterns:
        nodes = pattern.nodes
        if len(nodes) < 4:
            continue
        pc, pq = quad_index(nodes)
        if not len(pc):
            continue
        distance, target = hash_tree.query(pc, k=min(3,len(code)))
        distance, target = np.asarray(distance).reshape(len(pc),-1), np.asarray(target).reshape(len(pc),-1)
        a,b = np.nonzero(distance < .012)
        if not len(a):
            continue
        source = nodes[pq[a]]
        dest = points[target_quads[target[a,b]]]
        design = np.concatenate([source, np.ones((*source.shape[:2],1))], axis=2)
        # Batched pseudoinverse avoids an expensive Python hypothesis loop.
        fits = np.linalg.pinv(design) @ dest
        singular = np.linalg.svd(fits[:,:2,:], compute_uv=False)
        valid = (singular[:,1] > .05) & (singular[:,0]/np.maximum(singular[:,1],1e-8) < 8)
        fits = fits[valid]
        if not len(fits):
            continue
        augmented = np.column_stack([nodes,np.ones(len(nodes))])
        best = []
        for start in range(0,len(fits),1024):
            batch = fits[start:start+1024]
            projected = np.einsum('ni,bij->bnj',augmented,batch)
            d,ids = point_tree.query(projected)
            support = np.exp(-.5*(d/(tolerance*.5))**2)
            support[d >= tolerance] = 0
            for k in range(len(nodes)):
                if k:
                    duplicate = np.any((ids[:,k,None] == ids[:,:k]) & (d[:,:k]<tolerance),axis=1)
                    support[duplicate,k] = 0
            scores = support.sum(axis=1)
            for j in np.argsort(scores)[-5:]:
                best.append((float(scores[j]),batch[j]))
        best.sort(key=lambda r:r[0],reverse=True)
        records = []
        for _,fit in best[:10]:
            a,b,d = assignment(augmented@fit,points,tolerance)
            used_ransac = False
            if len(a) >= 3:
                consensus = ransac_affine(nodes[a], points[b], threshold=tolerance,
                                          trials=64, rng=rng)
                if consensus is not None and consensus.n_inliers >= 3:
                    refit = affine_from_points(nodes[a][consensus.inliers], points[b][consensus.inliers])
                    fit = refit if refit is not None else consensus.model
                    used_ransac = True
            if not used_ransac:
                for _ in range(3):
                    a,b,d = assignment(augmented@fit,points,tolerance)
                    if len(a)<4 or np.linalg.matrix_rank(augmented[a])<3:
                        break
                    updated = np.linalg.lstsq(augmented[a],points[b],rcond=None)[0]
                    singular = np.linalg.svd(updated[:2],compute_uv=False)
                    if singular[-1] < .05 or singular[0]/singular[-1]>8:
                        break
                    fit=updated
            a,b,d=assignment(augmented@fit,points,tolerance)
            score=float(np.exp(-.5*(d/(tolerance*.5))**2).sum()-.015*len(nodes))
            records.append(dict(name=pattern.name,score=score,matched_nodes=len(a),total_nodes=len(nodes),
                                residual_px=float(d.mean()) if len(d) else None,point_indices=b.tolist(),
                                tolerance_px=tolerance,matrix=fit[:2].T.tolist(),translation=fit[2].tolist()))
        if records:
            ranking.append(max(records,key=lambda r:r['score']))
    return sorted(ranking,key=lambda r:r['score'],reverse=True)


def rank_patterns_with_alternatives(
    patterns: list[Pattern],
    candidates: list[list[dict]],
    *,
    max_points: int = 40,
    iterations: int = 4,
    alternative_penalty: float = 0.0,
) -> list[dict]:
    """Refine affine fits by assigning distinct patches and nodes to alternatives.

    The ordinary ranker supplies an initialization. This cannot discover poses
    absent from the top-location initialization; the experiment tests whether
    its existing pose improves when ambiguous patch locations remain available.
    """
    candidates = candidates[:max_points]
    if not candidates or any(not group for group in candidates):
        return []
    initial_points = np.array([[group[0]['x'], group[0]['y']] for group in candidates], dtype=float)
    seeds = rank_patterns(patterns, initial_points, max_points=max_points)
    if not seeds:
        return []
    extent = float(np.linalg.norm(np.ptp(initial_points, axis=0)))
    tolerance = max(12.0, extent * .008)
    output = []
    for seed in seeds:
        pattern = next((p for p in patterns if p.name == seed['name']), None)
        if pattern is None:
            continue
        nodes = pattern.nodes
        matrix = np.asarray(seed['matrix'], dtype=float)
        translation = np.asarray(seed['translation'], dtype=float)
        for _ in range(iterations):
            projected = nodes @ matrix.T + translation
            distances = np.empty((len(nodes), len(candidates)), dtype=float)
            alt_ids = np.empty((len(nodes), len(candidates)), dtype=int)
            for j, group in enumerate(candidates):
                coords = np.array([[a['x'], a['y']] for a in group], dtype=float)
                d = np.linalg.norm(projected[:, None, :] - coords[None, :, :], axis=2)
                alt_ids[:, j] = d.argmin(axis=1)
                if alternative_penalty:
                    scores = np.array([a.get('score', 0.0) for a in group], dtype=float)
                    cost_d = d + alternative_penalty * (scores.max() - scores[None, :])
                    alt_ids[:, j] = cost_d.argmin(axis=1)
                    distances[:, j] = d[np.arange(len(nodes)), alt_ids[:, j]]
                else:
                    distances[:, j] = d.min(axis=1)
            cost = np.full((len(nodes), len(candidates) + len(nodes)), tolerance + 1e-5)
            cost[:, :len(candidates)] = distances
            rr, cc = linear_sum_assignment(cost)
            keep = (cc < len(candidates)) & (cost[rr, np.minimum(cc, len(candidates)-1)] < tolerance)
            rr, cc = rr[keep], cc[keep]
            if len(rr) < 3:
                break
            src = nodes[rr]
            dst = np.array([[candidates[j][alt_ids[i, j]]['x'],
                             candidates[j][alt_ids[i, j]]['y']] for i, j in zip(rr, cc)])
            design = np.column_stack([src, np.ones(len(src))])
            if np.linalg.matrix_rank(design) < 3:
                break
            fit = np.linalg.lstsq(design, dst, rcond=None)[0]
            updated = fit[:2].T
            singular = np.linalg.svd(updated, compute_uv=False)
            if singular[-1] < .05 or singular[0] / singular[-1] > 8:
                break
            matrix, translation = updated, fit[2]
        projected = nodes @ matrix.T + translation
        distances = np.empty((len(nodes), len(candidates)), dtype=float)
        alt_ids = np.empty((len(nodes), len(candidates)), dtype=int)
        for j, group in enumerate(candidates):
            coords = np.array([[a['x'], a['y']] for a in group], dtype=float)
            d = np.linalg.norm(projected[:, None, :] - coords[None, :, :], axis=2)
            if alternative_penalty:
                scores = np.array([a.get('score', 0.0) for a in group], dtype=float)
                cost_d = d + alternative_penalty * (scores.max() - scores[None, :])
                alt_ids[:, j] = cost_d.argmin(axis=1)
                distances[:, j] = d[np.arange(len(nodes)), alt_ids[:, j]]
            else:
                alt_ids[:, j] = d.argmin(axis=1)
                distances[:, j] = d.min(axis=1)
        cost = np.full((len(nodes), len(candidates) + len(nodes)), tolerance + 1e-5)
        cost[:, :len(candidates)] = distances
        rr, cc = linear_sum_assignment(cost)
        keep = (cc < len(candidates)) & (cost[rr, np.minimum(cc, len(candidates)-1)] < tolerance)
        rr, cc = rr[keep], cc[keep]
        chosen = [(int(i), int(j), int(alt_ids[i, j]), float(distances[i, j]))
                  for i, j in zip(rr, cc)]
        if len(chosen) < 3:
            continue
        d = np.array([item[3] for item in chosen])
        locations = {str(j): {'x': float(candidates[j][k]['x']),
                              'y': float(candidates[j][k]['y'])}
                     for _, j, k, _ in chosen}
        selected_scores = {str(j): float(candidates[j][k].get('score', 0.0))
                           for _, j, k, _ in chosen}
        output.append(dict(name=pattern.name,
                           score=float(np.exp(-.5*(d/(tolerance*.5))**2).sum() - .015*len(nodes)),
                           matched_nodes=len(chosen), total_nodes=len(nodes),
                           residual_px=float(d.mean()), point_indices=[j for _, j, _, _ in chosen],
                           point_locations=locations, point_candidate_scores=selected_scores,
                           tolerance_px=tolerance,
                           matrix=matrix.tolist(), translation=translation.tolist(),
                           seed_score=seed['score']))
    return sorted(output, key=lambda r: r['score'], reverse=True)
