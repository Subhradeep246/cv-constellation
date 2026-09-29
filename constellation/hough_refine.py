"""Training-only geometric refinement of Hough poses with unique query IDs."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .data import DEFAULT_DATA
from .copy_evidence import clustered_pairs
from .hough_probe import heatmap
from .identify import load_patterns
from .pattern_graph import extract_edges, selected_edge_count


@dataclass
class CandidateGroups:
    items: list[list[dict]]
    xy: np.ndarray
    valid: np.ndarray

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def prepare_groups(items: list[list[dict]]) -> CandidateGroups:
    width = max((len(group) for group in items), default=0)
    xy = np.zeros((len(items), max(1, width), 2), dtype=float)
    valid = np.zeros(xy.shape[:2], dtype=bool)
    for j, group in enumerate(items):
        for k, item in enumerate(group):
            xy[j, k] = (item['x'], item['y'])
            valid[j, k] = True
    return CandidateGroups(items, xy, valid)


def select_hits(hits: list[dict], limit: int) -> list[dict]:
    """Keep strong, distinct coarse poses before costly patch assignment."""
    if limit <= 0 or len(hits) <= limit:
        return hits
    selected = []
    for hit in sorted(hits, key=lambda item: -item['support']):
        if any(hit.get('reflected', False) == prior.get('reflected', False) and
               abs(hit['x'] - prior['x']) <= 16 and
               abs(hit['y'] - prior['y']) <= 16 and
               abs(hit['scale'] - prior['scale']) <= .25 and
               min(abs(hit['degrees'] - prior['degrees']),
                   360 - abs(hit['degrees'] - prior['degrees'])) <= 10
               for prior in selected):
            continue
        selected.append(hit)
        if len(selected) >= limit:
            break
    return selected


def assignment(nodes: np.ndarray, projected: np.ndarray,
               groups: CandidateGroups, tolerance: float):
    delta = projected[:, None, None, :] - groups.xy[None]
    squared = np.sum(delta * delta, axis=-1)
    squared = np.where(groups.valid[None], squared, np.inf)
    indices = np.argmin(squared, axis=2)
    distances = np.sqrt(np.take_along_axis(squared, indices[:, :, None], axis=2)[:, :, 0])
    augmented = np.concatenate([distances,
                                np.full((len(nodes), len(nodes)), tolerance)], axis=1)
    rr, cc = linear_sum_assignment(augmented)
    keep = (cc < len(groups)) & (augmented[rr, cc] < tolerance)
    return [(int(i), int(j), int(indices[i, j]), float(distances[i, j]))
            for i, j in zip(rr[keep], cc[keep])]


def fit_affine(src: np.ndarray, dst: np.ndarray):
    source_mean, target_mean = src.mean(axis=0), dst.mean(axis=0)
    x = src[:, 0] - source_mean[0]
    y = src[:, 1] - source_mean[1]
    dx = dst[:, 0] - target_mean[0]
    dy = dst[:, 1] - target_mean[1]
    xx, xy, yy = float(x @ x), float(x @ y), float(y @ y)
    determinant = xx * yy - xy * xy
    if determinant <= 1e-8 * (xx + yy) ** 2:
        return None
    xdx, ydx = float(x @ dx), float(y @ dx)
    xdy, ydy = float(x @ dy), float(y @ dy)
    matrix = np.array([[(yy * xdx - xy * ydx) / determinant,
                        (xx * ydx - xy * xdx) / determinant],
                       [(yy * xdy - xy * ydy) / determinant,
                        (xx * ydy - xy * xdy) / determinant]])
    minimum, ratio = singular_summary(matrix)
    if minimum < .1 or ratio > 2:
        return None
    translation = target_mean - source_mean @ matrix.T
    return matrix, translation


def singular_summary(matrix: np.ndarray) -> tuple[float, float]:
    squared_sum = float(np.sum(matrix * matrix))
    determinant = float(matrix[0, 0] * matrix[1, 1] -
                        matrix[0, 1] * matrix[1, 0])
    discriminant = max(squared_sum ** 2 - 4 * determinant ** 2, 0.) ** .5
    maximum = max((squared_sum + discriminant) / 2, 0.) ** .5
    minimum = max((squared_sum - discriminant) / 2, 0.) ** .5
    return minimum, maximum / max(minimum, 1e-8)


def score_pose(hit: dict, nodes: np.ndarray, groups: CandidateGroups,
               tolerance: float = 20, copy_tree: cKDTree | None = None,
               star_tree: cKDTree | None = None,
               star_tolerance: float = 20,
               image_shape: tuple[int, int] = (3000, 3000)):
    radians = np.deg2rad(hit['degrees'])
    co, si = np.cos(radians), np.sin(radians)
    matrix = hit['scale'] * np.array([[co, -si], [si, co]])
    if hit.get('reflected', False):
        matrix = matrix @ np.diag([-1., 1.])
    translation = np.array([hit['x'], hit['y']]) - nodes.mean(axis=0) @ matrix.T
    for _ in range(3):
        projected = nodes @ matrix.T + translation
        chosen = assignment(nodes, projected, groups, tolerance)
        if len(chosen) < 3:
            break
        src = nodes[[i for i, _, _, _ in chosen]]
        dst = np.array([[groups[j][k]['x'], groups[j][k]['y']]
                        for _, j, k, _ in chosen])
        fit = fit_affine(src, dst)
        if fit is None:
            break
        matrix, translation = fit
    projected = nodes @ matrix.T + translation
    in_frame = ((projected[:, 0] >= 0) & (projected[:, 0] < image_shape[1]) &
                (projected[:, 1] >= 0) & (projected[:, 1] < image_shape[0]))
    chosen = assignment(nodes, projected, groups, tolerance)
    distances = np.array([d for _, _, _, d in chosen])
    ranks = np.array([k for _, _, k, _ in chosen])
    weighted = float(np.exp(-.5 * (distances / 8) ** 2).sum())
    _, aspect_ratio = singular_summary(matrix)
    copy_nodes = (int(np.sum((copy_tree.query(projected)[0] <= 10) & in_frame))
                  if copy_tree is not None else 0)
    star_distances, star_indices = (star_tree.query(projected[in_frame])
                                    if star_tree is not None else
                                    (np.empty(0), np.empty(0, dtype=int)))
    star_score = float(np.exp(-.5 * (star_distances / star_tolerance) ** 2).sum())
    star_hits = int(np.sum(star_distances <= star_tolerance))
    return dict(hit=hit, matched=len(chosen), weighted=weighted,
                copy_nodes=copy_nodes,
                star_score=star_score, star_hits=star_hits,
                star_distances=star_distances.tolist(),
                star_ranks=(star_indices + 1).tolist(),
                in_frame_nodes=int(in_frame.sum()),
                aspect_ratio=float(aspect_ratio),
                mean_distance=float(distances.mean()) if len(distances) else None,
                mean_rank=float(ranks.mean()) if len(ranks) else None,
                chosen=[dict(node=i, patch=j + 1, rank=k + 1, distance=d,
                             x=groups[j][k]['x'], y=groups[j][k]['y'])
                        for i, j, k, d in chosen],
                matrix=matrix.tolist(), translation=translation.tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hits', type=Path)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--top', type=int, default=12)
    parser.add_argument('--copy-report', type=Path)
    parser.add_argument('--copy-weight', type=float, default=0)
    parser.add_argument('--missing-penalty', type=float, default=0)
    parser.add_argument('--patch-budget-weight', type=float, default=0)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-hits', type=int, default=0)
    args = parser.parse_args()
    report = json.loads(args.hits.read_text(encoding='utf-8'))
    patterns = {p.name: p for p in load_patterns(args.data / 'patterns')}
    _, _, candidates = heatmap(args.folder, 8, args.top, args.baseline)
    copy_tree = None
    if args.copy_report:
        copy_records = json.loads(args.copy_report.read_text(encoding='utf-8'))
        copy_record = next(item for item in copy_records
                           if item['scene'] == report['scene'])
        _, selected = clustered_pairs(copy_record['pairs'])
        anchors = np.asarray([xy for item in selected
                              for xy in (item['a'], item['b'])], dtype=float).reshape(-1, 2)
        if len(anchors):
            copy_tree = cKDTree(anchors)
    items = [[] for _ in range(max(item['patch'] for item in candidates))]
    for item in candidates:
        items[item['patch'] - 1].append(item)
    groups = prepare_groups(items)
    entries = (report['patterns'] if 'patterns' in report else
               [dict(pattern=report['pattern'], hits=report['hits'])])
    ranked = []
    for entry in entries:
        pattern = patterns[entry['pattern']]
        edges = extract_edges(args.data / 'patterns' / f'{pattern.name}_pattern.png',
                              pattern.nodes)
        for hit in select_hits(entry['hits'], args.max_hits):
            value = score_pose(hit, pattern.nodes, groups, copy_tree=copy_tree)
            value['pattern'] = pattern.name
            value['total_nodes'] = len(pattern.nodes)
            value['total_edges'] = len(edges)
            value['selected_edges'] = selected_edge_count(
                edges, [item['node'] for item in value['chosen']])
            in_frame = max(value['in_frame_nodes'], 1)
            budget_prior = max(-.5 * (np.log(len(groups) / in_frame / 2.97) / .35) ** 2,
                               -20.0)
            value['budget_prior'] = float(budget_prior)
            value['combined'] = (value['weighted']
                                 - args.missing_penalty * max(0, in_frame - value['matched'])
                                 + args.copy_weight * value['copy_nodes']
                                 + args.patch_budget_weight * budget_prior)
            ranked.append(value)
    ranked.sort(key=lambda item: (item['combined'], item['matched']), reverse=True)
    per_pattern = {}
    for item in ranked:
        per_pattern.setdefault(item['pattern'], item)
    summary = dict(scene=report['scene'], patterns=len(entries),
                   poses=len(ranked), top=ranked[:10],
                   per_pattern=list(per_pattern.values()))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(dict(scene=report['scene'], poses=len(ranked),
                              top=[dict(pattern=item['pattern'],
                                        combined=item['combined'])
                                   for item in list(per_pattern.values())[:3]])))
    else:
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
