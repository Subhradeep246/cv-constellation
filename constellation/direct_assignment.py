"""Training-only test of one-to-one assignment on direct-search candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    rows = [json.loads(path.read_text(encoding='utf-8'))
            for path in sorted(args.folder.glob('patch_*.json'))]
    if not rows:
        raise ValueError('No direct-search candidate files found')
    flat = [(i, item) for i, row in enumerate(rows) for item in row['candidates']]
    xy = np.asarray([[item['x'], item['y']] for _, item in flat], dtype=float)
    parent = np.arange(len(flat))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, j in cKDTree(xy).query_pairs(8):
        a, b = find(i), find(j)
        parent[b] = a
    roots = [find(i) for i in range(len(flat))]
    unique = {root: i for i, root in enumerate(sorted(set(roots)))}
    scores = np.full((len(rows), len(unique)), -10., dtype=float)
    chosen = [[None for _ in unique] for _ in rows]
    for (patch_id, item), root in zip(flat, roots):
        site = unique[root]
        if item['score'] > scores[patch_id, site]:
            scores[patch_id, site] = item['score']
            chosen[patch_id][site] = item
    rr, cc = linear_sum_assignment(-scores)
    degrees = np.sum(scores > -1, axis=0)
    hub_sweep = []
    for weight in (0, .002, .005, .01, .02, .04, .08):
        penalized = scores - weight * np.log(np.maximum(degrees, 1))[None, :]
        top_sites = np.argmax(penalized, axis=1)
        rra, cca = linear_sum_assignment(-penalized)
        correct_local = 0
        correct_global = 0
        for i, row in enumerate(rows):
            gt = row['truth']
            if not gt:
                continue
            local = chosen[i][int(top_sites[i])]
            assigned = chosen[i][int(cca[np.where(rra == i)[0][0]])]
            correct_local += int(local is not None and
                                 np.hypot(local['x'] - gt[0], local['y'] - gt[1]) < 12)
            correct_global += int(assigned is not None and
                                  np.hypot(assigned['x'] - gt[0], assigned['y'] - gt[1]) < 12)
        hub_sweep.append(dict(weight=weight, local_correct=correct_local,
                              assignment_correct=correct_global))
    correct_top, correct_assigned, covered = [], [], []
    records = []
    for i, row in enumerate(rows):
        gt = row['truth']
        top = row['candidates'][0]
        assigned = chosen[i][int(cc[np.where(rr == i)[0][0]])]
        if gt:
            good_top = np.hypot(top['x'] - gt[0], top['y'] - gt[1]) < 12
            good_assigned = (assigned is not None and
                             np.hypot(assigned['x'] - gt[0], assigned['y'] - gt[1]) < 12)
            good_covered = any(np.hypot(item['x'] - gt[0], item['y'] - gt[1]) < 12
                               for item in row['candidates'])
            correct_top.append(good_top)
            correct_assigned.append(good_assigned)
            covered.append(good_covered)
        records.append(dict(patch=row['patch'], present=bool(gt),
                            top_score=top['score'], assigned_score=assigned['score'] if assigned else None,
                            top_error=float(np.hypot(top['x'] - gt[0], top['y'] - gt[1])) if gt else None,
                            assigned_error=(float(np.hypot(assigned['x'] - gt[0],
                                                           assigned['y'] - gt[1]))
                                            if gt and assigned else None)))
    print(json.dumps(dict(scene=rows[0]['scene'], patches=len(rows), sites=len(unique),
                          true_present=len(correct_top), top_correct=int(sum(correct_top)),
                          assigned_correct=int(sum(correct_assigned)),
                          candidate_covered=int(sum(covered)), hub_sweep=hub_sweep,
                          records=records), indent=2))


if __name__ == '__main__':
    main()
