"""Extract visible green edges from the supplied constellation drawings."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def extract_edges(path: Path, nodes: np.ndarray) -> list[tuple[int, int]]:
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot decode pattern: {path}')
    blue, green, red = cv2.split(image)
    line = ((green > 80) & (green > red.astype(np.int16) * 1.5)
            & (green > blue.astype(np.int16) * 1.5)).astype(np.uint8)
    line = cv2.dilate(line, np.ones((3, 3), np.uint8))
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            length = float(np.linalg.norm(b - a))
            if length < 12:
                continue
            positions = np.linspace(5 / length, 1 - 5 / length,
                                    max(4, int(length - 10)))
            xy = np.rint(a[None] + positions[:, None] * (b - a)[None]).astype(int)
            xy[:, 0] = np.clip(xy[:, 0], 0, line.shape[1] - 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, line.shape[0] - 1)
            if line[xy[:, 1], xy[:, 0]].mean() >= .80:
                edges.append((i, j))
    return edges


def selected_edge_count(edges: list[tuple[int, int]], nodes: list[int]) -> int:
    selected = set(nodes)
    return sum(a in selected and b in selected for a, b in edges)
