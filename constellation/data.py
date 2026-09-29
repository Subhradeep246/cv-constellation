"""Dataset discovery and strict competition CSV handling."""
from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data/constellation-detection/participant"


@dataclass
class Scene:
    scene_id: str
    image_path: Path
    patches: list[Path]


def read_gray(path: Path) -> np.ndarray:
    # imdecode handles Unicode Windows paths consistently.
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def discover_scenes(folder: Path) -> list[Scene]:
    result = []
    for scene_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        images = sorted(scene_dir.glob("*_image.png"))
        patches = sorted((scene_dir / "patches").glob("patch_*.png"))
        if len(images) != 1 or not patches:
            raise ValueError(f"Expected one sky image and query patches: {scene_dir}")
        result.append(Scene(scene_dir.name, images[0], patches))
    if not result:
        raise ValueError(f"No scenes found: {folder}")
    return result


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not {"Id", "n_patches", "constellation"}.issubset(fields):
        raise ValueError(f"Invalid competition CSV: {path}")
    if len({row['Id'] for row in rows}) != len(rows):
        raise ValueError("Duplicate scene IDs")
    for row in rows:
        for i in range(1, int(row["n_patches"]) + 1):
            if f"patch_{i:02d}" not in fields:
                raise ValueError(f"Missing patch column {i} in {path}")
    return fields, rows


def parse_cell(value: str) -> tuple[float, float, int] | None:
    value = value.strip()
    if value == "-1":
        return None
    if value.startswith("(") or value.startswith("["):
        values = ast.literal_eval(value)
    else:
        values = value.replace(",", " ").split()
    if not isinstance(values, (tuple, list)) or len(values) not in (2, 3):
        raise ValueError(f"Invalid patch value: {value!r}")
    x, y = float(values[0]), float(values[1])
    m = float(values[2]) if len(values) == 3 else 0
    if not np.isfinite([x, y, m]).all() or m not in (0, 1):
        raise ValueError(f"Invalid coordinates or membership: {value!r}")
    return x, y, int(m)


def row_points(row: dict[str, str]) -> list[tuple[float, float, int] | None]:
    return [parse_cell(row[f"patch_{i:02d}"]) for i in range(1, int(row["n_patches"]) + 1)]


def write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def validate_submission(rows: list[dict[str,str]], schema: list[dict[str,str]],
                        allowed_names: set[str], image_sizes: dict[str,tuple[int,int]] | None = None) -> None:
    expected={r['Id']:r for r in schema}
    if len(rows)!=len(expected) or {r['Id'] for r in rows}!=set(expected):
        raise ValueError('Submission must contain every required scene exactly once')
    for row in rows:
        original=expected[row['Id']]
        if int(row['n_patches'])!=int(original['n_patches']):
            raise ValueError(f"Changed patch count: {row['Id']}")
        if row['constellation'] not in allowed_names | {'unknown'}:
            raise ValueError(f"Invalid constellation name: {row['constellation']}")
        for point in row_points(row):
            if point is None:
                continue
            x,y,_=point
            if x<0 or y<0:
                raise ValueError(f"Negative coordinates in {row['Id']}")
            if image_sizes:
                width,height=image_sizes[row['Id']]
                if x>=width or y>=height:
                    raise ValueError(f"Out-of-bounds coordinates in {row['Id']}")
