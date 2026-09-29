"""Read-only patch quality audit.

Reports exact duplicates, similar-looking patches, brightness, blur, and
low-information cases. Original files are never rewritten. Patch IDs are
never collapsed: even identical pixels would still need separate predictions.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .data import discover_scenes, read_gray, read_rows, row_points
from .localize import Localizer, LocalizerConfig, normalized_rows

# Intensity-normalized correlation. 0.99 is near-pixel identity after gain;
# 0.90 is a similar-looking aligned crop; 0.94 is used for connected groups.
NEAR_EXACT_NCC = 0.99
SIMILAR_NCC = 0.90
STRONG_SIMILAR_NCC = 0.94
LOW_STD = 8.0
LOW_ENTROPY = 4.0
SATURATED_MEAN = 200.0
FLAT_STD = 15.0
DARK_MEAN = 15.0
BLUR_LAPLACIAN = 200.0


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pixel_digest(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def histogram_entropy(image: np.ndarray) -> float:
    hist = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    probability = hist / max(hist.sum(), 1.0)
    probability = probability[probability > 0]
    return float(-(probability * np.log2(probability)).sum())


def laplacian_variance(image: np.ndarray) -> float:
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def quality_flags(mean: float, std: float, entropy: float, lapvar: float) -> list[str]:
    flags = []
    if std < LOW_STD or entropy < LOW_ENTROPY:
        flags.append("low_information")
    if mean >= SATURATED_MEAN and std < FLAT_STD:
        flags.append("saturated")
    if mean < DARK_MEAN and std < FLAT_STD:
        flags.append("dark")
    # Blur is only meaningful when some structure remains.
    if lapvar < BLUR_LAPLACIAN and std >= LOW_STD:
        flags.append("blurry")
    return flags


def _union_groups(keys: list[tuple], pairs: list[tuple]) -> list[list[tuple]]:
    parent = {key: key for key in keys}

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for left, right in pairs:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a
    buckets: dict[tuple, list[tuple]] = defaultdict(list)
    for key in keys:
        buckets[find(key)].append(key)
    return [members for members in buckets.values() if len(members) > 1]


def _hash_groups(items: list[dict], field: str) -> list[list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        buckets[item[field]].append({"split": item["split"], "scene": item["scene"], "patch": item["patch"]})
    return [members for members in buckets.values() if len(members) > 1]


def _candidate_views(image: np.ndarray) -> dict[str, np.ndarray]:
    blur = cv2.GaussianBlur(image, (0, 0), 0.8)
    return {
        "raw": image,
        "clahe": cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image),
        "gamma_0.6": np.clip(((image / 255.0) ** 0.6) * 255, 0, 255).astype(np.uint8),
        "bilateral": cv2.bilateralFilter(image, 5, 25, 25),
        "unsharp": np.clip(cv2.addWeighted(image, 1.5, blur, -0.5, 0), 0, 255).astype(np.uint8),
        "median": cv2.medianBlur(image, 3),
    }


def _ring_std(image: np.ndarray) -> float:
    yy, xx = np.mgrid[:image.shape[0], :image.shape[1]]
    radius = np.hypot(xx - (image.shape[1] - 1) / 2, yy - (image.shape[0] - 1) / 2)
    ring = (radius >= 6) & (radius <= 14)
    return float(image.astype(np.float32)[ring].std())


def probe_preprocess(data_root: Path, localizer: Localizer, limit_per_scene: int = 2) -> dict:
    """Oracle-location probe. Does not change inference or write images."""
    truth_path = data_root / "train_ground_truth.csv"
    train_dir = data_root / "train"
    if not truth_path.is_file() or not train_dir.is_dir():
        return {"available": False}
    _, rows = read_rows(truth_path)
    scenes = {scene.scene_id: scene for scene in discover_scenes(train_dir)}
    names = list(_candidate_views(np.zeros((32, 32), np.uint8)))
    totals = {name: {"score": [], "error_px": [], "ring_std": []} for name in names}
    examples = []
    for row in rows:
        scene = scenes.get(row["Id"])
        if scene is None:
            continue
        sky = read_gray(scene.image_path)
        filtered = localizer.filtered(sky)
        taken = 0
        for path, truth in zip(scene.patches, row_points(row)):
            if truth is None:
                continue
            image = read_gray(path)
            proposals = np.array([[truth[0], truth[1], scale] for scale in localizer.config.scales])
            entry = {"scene": scene.scene_id, "patch": path.stem, "methods": {}}
            for name, view in _candidate_views(image).items():
                match = localizer.verify(filtered, view, proposals)
                error = float(np.hypot(match.x - truth[0], match.y - truth[1]))
                ring = _ring_std(view)
                totals[name]["score"].append(float(match.score))
                totals[name]["error_px"].append(error)
                totals[name]["ring_std"].append(ring)
                entry["methods"][name] = {"score": float(match.score), "error_px": error, "ring_std": ring}
            examples.append(entry)
            taken += 1
            if taken >= limit_per_scene:
                break
    summary = {
        name: {
            "mean_score": float(np.mean(values["score"])) if values["score"] else None,
            "mean_error_px": float(np.mean(values["error_px"])) if values["error_px"] else None,
            "mean_ring_std": float(np.mean(values["ring_std"])) if values["ring_std"] else None,
        }
        for name, values in totals.items()
    }
    raw = summary.get("raw") or {}
    better = [
        name for name, stats in summary.items()
        if name != "raw" and raw.get("mean_score") is not None and stats["mean_score"] is not None
        and stats["mean_score"] > raw["mean_score"] + 0.01
    ]
    return {
        "available": True,
        "patches_probed": len(examples),
        "summary": summary,
        "methods_beating_raw_mean_score": better,
        "enabled_in_matcher": False,
        "examples": examples,
        "note": (
            "The matcher already subtracts a radial background and L2-normalizes "
            "contrast. Stronger brightening or denoising can flatten the faint "
            "neighboring stars that distinguish similar patches."
        ),
    }


def audit_dataset(data_root: Path, *, include_preprocess_probe: bool = True) -> dict:
    localizer = Localizer(LocalizerConfig())
    records = []
    truth = {}
    train_csv = data_root / "train_ground_truth.csv"
    if train_csv.is_file():
        _, rows = read_rows(train_csv)
        truth = {row["Id"]: row_points(row) for row in rows}
    for split in ("train", "validation"):
        folder = data_root / split
        if not folder.is_dir():
            continue
        for scene in discover_scenes(folder):
            labels = truth.get(scene.scene_id)
            for index, path in enumerate(scene.patches):
                image = read_gray(path)
                mean = float(image.mean())
                std = float(image.std())
                entropy = histogram_entropy(image)
                lapvar = laplacian_variance(image)
                filtered = localizer.filtered(image)
                polar = localizer.sample(filtered, ((np.array(image.shape[::-1]) - 1) / 2)[None], 1)
                label = labels[index] if labels is not None else None
                records.append({
                    "split": split,
                    "scene": scene.scene_id,
                    "patch": path.stem,
                    "path": str(path),
                    "mean": mean,
                    "std": std,
                    "min": int(image.min()),
                    "max": int(image.max()),
                    "entropy": entropy,
                    "laplacian_variance": lapvar,
                    "flags": quality_flags(mean, std, entropy, lapvar),
                    "file_sha256": file_digest(path),
                    "pixel_sha256": pixel_digest(image),
                    "vector": normalized_rows(image.astype(np.float32).ravel()[None])[0],
                    "descriptor": localizer.descriptor(polar)[0],
                    "truth_present": None if labels is None else label is not None,
                    "truth_xy": None if label is None else [float(label[0]), float(label[1])],
                })
    keys = [(item["split"], item["scene"], item["patch"]) for item in records]
    vectors = np.stack([item["vector"] for item in records]) if records else np.empty((0, 1))
    descriptors = np.stack([item["descriptor"] for item in records]) if records else np.empty((0, 1))
    similar = []
    near_exact = []
    if len(records) >= 2:
        similarity = vectors @ vectors.T
        polar_sim = descriptors @ descriptors.T
        np.fill_diagonal(similarity, -1)
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                ncc = float(similarity[i, j])
                if ncc < SIMILAR_NCC:
                    continue
                left, right = records[i], records[j]
                if left["pixel_sha256"] == right["pixel_sha256"] or left["file_sha256"] == right["file_sha256"]:
                    continue
                pair = {
                    "ncc": ncc,
                    "polar_cosine": float(polar_sim[i, j]),
                    "mean_delta": abs(left["mean"] - right["mean"]),
                    "same_scene": left["scene"] == right["scene"],
                    "a": {"split": left["split"], "scene": left["scene"], "patch": left["patch"]},
                    "b": {"split": right["split"], "scene": right["scene"], "patch": right["patch"]},
                }
                if left["truth_present"] is not None and right["truth_present"] is not None:
                    pair["different_presence"] = left["truth_present"] != right["truth_present"]
                    if left["truth_xy"] is not None and right["truth_xy"] is not None:
                        pair["location_delta_px"] = float(np.hypot(
                            left["truth_xy"][0] - right["truth_xy"][0],
                            left["truth_xy"][1] - right["truth_xy"][1]))
                similar.append(pair)
                if ncc >= NEAR_EXACT_NCC:
                    near_exact.append(pair)
    similar.sort(key=lambda item: item["ncc"], reverse=True)
    strong = [pair for pair in similar if pair["ncc"] >= STRONG_SIMILAR_NCC]
    similar_groups = [
        [{"split": split, "scene": scene, "patch": patch} for split, scene, patch in group]
        for group in _union_groups(keys, [
            ((pair["a"]["split"], pair["a"]["scene"], pair["a"]["patch"]),
             (pair["b"]["split"], pair["b"]["scene"], pair["b"]["patch"]))
            for pair in strong
        ])
    ]
    flag_counts: dict[str, int] = defaultdict(int)
    flag_counts_by_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in records:
        if item["flags"]:
            flag_counts_by_split[item["split"]]["any"] += 1
        for flag in item["flags"]:
            flag_counts[flag] += 1
            flag_counts_by_split[item["split"]][flag] += 1
    patches = []
    for item in records:
        patches.append({key: item[key] for key in (
            "split", "scene", "patch", "mean", "std", "min", "max", "entropy",
            "laplacian_variance", "flags", "file_sha256", "pixel_sha256",
            "truth_present",
        )})
    probe = probe_preprocess(data_root, localizer) if include_preprocess_probe else {"available": False, "skipped": True}
    values = lambda key: np.array([item[key] for item in records], dtype=float)
    percentiles = {}
    if records:
        for key in ("mean", "std", "entropy", "laplacian_variance"):
            series = values(key)
            percentiles[key] = {
                "min": float(series.min()),
                "p5": float(np.percentile(series, 5)),
                "median": float(np.median(series)),
                "p95": float(np.percentile(series, 95)),
                "max": float(series.max()),
            }
    return {
        "policy": {
            "original_files_unchanged": True,
            "duplicate_patch_ids_kept": True,
            "extra_preprocess_enabled": False,
            "exact_vs_similar": (
                "Exact duplicates share a file or pixel SHA-256. Similar-looking "
                f"patches have aligned intensity-normalized correlation ≥ {SIMILAR_NCC:.2f}. "
                f"Connected groups use a stricter ≥ {STRONG_SIMILAR_NCC:.2f} cut so chained "
                "generic star cores are not treated as one duplicate set. Patch IDs stay "
                "separate because labels and coordinates can still differ."
            ),
        },
        "thresholds": {
            "near_exact_ncc": NEAR_EXACT_NCC,
            "similar_ncc": SIMILAR_NCC,
            "strong_similar_ncc": STRONG_SIMILAR_NCC,
            "low_std": LOW_STD,
            "low_entropy": LOW_ENTROPY,
            "saturated_mean": SATURATED_MEAN,
            "flat_std": FLAT_STD,
            "dark_mean": DARK_MEAN,
            "blur_laplacian": BLUR_LAPLACIAN,
        },
        "summary": {
            "n_patches": len(records),
            "n_scenes": len({(item["split"], item["scene"]) for item in records}),
            "exact_file_groups": len(_hash_groups(records, "file_sha256")),
            "exact_pixel_groups": len(_hash_groups(records, "pixel_sha256")),
            "near_exact_pairs": len(near_exact),
            "similar_pairs": len(similar),
            "strong_similar_pairs": len(strong),
            "same_scene_similar_pairs": sum(1 for pair in similar if pair["same_scene"]),
            "similar_groups": len(similar_groups),
            "flagged_patches": sum(1 for item in records if item["flags"]),
            "flag_counts": dict(flag_counts),
            "flag_counts_by_split": {split: dict(counts) for split, counts in flag_counts_by_split.items()},
        },
        "percentiles": percentiles,
        "exact_duplicates": {
            "file": _hash_groups(records, "file_sha256"),
            "pixels": _hash_groups(records, "pixel_sha256"),
        },
        "similar_pairs": similar,
        "similar_groups": similar_groups,
        "patches": patches,
        "preprocess_probe": probe,
    }
