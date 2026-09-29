"""Package-facing runner for the robust detection engine.

The selected CSV from ``demo_reference`` is never read by inference. It is
only an external comparison artifact; all predictions here come from pixels.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import robust_engine
from .data import discover_scenes, read_gray, read_rows, validate_submission, write_rows
from .metrics import score_scene


def validate_robust_predictions(
    fields: list[str],
    rows: list[dict[str, str]],
    expected_fields: list[str],
    schema: list[dict[str, str]],
    allowed_names: set[str],
    image_sizes: dict[str, tuple[int, int]],
) -> None:
    """Check the engine's complete output before naming it a submission."""
    if fields != expected_fields:
        raise ValueError("Robust output columns differ from the competition template")
    if [row["Id"] for row in rows] != [row["Id"] for row in schema]:
        raise ValueError("Robust output scene order differs from the competition template")
    validate_submission(rows, schema, allowed_names, image_sizes)
    for row in rows:
        for number in range(int(row["n_patches"]) + 1, 1 + len(expected_fields) - 3):
            if row[f"patch_{number:02d}"].strip() != "-1":
                raise ValueError(f"Unused patch column is not -1: {row['Id']} patch_{number:02d}")


def run_robust(
    *,
    data: Path,
    output: Path,
    cache: Path,
    evaluate: bool,
    scene: str | None = None,
    probe: bool = False,
    channel_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Path:
    """Run robust inference through this package's CLI and CSV contract.

    Training labels are used only for post-inference scoring. The validation
    path never opens the training ground-truth file or selected reference CSV.
    """
    if probe and (evaluate or not scene):
        raise ValueError("A validation probe needs one scene and cannot evaluate training labels")
    if scene and not (evaluate or probe):
        raise ValueError("A partial scene run cannot create a submission")
    if (len(channel_weights) != 3 or
            not all(math.isfinite(weight) and weight >= 0 for weight in channel_weights) or
            sum(channel_weights) <= 0):
        raise ValueError("channel_weights must contain three finite, nonnegative values, not all zero")
    split = "train" if evaluate else "validation"
    schema_path = data / ("train_ground_truth.csv" if evaluate else "sample_submission.csv")
    expected_fields, schema = read_rows(schema_path)
    if scene:
        schema = [row for row in schema if row["Id"] == scene]
        if not schema:
            raise ValueError(f"Unknown {split} scene: {scene}")
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    candidate_path = output / ("robust-train-candidate.csv" if evaluate else "robust-validation-candidate.csv")
    engine_args = [
        "--data", str(data), "--split", split,
        "--out", str(candidate_path), "--cache", str(cache), "--no-fb-extra",
        "--checkpoint", str(output / "robust-checkpoint.csv"),
        "--channel-weights", *(str(value) for value in channel_weights),
    ]
    if evaluate:
        engine_args.append("--eval")
    if scene:
        engine_args.extend(["--scenes", scene])
    robust_engine.main(engine_args)

    fields, predictions = read_rows(candidate_path)
    scene_paths = {item.scene_id: item for item in discover_scenes(data / split)}
    image_sizes = {}
    for row in schema:
        item = scene_paths.get(row["Id"])
        if item is None or len(item.patches) != int(row["n_patches"]):
            raise ValueError(f"Missing scene or patch count mismatch: {row['Id']}")
        sky = read_gray(item.image_path)
        image_sizes[row["Id"]] = (sky.shape[1], sky.shape[0])
    names = set(robust_engine.P)
    validate_robust_predictions(fields, predictions, expected_fields, schema, names, image_sizes)

    final_path = output / ("robust-train-predictions.csv" if evaluate else
                           "robust-probe-predictions.csv" if probe else "submission.csv")
    write_rows(final_path, fields, predictions)
    report = {
        "method": "robust engine integrated into constellation package",
        "split": split,
        "scenes": len(predictions),
        "patches": sum(int(row["n_patches"]) for row in predictions),
        "cache": str(cache.resolve()),
        "output": str(final_path.resolve()),
        "channel_weights": dict(zip(("raw", "bandpass", "highpass"), channel_weights)),
        "experimental_channel_weights": tuple(channel_weights) != (1.0, 1.0, 1.0),
    }
    if evaluate:
        by_id = {row["Id"]: row for row in schema}
        report["scores"] = {row["Id"]: score_scene(by_id[row["Id"]], row) for row in predictions}
        report["mean_score"] = sum(value["score"] for value in report["scores"].values()) / len(predictions)
    (output / "robust-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return final_path
