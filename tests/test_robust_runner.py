"""Small synthetic contract tests; no synthetic data enters inference."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

from constellation.data import write_rows
from constellation.robust_engine import build_F
from constellation.robust_runner import run_robust, validate_robust_predictions


class RobustOutputTests(unittest.TestCase):
    def setUp(self):
        self.fields = ["Id", "n_patches", "patch_01", "patch_02", "constellation"]
        self.schema = [{"Id": "scene", "n_patches": "1", "patch_01": "-1",
                        "patch_02": "-1", "constellation": "orion"}]
        self.row = {"Id": "scene", "n_patches": "1", "patch_01": "(10, 12, 0)",
                    "patch_02": "-1", "constellation": "orion"}

    def check(self, row=None, fields=None):
        validate_robust_predictions(
            fields or self.fields, [row or self.row], self.fields,
            self.schema, {"orion"}, {"scene": (30, 30)},
        )

    def test_valid_submission_shape(self):
        self.check()

    def test_unused_patch_column_must_be_absent(self):
        with self.assertRaisesRegex(ValueError, "Unused patch column"):
            self.check(row=dict(self.row, patch_02="(15, 15, 1)"))

    def test_out_of_bounds_position_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Out-of-bounds"):
            self.check(row=dict(self.row, patch_01="(30, 12, 0)"))

    def test_wrong_column_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "columns differ"):
            self.check(fields=list(reversed(self.fields)))

    def test_validation_probe_never_creates_submission(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene_dir = root / "validation" / "scene"
            (scene_dir / "patches").mkdir(parents=True)
            cv2.imwrite(str(scene_dir / "scene_image.png"), np.zeros((30, 30), np.uint8))
            cv2.imwrite(str(scene_dir / "patches" / "patch_01.png"), np.zeros((32, 32), np.uint8))
            write_rows(root / "sample_submission.csv", self.fields, self.schema)

            def fake_engine(args):
                self.assertIn("--scenes", args)
                self.assertNotIn("--eval", args)
                destination = Path(args[args.index("--out") + 1])
                write_rows(destination, self.fields, [self.row])

            output = root / "output"
            with patch("constellation.robust_runner.robust_engine.main", side_effect=fake_engine), \
                 patch("constellation.robust_runner.robust_engine.P", {"orion": None}, create=True):
                result = run_robust(data=root, output=output, cache=root / "cache",
                                    evaluate=False, scene="scene", probe=True)
            self.assertEqual(result.name, "robust-probe-predictions.csv")
            self.assertTrue(result.exists())
            self.assertFalse((output / "submission.csv").exists())

    def test_channel_weights_reach_engine_and_report(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene_dir = root / "validation" / "scene"
            (scene_dir / "patches").mkdir(parents=True)
            cv2.imwrite(str(scene_dir / "scene_image.png"), np.zeros((30, 30), np.uint8))
            cv2.imwrite(str(scene_dir / "patches" / "patch_01.png"), np.zeros((32, 32), np.uint8))
            write_rows(root / "sample_submission.csv", self.fields, self.schema)

            def fake_engine(args):
                index = args.index("--channel-weights")
                self.assertEqual(args[index + 1:index + 4], ["0.5", "1.0", "1.5"])
                write_rows(Path(args[args.index("--out") + 1]), self.fields, [self.row])

            output = root / "output"
            with patch("constellation.robust_runner.robust_engine.main", side_effect=fake_engine), \
                 patch("constellation.robust_runner.robust_engine.P", {"orion": None}, create=True):
                run_robust(data=root, output=output, cache=root / "cache", evaluate=False,
                           scene="scene", probe=True, channel_weights=(0.5, 1.0, 1.5))
            import json
            report = json.loads((output / "robust-report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["channel_weights"], {"raw": 0.5, "bandpass": 1.0, "highpass": 1.5})
            self.assertTrue(report["experimental_channel_weights"])

    def test_invalid_channel_weights_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for weights in ((0, 0, 0), (1, -1, 1), (1, float("nan"), 1), (1, 2)):
                with self.subTest(weights=weights), self.assertRaisesRegex(ValueError, "channel_weights"):
                    run_robust(data=root, output=root / "out", cache=root / "cache",
                               evaluate=False, channel_weights=weights)

    def test_channel_weights_change_candidate_order(self):
        # Synthetic match scores exercise ranking only; no synthetic image is used for inference.
        location = np.array([[100, 100, 0, 0], [200, 200, 0, 0]], dtype=np.float32)
        sample = {"S": {
            "raw": np.column_stack(([0.9, 0.5], location)),
            "bp": np.column_stack(([0.7, 0.7], location)),
            "hp": np.column_stack(([0.5, 0.9], location)),
        }}
        raw_first = build_F(sample, {"raw": 1.5, "bp": 1.0, "hp": 0.5})
        hp_first = build_F(sample, {"raw": 0.5, "bp": 1.0, "hp": 1.5})
        self.assertEqual(tuple(raw_first[0, 1:3]), (100, 100))
        self.assertEqual(tuple(hp_first[0, 1:3]), (200, 200))


if __name__ == "__main__":
    unittest.main()
