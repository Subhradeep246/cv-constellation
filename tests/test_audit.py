import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from constellation.audit import (
    SIMILAR_NCC,
    STRONG_SIMILAR_NCC,
    audit_dataset,
    file_digest,
    histogram_entropy,
    laplacian_variance,
    pixel_digest,
    quality_flags,
)


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def star_patch(seed: int, shift: int = 0, blur: float = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((32, 32), 18, dtype=np.uint8)
    yy, xx = np.mgrid[:32, :32]
    image = np.clip(image.astype(np.float32) + 180 * np.exp(-((xx - 15.5 - shift) ** 2 + (yy - 15.5) ** 2) / 8), 0, 255)
    image = image + rng.normal(0, 4, image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    if blur > 0:
        image = cv2.GaussianBlur(image, (0, 0), blur)
    return image


class AuditTests(unittest.TestCase):
    def test_quality_flags_distinguish_information_from_blur(self):
        self.assertIn("low_information", quality_flags(5, 2, 2.1, 10))
        self.assertIn("saturated", quality_flags(248, 4, 2.5, 20))
        self.assertIn("dark", quality_flags(8, 6, 3.2, 12))
        self.assertIn("blurry", quality_flags(80, 20, 6.0, 40))
        self.assertNotIn("blurry", quality_flags(8, 3, 2.0, 10))
        self.assertEqual(quality_flags(80, 25, 6.2, 1400), [])

    def test_digests_separate_identical_pixels_from_similar_crops(self):
        original = star_patch(3)
        copy = original.copy()
        shifted = star_patch(3, shift=2)
        self.assertEqual(pixel_digest(original), pixel_digest(copy))
        self.assertNotEqual(pixel_digest(original), pixel_digest(shifted))
        self.assertGreater(histogram_entropy(original), 3)
        self.assertGreater(laplacian_variance(original), laplacian_variance(star_patch(3, blur=2.5)))

    def test_audit_keeps_ids_and_does_not_rewrite_files(self):
        with tempfile.TemporaryDirectory(prefix="constellation-audit-") as temporary:
            root = Path(temporary)
            scene = root / "train" / "demo"
            original = star_patch(11)
            similar = star_patch(11, shift=1)
            blank = np.full((32, 32), 6, dtype=np.uint8)
            saturated = np.full((32, 32), 250, dtype=np.uint8)
            paths = [
                scene / "patches" / "patch_01.png",
                scene / "patches" / "patch_02.png",
                scene / "patches" / "patch_03.png",
                scene / "patches" / "patch_04.png",
                scene / "patches" / "patch_05.png",
            ]
            write_png(scene / "demo_image.png", np.zeros((64, 64), np.uint8))
            write_png(paths[0], original)
            write_png(paths[1], original)
            write_png(paths[2], similar)
            write_png(paths[3], blank)
            write_png(paths[4], saturated)
            before = {path: (file_digest(path), path.stat().st_mtime_ns) for path in paths}
            report = audit_dataset(root, include_preprocess_probe=False)
            after = {path: (file_digest(path), path.stat().st_mtime_ns) for path in paths}
            self.assertEqual(before, after)
            self.assertEqual(report["summary"]["n_patches"], 5)
            self.assertEqual(report["summary"]["exact_pixel_groups"], 1)
            self.assertEqual(len(report["exact_duplicates"]["pixels"][0]), 2)
            similar_pairs = report["similar_pairs"]
            self.assertTrue(any(
                {pair["a"]["patch"], pair["b"]["patch"]} == {"patch_01", "patch_03"}
                or {pair["a"]["patch"], pair["b"]["patch"]} == {"patch_02", "patch_03"}
                for pair in similar_pairs
            ))
            self.assertFalse(any(
                {pair["a"]["patch"], pair["b"]["patch"]} == {"patch_01", "patch_02"}
                for pair in similar_pairs
            ))
            flags = {item["patch"]: item["flags"] for item in report["patches"]}
            self.assertIn("low_information", flags["patch_04"])
            self.assertIn("saturated", flags["patch_05"])
            self.assertTrue(report["policy"]["original_files_unchanged"])
            self.assertTrue(report["policy"]["duplicate_patch_ids_kept"])
            self.assertFalse(report["policy"]["extra_preprocess_enabled"])
            self.assertGreaterEqual(SIMILAR_NCC, 0.9)
            self.assertGreater(STRONG_SIMILAR_NCC, SIMILAR_NCC)


if __name__ == "__main__":
    unittest.main()
