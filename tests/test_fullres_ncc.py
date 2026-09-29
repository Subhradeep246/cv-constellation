import unittest

import cv2
import numpy as np

from constellation.fullres_ncc import masked_ncc_candidates, relative_gap


class FullResolutionNCCTests(unittest.TestCase):
    def test_masked_ncc_recovers_rotated_noisy_synthetic_crop(self):
        rng = np.random.default_rng(29)
        sky = rng.normal(18, 2, (192, 192)).astype(np.float32)
        for x, y, flux in [(93, 105, 170), (99, 107, 95), (88, 114, 80),
                           (107, 93, 65), (59, 43, 180), (151, 138, 160)]:
            cv2.circle(sky, (x, y), 2, float(flux), -1)
        sky = cv2.GaussianBlur(sky, (0, 0), .9)
        cx, cy = 96, 104
        crop = sky[cy - 16:cy + 16, cx - 16:cx + 16]
        patch = cv2.warpAffine(crop, cv2.getRotationMatrix2D((15.5, 15.5), 30, 1),
                               (32, 32), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
        patch += rng.normal(0, 1, patch.shape).astype(np.float32)
        found = masked_ncc_candidates(sky, patch, scales=(1.0,), angle_step=10, keep=8)
        self.assertTrue(found)
        self.assertTrue(any(np.hypot(c['x'] - cx, c['y'] - cy) <= 2 for c in found))


    def test_masked_ncc_input_validation(self):
        sky = np.zeros((64, 64), dtype=np.float32)
        patch = np.zeros((32, 32), dtype=np.float32)
        with self.assertRaises(ValueError):
            masked_ncc_candidates(sky, patch, angle_step=7)
        self.assertEqual(masked_ncc_candidates(sky, patch), [])

    def test_relative_gap_has_no_evidence_without_runner_up(self):
        self.assertEqual(relative_gap([]), 0)
        self.assertEqual(relative_gap([{'score': .9}]), 0)
        self.assertAlmostEqual(relative_gap([{'score': .8}, {'score': .6}]), 1.0)
