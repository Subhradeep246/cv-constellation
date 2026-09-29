import unittest
import tempfile
from pathlib import Path
import numpy as np
from constellation.cleaning import (MODES, prepare_image, starlet_denoise, normalized_view,
                                     unique_patches, quality, copy_neighborhoods, clean_dataset)


class CleaningTests(unittest.TestCase):
    def test_full_export_preserves_sources_and_ids(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp) / 'data', Path(tmp) / 'clean'
            folder = root / 'validation' / 'unseen'
            (folder / 'patches').mkdir(parents=True)
            cv2.imwrite(str(folder / 'unseen_image.png'), np.zeros((128, 128), np.uint8))
            for i in (1, 2):
                cv2.imwrite(str(folder / 'patches' / f'patch_{i:02d}.png'), np.zeros((32, 32), np.uint8))
            before = {p: p.read_bytes() for p in root.rglob('*.png')}
            result = clean_dataset(root, output)
            self.assertEqual(result['summary']['exact_duplicate_groups'], 1)
            self.assertEqual(result['source_files_verified'], 3)
            with np.load(output / 'normalized/validation/unseen.npz') as archive:
                self.assertEqual(archive['patch_ids'].tolist(), ['patch_01', 'patch_02'])
                self.assertEqual(archive['normalized'].shape, (2, 32, 32))
            self.assertTrue(all(p.read_bytes() == contents for p, contents in before.items()))
            with self.assertRaises(ValueError):
                clean_dataset(root, root / 'derived')

    def test_views_do_not_mutate_or_move_pixels(self):
        arr = np.arange(1024, dtype=np.uint8).reshape(32, 32)
        before = arr.copy()
        for mode in MODES:
            view = prepare_image(arr, mode)
            self.assertEqual(view.shape, arr.shape)
            self.assertEqual(view.dtype, np.uint8)
        np.testing.assert_array_equal(arr, before)
        view = normalized_view(arr)
        self.assertEqual(view.dtype, np.float32)
        self.assertTrue(np.isfinite(view).all())

    def test_flat_views_remain_finite(self):
        for value in (0, 42, 255):
            arr = np.full((32, 32), value, np.uint8)
            np.testing.assert_allclose(normalized_view(arr), 0, atol=1e-4)
            self.assertIn('almost_constant', quality(arr)['flags'])

    def test_starlet_reduces_noise_without_erasing_point_source(self):
        rng = np.random.default_rng(42)
        yy, xx = np.mgrid[:64, :64]
        star = 100 * np.exp(-((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 1.2 ** 2))
        raw = np.clip(np.rint(70 + star + rng.normal(0, 9, (64, 64))),
                      0, 255).astype(np.uint8)
        before = raw.copy()
        clean = starlet_denoise(raw)
        self.assertLess(clean[:16, :16].std(), raw[:16, :16].std())
        self.assertGreater(clean[32, 32] - 70, .7 * (raw[32, 32] - 70))
        np.testing.assert_array_equal(raw, before)

    def test_duplicate_expansion_preserves_ids(self):
        a = np.zeros((32, 32), np.uint8)
        b = a.copy()
        b[5, 5] = 1
        unique, inverse = unique_patches([a, b, a.copy()])
        self.assertEqual(len(unique), 2)
        self.assertEqual(inverse, [0, 1, 0])

    def test_copy_search_ignores_core_only_similarity(self):
        import cv2
        rng = np.random.default_rng(1)
        sky = rng.integers(15, 35, (350, 550), dtype=np.uint8)
        cv2.circle(sky, (100, 150), 3, 255, -1)
        cv2.circle(sky, (400, 150), 3, 255, -1)
        self.assertEqual(copy_neighborhoods(sky, 2)['pairs'], [])
        sky[100:201, 350:451] = sky[100:201, 50:151]
        result = copy_neighborhoods(sky, 2)
        self.assertEqual(len(result['pairs']), 1)
        self.assertGreater(result['pairs'][0]['highpass_ncc'], .99)
