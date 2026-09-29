import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from constellation.candidate_cache import get_fullres_candidates


class CandidateCacheTests(unittest.TestCase):
    def test_cache_is_label_free_reused_and_invalidated_by_pixels(self):
        sky = np.zeros((64, 64), dtype=np.uint8)
        query = np.ones((32, 32), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'candidates.json'
            with patch('constellation.candidate_cache.masked_ncc_candidates',
                       return_value=[{'x': 20, 'y': 30, 'score': .8}]) as search:
                first = get_fullres_candidates(sky, [query], target, log=lambda _: None)
                self.assertEqual(search.call_count, 1)
                second = get_fullres_candidates(sky, [query], target, log=lambda _: None)
                self.assertEqual(search.call_count, 1)
                self.assertEqual(first, second)
                modified = sky.copy()
                modified[20, 30] = 1
                get_fullres_candidates(modified, [query], target, log=lambda _: None)
                self.assertEqual(search.call_count, 2)
            data = target.read_text(encoding='utf-8')
            self.assertNotIn('truth', data)
            self.assertNotIn('synthetic', data)
