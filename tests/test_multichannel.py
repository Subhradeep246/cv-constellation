import unittest

import numpy as np

from constellation.multichannel import local_ncc, rank_candidates, select_candidate


class MultichannelTests(unittest.TestCase):
    def test_raw_match_prefers_injected_patch(self):
        rng = np.random.default_rng(23)
        patch = rng.integers(0, 255, (32, 32), dtype=np.uint8)
        sky = rng.integers(0, 255, (100, 100), dtype=np.uint8)
        sky[35:66, 35:66] = patch[:31, :31]
        candidate = dict(x=50, y=50, scale=1., degrees=0, score=.7)
        far = dict(x=75, y=75, scale=1., degrees=0, score=.72)
        self.assertGreater(local_ncc(sky, patch, candidate),
                           local_ncc(sky, patch, far) + .4)
        ranked = rank_candidates(sky, patch, [far, candidate], raw_weight=.5)
        self.assertEqual((ranked[0]['x'], ranked[0]['y']), (50, 50))

    def test_gap_gate_and_empty_candidates(self):
        ranked = [dict(combined=1.2), dict(combined=.9)]
        self.assertIs(select_candidate(ranked, .5, .16), ranked[0])
        self.assertIsNone(select_candidate(ranked, .5, 1.1))
        self.assertIsNone(select_candidate([], .5, .16))


if __name__ == '__main__':
    unittest.main()
