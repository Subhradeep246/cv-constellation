"""Synthetic decision tests only; no synthetic image enters inference."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from constellation.conservative_fusion import fuse_row


def matches(*sites):
    return np.array([[score, x, y, 0, 1] for score, x, y in sites], dtype=np.float32)


class ConservativeFusionTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((256, 256), np.uint8)
        self.patches = [np.zeros((32, 32), np.uint8) for _ in range(2)]

    def test_weak_fourth_rank_figure_claim_abstains(self):
        base = {"Id": "scene", "n_patches": "2", "patch_01": "(100, 100, 1)",
                "patch_02": "-1", "constellation": "orion"}
        raw = dict(base, patch_01="-1")
        F = matches((.90, 20, 20), (.89, 40, 40), (.88, 60, 60), (.86, 100, 100))
        ev = {"pf": [{"F": F}, {"F": F}]}
        with patch("constellation.conservative_fusion._ring", return_value=.3):
            result, changes = fuse_row(base, raw, ev, ev, self.image, self.patches)
        self.assertEqual(result["patch_01"], "-1")
        self.assertEqual(changes[0]["action"], "abstain_weak_geometry")

    def test_stronger_patch_takes_one_star_without_duplicate(self):
        base = {"Id": "scene", "n_patches": "2", "patch_01": "(100, 100, 1)",
                "patch_02": "-1", "constellation": "orion"}
        raw = dict(base, patch_01="-1", patch_02="(102, 100, 1)")
        equal = {"pf": [{"F": matches((.87, 100, 100))},
                        {"F": matches((.90, 102, 100))}]}
        heavy = {"pf": [{"F": matches((.86, 100, 100))},
                        {"F": matches((.92, 102, 100))}]}
        with patch("constellation.conservative_fusion._ring", return_value=.7):
            result, changes = fuse_row(base, raw, equal, heavy, self.image, self.patches)
        self.assertEqual(result["patch_01"], "-1")
        self.assertEqual(result["patch_02"], "(102, 100, 1)")
        self.assertEqual([change["action"] for change in changes], ["release_star", "claim_star"])

    def test_weak_raw_only_match_is_not_added(self):
        base = {"Id": "scene", "n_patches": "2", "patch_01": "-1",
                "patch_02": "-1", "constellation": "orion"}
        raw = dict(base, patch_01="(100, 100, 1)")
        ev = {"pf": [{"F": matches((.65, 100, 100))},
                      {"F": matches((.90, 102, 100))}]}
        result, changes = fuse_row(base, raw, ev, ev, self.image, self.patches)
        self.assertEqual(result, base)
        self.assertEqual(changes, [])

    def test_relative_rank_can_transfer_a_star(self):
        base = {"Id": "scene", "n_patches": "2", "patch_01": "(100, 100, 1)",
                "patch_02": "-1", "constellation": "orion"}
        raw = dict(base, patch_01="-1", patch_02="(102, 100, 1)")
        incumbent = matches((.95, 20, 20), (.945, 40, 40), (.94, 60, 60),
                            (.93, 80, 80), (.925, 150, 150), (.915, 100, 100))
        challenger = matches((.914, 20, 20), (.912, 40, 40), (.911, 102, 100))
        equal = {"pf": [{"F": incumbent}, {"F": challenger}]}
        heavy = {"pf": [{"F": incumbent}, {"F": challenger}]}
        with patch("constellation.conservative_fusion._ring", return_value=.7):
            result, changes = fuse_row(base, raw, equal, heavy, self.image, self.patches)
        self.assertEqual(result["patch_01"], "-1")
        self.assertEqual(result["patch_02"], "(102, 100, 1)")
        self.assertEqual(len(changes), 2)


if __name__ == "__main__":
    unittest.main()
