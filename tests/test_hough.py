import unittest

import numpy as np
from scipy.spatial import cKDTree

from constellation.copy_evidence import clustered_pairs
from constellation.hough_infer import confident_extra_candidates, merge_extra_candidates
from constellation.hough_probe import search_pattern
from constellation.hough_refine import fit_affine, prepare_groups, score_pose, select_hits
from constellation.star_catalog import broad_star_centres


class HoughTests(unittest.TestCase):
    def test_independent_candidate_stream_is_bounded_and_deduplicated(self):
        coarse = [dict(x=10, y=10)]
        extra = [dict(x=11, y=10)] + [dict(x=30 + 20*i, y=40) for i in range(7)]
        result = merge_extra_candidates(coarse, extra)
        self.assertEqual(len(result), 5)
        self.assertEqual((result[1]['x'], result[1]['y']), (30, 40))

    def test_only_distinct_top_ncc_site_enters_confident_hough(self):
        strong = [dict(x=10, y=20, score=.80),
                  dict(x=80, y=90, score=.70)]
        weak = [dict(x=10, y=20, score=.80),
                dict(x=80, y=90, score=.79)]
        self.assertEqual(confident_extra_candidates(strong, .16), strong[:1])
        self.assertEqual(confident_extra_candidates(weak, .16), [])
        self.assertEqual(confident_extra_candidates(strong[:1], .16), [])

    def test_affine_refit_recovers_pose(self):
        source = np.array([[0., 0.], [10., 0.], [0., 20.], [8., 9.]])
        expected = np.array([[1.2, -.3], [.4, 1.1]])
        translation = np.array([40., 70.])
        matrix, offset = fit_affine(source, source @ expected.T + translation)
        np.testing.assert_allclose(matrix, expected, atol=1e-10)
        np.testing.assert_allclose(offset, translation, atol=1e-10)

    def test_pose_uses_distinct_patches(self):
        nodes = np.array([[0., 0.], [10., 0.], [0., 10.], [10., 10.]])
        items = [[dict(x=float(x), y=float(y))] for x, y in nodes + [100, 100]]
        groups = prepare_groups(items)
        result = score_pose(dict(x=105., y=105., scale=1., degrees=0.),
                            nodes, groups, tolerance=3, image_shape=(300, 300))
        self.assertEqual(result['matched'], 4)
        self.assertEqual(len({entry['patch'] for entry in result['chosen']}), 4)

    def test_pose_pruning_keeps_strong_distinct_hypotheses(self):
        hits = [dict(x=100., y=100., scale=4., degrees=20., support=7.),
                dict(x=108., y=100., scale=4., degrees=25., support=6.),
                dict(x=300., y=100., scale=4., degrees=20., support=6.)]
        retained = select_hits(hits, 2)
        self.assertEqual(len(retained), 2)
        self.assertEqual({item['x'] for item in retained}, {100., 300.})

    def test_pose_pruning_keeps_mirrored_alternative(self):
        hits = [dict(x=100., y=100., scale=4., degrees=20., support=7., reflected=False),
                dict(x=100., y=100., scale=4., degrees=20., support=6., reflected=True)]
        self.assertEqual(len(select_hits(hits, 2)), 2)

    def test_synthetic_mirrored_figure_is_searched_and_refined(self):
        # Synthetic geometry is test-only; inference still reads supplied patterns.
        nodes = np.array([[0., 0.], [9., 0.], [3., 7.], [17., 10.],
                          [8., 18.], [21., 24.]])
        angle = np.deg2rad(30.)
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        matrix = 2. * rotation @ np.diag([-1., 1.])
        points = (nodes - nodes.mean(axis=0)) @ matrix.T + [140., 130.]
        image = np.zeros((300, 300), dtype=np.float32)
        for x, y in np.rint(points).astype(int):
            image[y, x] = 1.
        _, plain_hits = search_pattern(image, nodes, 1, np.array([2.]),
                                       np.array([30.]), min_support=6)
        _, mirror_hits = search_pattern(image, nodes, 1, np.array([2.]),
                                        np.array([30.]), min_support=6,
                                        include_reflection=True)
        self.assertEqual(len(plain_hits), 0)
        mirror = next(hit for hit in mirror_hits if hit['reflected'])
        groups = prepare_groups([[dict(x=float(x), y=float(y))] for x, y in points])
        result = score_pose(mirror, nodes, groups, tolerance=3,
                            image_shape=image.shape)
        self.assertEqual(result['matched'], len(nodes))
        self.assertEqual(len({item['patch'] for item in result['chosen']}), len(nodes))

    def test_synthetic_four_node_figure_can_form_hough_hit(self):
        nodes = np.array([[0., 0.], [12., 1.], [3., 9.], [21., 14.]])
        projected = (nodes - nodes.mean(axis=0)) * 2 + [80., 90.]
        image = np.zeros((180, 180), dtype=np.float32)
        for x, y in np.rint(projected).astype(int):
            image[y, x] = 1.
        _, hits = search_pattern(image, nodes, 1, np.array([2.]),
                                 np.array([0.]), min_support=4)
        self.assertTrue(any(hit['support'] == 4 for hit in hits))

    def test_synthetic_star_seeded_search_bounds_hypotheses(self):
        nodes = np.array([[0., 0.], [9., 0.], [3., 7.],
                          [17., 10.], [8., 18.], [21., 24.]])
        image = np.zeros((180, 180), dtype=np.float32)
        for centre in ([70., 80.], [125., 115.]):
            points = (nodes - nodes.mean(axis=0)) * 2 + centre
            for x, y in np.rint(points).astype(int):
                image[y, x] = 1.
        _, hits = search_pattern(image, nodes, 1, np.array([2.]),
                                 np.array([0.]), min_support=6,
                                 max_hits_per_transform=2)
        self.assertLessEqual(len(hits), 2)
        self.assertTrue(any(hit['support'] == 6 for hit in hits))

    def test_broad_star_catalog_recovers_saturated_synthetic_blob(self):
        import cv2
        sky = np.full((128, 128), 100, dtype=np.uint8)
        cv2.circle(sky, (64, 70), 7, 255, -1)
        stars = broad_star_centres(sky, count=10)
        self.assertLess(cKDTree(stars).query([[64, 70]])[0][0], 2)

    def test_pose_scores_independent_synthetic_star_support(self):
        nodes = np.array([[0., 0.], [10., 0.], [0., 10.], [10., 10.]])
        points = nodes + [100., 100.]
        groups = prepare_groups([[dict(x=float(x), y=float(y))] for x, y in points])
        result = score_pose(dict(x=105., y=105., scale=1., degrees=0.),
                            nodes, groups, star_tree=cKDTree(points),
                            image_shape=(300, 300))
        self.assertEqual(result['star_hits'], 4)
        self.assertGreater(result['star_score'], 3.9)

    def test_copy_displacement_requires_repetition(self):
        pairs = [dict(a=[float(i), 20.], b=[float(i + 100), 40.], annulus_ncc=.95)
                 for i in range(6)]
        pairs.append(dict(a=[50., 80.], b=[200., 80.], annulus_ncc=.99))
        _, selected = clustered_pairs(pairs, min_support=5)
        self.assertEqual(len(selected), 6)
        self.assertTrue(all(item['shift'] == (25, 5) for item in selected))


if __name__ == '__main__':
    unittest.main()
