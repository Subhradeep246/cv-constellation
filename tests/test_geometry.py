import unittest

import numpy as np

from constellation.geometry import (
    affine_from_points,
    homography_from_points,
    ransac_affine,
    ransac_similarity,
    similarity_from_points,
    transform_points,
)


class GeometryTests(unittest.TestCase):
    def test_similarity_recovers_rotation_scale_translation(self):
        source = np.array([[0., 0.], [10., 0.], [0., 8.], [6., 4.]])
        angle, scale = 0.4, 1.3
        rotation = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        destination = source @ rotation.T + np.array([12.0, -5.0])
        fit = similarity_from_points(source, destination)
        self.assertIsNotNone(fit)
        recovered = transform_points(fit, source)
        np.testing.assert_allclose(recovered, destination, atol=1e-6)

    def test_affine_recovers_nonuniform_scale_and_shear(self):
        source = np.array([[2., 3.], [8., 4.], [3., 10.], [11., 13.]])
        matrix = np.array([[-2., .8], [.3, 1.7]])
        destination = source @ matrix.T + np.array([24.0, -17.0])
        fit = affine_from_points(source, destination)
        self.assertIsNotNone(fit)
        np.testing.assert_allclose(transform_points(fit, source), destination, atol=1e-6)

    def test_ransac_similarity_ignores_outliers(self):
        rng = np.random.default_rng(3)
        source = rng.normal(size=(24, 2))
        fit = similarity_from_points(source[:8], source[:8] @ np.array([[0, -1.2], [1.2, 0]]) + [4, 9])
        destination = transform_points(fit, source)
        destination[:9] += rng.normal(scale=25, size=(9, 2))
        result = ransac_similarity(source, destination, threshold=0.5, trials=80, rng=rng)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.n_inliers, 12)
        np.testing.assert_allclose(transform_points(result.model, source[result.inliers]),
                                   destination[result.inliers], atol=0.5)

    def test_ransac_affine_ignores_outliers(self):
        rng = np.random.default_rng(4)
        source = rng.normal(size=(20, 2))
        true = affine_from_points(
            np.array([[0., 0.], [1., 0.], [0., 1.]]),
            np.array([[2., 1.], [3.5, 1.2], [1.8, 2.4]]))
        destination = transform_points(true, source)
        destination[:6] += 40
        result = ransac_affine(source, destination, threshold=0.4, trials=100, rng=rng)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.n_inliers, 12)

    def test_homography_is_available_but_not_required(self):
        source = np.array([[0., 0.], [10., 0.], [10., 8.], [0., 8.]], dtype=float)
        destination = np.array([[1., 2.], [12., 1.], [14., 11.], [0., 9.]], dtype=float)
        homography = homography_from_points(source, destination)
        self.assertIsNotNone(homography)
        ones = np.ones((len(source), 1))
        mapped = np.column_stack([source, ones]) @ homography.T
        mapped = mapped[:, :2] / mapped[:, 2:]
        np.testing.assert_allclose(mapped, destination, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
