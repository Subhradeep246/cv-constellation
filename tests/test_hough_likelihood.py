import unittest

from constellation.hough_likelihood import likelihood_score


class HoughLikelihoodTests(unittest.TestCase):
    def test_unsupported_nodes_reduce_score(self):
        supported = dict(in_frame_nodes=3, chosen=[dict(patch=i+1, distance=0.)
                                                   for i in range(3)],
                         star_distances=[1., 1., 1.], star_ranks=[1, 2, 3],
                         copy_nodes=0)
        missing = dict(supported, in_frame_nodes=6,
                       star_distances=[1., 1., 1., 100., 100., 100.],
                       star_ranks=[1, 2, 3, 300, 300, 300])
        gaps = [.2, .2, .2]
        self.assertGreater(likelihood_score(supported, 6, gaps),
                           likelihood_score(missing, 6, gaps))

    def test_clear_patch_and_star_rank_add_evidence(self):
        pose = dict(in_frame_nodes=1, chosen=[dict(patch=1, distance=0.)],
                    star_distances=[3.], star_ranks=[150], copy_nodes=0)
        strong = likelihood_score(pose, 1, [.3])
        weak = likelihood_score(pose, 1, [.01])
        self.assertAlmostEqual(strong - weak, 1.)
        brighter = dict(pose, star_ranks=[10])
        self.assertGreater(likelihood_score(brighter, 1, [.3]), strong)
