import unittest
import csv
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from constellation.data import parse_cell, validate_submission
from constellation.identify import Pattern, assignment, quad_index, rank_patterns_with_alternatives
from constellation.localize import Localizer, LocalizerConfig
from constellation.metrics import reward, score_scene


def row(values, name='a'):
    return dict(Id='scene', n_patches=str(len(values)), constellation=name,
                **{f'patch_{i+1:02d}':v for i,v in enumerate(values)})


class CoreTests(unittest.TestCase):
    def test_parser(self):
        self.assertIsNone(parse_cell('-1'))
        self.assertEqual(parse_cell('(12, 34, 1)'), (12,34,1))
        self.assertEqual(parse_cell('12 34'), (12,34,0))
        for invalid in ['nan 2 1', '(1,2,3)', '(1,2,.5)', 'oops']:
            with self.assertRaises((ValueError, SyntaxError)):
                parse_cell(invalid)

    def test_localization_ramp(self):
        np.testing.assert_allclose(reward([0,12,24,36,50]), [1,1,.5,0,0])

    def test_perfect_and_empty(self):
        truth = row(['(10,20,1)', '-1', '(80,90,0)'])
        self.assertEqual(score_scene(truth, truth)['score'], 1)
        absent = row(['-1','-1','-1'], 'unknown')
        scores = score_scene(truth, absent)
        self.assertEqual(scores['localization'], 0)
        self.assertEqual(scores['geometric_recovery'], 0)
        self.assertAlmostEqual(scores['presence'], .25)

    def test_geometry_all_present_one_to_one(self):
        truth = row(['(10,20,1)', '(11,20,1)', '-1'])
        pred = row(['-1', '-1', '(10,20,0)'])
        scores = score_scene(truth, pred)
        self.assertEqual(scores['localization'], 0)
        self.assertEqual(scores['geometric_recovery'], .5)

    def test_assignment_allows_missing(self):
        a,b,d = assignment(np.array([[0,0],[1,0],[100,100]]), np.array([[0,0],[99,99]]), 3)
        self.assertEqual(len(a), 2)
        self.assertEqual(len(set(b)), 2)

    def test_joint_geometry_can_select_lower_ranked_patch_location(self):
        nodes = np.array([[0,0],[3,0],[0,2],[5,4],[2,6],[7,1]], dtype=float)
        points = nodes * 20 + [150, 100]
        candidates = [[dict(x=float(p[0]+70), y=float(p[1]), score=.98),
                       dict(x=float(p[0]), y=float(p[1]), score=.90)] if i == 0 else
                      [dict(x=float(p[0]), y=float(p[1]), score=.90)]
                      for i, p in enumerate(points)]
        ranked = rank_patterns_with_alternatives([Pattern('toy', nodes)], candidates)
        self.assertEqual(ranked[0]['name'], 'toy')
        self.assertEqual(ranked[0]['matched_nodes'], 6)
        self.assertAlmostEqual(ranked[0]['point_locations']['0']['x'], points[0,0])

    def test_quad_affine_invariance_and_reflection(self):
        points=np.array([[2.,3.],[8.,4.],[3.,10.],[11.,13.]])
        transformed=points@np.array([[-2.,.8],[.3,1.7]]).T+[24,-17]
        code,vertices=quad_index(points)
        new_code,new_vertices=quad_index(transformed)
        self.assertGreater(len(code),0)
        np.testing.assert_allclose(code,new_code,atol=1e-12)
        np.testing.assert_array_equal(vertices,new_vertices)

    def test_blank_patch_is_rejected(self):
        loc=Localizer(LocalizerConfig())
        match=loc.verify(np.zeros((100,100),np.float32),np.zeros((32,32),np.uint8),
                         np.array([[50,50,1]],dtype=np.float32))
        self.assertFalse(match.present)

    def test_coordinate_convention(self):
        truth=row(['(10,70,1)','-1'])
        swapped=row(['(70,10,1)','-1'])
        self.assertEqual(score_scene(truth,swapped)['localization'],0)

    def test_submission_contract(self):
        schema=[row(['-1','-1'],'unknown')]
        good=[row(['(10,70,1)','-1'])]
        validate_submission(good,schema,{'a'},{'scene':(100,100)})
        for bad in [[], good+good, [row(['(100,70,1)','-1'])], [row(['-1'])], [row(['-1','-1'],'BAD')]]:
            with self.assertRaises(ValueError):
                validate_submission(bad,schema,{'a'},{'scene':(100,100)})

    def test_predict_without_training_files(self):
        """A new scene root needs no labels; enforce the real CLI contract."""
        import cv2
        with tempfile.TemporaryDirectory(prefix='constellation-test-') as temporary:
            root=Path(temporary)
            (root/'patterns').mkdir()
            scene=root/'validation'/'unseen'
            (scene/'patches').mkdir(parents=True)
            drawing=np.zeros((64,64,4),dtype=np.uint8)
            for point in [(10,10),(50,15),(20,45),(45,50)]:
                cv2.circle(drawing,point,3,(255,255,255,255),-1)
            cv2.imwrite(str(root/'patterns'/'example_pattern.png'),drawing)
            cv2.imwrite(str(scene/'unseen_image.png'),np.zeros((100,100),np.uint8))
            cv2.imwrite(str(scene/'patches'/'patch_01.png'),np.zeros((32,32),np.uint8))
            with (root/'sample_submission.csv').open('w',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['Id','n_patches','patch_01','constellation'])
                writer.writerow(['unseen',1,-1,'unknown'])
            output=root/'output'
            result=subprocess.run([sys.executable,'-m','constellation','predict','--data',str(root),
                                   '--output',str(output),'--no-sift','--sources','10','--top-k','2'],
                                  capture_output=True,text=True,timeout=30,
                                  cwd=Path(__file__).resolve().parents[1])
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            with (output/'submission.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(rows[0]['Id'],'unseen')
            self.assertEqual(rows[0]['patch_01'],'-1')
            hough_output = root/'hough-output'
            hough = subprocess.run([sys.executable, '-m', 'constellation', 'predict',
                                    '--data', str(root), '--output', str(hough_output),
                                    '--no-sift', '--sources', '10', '--top-k', '2',
                                    '--hough-identification', '--mirror-hough',
                                    '--small-hough', '--bright-star-weight', '1',
                                    '--star-seeded-hough'],
                                   capture_output=True, text=True, timeout=60,
                                   cwd=Path(__file__).resolve().parents[1])
            self.assertEqual(hough.returncode, 0, hough.stdout+hough.stderr)
            with (hough_output/'submission.csv').open() as stream:
                hough_rows = list(csv.DictReader(stream))
            self.assertEqual(hough_rows[0]['Id'], 'unseen')
            self.assertEqual(hough_rows[0]['patch_01'], '-1')

    def test_polar_rotation_sign_and_verification(self):
        import cv2
        rng = np.random.default_rng(8)
        image = cv2.GaussianBlur(rng.normal(size=(160,160)).astype(np.float32), (0,0), 1)
        image = np.clip(image*55+120,0,255).astype(np.uint8)
        yy,xx = np.mgrid[:32,:32].astype(np.float32)
        angle,scale,x,y = .7,1.15,90.,65.
        u,v = xx-15.5,yy-15.5
        patch = cv2.remap(image, (x+scale*(np.cos(angle)*u-np.sin(angle)*v)).astype(np.float32),
                          (y+scale*(np.sin(angle)*u+np.cos(angle)*v)).astype(np.float32), cv2.INTER_LINEAR)
        loc = Localizer(LocalizerConfig(refine_candidates=2))
        match = loc.verify(loc.filtered(image), patch, np.array([[90,65,1.1],[35,35,1.1]], dtype=np.float32))
        self.assertLess(np.hypot(match.x-x,match.y-y), 1)
        self.assertGreater(match.score, .9)
        self.assertLess(abs(match.scale-scale), .1)
        self.assertGreaterEqual(len(match.alternatives), 2)
        self.assertLess(np.hypot(match.alternatives[0]['x']-x, match.alternatives[0]['y']-y), 1)

    def test_synthetic_blurred_noisy_illumination_patch(self):
        """Exercise photometric nuisance handling without adding generated data to inference."""
        import cv2
        rng = np.random.default_rng(21)
        sky = cv2.GaussianBlur(rng.normal(size=(180, 180)).astype(np.float32),
                               (0, 0), .8) * 24 + 110
        for cx, cy, bright in [(80, 70, 120), (92, 78, 65), (72, 88, 85),
                               (125, 125, 130), (132, 117, 60)]:
            cv2.circle(sky, (cx, cy), 2, float(bright), -1)
        sky = np.clip(sky, 0, 255).astype(np.uint8)
        yy, xx = np.mgrid[:32, :32].astype(np.float32)
        u, v = xx - 15.5, yy - 15.5
        x, y, angle, scale = 83., 79., .55, 1.12
        patch = cv2.remap(sky, (x + scale * (np.cos(angle) * u - np.sin(angle) * v)).astype(np.float32),
                          (y + scale * (np.sin(angle) * u + np.cos(angle) * v)).astype(np.float32),
                          cv2.INTER_LINEAR)
        patch = cv2.GaussianBlur(patch, (0, 0), .65).astype(np.float32)
        patch = patch * (.9 + .2 * xx / 31) + rng.normal(0, 1.5, patch.shape)
        patch = np.clip(patch, 0, 255).astype(np.uint8)
        loc = Localizer(LocalizerConfig(refine_candidates=2))
        match = loc.verify(loc.filtered(sky), patch,
                           np.array([[x, y, scale], [127, 122, scale]], dtype=np.float32))
        self.assertTrue(np.isfinite(match.score))
        self.assertLess(np.hypot(match.x - x, match.y - y), 3)

    def test_ransac_alignment_drives_location_when_matches_exist(self):
        import cv2
        rng = np.random.default_rng(2)
        sky = np.clip(rng.integers(18, 36, (220, 220)), 0, 255).astype(np.uint8)
        for cx, cy in [(40, 50), (80, 45), (120, 70), (70, 110), (150, 130), (55, 160), (170, 60), (100, 175)]:
            cv2.circle(sky, (cx, cy), 2, 230, -1)
        x, y, angle, scale = 100., 90., .25, 1.08
        yy, xx = np.mgrid[:32, :32].astype(np.float32)
        u, v = xx-15.5, yy-15.5
        patch = cv2.remap(sky, (x+scale*(np.cos(angle)*u-np.sin(angle)*v)).astype(np.float32),
                          (y+scale*(np.sin(angle)*u+np.cos(angle)*v)).astype(np.float32), cv2.INTER_LINEAR)
        loc = Localizer(LocalizerConfig(max_sources=1500, candidates_per_scale=8, harris=False, refine_candidates=3))
        matches, stats = loc.match_scene(sky, [patch], log=lambda *_: None)
        self.assertIn("ransac_aligned", stats)
        self.assertEqual(stats["ransac_accepted"] + stats["polar_fallback"], 1)
        if matches[0].method == "ransac":
            self.assertLess(np.hypot(matches[0].x-x, matches[0].y-y), 12)


if __name__ == '__main__':
    unittest.main()
