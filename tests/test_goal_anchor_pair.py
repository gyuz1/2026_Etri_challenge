"""CPU-only tests for TRAIN anchor geometry and assignment safeguards."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

_path = Path(__file__).resolve().parents[1] / 'tools' / 'make_goal_anchor_pair.py'
_spec = importlib.util.spec_from_file_location('make_goal_anchor_pair', _path)
anchors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(anchors)


class GoalAnchorPairTest(unittest.TestCase):
    def test_nearest_uses_width_not_rectangular_membership(self):
        table = [[0., 1., 0., 1.], [10., 100., 0., 1.]]
        np.testing.assert_array_equal(anchors.nearest_labels([[1., 0.]], table), [1])

    def test_mirror_preserves_assignment(self):
        points = np.array([[1., -2.], [5., 3.]], dtype=np.float32)
        table = np.array([[0., 2., -2., 3.], [6., 2., 4., 3.]])
        reflected = points * [1., -1.]
        np.testing.assert_array_equal(
            anchors.nearest_labels(points, table),
            anchors.nearest_labels(reflected, anchors.mirror_table(table)))

    def test_supported_split_keeps_forward_geometry(self):
        points = np.array([[1., y] for _ in range(30) for y in [-3., 3.]],
                          dtype=np.float32)
        commands = np.zeros(60, dtype=int)
        scenes = np.array(['s{}'.format(i % 3) for i in range(60)])
        lat1, adaptive, report = anchors.fit_group(
            points, commands, scenes, 'test', (0,), False,
            np.array([0., 2.]), 2, min_frames=5, min_scenes=3)
        self.assertEqual(len(lat1), 1)
        self.assertEqual(len(adaptive), 2)
        np.testing.assert_array_equal(adaptive[:, :2], np.repeat(lat1[:, :2], 2, axis=0))
        np.testing.assert_array_equal(adaptive[:, 3], np.repeat(lat1[:, 3], 2))
        self.assertEqual(report['variants']['adaptive']['weak_anchors'], [])

    def test_frames_cannot_replace_scene_diversity(self):
        points = np.array([[1., y] for _ in range(50) for y in [-3., 3.]],
                          dtype=np.float32)
        _, adaptive, report = anchors.fit_group(
            points, np.zeros(100, int), np.array(['same'] * 100),
            'test', (0,), False, np.array([0., 1., 2.]), 2,
            min_frames=5, min_scenes=3)
        self.assertEqual(len(adaptive), 1)
        self.assertTrue(report['fallback_below_min_support'])

    def test_empty_forward_bins_merge_in_both_variants(self):
        points = np.array([[1., 0.]] * 30, dtype=np.float32)
        lat1, adaptive, report = anchors.fit_group(
            points, np.zeros(30, int), np.array(['s{}'.format(i % 3) for i in range(30)]),
            'test', (0,), False, np.array([0., 2., 4., 6.]), 2,
            min_frames=5, min_scenes=3)
        self.assertEqual(report['forward_edges'], [0., 6.])
        np.testing.assert_array_equal(lat1, adaptive)


if __name__ == '__main__':
    unittest.main()
