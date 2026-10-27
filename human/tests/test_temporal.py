"""Regression checks for automatic boundary selection, without GPU or assets."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CODE_ROOT, read
from frontend import classify_temporal_signals


class TemporalPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = read(CODE_ROOT / "config/temporal_policy.json")

    def signals(self, count=61):
        return np.full((count, 4), .9), np.full((count, 4), .02), np.full((count, 4), .02)

    def classify(self, values):
        return classify_temporal_signals(*values, self.policy)

    def test_normal_keeps_all(self):
        r = self.classify(self.signals())
        self.assertEqual((r['decision'], r['start_frame'], r['end_frame_exclusive']), ('keep_full', 0, 61))
        self.assertFalse(r['quality_approved'])

    def test_occlusion_without_pose_failure_is_retained(self):
        x = self.signals(); x[0][-8:] = .3
        self.assertEqual(self.classify(x)['decision'], 'keep_full')

    def test_fast_visible_action_is_retained(self):
        x = self.signals(); x[1][-8:] = .8
        self.assertEqual(self.classify(x)['decision'], 'keep_full')

    def test_tail_only(self):
        x = self.signals(); x[0][-5:] = .3; x[1][-5:] = .8
        r = self.classify(x)
        self.assertEqual((r['decision'], r['start_frame'], r['end_frame_exclusive']), ('trim_edges', 0, 54))

    def test_head_only(self):
        x = self.signals(); x[0][:5] = .3; x[1][:5] = .8
        r = self.classify(x)
        self.assertEqual((r['decision'], r['start_frame'], r['end_frame_exclusive']), ('trim_edges', 7, 61))

    def test_both_edges_share_one_budget(self):
        x = self.signals(121)
        x[0][:3] = x[0][-3:] = .3; x[1][:3] = x[1][-3:] = .8
        r = self.classify(x)
        self.assertEqual((r['decision'], r['start_frame'], r['end_frame_exclusive']), ('trim_edges', 5, 116))
        x = self.signals(); x[0][:3] = x[0][-3:] = .3; x[1][:3] = x[1][-3:] = .8
        self.assertEqual(self.classify(x)['decision'], 'skip')

    def test_middle_is_not_spliced(self):
        x = self.signals(); x[0][25:30] = .3; x[1][25:30] = .8
        self.assertEqual(self.classify(x)['decision'], 'skip')

    def test_single_noisy_frame_is_not_cut(self):
        x = self.signals(); x[0][-1] = .3; x[1][-1] = .8
        self.assertEqual(self.classify(x)['decision'], 'keep_full')

    def test_too_long_or_no_reference_is_skipped(self):
        x = self.signals(); x[0][-16:] = .3; x[1][-16:] = .8
        self.assertEqual(self.classify(x)['decision'], 'skip')
        x[0][:] = .2
        self.assertEqual(self.classify(x)['decision'], 'skip')


if __name__ == '__main__':
    unittest.main()
