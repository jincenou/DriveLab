import unittest
from motion_review import validate_result, crosscheck


class MotionReviewTests(unittest.TestCase):
    def example(self):
        return dict(action='unknown',direction='unknown',confidence=.2,evidence=[],
                    feet=[dict(frame=0,left_visible=False,right_visible=False,support='unknown',
                               left_xy=None,right_xy=None,reason='occluded')],issues=[],uncertainties=['occluded'],next_check='inspect')

    def test_unknown_is_valid(self):
        validate_result(self.example(),[0])

    def test_reject_invented_frame(self):
        value=self.example();value['evidence']=[dict(frame=1,observation='unsupported frame')]
        with self.assertRaises(AssertionError):validate_result(value,[0])

    def test_reject_location_of_invisible_foot(self):
        value=self.example();value['feet'][0]['left_xy']=[.5,.8]
        with self.assertRaises(AssertionError):validate_result(value,[0])

    def test_reject_missing_frame_or_unapproved_field(self):
        with self.assertRaises(AssertionError):validate_result(self.example(),[0,1])
        value=self.example();value['loopable']=True
        with self.assertRaises(AssertionError):validate_result(value,[0])

    def test_unseen_contact_claim_is_flagged_not_promoted(self):
        value=self.example();value['feet'][0]['support']='neither'
        check=crosscheck('unused','observed',value)
        self.assertEqual(check['support_asserted_without_visible_feet'],[0])
        self.assertFalse(check['contact_verified'])
        self.assertFalse(check['metric_speed_available'])
