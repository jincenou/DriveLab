import unittest
import numpy as np
from playback import compose,sample


class PlaybackTests(unittest.TestCase):
    def test_single_timestamp_does_not_reset_to_first_frame(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            np.savez(p/'placement_playback.npz',frame_timestamps=[0.,1.],root_xy=np.zeros((2,2)),root_yaw=np.zeros(2),asset_to_standard=np.eye(4),ground_offset=0.)
            with patch('playback.validate_estimate'):
                _,frame=sample(p,[.5],[[0,0,0]],[0])
            np.testing.assert_allclose(frame,[.5])

    def test_root_removed_once_and_world_path_added(self):
        times=np.array([0.,1.,2.]);root=np.stack([times,np.ones(3)],axis=1)
        yaw=np.array([0.,.2,.4]);query=np.array([.5,1.5]);world=np.array([[10,20,0],[11,20,.1]])
        M,frames=compose(times,root,yaw,np.eye(4),-.02,query,world,np.array([.3,.5]))
        center=np.c_[query,np.ones(2),np.ones(2),np.ones(2)]
        actual=np.einsum('fij,fj->fi',M,center)
        np.testing.assert_allclose(actual[:,:2],world[:,:2],atol=1e-12)
        np.testing.assert_allclose(actual[:,2],world[:,2]+.98,atol=1e-12)
        np.testing.assert_allclose(frames,query)

    def test_no_implicit_loop_or_hold(self):
        with self.assertRaises(ValueError):
            compose([0,1],np.zeros((2,2)),np.zeros(2),np.eye(4),0,[2],np.zeros((1,3)),[0])

    def test_similarity_and_heading_are_applied_once(self):
        T=np.diag([2.,2.,2.,1.]);T[:3,3]=[3,4,5]
        M,_=compose([0,1],np.array([[3,4],[3,4]]),np.zeros(2),T,0,[0],[[10,20,0]],[np.pi/2])
        point=M[0]@np.array([1,0,0,1]);np.testing.assert_allclose(point,[10,22,5,1],atol=1e-12)
