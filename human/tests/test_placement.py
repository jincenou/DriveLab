import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from placement import standard_transform, transform_points, root_motion, bind_proxy, LOCAL_TO_STANDARD, camera_in_native_frame, vertical_sigma, action_diagnostics
from common import read, CODE_ROOT
from placement import foot_pair_diagnostic, motion_review_route, inplace_foot_velocities, validate_output_contract, source_contact_hints, validate_source_hint_package
from placement import contact_playback_curve


class PlacementTests(unittest.TestCase):
    def test_contact_clock_recovers_constant_speed_without_time_warp(self):
        times=np.arange(51)/25
        feet=np.zeros((51,2,3));feet[:,:,0]=-times[:,None]*.8
        support=np.ones((51,2),bool)
        result=contact_playback_curve(times,feet,support,read(CODE_ROOT/'config/placement_trial.json'))
        np.testing.assert_allclose(result['sample_times'],times,atol=1e-12)
        np.testing.assert_allclose(result['root_xy'][:,0],times*.8,atol=1e-12)
        self.assertAlmostEqual(result['native_speed'],.8)
        self.assertTrue(result['bounded'])

    def test_contact_clock_marks_short_gaps_and_rejects_long_ones(self):
        times=np.arange(51)/25;feet=np.zeros((51,2,3));feet[:,:,0]=-times[:,None]
        support=np.ones((51,2),bool);support[10:13]=False
        policy=read(CODE_ROOT/'config/placement_trial.json')
        result=contact_playback_curve(times,feet,support,policy)
        self.assertEqual(result['limits']['interpolated_frames'],3)
        support[10:25]=False
        with self.assertRaises(ValueError):contact_playback_curve(times,feet,support,policy)

    def test_contact_clock_rejects_stationary_and_nonfinite(self):
        times=np.arange(51)/25;feet=np.zeros((51,2,3));support=np.ones((51,2),bool)
        policy=read(CODE_ROOT/'config/placement_trial.json')
        with self.assertRaises(ValueError):contact_playback_curve(times,feet,support,policy)
        feet[0,0,0]=np.nan
        with self.assertRaises(ValueError):contact_playback_curve(times,feet,support,policy)

    def test_source_hint_readback_rejects_wrong_frame_time_even_with_valid_hash(self):
        import tempfile,json
        from common import sha
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);refs=p/'refs';refs.mkdir()
            for name in ('root_motion.npz','motion.safetensors'):(refs/name).write_bytes(name.encode())
            names=[s+'-'+part for s in ('left','right') for part in ('heel','big-toe-tip','small-toe-tip')]
            metadata={'provenance':{'keypoint_names':names,'selected_condition_frames':[2,3,4]}}
            (p/'placement.json').write_text(json.dumps(metadata))
            np.savez(p/'placement_motion.npz',frame_timestamps=np.array([0,.04,.08]),
                     condition_keypoints_standard_model_m=np.zeros((3,6,3)),
                     source_removed_horizontal_root_estimate=np.zeros((3,3)))
            values=dict(frame_timestamps=np.array([.08,.12,.16]),selected_condition_frames=np.array([2,3,4]),
                        source_foot_landmarks=np.zeros((3,2,3,3)),source_removed_horizontal_translation_estimate=np.zeros((3,3)))
            info={'frame_count':3,'signature':{'source_sha256':{'root_motion':sha(refs/'root_motion.npz'),
                                                             'pose_parameters':sha(refs/'motion.safetensors')}}}
            def write():
                np.savez(refs/'source_motion_hints.npz',**values)
                info['array_sha256']=sha(refs/'source_motion_hints.npz')
                (refs/'source_motion_hints.json').write_text(json.dumps(info))
            write()
            self.assertAlmostEqual(validate_source_hint_package(p)['source_to_asset_time_offset_seconds'],-.08)
            values['frame_timestamps'][1]+=.01;write()
            with self.assertRaises(AssertionError):validate_source_hint_package(p)

    def test_source_contact_candidates_need_observation_and_allow_swing(self):
        times=np.arange(10)/25
        feet=np.zeros((10,2,3,3));feet[:,1,:,2]=.2
        confidence=np.ones((10,2));confidence[4,0]=.1
        error=np.zeros((10,2));error[6,0]=.3
        result=source_contact_hints(feet,times,confidence,error,read(CODE_ROOT/'config/placement.json'))
        self.assertTrue(result['source_support_candidate'][0,0])
        self.assertFalse(result['source_support_candidate'][:,1].any())
        self.assertFalse(result['source_support_candidate'][4,0])
        self.assertFalse(result['source_support_candidate'][6,0])
        np.testing.assert_array_equal(result['source_foot_landmarks'],feet)

    def test_export_contract_rejects_changed_payload_and_stale_render(self):
        import tempfile,json
        from common import sha
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            (p/'asset_4dgs.npz').write_bytes(b'original-payload')
            (p/'placement_motion.npz').write_bytes(b'original-motion')
            (p/'insertion.json').write_text('{}')
            record={'asset_sha256':sha(p/'asset_4dgs.npz'),'array_sha256':sha(p/'placement_motion.npz'),
                    'coordinates':{'convention':'right_handed_x_forward_y_left_z_up'},
                    'scale':{'asset_unit_to_model_meter':2},'motion':{'planned_trajectory_ready':False},
                    'motion_review_group':'separate_motion_review',
                    'feet':{'contact_verified':False,'ground_verified':False},
                    'playback':{'loopable':False},'action':{'confidence':'estimated'}}
            (p/'placement.json').write_text(json.dumps(record))
            for name,state in [('placement_validation.json','exported_coordinate_and_binding_readback_passed'),
                               ('placement_replay.json','native_camera_placement_render_passed')]:
                (p/name).write_text(json.dumps({'state':state,'placement_sha256':sha(p/'placement.json'),
                                              'asset_sha256':record['asset_sha256']}))
            self.assertFalse(validate_output_contract(p)['planned_trajectory_ready'])
            (p/'asset_4dgs.npz').write_bytes(b'changed-payload')
            with self.assertRaises(AssertionError):validate_output_contract(p)
            (p/'asset_4dgs.npz').write_bytes(b'original-payload')
            record['scale']['asset_unit_to_model_meter']=3
            (p/'placement.json').write_text(json.dumps(record))
            with self.assertRaises(AssertionError):validate_output_contract(p)

    def test_foot_velocity_uses_same_inplace_frame_as_insertion(self):
        times=np.arange(21)/25
        yaw=times*.7
        rotation=np.zeros((len(times),3,3));rotation[:,0,0]=np.cos(yaw);rotation[:,0,1]=-np.sin(yaw)
        rotation[:,1,0]=np.sin(yaw);rotation[:,1,1]=np.cos(yaw);rotation[:,2,2]=1
        translation=np.stack([2*times,times,np.zeros(len(times))],axis=1)
        local=np.broadcast_to([[.2,.1,0],[-.2,-.1,0]],(len(times),2,3)).copy()
        standard=np.einsum('fij,fsj->fsi',rotation,local)+translation[:,None]
        into=np.repeat(np.eye(4)[None],len(times),axis=0)
        into[:,:3,:3]=rotation.transpose(0,2,1)
        into[:,:3,3]=-np.einsum('fij,fj->fi',into[:,:3,:3],translation)
        actual,velocity=inplace_foot_velocities(standard[:,0],standard[:,1],into,times)
        np.testing.assert_allclose(actual,local,atol=1e-12)
        np.testing.assert_allclose(velocity,0,atol=1e-12)
        with self.assertRaises(ValueError):inplace_foot_velocities(standard[:,0],standard[:,1],into,times[::-1])

    def test_routing_separates_unreliable_motion_without_approval(self):
        import copy
        names=('left-foot','right-foot','left-hip','right-hip','left-shoulder','right-shoulder')
        record={'motion':{'associations':{n:{'numerically_supported':True} for n in names},
                          'gait':{'native_speed_model_mps_estimate':1.}},'action':{'label':'walking'}}
        self.assertEqual(motion_review_route(record),'walking_validation_candidate')
        broken=copy.deepcopy(record);broken['motion']['associations']['left-foot']['numerically_supported']=False
        self.assertEqual(motion_review_route(broken),'separate_motion_review')
        broken=copy.deepcopy(record);broken['feet']={'pair_assignment_diagnostic':{'state':'possible_left_right_exchange'}}
        self.assertEqual(motion_review_route(broken),'separate_motion_review')
        record['motion']['gait']['native_speed_model_mps_estimate']=None
        self.assertEqual(motion_review_route(record),'contact_support_review')
        self.assertEqual(motion_review_route(None),'pending_motion_metadata')

    def test_pair_exchange_is_detected_without_relabeling(self):
        left=np.tile([0,.2,0],(6,1));right=np.tile([0,-.2,0],(6,1))
        tracked_left=left.copy();tracked_right=right.copy()
        tracked_left[2:4]=right[2:4];tracked_right[2:4]=left[2:4]
        report=foot_pair_diagnostic(tracked_left,tracked_right,left,right)
        self.assertEqual(report['suspect_frames'],[2,3])
        self.assertFalse(report['automatic_relabeling'])
        self.assertFalse(report['semantic_identity_verified'])
        np.testing.assert_array_equal(tracked_left[2:4],right[2:4])
        self.assertEqual(foot_pair_diagnostic(left,right,left,right)['suspect_frames'],[])

    def test_pair_unknown_and_nonfinite(self):
        self.assertEqual(foot_pair_diagnostic(None,None,None,None)['state'],'unknown_missing_fixed_proxy')
        invalid=np.full((3,3),np.nan)
        with self.assertRaises(ValueError):foot_pair_diagnostic(invalid,invalid,invalid,invalid)

    def test_tilt_scale_translation_roundtrip(self):
        angle=np.deg2rad(15)
        a=np.eye(4)
        a[:3,:3]=3*np.array([[1,0,0],[0,-np.cos(angle),-np.sin(angle)],[0,np.sin(angle),-np.cos(angle)]])
        a[:3,3]=[.2,1.7,3]
        t,scale,error=standard_transform(a)
        self.assertAlmostEqual(scale,3)
        self.assertAlmostEqual(error,15)
        p=np.array([[1,2,3],[.1,-.2,.3]])
        np.testing.assert_allclose(transform_points(p,t),transform_points(p,a)@LOCAL_TO_STANDARD.T)
        np.testing.assert_allclose(transform_points(transform_points(p,t),np.linalg.inv(t)),p,atol=1e-12)
        np.testing.assert_allclose(LOCAL_TO_STANDARD@[0,1,0],[0,0,1])
        np.testing.assert_allclose(LOCAL_TO_STANDARD@[0,0,1],[1,0,0])
        self.assertAlmostEqual(np.linalg.det(t[:3,:3]/scale),1)

    def test_reject_shear(self):
        a=np.eye(4);a[0,1]=.1
        with self.assertRaises(AssertionError):standard_transform(a)

    def test_motion_from_gaussians_not_source_motion(self):
        times=np.arange(11)/10
        root=np.stack([2*times,np.zeros(11),np.ones(11)],axis=1)
        result=root_motion(root+[0,.2,0],root-[0,.2,0],root+[0,.3,.5],root+[0,-.3,.5],times)
        np.testing.assert_allclose(result['gaussian_root_speed'],2)
        np.testing.assert_allclose(result['gaussian_heading_radians'],0)

    def test_binding_retains_point_identity(self):
        policy={'association_probe_frames':3,'opacity_min':.5,'association_radius_model_m':.15,
                'association_points_max':32,'association_points_min':8,'association_error_p95_model_m':.15}
        points=np.zeros((5,20,3));points[:,:10,0]=np.arange(10)*.001;points[:,10:,0]=10
        trajectory,record,ids,weights=bind_proxy(points,np.zeros((5,3)),np.ones(20),policy)
        self.assertTrue(record['numerically_supported'])
        np.testing.assert_array_equal(ids,np.arange(10))
        np.testing.assert_allclose(trajectory,np.einsum('fnc,n->fc',points[:,ids],weights))
        result=bind_proxy(points,np.full((5,3),100),np.ones(20),policy)
        self.assertIsNone(result[0])

    def test_placed_world_camera_inverse_with_translation_yaw_scale(self):
        asset=np.eye(4);asset[:3,:3]=np.diag([2.,2.,2.]);asset[:3,3]=[1,2,3]
        placement=np.eye(4);placement[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];placement[:3,3]=[10,20,5]
        native_camera=np.eye(4);native_camera[:3,3]=[2,3,4]
        world=placement@asset@native_camera;world[:3,:3]/=2
        recovered,scale=camera_in_native_frame(world,placement,asset,np.eye(4))
        self.assertEqual(scale,2)
        np.testing.assert_allclose(recovered,native_camera,atol=1e-12)

    def test_support_sigma_includes_scale(self):
        q=np.array([[[1.,0,0,0]]]);s=np.array([[[.01,.02,.03]]])
        np.testing.assert_allclose(vertical_sigma(q,s,[0,0,3]),[[.09]])

    def test_stationary_action_does_not_become_walk_from_source_speed(self):
        names=['left-hip','right-hip','left-ankle','right-ankle','left-wrist','right-wrist']
        kp=np.tile(np.array([[0,.1,1],[0,-.1,1],[0,.1,0],[0,-.1,0],[0,.3,1],[0,-.3,1]])[None],(20,1,1))
        result=action_diagnostics(kp,names,np.arange(20)/25,np.ones(20)*10,read(CODE_ROOT/'config/placement.json'))
        self.assertEqual(result['label'],'standing')


if __name__=='__main__':unittest.main()
