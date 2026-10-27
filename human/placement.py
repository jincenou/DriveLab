"""CPU-only insertion coordinates and Gaussian-bound motion diagnostics.

This module never changes the reconstructed Gaussian payload or its animation.
Source skeletons provide semantic hints; exported motion is measured on explicit
Gaussian indices, and is not promoted to reliable foot contacts without review.
"""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import time
import sys
import numpy as np
from common import ROOT, CODE_ROOT, ASSET_ROOT, read, save, sha, case

LOCAL_TO_STANDARD = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])


def transform_points(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def camera_in_native_frame(world_camera_to_world, world_from_standard, asset_to_standard, asset_from_native):
    """Render the untouched native Gaussians from a placed world's camera.

    No separate SH/covariance rotation is needed: evaluate rays and SH in the
    original native frame. Camera translations are in native units; world depth,
    if compositing with a separate scene renderer, needs the similarity scale.
    """
    world_from_native=np.asarray(world_from_standard) @ np.asarray(asset_to_standard) @ np.asarray(asset_from_native)
    singular=np.linalg.svd(world_from_native[:3,:3],compute_uv=False)
    scale=float(singular.mean())
    np.testing.assert_allclose(singular,scale,rtol=2e-5)
    assert np.linalg.det(world_from_native[:3,:3])>0
    camera=np.linalg.inv(world_from_native) @ np.asarray(world_camera_to_world)
    camera[:3,:3]*=scale
    return camera, scale


def standard_transform(asset_to_local):
    """Recover model meters and generator gravity, without using camera up."""
    a = np.asarray(asset_to_local, np.float64)
    assert a.shape == (4, 4) and np.isfinite(a).all()
    np.testing.assert_allclose(a[3], [0, 0, 0, 1], atol=1e-10)
    singular = np.linalg.svd(a[:3, :3], compute_uv=False)
    scale = float(singular.mean())
    assert scale > 0 and np.linalg.det(a[:3, :3]) > 0
    np.testing.assert_allclose(singular, scale, rtol=2e-5)
    b = np.eye(4)
    b[:3, :3] = LOCAL_TO_STANDARD
    transform = b @ a
    rotation = transform[:3, :3] / scale
    up = np.linalg.solve(a[:3, :3], [0., 1., 0.])
    up /= np.linalg.norm(up)
    old_error = float(np.degrees(np.arccos(np.clip(up @ [0., -1., 0.], -1, 1))))
    np.testing.assert_allclose(rotation @ up, [0, 0, 1], atol=2e-5)
    return transform, scale, old_error


def bind_proxy(points, target, opacity, policy):
    """Fixed point identities; no framewise relabeling that hides sliding."""
    probe = np.unique(np.rint(np.linspace(0, len(points)-1, policy['association_probe_frames'])).astype(int))
    distance = np.median(np.linalg.norm(points[probe] - target[probe, None], axis=-1), axis=0)
    candidates = np.flatnonzero((opacity >= policy['opacity_min']) &
                               (distance <= policy['association_radius_model_m']))
    selected = candidates[np.argsort(distance[candidates], kind='stable')[:policy['association_points_max']]]
    if len(selected) < policy['association_points_min']:
        return None, dict(state='unknown_insufficient_gaussian_support', count=len(selected)), selected, None
    weights = opacity[selected] / np.maximum(distance[selected], .01)
    weights /= weights.sum()
    trajectory = np.einsum('fnc,n->fc', points[:, selected], weights)
    error = np.linalg.norm(trajectory - target, axis=-1)
    accepted = float(np.quantile(error, .95)) <= policy['association_error_p95_model_m']
    return trajectory, dict(state='geometric_proxy_pending_semantic_review' if accepted else 'association_error',
                            count=len(selected), error_median_model_m=float(np.median(error)),
                            error_p95_model_m=float(np.quantile(error, .95)),
                            numerically_supported=accepted), selected, weights


def root_motion(left_hip, right_hip, left_shoulder, right_shoulder, times):
    root = (left_hip + right_hip) / 2
    left = (left_hip-right_hip) + (left_shoulder-right_shoulder)
    forward = np.cross(left, np.array([0., 0., 1.]))
    norm = np.linalg.norm(forward[:, :2], axis=1)
    if np.any(norm < .05):
        raise ValueError('Body heading is undefined: insufficient lateral span')
    yaw = np.unwrap(np.arctan2(forward[:, 1], forward[:, 0]))
    velocity = np.gradient(root, times, axis=0)
    rotation=np.zeros((len(root),3,3))
    rotation[:,0,0]=np.cos(yaw);rotation[:,0,1]=np.sin(yaw)
    rotation[:,1,0]=-np.sin(yaw);rotation[:,1,1]=np.cos(yaw);rotation[:,2,2]=1
    into=np.repeat(np.eye(4)[None],len(root),axis=0);into[:,:3,:3]=rotation
    horizontal=root.copy();horizontal[:,2]=0
    into[:,:3,3]=-np.einsum('fij,fj->fi',rotation,horizontal)
    return dict(gaussian_root_position=root, gaussian_root_displacement=root-root[:1],
                gaussian_heading_radians=yaw, gaussian_root_velocity=velocity,
                gaussian_root_speed=np.linalg.norm(velocity[:, :2], axis=1),
                standard_to_gaussian_inplace=into, gaussian_inplace_to_standard=np.linalg.inv(into))


def action_diagnostics(kp, names, times, source_speed, policy):
    """Explicit rule-based estimates; source root speed alone cannot define gait."""
    def joints(group):
        return kp[:, [names.index(n) for n in group]].mean(1)
    pelvis=joints(['left-hip','right-hip'])
    left=joints(['left-hip'])-joints(['right-hip'])
    forward=np.cross(left,[0.,0.,1.]); forward/=np.maximum(np.linalg.norm(forward,axis=1,keepdims=True),1e-8)
    def relative(side, joint):return joints([side+'-'+joint])-pelvis
    def span(value):return float(np.linalg.norm(np.quantile(value,.95,axis=0)-np.quantile(value,.05,axis=0)))
    foot_spans=[span(relative(side,'ankle')) for side in ['left','right']]
    wrist_spans=[span(relative(side,'wrist')) for side in ['left','right']]
    strides=[np.sum(relative(side,'ankle')*forward,axis=1) for side in ['left','right']]
    correlation=float(np.corrcoef(*strides)[0,1]) if min(np.std(x) for x in strides)>.005 else None
    foot=max(foot_spans);wrist=max(wrist_spans)
    if foot <= policy['action_stationary_foot_span_m']:
        label='standing' if wrist<=policy['action_stationary_wrist_span_m'] else 'gesturing' if wrist>=policy['action_gesture_wrist_span_m'] else 'standing_or_small_gesture'
    elif min(foot_spans)>=policy['action_gait_foot_span_m'] and correlation is not None and correlation<=policy['action_gait_opposition_correlation_max']:
        label='walking_or_running' if np.median(source_speed)>=policy['action_running_source_speed_threshold_mps'] else 'walking'
    else:label='unclassified_motion'
    return dict(label=label, method='frozen_rules_on_condition_skeleton_not_human_review',
                confidence='estimated', human_verified=False, foot_spans_model_m=foot_spans,
                wrist_spans_model_m=wrist_spans, stride_opposition_correlation=correlation,
                source_speed_alone_used_for_classification=False,
                animation_space='root_local_original_horizontal_path_removed',
                limitations='Occluded/wrong skeletons can mislabel actions. Running is not certified from speed alone.')


def vertical_sigma(quaternions, scales, up_in_asset):
    # Same non-normalized native quaternion polynomial as the rasterizer.
    w,x,y,z=np.moveaxis(np.asarray(quaternions,np.float64),-1,0)
    rotation=np.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                       2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                       2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],-1).reshape(quaternions.shape[:-1]+(3,3))
    projected=np.einsum('i,...ij->...j',up_in_asset,rotation)*scales
    return np.linalg.norm(projected,axis=-1)


def camera_validation(package, transform):
    cameras = read(package/'cameras.json')['cameras']
    centers = np.asarray([c['camtoworld'] for c in cameras])[:, :3, 3]
    _, _, v = np.linalg.svd(centers-centers.mean(0), full_matrices=False)
    normal = transform[:3, :3] @ v[-1]
    normal /= np.linalg.norm(normal)
    normal_error = float(np.degrees(np.arccos(np.clip(abs(normal[2]), -1, 1))))
    # Similarity transforms preserve projection if the same transform is applied
    # to the camera. Use affine inverse; normalized camera rotations are handled
    # by the rendering adapter, not by silently dropping the metric scale.
    probe = np.array([[0.,0.,2.], [.1,.2,2.5], [-.2,-.1,3.]])
    inverse_error = float(np.abs(transform_points(transform_points(probe, transform), np.linalg.inv(transform))-probe).max())
    projection_errors = []
    for c in cameras:
        c2w = np.asarray(c['camtoworld'])
        original = transform_points(probe, np.linalg.inv(c2w))
        changed = transform_points(transform_points(probe, transform), np.linalg.inv(transform @ c2w))
        projection_errors.append(float(np.max(np.abs(original-changed))))
    assert inverse_error < 1e-8 and max(projection_errors) < 1e-8 and normal_error < .01
    return dict(orbit_plane_up_error_degrees=normal_error, inverse_max_abs=inverse_error,
                camera_space_max_abs=max(projection_errors),
                scope='Transform algebra and camera ring only; not rendered grounding or real-world gravity')


def foot_pair_diagnostic(left, right, target_left, target_right):
    """Detect possible identity exchanges; never relabel points or certify contact.

    Thresholds are engineering diagnostics (5 cm advantage, 10 cm paired error),
    measured against the existing condition, not independent anatomical truth.
    """
    if left is None or right is None:
        return dict(state='unknown_missing_fixed_proxy', semantic_identity_verified=False)
    values = [np.asarray(v, dtype=np.float64) for v in
              (left, right, target_left, target_right)]
    if (values[0].ndim != 2 or values[0].shape[1] != 3 or not len(values[0])
            or any(v.shape != values[0].shape or not np.isfinite(v).all() for v in values)):
        raise ValueError('Foot pair diagnostics require matching finite F x 3 trajectories')
    left, right, target_left, target_right = values
    direct = (np.linalg.norm(left-target_left, axis=1)+np.linalg.norm(right-target_right, axis=1))/2
    swapped = (np.linalg.norm(left-target_right, axis=1)+np.linalg.norm(right-target_left, axis=1))/2
    suspect = (direct-swapped > .05) & (swapped < .10)
    return dict(state='possible_left_right_exchange' if suspect.any() else 'no_exchange_detected_by_condition_proxy',
                suspect_frames=np.flatnonzero(suspect).tolist(),
                direct_pair_error_p95_model_m=float(np.quantile(direct,.95)),
                best_assignment_error_p95_model_m=float(np.quantile(np.minimum(direct,swapped),.95)),
                improvement_threshold_model_m=.05, swapped_error_threshold_model_m=.10,
                threshold_basis='Engineering diagnostic, not a quality acceptance gate',
                semantic_identity_verified=False, automatic_relabeling=False,
                limitation='Condition-guided comparison; no detected swap does not certify semantic feet or contact')


def motion_review_route(record):
    """Separate difficult assets without confusing diagnostic support with approval."""
    if not record:
        return 'pending_motion_metadata'
    motion=record.get('motion',{})
    associations=motion.get('associations',{})
    required=('left-foot','right-foot','left-hip','right-hip','left-shoulder','right-shoulder')
    supported=all(associations.get(name,{}).get('numerically_supported') is True for name in required)
    pair=record.get('feet',{}).get('pair_assignment_diagnostic',{})
    if not supported or pair.get('state')=='possible_left_right_exchange':
        return 'separate_motion_review'
    if motion.get('gait',{}).get('native_speed_model_mps_estimate') is None:
        return 'contact_support_review'
    action=record.get('action',{}).get('label')
    if action in ('walking','walking_or_running','running'):
        return 'walking_validation_candidate'
    return 'other_motion_validation_candidate'


def inplace_foot_velocities(left, right, standard_to_inplace, times):
    """Velocities after the exact root/yaw removal used by planned placement."""
    feet=np.stack([left,right],axis=1).astype(np.float64)
    matrices=np.asarray(standard_to_inplace,np.float64)
    times=np.asarray(times,np.float64)
    if (feet.shape!=(len(times),2,3) or matrices.shape!=(len(times),4,4)
            or len(times)<2 or not np.all(np.diff(times)>0)
            or not np.isfinite(feet).all() or not np.isfinite(matrices).all()):
        raise ValueError('Invalid timed foot trajectories or root-removal transforms')
    local=np.einsum('fij,fsj->fsi',matrices[:,:3,:3],feet)+matrices[:,None,:3,3]
    return local,np.gradient(local,times,axis=0)


def source_contact_hints(feet, times, ankle_confidence, ankle_error, policy):
    """Condition-derived contact candidates, gated by observed ankle evidence.

    feet: F x 2 sides x 3 landmarks (heel/big toe/small toe) x XYZ.
    Ankle confidence is not a direct observation of the heel or sole.
    """
    feet=np.asarray(feet,np.float64);times=np.asarray(times,np.float64)
    confidence=np.asarray(ankle_confidence,np.float64);error=np.asarray(ankle_error,np.float64)
    if (feet.shape!=(len(times),2,3,3) or confidence.shape!=(len(times),2)
            or error.shape!=confidence.shape or len(times)<2 or not np.all(np.diff(times)>0)
            or not all(np.isfinite(v).all() for v in (feet,times,confidence,error))):
        raise ValueError('Invalid source foot/time/observation evidence')
    velocity=np.gradient(feet,times,axis=0)
    evidence=(confidence>=policy['source_hint_ankle_confidence_min']) & (error<=policy['source_hint_projection_error_height_max'])
    low=(np.abs(feet[:,:,:,2])<=policy['contact_height_model_m'])
    slow=(np.abs(velocity[:,:,:,2])<=policy['contact_vertical_speed_model_mps'])
    return dict(source_foot_landmarks=feet,source_foot_landmark_velocity=velocity,
                source_ankle_confidence=confidence,source_ankle_projection_error_height=error,
                source_support_evidence_valid=evidence,
                source_landmark_support_candidate=low & slow & evidence[:,:,None],
                source_support_candidate=(low & slow).any(2) & evidence)


def write_source_motion_hints(identity, destination):
    """Small CPU sidecar before diffusion; same operation supports old packets.

    Does not modify the frozen G0 pose, images, models, or acceptance decisions.
    """
    spec=case(identity);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    condition=Path(spec['conditioning']);metadata=read(condition/'metadata.json')
    camera_meta=read(condition/'cameras.json')
    assert camera_meta['world_frame']['units']=='meters' and camera_meta['world_frame']['axes']['y']=='up'
    names=metadata['keypoint_names'];count=spec['prepared_frames']
    retained=spec.get('temporal_subset',{}).get('retained_source_frames',list(range(count)))
    def selected(value):
        if len(value)==count:return value
        if len(retained)!=count or min(retained)<0 or max(retained)>=len(value):
            raise ValueError('Source hint time mapping unavailable')
        return value[np.asarray(retained,dtype=int)]
    policy=read(CODE_ROOT/'config/placement.json')
    sources={'keypoints':condition/'keypoints_3d.npy','condition_metadata':condition/'metadata.json',
             'condition_cameras':condition/'cameras.json','root_motion':Path(spec['root_motion'])}
    # Temporal subsets retain their original calibrated observation packet;
    # select its exact registered frames rather than requiring a duplicate copy.
    observation_spec=spec
    if 'temporal_subset' in spec:
        subset=spec['temporal_subset']
        observation_spec=read(Path(subset['source_root'])/'config/cases.json')[subset['source_identity']]
        assert len(retained)==count and all(0<=i<subset['source_frame_count'] for i in retained)
        assert all(b>a for a,b in zip(retained[:-1],retained[1:]))
    projection=Path(observation_spec['selection']['package'])/'clip0/motion_pair/calibrated_moving/projection.npz'
    pose_path=Path(spec['motion'])/'motion.safetensors'
    if projection.is_file():sources['observed_projection']=projection
    if pose_path.is_file():sources['pose_parameters']=pose_path
    signature=dict(source_sha256={key:sha(path) for key,path in sources.items()},
                   policy_sha256=sha(CODE_ROOT/'config/placement.json'),
                   implementation_sha256=sha(Path(__file__)),selected_condition_frames=retained,
                   temporal_policy_sha256=sha(CODE_ROOT/'config/temporal_policy.json'),
                   temporal_classifier_sha256=sha(CODE_ROOT/'frontend.py'))
    info=destination/'source_motion_hints.json';payload=destination/'source_motion_hints.npz'
    if info.exists() and payload.exists():
        old=read(info)
        if old.get('signature')==signature and old['array_sha256']==sha(payload):return info,payload
    kp=selected(np.load(sources['keypoints'],allow_pickle=False)) @ LOCAL_TO_STANDARD.T
    with np.load(sources['root_motion'],allow_pickle=False) as root:
        times=selected(root['frame_times']);removed=selected(root['local_to_canonical'])[:,:3,3] @ LOCAL_TO_STANDARD.T
    assert len(kp)==count and np.isfinite(kp).all()
    feet=np.stack([kp[:,[names.index(side+'-'+part) for part in ('heel','big-toe-tip','small-toe-tip')]]
                   for side in ('left','right')],axis=1)
    confidence=np.zeros((count,2));error=np.ones((count,2))
    source_audit=dict(state='unknown_missing_pose_or_projection',decision='unknown',
                      recompute_recommendation='Inspect missing evidence; do not automatically regenerate')
    if projection.is_file():
        with np.load(projection,allow_pickle=False) as observed:
            boxes=selected(observed['source_boxes']);height=boxes[:,3]-boxes[:,1]
            assert (height>0).all()
            observed_joints=selected(observed['observed_coco17'])
            distances=selected(observed['detector_distance_px'])/height[:,None]
            confidence=observed_joints[:,[15,16],2]
            error=distances[:,[15,16]]
        if pose_path.is_file():
            from safetensors.numpy import load
            from scipy.spatial.transform import Rotation
            from frontend import classify_temporal_signals
            tensors=load(pose_path.read_bytes())
            pose=selected(tensors['smpl_params_incam.body_pose'])
            np.testing.assert_allclose(selected(tensors['observed_keypoints_2d']),observed_joints,atol=1e-5,rtol=0)
            rotations=Rotation.from_rotvec(pose.reshape(-1,3)).as_matrix().reshape(count,21,3,3)
            steps=Rotation.from_matrix((rotations[1:]@rotations[:-1].swapaxes(-1,-2)).reshape(-1,3,3)).magnitude().reshape(count-1,21)
            steps=np.concatenate([np.zeros((1,21)),steps])
            groups=[([11,13,15],[0,3,6,9]),([12,14,16],[1,4,7,10]),
                    ([5,7,9],[12,15,17,19]),([6,8,10],[13,16,18,20])]
            result=classify_temporal_signals(
                np.stack([observed_joints[:,ids,2].mean(1) for ids,_ in groups],axis=1),
                np.stack([steps[:,ids].max(1) for _,ids in groups],axis=1),
                np.stack([np.quantile(distances[:,ids],.75,axis=1) for ids,_ in groups],axis=1),
                read(CODE_ROOT/'config/temporal_policy.json'))
            source_audit=dict(state='no_corroborated_anomaly_detected' if result['decision']=='keep_full' else 'source_skeleton_review_needed',
                decision=result['decision'],reason=result['reason'],bad_frames=result.get('bad_frames',[]),
                recompute_recommendation='Not indicated by these checks; missing export fields do not require refitting' if result['decision']=='keep_full'
                                       else 'Review existing bone overlay first; rerun only confirmed pose defects, not Gaussian-only errors',
                limitations='Weak confidence plus fitting anomaly screen only; confidently wrong/gradually drifting poses may escape. Not final Gaussian motion verification.')
    arrays=source_contact_hints(feet,times,confidence,error,policy)
    pelvis=(kp[:,names.index('left-hip')]+kp[:,names.index('right-hip')])/2
    lateral=(kp[:,names.index('left-hip')]-kp[:,names.index('right-hip')]
             +kp[:,names.index('left-shoulder')]-kp[:,names.index('right-shoulder')])
    forward=np.cross(lateral,[0.,0.,1.]);heading_valid=np.linalg.norm(forward[:,:2],axis=1)>=.05
    heading=np.unwrap(np.arctan2(forward[:,1],forward[:,0]))
    arrays.update(frame_timestamps=times,selected_condition_frames=np.asarray(retained),
                  source_pelvis_root_local=pelvis,source_body_heading_radians=heading,
                  source_body_heading_valid=heading_valid,
                  source_removed_horizontal_translation_estimate=removed,
                  source_removed_root_velocity_estimate=np.gradient(removed,times,axis=0))
    temporary=payload.with_suffix('.writing')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**arrays)
    temporary.replace(payload)
    report=dict(version=1,identity=identity,state='source_skeleton_hints_not_gaussian_motion',signature=signature,
                arrays=payload.name,array_sha256=sha(payload),coordinate_convention='right_handed_x_forward_y_left_z_up',
                units='fitted_body_model_meters_not_independent_measurement',frame_count=count,
                side_order=['left','right'],landmark_order=['heel','big-toe-tip','small-toe-tip'],
                ground_plane=[0,0,1,0],ground_basis='Frozen G0 fitted-body ground; not verified Gaussian sole',
                observation_evidence_available=projection.is_file(),direct_heel_toe_observation_available=False,
                source_skeleton_audit=source_audit,
                contact_verified=False,loopable=False,loop_interval=None,trajectory_ready=False,
                use='Initialization, phase proposals and source-versus-rendered diagnostics only; validate against final Gaussian renders',
                limitations='Ankle evidence does not certify a visible sole. Unknown/weak evidence gives invalid candidates, not evidence of no contact. Removed source root is not a verified world path. No automatic pose repair or production rejection.')
    save(info,report)
    with np.load(payload,allow_pickle=False) as check:
        assert all(np.isfinite(check[key]).all() for key in check.files)
        np.testing.assert_array_equal(check['frame_timestamps'],times)
    return info,payload


def backfill_source_hints(identity):
    package=ASSET_ROOT/identity
    with (ROOT/'jobs'/f'{identity}_placement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        complete=read(package/'package_complete.json')
        assert complete['state']=='complete_pending_visual_review'
        original_hash=sha(package/'asset_4dgs.npz')
        info,payload=write_source_motion_hints(identity,package/'refs')
        manifest=read(package/'manifest.json')
        provenance=manifest['provenance']
        provenance['source_refs']=sorted(set(provenance['source_refs'])|{info.name,payload.name})
        manifest['source_motion_hints']='refs/'+info.name
        save(package/'manifest.json',manifest)
        insertion=read(package/'insertion.json')
        insertion['source_motion_hints']='refs/'+info.name
        insertion['source_hints_are_not_final_gaussian_motion']=True
        save(package/'insertion.json',insertion)
        complete['refs_sha256'].update({path.name:sha(path) for path in (info,payload)})
        assert sha(package/'asset_4dgs.npz')==original_hash==complete['asset_sha256']
        save(package/'package_complete.json',complete)
        result=read(info)
        return dict(identity=identity,asset_sha256=original_hash,source_skeleton_audit=result['source_skeleton_audit'],
                    state='source_hints_backfilled_without_pose_or_asset_regeneration')


def enrich(identity):
    package = ASSET_ROOT / identity
    lock_dir = ROOT/'jobs'; lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir/f'{identity}_placement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _enrich(identity, package)


def contact_playback_curve(times, feet, support, policy):
    """Exploratory straight-path timing plus bounded lateral body motion.

    Contact hypotheses infer a native path. Its longitudinal distance becomes
    the playback clock, rather than pretending one median speed fits each phase.
    Unsupported gaps are explicitly marked as interpolated, never measured.
    """
    times=np.asarray(times,np.float64);feet=np.asarray(feet,np.float64)
    support=np.asarray(support,bool)
    if (len(times)<3 or feet.shape!=(len(times),2,3) or support.shape!=(len(times),2)
            or not np.isfinite(feet).all() or not np.all(np.diff(times)>0)):
        raise ValueError('Invalid contact playback inputs')
    velocity=np.gradient(feet,times,axis=0)
    count=support.sum(1);valid=count>0
    if valid.sum()<3:raise ValueError('Insufficient contact hypotheses')
    estimate=-(velocity[:,:,:2]*support[:,:,None]).sum(1)/np.maximum(count[:,None],1)
    missing=np.flatnonzero(~valid)
    runs=np.split(missing,np.flatnonzero(np.diff(missing)>1)+1)
    longest=max((float(times[r[-1]]-times[r[0]]+np.median(np.diff(times))) for r in runs if len(r)),default=0.)
    if longest>policy['maximum_interpolated_gap_seconds']:
        raise ValueError('Contact gaps exceed frozen interpolation limit')
    filled=np.stack([np.interp(times,times[valid],estimate[valid,axis]) for axis in (0,1)],axis=1)
    travel=np.concatenate([np.zeros((1,2)),np.cumsum((filled[1:]+filled[:-1])*.5*np.diff(times)[:,None],axis=0)])
    angle=-float(np.arctan2(travel[-1,1],travel[-1,0]))
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    aligned=travel@rotation.T
    if not np.all(np.diff(aligned[:,0])>1e-6):raise ValueError('Native forward travel is not strictly increasing')
    speed=float(aligned[-1,0]/(times[-1]-times[0]))
    target=speed*(times-times[0]);sample_times=np.interp(target,aligned[:,0],times)
    sampled_lateral=np.interp(sample_times,times,aligned[:,1])
    rate=np.gradient(sample_times,times)
    limits=dict(lateral_excursion_model_m=float(np.max(np.abs(sampled_lateral))),
                heading_alignment_degrees=float(np.degrees(angle)),
                local_time_scale_min=float(rate.min()),local_time_scale_max=float(rate.max()),
                interpolated_frames=int((~valid).sum()),maximum_gap_seconds=longest)
    bounded=(limits['lateral_excursion_model_m']<=policy['maximum_lateral_excursion_model_m']
             and abs(limits['heading_alignment_degrees'])<=policy['maximum_heading_alignment_degrees']
             and rate.max()<=policy['maximum_local_time_scale'] and rate.min()>=1/policy['maximum_local_time_scale'])
    return dict(sample_times=sample_times,sample_frames=np.interp(sample_times,times,np.arange(len(times))),
                root_xy=np.stack([target,sampled_lateral],axis=1),native_speed=speed,alignment_yaw=angle,
                native_contact_path=travel,interpolated_velocity_frames=~valid,limits=limits,bounded=bool(bounded))


def build_placement_trial(identity, mode):
    """Write candidate transforms under work/reports only, never modify assets."""
    package=ASSET_ROOT/identity;meta=read(package/'placement.json')
    assert mode in ('standing','walking')
    assert all(meta['motion']['associations'][side+'-foot']['numerically_supported'] for side in ('left','right'))
    policy=read(CODE_ROOT/'config/placement_trial.json')
    with np.load(package/'placement_motion.npz',allow_pickle=False) as z:values={k:z[k] for k in z.files}
    times=values['frame_timestamps'];F=len(times)
    heights=np.stack([values[s+'_foot_2sigma_support_height'] for s in ('left','right')],axis=1)
    offset=-float(np.median(heights.min(1)))
    if mode=='walking':
        curve=contact_playback_curve(times,values['gaussian_inplace_foot_positions'],values['support_foot_candidate'],policy)
    else:
        curve=dict(sample_times=times,sample_frames=np.arange(F,dtype=float),root_xy=np.zeros((F,2)),
                   native_speed=0.,alignment_yaw=0.,native_contact_path=np.zeros((F,2)),
                   interpolated_velocity_frames=np.zeros(F,bool),limits={},bounded=True)
    rotation=np.eye(4);a=curve['alignment_yaw'];rotation[:2,:2]=[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]
    root=values['gaussian_root_position'];yaw=values['gaussian_heading_radians']
    def matrix_at(query):
        h=np.interp(query,times,yaw);r=np.array([[np.cos(h),np.sin(h),0],[-np.sin(h),np.cos(h),0],[0,0,1]])
        p=np.array([np.interp(query,times,root[:,0]),np.interp(query,times,root[:,1]),0.])
        result=np.eye(4);result[:3,:3]=r;result[:3,3]=-r@p
        return result
    baseline=[];candidate=[]
    for i,t in enumerate(times):
        before=rotation@matrix_at(t);before[:2,3]+=np.array([curve['native_speed']*(t-times[0]),0.])
        after=rotation@matrix_at(curve['sample_times'][i]);after[:2,3]+=curve['root_xy'][i];after[2,3]+=offset
        baseline.append(before);candidate.append(after)
    destination=ROOT/'reports/placement_trials'/identity;destination.mkdir(parents=True,exist_ok=True)
    payload=destination/'candidate.npz'
    np.savez_compressed(payload,times=times,baseline_frames=np.arange(F,dtype=float),candidate_frames=curve['sample_frames'],
                        baseline_world_from_standard=np.stack(baseline),candidate_world_from_standard=np.stack(candidate),
                        native_contact_path=curve['native_contact_path'],interpolated_velocity_frames=curve['interpolated_velocity_frames'])
    result=dict(identity=identity,mode=mode,state='experimental_candidate_not_production_accepted',
                asset_sha256=sha(package/'asset_4dgs.npz'),placement_sha256=sha(package/'placement.json'),
                candidate_sha256=sha(payload),policy_sha256=sha(CODE_ROOT/'config/placement_trial.json'),
                ground_offset_model_m=offset,native_forward_speed_model_mps=curve['native_speed'],
                limits=curve['limits'],bounded=curve['bounded'],loopable=False,
                ground_basis='Single constant offset from median lower fixed-foot finite 2-sigma envelope; not per-frame foot flattening',
                limitation='Contact proxies fit this candidate; cancellation on those same points is not independent anti-skating evidence. Source assets remain untouched.')
    save(destination/'candidate.json',result)
    return result


def _enrich(identity, package):
    begin = time.time()
    policy_path = CODE_ROOT/'config/placement.json'
    policy = read(policy_path)
    spec = case(identity)
    conditioning = Path(spec['conditioning'])
    metadata = read(conditioning/'metadata.json')
    condition_cameras = read(conditioning/'cameras.json')
    assert condition_cameras['world_frame']['axes']['y'] == 'up'
    assert condition_cameras['world_frame']['units'] == 'meters'
    assert 'root_local' in condition_cameras['world_frame']['name']
    replay = read(package/'replay_validation.json')
    assert replay['state'] == 'NPZ_integer_frame_replay_passed'
    payload_hash = sha(package/'asset_4dgs.npz')
    with np.load(package/'insertion_transforms.npz', allow_pickle=False) as z:
        asset_to_local = z['asset_camera_to_root_local']
    transform, scale, tilt = standard_transform(asset_to_local)
    proof = camera_validation(package, transform)
    with np.load(package/'asset_4dgs.npz', allow_pickle=False) as z:
        frames = int(z['frame_count'][0])
        times = z['frame_timestamps'].copy()
        xyz = z['frame_xyz']
        opacity = z['opacity'].reshape(-1)
        canonical = z['canonical_xyz']
        quaternions = z['frame_rotation']
        scales = z['frame_scaling']
    assert xyz.shape[0] == frames and np.all(np.diff(times) > 0)
    # Float32 motion data limit memory while keeping matrix validation float64.
    standard = transform_points(xyz, transform).astype(np.float32)
    del xyz
    kp = np.load(conditioning/'keypoints_3d.npy', allow_pickle=False)
    if len(kp) != frames:
        retained = spec.get('temporal_subset', {}).get('retained_source_frames')
        if retained is None or len(retained) != frames:
            raise ValueError('Missing exact source-frame map for condition keypoints')
        kp = kp[np.asarray(retained, dtype=int)]
    kp = kp @ LOCAL_TO_STANDARD.T
    names = metadata['keypoint_names']
    groups = {n:[n] for n in ['left-hip','right-hip','left-shoulder','right-shoulder']}
    groups.update({side+'-foot':[side+'-heel',side+'-big-toe-tip',side+'-small-toe-tip']
                   for side in ['left','right']})
    arrays = dict(frame_timestamps=times, asset_to_standard_model_meters=transform,
                  standard_model_meters_to_asset=np.linalg.inv(transform),
                  condition_keypoints_standard_model_m=kp.astype(np.float32))
    associations, trajectories = {}, {}
    for name, joints in groups.items():
        target = kp[:, [names.index(n) for n in joints]].mean(1)
        trajectory, result, ids, weights = bind_proxy(standard, target, opacity, policy)
        associations[name] = result
        arrays[name+'_gaussian_indices'] = ids
        if trajectory is not None:
            trajectories[name] = trajectory
            arrays[name+'_gaussian_weights'] = weights
            arrays[name+'_proxy_position'] = trajectory
    root_names = ['left-hip','right-hip','left-shoulder','right-shoulder']
    root_status = 'unknown_insufficient_or_unreliable_support'
    if all(associations[n].get('numerically_supported') for n in root_names):
        try:
            arrays.update(root_motion(*(trajectories[n] for n in root_names), times))
            root_status = 'gaussian_bound_geometric_proxy_pending_semantic_review'
        except ValueError:
            root_status = 'unknown_degenerate_body_heading'
    with np.load(package/'refs/root_motion.npz', allow_pickle=False) as source:
        source_motion = source['local_to_canonical'][:, :3, 3] @ LOCAL_TO_STANDARD.T
        assert len(source_motion) == frames
    arrays['source_removed_horizontal_root_estimate'] = source_motion
    arrays['source_removed_root_speed_estimate'] = np.linalg.norm(np.gradient(source_motion, times, axis=0)[:, :2], axis=1)
    contacts = {}
    for side in ['left','right']:
        name = side+'-foot'
        if not associations[name].get('numerically_supported'):
            contacts[side] = dict(state='unknown', reason='unreliable_gaussian_association')
            continue
        ids = arrays[name+'_gaussian_indices']
        # Centers only: explicit diagnostic, never asserted to be a sole surface.
        heights = np.quantile(standard[:, ids, 2], .1, axis=1)
        sigma=vertical_sigma(quaternions[:,ids],scales[:,ids],transform[2,:3])
        support_height=np.quantile(standard[:,ids,2]-policy['foot_support_sigma']*sigma,.1,axis=1)
        velocity = np.gradient(trajectories[name], times, axis=0)
        contact = (np.abs(heights) <= policy['contact_height_model_m']) & (np.abs(velocity[:,2]) <= policy['contact_vertical_speed_model_mps'])
        speed = np.linalg.norm(velocity[:,:2], axis=1)
        arrays[side+'_foot_center_low_height'] = heights
        arrays[side+'_foot_contact_candidate'] = contact
        arrays[side+'_foot_velocity'] = velocity
        arrays[side+'_foot_2sigma_support_height'] = support_height
        contacts[side] = dict(state='center_proxy_only_not_validated_contact',
            candidate_frames=int(contact.sum()),
            low_center_height_min_model_m=float(heights.min()),
            low_center_height_max_model_m=float(heights.max()),
            below_ground_center_frames=int((heights < -policy['penetration_tolerance_model_m']).sum()),
            above_ground_center_frames=int((heights > policy['contact_height_model_m']).sum()),
            candidate_contact_horizontal_speed_p95_model_mps=float(np.quantile(speed[contact],.95)) if contact.any() else None,
            support_2sigma_min_model_m=float(support_height.min()),support_2sigma_max_model_m=float(support_height.max()),
            support_2sigma_below_tolerance_frames=int((support_height < -policy['penetration_tolerance_model_m']).sum()),
            support_2sigma_above_tolerance_frames=int((support_height > policy['contact_height_model_m']).sum()))
    gait=dict(state='unknown_insufficient_contact_support',retiming_ready=False)
    if 'gaussian_heading_radians' in arrays and all(side+'_foot_velocity' in arrays for side in ['left','right']):
        # Estimate the rigid XY translation that would cancel support-foot motion.
        # Do not bake it into the asset or confuse it with a measured world path.
        inplace_feet,inplace_velocity=inplace_foot_velocities(
            trajectories['left-foot'],trajectories['right-foot'],
            arrays['standard_to_gaussian_inplace'],times)
        arrays['gaussian_inplace_foot_positions']=inplace_feet
        arrays['gaussian_inplace_foot_velocities']=inplace_velocity
        velocities=inplace_velocity[:,:,:2]
        height=np.stack([arrays[side+'_foot_2sigma_support_height'] for side in ['left','right']],axis=1)
        vz=np.stack([arrays[side+'_foot_velocity'][:,2] for side in ['left','right']],axis=1)
        support=(np.abs(height)<=policy['contact_height_model_m']) & (np.abs(vz)<=policy['contact_vertical_speed_model_mps'])
        count=support.sum(1);valid=count>0
        estimate=-(velocities*support[:,:,None]).sum(1)/np.maximum(count[:,None],1)
        slip=np.linalg.norm(velocities+estimate[:,None],axis=-1)
        arrays['support_translation_velocity_candidate']=estimate
        arrays['support_translation_velocity_valid']=valid
        arrays['support_foot_candidate']=support
        arrays['support_slip_residual_model_mps']=slip
        enough=int(valid.sum())>=policy['gait_speed_min_support_frames']
        gait=dict(state='contact_based_speed_estimate_pending_visual_validation' if enough else 'unknown_insufficient_contact_support',
                  support_frames=int(valid.sum()),total_frames=frames,
                  native_speed_model_mps_estimate=float(np.median(np.linalg.norm(estimate[valid],axis=1))) if enough else None,
                  supported_foot_residual_speed_p95_model_mps=float(np.quantile(slip[support],.95)) if support.any() else None,
                  retiming_ready=False,
                  candidate_velocity_frame='Gaussian inplace X-forward/Y-left after the exported root AND heading removal',
                  planned_world_velocity_rule='Rotate candidate XY velocity by the planned world yaw; scaled playback needs the same time derivative multiplier',
                  method='Negative Gaussian support-foot velocity AFTER root/yaw removal; no integration through unsupported frames',
                  limitation='Contact inferred from finite ellipsoid envelope, not observed force/contact. A single support foot makes residual zero by construction; this is not anti-skating validation.')
    active = opacity >= policy['opacity_min']
    assert active.any()
    velocity0 = (standard[1,active]-standard[0,active])/(times[1]-times[0])
    velocity1 = (standard[-1,active]-standard[-2,active])/(times[-1]-times[-2])
    seam = dict(position_p95_model_m=float(np.quantile(np.linalg.norm(standard[-1,active]-standard[0,active],axis=1),.95)),
                velocity_jump_p95_model_mps=float(np.quantile(np.linalg.norm(velocity1-velocity0,axis=1),.95)))
    seam['center_numeric_candidate'] = seam['position_p95_model_m'] <= policy['loop_position_p95_model_m'] and seam['velocity_jump_p95_model_mps'] <= policy['loop_velocity_p95_model_mps']
    std_canonical = transform_points(canonical,transform)
    bounds = np.quantile(std_canonical[active], [.01,.99],axis=0)
    output = dict(version=1, identity=identity, state='coordinate_calibrated_motion_diagnostics_pending_review',
        arrays='placement_motion.npz', array_sha256=None,
        asset_sha256=payload_hash, policy_sha256=sha(policy_path), code_sha256=sha(Path(__file__)),
        provenance=dict(condition_keypoints_sha256=sha(conditioning/'keypoints_3d.npy'),
            condition_metadata_sha256=sha(conditioning/'metadata.json'),
            condition_cameras_sha256=sha(conditioning/'cameras.json'),
            insertion_transforms_sha256=sha(package/'insertion_transforms.npz'),
            source_root_sha256=sha(package/'refs/root_motion.npz'),
            selected_condition_frames=spec.get('temporal_subset',{}).get('retained_source_frames',list(range(frames))),
            keypoint_names=names),
        coordinates=dict(convention='right_handed_x_forward_y_left_z_up',
            asset_to_standard_model_meters=transform.tolist(),
            standard_model_meters_to_asset=np.linalg.inv(transform).tolist(),
            ground_plane=[0,0,1,0], ground_basis='Original fitted SMPL-X sequence minimum vertex height; Gaussian sole not calibrated',
            forward_basis='Generator initial anatomical front +Z converted to +X; per-frame Gaussian body-heading proxy stored separately',
            previous_catalogue_up_error_degrees=tilt, validation=proof,
            consumer_rule='Apply this similarity once to asset frame_xyz; transform covariance and full SH consistently. Do not also apply old UP_FIX. Native arrays require asset_from_native first.'),
        scale=dict(asset_unit_to_model_meter=scale, estimated_gaussian_height_model_m=float(bounds[1,2]-bounds[0,2]),
            basis='Inverse saved Grid4D normalization into frozen GVHMR/SMPL-X conditioning meters',
            personal_height_measured=False, real_world_metric_verified=False,
            calibration_status='body_model_scale_estimate_not_independent_measurement'),
        motion=dict(root_motion_state=root_status, associations=associations,
            source_horizontal_root_removed_before_generation=True,
            residual_gaussian_translation_and_heading_baked=True,
            source_removed_root_is_unverified_estimate=True,
            heading_units='radians_counterclockwise_about_positive_Z',
            inplace_transform='standard_to_gaussian_inplace removes measured Gaussian horizontal root AND heading, preserves vertical body motion; inverse is stored',
            planned_transform_order='world_from_planned_pose(t) @ standard_to_gaussian_inplace(frame) @ asset_to_standard_model_meters; native arrays additionally require asset_from_native. This is spatial alignment, not gait synthesis.',
            fps=float(1/np.median(np.diff(times))),
            arbitrary_retargeting=False, validated_anti_skating=False,
            planned_trajectory_ready=False, gait=gait),
        action=action_diagnostics(kp,names,times,arrays['source_removed_root_speed_estimate'],policy),
        feet=dict(contact_verified=False, ground_verified=False, diagnostics=contacts,
            pair_assignment_diagnostic=foot_pair_diagnostic(
                trajectories.get('left-foot'), trajectories.get('right-foot'),
                kp[:,[names.index(n) for n in groups['left-foot']]].mean(1),
                kp[:,[names.index(n) for n in groups['right-foot']]].mean(1)),
            limitations='Fixed Gaussian-center proxies only. Low centers do not establish opaque sole contact, support phase or perceptual absence of sliding.'),
        playback=dict(loopable=False, loop_interval=None, loop_seam_verified=False,
            policy='Play once; do not silently repeat or extrapolate. No cycle selected.', diagnostic_full_clip_seam=seam),
        validation_scope='Coordinate/scale provenance and fixed Gaussian bindings; semantic motion, soles and loop visuals need independent review',
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),elapsed_seconds=time.time()-begin)
    output['motion_review_group']=motion_review_route(output)
    target=package/'placement_motion.npz'; temp=target.with_suffix('.writing')
    with temp.open('wb') as stream:
        np.savez_compressed(stream,**arrays)
    temp.replace(target); output['array_sha256']=sha(target)
    assert sha(package/'asset_4dgs.npz') == payload_hash
    save(package/'placement.json',output)
    for name in ['manifest.json','insertion.json','motion_metadata.json']:
        record=read(package/name)
        record['placement_metadata']='placement.json'
        record['placement_motion']='placement_motion.npz'
        save(package/name,record)
    print(identity, output['state'], 'tilt_deg',round(tilt,6), 'scale',round(scale,6),flush=True)
    validate_package(identity)
    return output


def foot_correspondence_probe(identity):
    """Compare fixed bindings with per-frame surface neighborhoods, read-only.

    Dynamic neighborhoods follow a source-conditioned region, so lower fitting
    error alone is not independent evidence of semantic tracking or foot contact.
    """
    from scipy.spatial import cKDTree
    package=ASSET_ROOT/identity
    metadata=read(package/'placement.json')
    with np.load(package/'placement_motion.npz',allow_pickle=False) as z:
        values={k:z[k] for k in z.files}
    with np.load(package/'asset_4dgs.npz',allow_pickle=False) as z:
        points=transform_points(z['frame_xyz'],values['asset_to_standard_model_meters'])
        opacity=z['opacity'].ravel()
    spec=case(identity)
    names=read(Path(spec['conditioning'])/'metadata.json')['keypoint_names']
    kp=values['condition_keypoints_standard_model_m']
    valid_ids=np.flatnonzero(opacity>=.5)
    assert len(valid_ids)>=32
    results={};curves={}
    trees=[cKDTree(p[valid_ids]) for p in points]
    for side in ['left','right']:
        target=kp[:,[names.index(side+'-'+joint) for joint in ['heel','big-toe-tip','small-toe-tip']]].mean(1)
        near=np.stack([tree.query(t,k=32)[1] for tree,t in zip(trees,target)])
        ids=valid_ids[near]
        candidates=points[np.arange(len(points))[:,None],ids]
        distance=np.linalg.norm(candidates-target[:,None],axis=-1)
        weights=opacity[ids]/np.maximum(distance,.01);weights/=weights.sum(1,keepdims=True)
        moving=np.sum(candidates*weights[:,:,None],axis=1)
        error=np.linalg.norm(moving-target,axis=-1)
        fixed=values.get(side+'-foot_proxy_position')
        fixed_error=np.linalg.norm(fixed-target,axis=-1) if fixed is not None else None
        overlap=np.array([len(set(a)&set(b))/32 for a,b in zip(ids[:-1],ids[1:])])
        frame_fail=np.flatnonzero(fixed_error>.15) if fixed_error is not None else np.arange(len(points))
        results[side]=dict(fixed_p95_model_m=float(np.quantile(fixed_error,.95)) if fixed_error is not None else None,
             dynamic_neighborhood_p95_model_m=float(np.quantile(error,.95)),
             dynamic_neighbor_distance_p95_model_m=float(np.quantile(distance,.95)),
             adjacent_point_id_overlap_mean=float(overlap.mean()),
             fixed_error_over_15cm_frames=frame_fail.tolist(),
             warning='Dynamic query follows source condition; low error is circular as tracking validation. Must check rendered foot evidence and cross-view/time consistency.')
        curves.update({side+'_target':target,side+'_dynamic_proxy':moving,
                       side+'_dynamic_indices':ids,side+'_dynamic_weights':weights,side+'_error':error})
    root=ROOT/'reports/foot_correspondence';root.mkdir(exist_ok=True)
    np.savez_compressed(root/f'{identity}.npz',**curves)
    result=dict(identity=identity,asset_sha256=sha(package/'asset_4dgs.npz'),
          state='diagnostic_only_not_a_repaired_asset',feet=results,
          source_condition_role='Spatial search seed, not independent ground truth',
          tracking_validated=False,anti_skating_validated=False,
          pair_assignment_diagnostic=foot_pair_diagnostic(
              values.get('left-foot_proxy_position'),values.get('right-foot_proxy_position'),
              curves['left_target'],curves['right_target']))
    save(root/f'{identity}.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def track_rendered_feet(identity, view, checkpoint):
    """CoTracker reads final rendered pixels; pose is used only to seed queries."""
    import av
    import torch
    from common import gpu_identity
    device=gpu_identity()
    assert view in (0,8)
    package=ASSET_ROOT/identity
    checkpoint=Path(checkpoint)
    assert checkpoint.is_file()
    expected='2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834'
    assert sha(checkpoint)==expected
    source=CODE_ROOT/'vendor/cotracker'
    pin=read(source/'VENDORED_SOURCE.json')
    assert all(sha(source/name)==digest for name,digest in pin['files'].items())
    sys.path.insert(0,str(source))
    from cotracker.predictor import CoTrackerPredictor
    with np.load(package/'placement_motion.npz',allow_pickle=False) as z:
        kp=z['condition_keypoints_standard_model_m'];T=z['standard_model_meters_to_asset']
    names=read(Path(case(identity)['conditioning'])/'metadata.json')['keypoint_names']
    landmark_names=[side+'-'+joint for side in ['left','right'] for joint in ['heel','big-toe-tip','ankle']]
    xyz=transform_points(kp[:,[names.index(n) for n in landmark_names]],T)
    c=next(c for c in read(package/'cameras.json')['cameras'] if c['view']==view)
    camera_points=transform_points(xyz,np.linalg.inv(np.asarray(c['camtoworld'])))
    projected=camera_points@np.asarray(c['K']).T
    projected=projected[...,:2]/projected[...,2:3]
    frames=[]
    with av.open(str(package/'demo.mp4')) as container:
        for frame in container.decode(video=0):
            image=frame.to_ndarray(format='rgb24')
            assert image.shape==(672,1760,3),image.shape
            column=3 if view==0 else 4
            frames.append(image[32:672,column*352:(column+1)*352].copy())
    assert len(frames)==len(kp)
    F=len(frames);seed_frames=[0,F//2]
    queries=np.concatenate([np.column_stack([np.full(len(landmark_names),f),projected[f]]) for f in seed_frames]).astype(np.float32)
    assert ((queries[:,1]>=0)&(queries[:,1]<352)&(queries[:,2]>=0)&(queries[:,2]<640)).all()
    begin=time.monotonic()
    model=CoTrackerPredictor(checkpoint=str(checkpoint),offline=True,v2=False).cuda().eval()
    video=torch.from_numpy(np.stack(frames)).permute(0,3,1,2)[None].float().cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        tracks,visible=model(video,queries=torch.as_tensor(queries[None],device='cuda'),backward_tracking=True)
        # Independent reverse pass from predicted endpoints, not source poses.
        reverse_queries=torch.cat([torch.zeros_like(tracks[:,-1,:,:1]),tracks[:,-1]],dim=-1)
        reverse,reverse_visible=model(video.flip(1),queries=reverse_queries,backward_tracking=False)
    tracks=tracks[0].cpu().numpy();visible=visible[0].cpu().numpy()
    reverse=reverse[0].flip(0).cpu().numpy();reverse_visible=reverse_visible[0].flip(0).cpu().numpy()
    cycle=np.linalg.norm(tracks-reverse,axis=-1)
    reference=np.tile(projected,(1,len(seed_frames),1))
    reference_error=np.linalg.norm(tracks-reference,axis=-1)
    valid=visible & reverse_visible
    # Upstream forces query visibility and exact query coordinates; exclude those
    # observations and the reverse initialization from any agreement statistic.
    for j,q in enumerate(queries):valid[int(q[0]),j]=False
    valid[-1]=False
    root=ROOT/'reports/foot_tracking';root.mkdir(exist_ok=True)
    output=root/f'{identity}_v{view:02d}.npz'
    np.savez_compressed(output,tracks=tracks,visible=visible,reverse_tracks=reverse,
        reverse_visible=reverse_visible,cycle_error_px=cycle,comparison_valid=valid,
        condition_projection=reference,queries=queries,frame_timestamps=np.arange(F)/25)
    report=dict(identity=identity,view=view,state='tracking_diagnostic_not_contact_validation',
        model='CoTracker3 scaled_offline',source_commit=pin['commit'],checkpoint_sha256=expected,
        demo_sha256=sha(package/'demo.mp4'),array_sha256=sha(output),frames=F,
        landmark_names=landmark_names,seed_frames=seed_frames,
        source_condition_used_only_for_initial_queries=True,
        input='Final asset rendered columns of the existing demo; original observations and generated reference columns are not read by model',
        visible_fraction=float(visible.mean()),valid_fraction=float(valid.mean()),
        forward_reverse_cycle_p50_px=float(np.median(cycle[valid])) if valid.any() else None,
        forward_reverse_cycle_p95_px=float(np.quantile(cycle[valid],.95)) if valid.any() else None,
        projected_condition_disagreement_p95_px=float(np.quantile(reference_error[valid],.95)) if valid.any() else None,
        explicit_limits=['Query points are estimated anatomical projections and may start occluded or away from visible surface.',
          'Cycle consistency is a self-consistency check, not independent ground truth.',
          'Visibility and forward/backward agreement do not prove persistent left/right identity.',
          'No 3D triangulation, contact or speed accepted by this diagnostic.'],
        elapsed_seconds=time.monotonic()-begin,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,device=device)
    save(root/f'{identity}_v{view:02d}.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


def validate_package(identity):
    """Independent read-back from exported files; no source poses or model needed."""
    package=ASSET_ROOT/identity
    record=read(package/'placement.json')
    assert record['asset_sha256']==sha(package/'asset_4dgs.npz')
    assert record['array_sha256']==sha(package/'placement_motion.npz')
    with np.load(package/'placement_motion.npz',allow_pickle=False) as z:
        arrays={k:z[k] for k in z.files}
    assert all(np.isfinite(value).all() for value in arrays.values())
    T=np.asarray(record['coordinates']['asset_to_standard_model_meters'])
    np.testing.assert_allclose(arrays['asset_to_standard_model_meters'],T)
    np.testing.assert_allclose(T@arrays['standard_model_meters_to_asset'],np.eye(4),atol=1e-8)
    with np.load(package/'asset_4dgs.npz',allow_pickle=False) as z:
        points=transform_points(z['frame_xyz'],T).astype(np.float32)
        np.testing.assert_array_equal(arrays['frame_timestamps'],z['frame_timestamps'])
    errors={}
    for name in record['motion']['associations']:
        if name+'_proxy_position' not in arrays:continue
        proxy=np.einsum('fnc,n->fc',points[:,arrays[name+'_gaussian_indices']],arrays[name+'_gaussian_weights'])
        errors[name]=float(np.abs(proxy-arrays[name+'_proxy_position']).max())
        assert errors[name]<1e-6
    root_error=None
    if 'standard_to_gaussian_inplace' in arrays:
        M=arrays['standard_to_gaussian_inplace'];inverse=arrays['gaussian_inplace_to_standard']
        np.testing.assert_allclose(M@inverse,np.broadcast_to(np.eye(4),M.shape),atol=1e-8)
        sample=points[:,::max(1,points.shape[1]//100)]
        inplace=np.einsum('fij,fnj->fni',M[:,:3,:3],sample)+M[:,None,:3,3]
        restored=np.einsum('fij,fnj->fni',inverse[:,:3,:3],inplace)+inverse[:,None,:3,3]
        root_error=float(np.abs(restored-sample).max());assert root_error<1e-8
    report=dict(state='exported_coordinate_and_binding_readback_passed',asset_sha256=record['asset_sha256'],
         placement_sha256=sha(package/'placement.json'),binding_max_errors=errors,
         inplace_roundtrip_error=root_error,
         scope='Independent numeric provenance/binding readback; no semantic contact, world-scale or loop certification')
    save(package/'placement_validation.json',report)
    return report


def validate_output_contract(package):
    """Require current coordinate/motion evidence; unknown gait remains explicit."""
    package=Path(package)
    record=read(package/'placement.json')
    digest=sha(package/'placement.json')
    assert record['asset_sha256']==sha(package/'asset_4dgs.npz')
    assert record['array_sha256']==sha(package/'placement_motion.npz')
    assert record['coordinates']['convention']=='right_handed_x_forward_y_left_z_up'
    assert record['scale']['asset_unit_to_model_meter']>0
    assert record['motion_review_group']==motion_review_route(record)
    assert isinstance(record['motion']['planned_trajectory_ready'],bool)
    assert isinstance(record['feet']['contact_verified'],bool)
    assert isinstance(record['feet']['ground_verified'],bool)
    assert isinstance(record['playback']['loopable'],bool)
    assert record['action']['confidence'] in ('estimated','verified')
    for name,state in [('placement_validation.json','exported_coordinate_and_binding_readback_passed'),
                       ('placement_replay.json','native_camera_placement_render_passed')]:
        evidence=read(package/name)
        assert evidence['state']==state and evidence['placement_sha256']==digest, name
        assert evidence['asset_sha256']==record['asset_sha256'], name
    hints=validate_source_hint_package(package) if (package/'refs/source_motion_hints.json').exists() else None
    practical=None
    if 'practical_playback' in read(package/'insertion.json'):
        from playback import validate_estimate
        practical=validate_estimate(package)
    return dict(state='placement_output_contract_passed',placement_sha256=digest,
                motion_review_group=record['motion_review_group'],
                practical_playback=practical,
                source_motion_hints=hints,
                planned_trajectory_ready=record['motion']['planned_trajectory_ready'])


def validate_source_hint_package(package):
    """Read final package only; prove source-hint frame and coordinate alignment."""
    package=Path(package);info=read(package/'refs/source_motion_hints.json')
    payload=package/'refs/source_motion_hints.npz'
    assert info['array_sha256']==sha(payload)
    assert info['signature']['source_sha256']['root_motion']==sha(package/'refs/root_motion.npz')
    assert info['signature']['source_sha256']['pose_parameters']==sha(package/'refs/motion.safetensors')
    metadata=read(package/'placement.json');names=metadata['provenance']['keypoint_names']
    with np.load(payload,allow_pickle=False) as source,np.load(package/'placement_motion.npz',allow_pickle=False) as final:
        assert all(np.isfinite(source[key]).all() for key in source.files)
        assert len(source['frame_timestamps'])==info['frame_count']==len(final['frame_timestamps'])
        np.testing.assert_allclose(source['frame_timestamps']-source['frame_timestamps'][0],
                                   final['frame_timestamps']-final['frame_timestamps'][0],atol=1e-8,rtol=0)
        np.testing.assert_array_equal(source['selected_condition_frames'],metadata['provenance']['selected_condition_frames'])
        kp=final['condition_keypoints_standard_model_m']
        feet=np.stack([kp[:,[names.index(side+'-'+part) for part in ('heel','big-toe-tip','small-toe-tip')]]
                       for side in ('left','right')],axis=1)
        error=float(np.max(np.abs(feet-source['source_foot_landmarks'])))
        assert error<5e-6
        np.testing.assert_allclose(source['source_removed_horizontal_translation_estimate'],
                                   final['source_removed_horizontal_root_estimate'],atol=1e-6,rtol=0)
        offset=float(final['frame_timestamps'][0]-source['frame_timestamps'][0])
    return dict(state='source_hint_time_coordinate_readback_passed',metadata_sha256=sha(package/'refs/source_motion_hints.json'),
                frame_count=info['frame_count'],source_to_asset_time_offset_seconds=offset,foot_coordinate_max_error=error,
                final_gaussian_foot_identity_verified=False)


async def batch_enrich(identities, concurrency):
    semaphore=asyncio.Semaphore(concurrency)
    async def one(identity):
        async with semaphore:
            process=await asyncio.create_subprocess_exec(sys.executable, str(Path(__file__).resolve()), identity)
            code=await process.wait()
            return dict(identity=identity,exit_code=code)
    results=await asyncio.gather(*(one(i) for i in identities))
    save(ROOT/'reports/placement_backfill.json', dict(results=results,
         state='coordinate_backfill_complete_pending_motion_acceptance' if all(r['exit_code']==0 for r in results) else 'needs_repair',
         code_sha256=sha(Path(__file__)), finished_at=time.time()))
    if any(r['exit_code'] for r in results):
        raise RuntimeError('Some placement backfills failed; see placement_backfill.json')


async def verify_batch(identities, gpus):
    from common import worker
    available=asyncio.Queue()
    for gpu in gpus:
        available.put_nowait(gpu)
    async def one(identity):
        gpu=await available.get()
        try:
            await worker('assets',['placement-check',identity],gpu,log_name=f'{identity}_placement_check.log')
            return dict(identity=identity,state='passed',gpu=gpu)
        except Exception as error:
            return dict(identity=identity,state='failed',error=str(error),gpu=gpu)
        finally:
            available.put_nowait(gpu)
    results=await asyncio.gather(*(one(i) for i in identities))
    save(ROOT/'reports/placement_render_batch.json',dict(results=results,finished_at=time.time(),
         state='passed' if all(r['state']=='passed' for r in results) else 'needs_repair'))
    if any(r['state']!='passed' for r in results):
        raise RuntimeError('Placement rendering checks failed; see placement_render_batch.json')


async def tracking_batch(identities, gpus, checkpoint):
    from common import worker
    available=asyncio.Queue()
    for gpu in gpus:available.put_nowait(gpu)
    async def one(identity,view):
        gpu=await available.get()
        try:
            await worker('placement',[identity,'--track-view',str(view),'--checkpoint',str(checkpoint)],gpu,
                         library='4danyone',log_name=f'{identity}_foot_tracking_v{view:02d}.log')
            return dict(identity=identity,view=view,state='executed_pending_visual_validation',gpu=gpu)
        except Exception as error:
            return dict(identity=identity,view=view,state='failed',error=str(error),gpu=gpu)
        finally:available.put_nowait(gpu)
    results=await asyncio.gather(*(one(i,v) for i in identities for v in [0,8]))
    save(ROOT/'reports/foot_tracking_batch.json',dict(results=results,finished_at=time.time(),
         state='executed_pending_visual_validation' if all(r['state']!='failed' for r in results) else 'needs_repair'))
    if any(r['state']=='failed' for r in results):raise RuntimeError('Foot tracking diagnostics failed')


async def source_hint_batch(identities, concurrency):
    semaphore=asyncio.Semaphore(concurrency)
    async def one(identity):
        async with semaphore:
            try:
                result=await asyncio.to_thread(backfill_source_hints,identity)
            except Exception as error:
                result=dict(identity=identity,state='failed',error=repr(error))
            print(json.dumps(result,ensure_ascii=False),flush=True)
            return result
    results=await asyncio.gather(*map(one,identities))
    save(ROOT/'reports/source_hints_backfill.json',dict(results=results,finished_at=time.time(),
         scope='Existing G0 evidence; no pose, diffusion or reconstruction rerun'))
    if any(r['state']=='failed' for r in results):raise RuntimeError('Source hint backfill needs review')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('identities',nargs='*')
    parser.add_argument('--all-complete',action='store_true')
    parser.add_argument('--concurrency',type=int,default=2)
    parser.add_argument('--verify-gpus',nargs='+',type=int)
    parser.add_argument('--foot-probe',action='store_true')
    parser.add_argument('--backfill-source-hints',action='store_true')
    parser.add_argument('--placement-trial',choices=['standing','walking'])
    parser.add_argument('--track-view',type=int,choices=[0,8])
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--track-gpus',nargs='+',type=int)
    args=parser.parse_args()
    assert 1 <= args.concurrency <= 4
    if args.all_complete:
        assert not args.identities
        args.identities=sorted(p.parent.name for p in ASSET_ROOT.glob('H*/package_complete.json')
                              if read(p).get('state')=='complete_pending_visual_review')
    if not args.identities:
        parser.error('Provide asset IDs or --all-complete')
    if args.placement_trial:
        assert len(args.identities)==1
        print(json.dumps(build_placement_trial(args.identities[0],args.placement_trial),ensure_ascii=False))
    elif args.backfill_source_hints:
        asyncio.run(source_hint_batch(args.identities,args.concurrency))
    elif args.track_gpus:
        assert args.checkpoint is not None and len(set(args.track_gpus))==len(args.track_gpus) and all(0<=g<8 for g in args.track_gpus)
        asyncio.run(tracking_batch(args.identities,args.track_gpus,args.checkpoint))
    elif args.track_view is not None:
        assert len(args.identities)==1 and args.checkpoint is not None
        track_rendered_feet(args.identities[0],args.track_view,args.checkpoint)
    elif args.foot_probe:
        assert len(args.identities)==1 and not args.verify_gpus
        foot_correspondence_probe(args.identities[0])
    elif args.verify_gpus:
        assert len(set(args.verify_gpus))==len(args.verify_gpus) and all(0<=g<8 for g in args.verify_gpus)
        asyncio.run(verify_batch(args.identities,args.verify_gpus))
    elif len(args.identities)==1 and not args.all_complete:
        enrich(args.identities[0])
    else:
        asyncio.run(batch_enrich(args.identities,args.concurrency))
