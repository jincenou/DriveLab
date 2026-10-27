"""Engineering placement estimates and one-clip trajectory playback.

Does not change Gaussian payloads or claim measured scale, contact, or looping.
Consumers use continuous native runtime at sample_frame and the returned matrix.
"""
import argparse
import asyncio
import fcntl
import json
import os
import sys
from pathlib import Path
import time
import numpy as np
from common import ASSET_ROOT, ROOT, read, save, sha


def compose(times, root_xy, root_yaw, asset_to_standard, ground_offset,
            sample_times, world_positions, world_yaw):
    times=np.asarray(times,float);query=np.asarray(sample_times,float)
    xyz=np.asarray(world_positions,float);yaw=np.asarray(world_yaw,float)
    if times.ndim!=1 or len(times)<2 or not np.all(np.diff(times)>0):raise ValueError('Invalid source clock')
    if query.ndim!=1 or xyz.shape!=(len(query),3) or yaw.shape!=(len(query),):raise ValueError('Trajectory shape mismatch')
    if not all(np.isfinite(v).all() for v in (times,query,xyz,yaw,root_xy,root_yaw,asset_to_standard)):raise ValueError('Nonfinite trajectory')
    if np.any(query<times[0]-1e-8) or np.any(query>times[-1]+1e-8):raise ValueError('One-clip playback only; no implicit wrap or endpoint hold')
    if np.asarray(root_xy).shape!=(len(times),2) or np.asarray(root_yaw).shape!=(len(times),):raise ValueError('Invalid root arrays')
    h=np.interp(query,times,np.unwrap(root_yaw));p=np.stack([np.interp(query,times,root_xy[:,i]) for i in (0,1)],axis=1)
    c,s=np.cos(yaw-h),np.sin(yaw-h);matrices=np.broadcast_to(np.eye(4),(len(query),4,4)).copy()
    matrices[:,0,0]=c;matrices[:,0,1]=-s;matrices[:,1,0]=s;matrices[:,1,1]=c
    matrices[:,:3,3]=xyz;matrices[:,0,3]-=c*p[:,0]-s*p[:,1];matrices[:,1,3]-=s*p[:,0]+c*p[:,1]
    matrices[:,2,3]+=float(ground_offset)
    return matrices@np.asarray(asset_to_standard),np.interp(query,times,np.arange(len(times)))


def sample(package, world_times, world_positions, world_yaw, retime_by_distance=False):
    """Return world_from_asset and frames. world_times are elapsed playback seconds.

    Distance retiming needs the entire path prefix, not an isolated target point.
    """
    package=Path(package);validate_estimate(package)
    target_times=np.asarray(world_times,float);positions=np.asarray(world_positions,float)
    if target_times.ndim!=1 or not len(target_times) or not np.isfinite(target_times).all() or np.any(target_times<0) or np.any(np.diff(target_times)<=0):
        raise ValueError('Requested times must be finite and strictly increasing')
    if positions.shape!=(len(target_times),3):raise ValueError('Expected Nx3 planned positions')
    with np.load(package/'placement_playback.npz',allow_pickle=False) as z:
        if retime_by_distance:
            if not bool(z['speed_available']) or float(z['native_speed_model_mps'])<=.02:
                raise ValueError('No usable stride-speed estimate: use explicit clock playback')
            distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(positions[:,:2],axis=0),axis=1))]
            query=z['frame_timestamps'][0]+distance/float(z['native_speed_model_mps'])
        else:query=z['frame_timestamps'][0]+target_times
        return compose(z['frame_timestamps'],z['root_xy'],z['root_yaw'],z['asset_to_standard'],z['ground_offset'],query,positions,world_yaw)


def build_estimates(package):
    """Small sidecar, no retraining; conservative capability labels remain intact."""
    from placement import transform_points
    package=Path(package);identity=package.name
    lock_dir=ROOT/'jobs';lock_dir.mkdir(parents=True,exist_ok=True)
    with (lock_dir/f'{identity}_placement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        meta=read(package/'placement.json');source_digest=sha(package/'placement_motion.npz')
        with np.load(package/'placement_motion.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
        times=a['frame_timestamps'];T=a['asset_to_standard_model_meters']
        reliable_root='standard_to_gaussian_inplace' in a
        height_keys=[s+'_foot_2sigma_support_height' for s in ('left','right') if s+'_foot_2sigma_support_height' in a]
        if not reliable_root or not height_keys:
            with np.load(package/'asset_4dgs.npz',allow_pickle=False) as z:
                mask=z['opacity'].reshape(-1)>=.5
                if mask.sum()<8:raise ValueError('Insufficient opaque Gaussian geometry')
                points=transform_points(z['frame_xyz'][:,mask],T)
            if not reliable_root:
                # Per-frame torso center: geometrically aligned, no semantic joint claim.
                low=np.quantile(points[:,:,2],.4,axis=1);high=np.quantile(points[:,:,2],.8,axis=1)
                torso=(points[:,:,2]>=low[:,None])&(points[:,:,2]<=high[:,None])
                center=np.nanmedian(np.where(torso[:,:,None],points,np.nan),axis=1)
                root=center[:,:2];root_z=center[:,2]
            if not height_keys:height=np.quantile(points[:,:,2],.02,axis=1)
            del points
        if reliable_root:root=a['gaussian_root_position'][:,:2];root_z=a['gaussian_root_position'][:,2];heading=a['gaussian_heading_radians']
        else:heading=np.zeros(len(times))  # Preserve unknown residual yaw; never invent a measured heading.
        if height_keys:height=np.min(np.stack([a[k] for k in height_keys]),axis=0)
        ground=-float(np.median(height))
        speed=meta['motion']['gait'].get('native_speed_model_mps_estimate')
        speed_basis='Gaussian support-foot candidate after root/yaw removal'
        if speed is None:
            names=meta['provenance']['keypoint_names'];kp=a['condition_keypoints_standard_model_m']
            feet=np.stack([kp[:,[names.index(s+'-'+k) for k in ('heel','big-toe-tip','small-toe-tip')]].mean(1) for s in ('left','right')],axis=1)
            velocity=np.gradient(feet,times,axis=0)
            support=(feet[:,:,2]<=np.quantile(feet[:,:,2],.35))&(np.abs(velocity[:,:,2])<=.15)
            speed=float(np.median(np.linalg.norm(velocity[:,:,:2][support],axis=-1))) if support.sum()>=12 else None
            speed_basis='Source conditioning foot motion estimate; final Gaussian stride agreement unverified'
        if speed is not None and (not np.isfinite(speed) or speed<=.02):speed=None
        if speed is None and meta['action']['label'] in ('standing','standing_or_small_gesture','gesturing'):
            speed=0.
            speed_basis='Nominal zero locomotion for estimated stationary/gesture action; not measured residual body motion'
        if 'support_foot_candidate' in a:
            contacts=a['support_foot_candidate'];contact_available=np.ones((len(times),2),bool)
            contact_basis='Gaussian finite-envelope/vertical-velocity hypothesis'
        else:
            names=meta['provenance']['keypoint_names'];kp=a['condition_keypoints_standard_model_m']
            feet=np.stack([kp[:,[names.index(s+'-'+k) for k in ('heel','big-toe-tip','small-toe-tip')]].mean(1) for s in ('left','right')],axis=1)
            velocity=np.gradient(feet,times,axis=0)
            contacts=(feet[:,:,2]<=np.quantile(feet[:,:,2],.35))&(np.abs(velocity[:,:,2])<=.15)
            contact_available=np.ones((len(times),2),bool)
            contact_basis='Source conditioning estimate; foot identity/contact on final Gaussians not established'
        default_move=meta['action']['label'] in ('walking','running','walking_or_running') and speed is not None
        target=package/'placement_playback.npz';temporary=target.with_suffix('.writing')
        with temporary.open('wb') as stream:np.savez_compressed(stream,frame_timestamps=times,root_xy=root,root_yaw=heading,
               asset_to_standard=T,ground_offset=np.array(ground),support_height_before=height,
               gaussian_root_position=np.column_stack([root,root_z]),
               gaussian_root_displacement=np.column_stack([root,root_z])-np.r_[root[0],root_z[0]],
               gaussian_root_velocity=np.gradient(np.column_stack([root,root_z]),times,axis=0),
               contact_estimate=contacts,contact_estimate_available=contact_available,
               native_speed_model_mps=np.array(speed if speed is not None else 0.),speed_available=np.array(speed is not None))
        temporary.replace(target)
        record=dict(version=1,identity=identity,state='engineering_estimate_user_accepts_residual_error',
            asset_sha256=meta['asset_sha256'],placement_sha256=sha(package/'placement.json'),source_motion_sha256=source_digest,
            arrays='placement_playback.npz',array_sha256=sha(target),
            coordinate_convention='right_handed_x_forward_y_left_z_up',scale_basis=meta['scale'],
            ground_offset_model_m=ground,ground_basis='Median of lower finite 2sigma foot proxies' if height_keys else 'Median 2nd percentile opaque Gaussian centers; not sole surface',
            support_height_after_p05_model_m=float(np.quantile(height+ground,.05)),support_height_after_p95_model_m=float(np.quantile(height+ground,.95)),
            root_basis='Existing Gaussian-bound hip/shoulder proxies' if reliable_root else 'Per-frame opaque torso center, estimated geometric center',
            heading_removal='Gaussian heading proxy removed' if reliable_root else 'Residual heading remains baked; only planned yaw is applied',
            body_forward_at_reference=[1,0,0],body_heading_from_gaussians_available=reliable_root,
            action=meta['action'],native_speed_model_mps=speed,speed_basis=speed_basis,
            contact_basis=contact_basis,root_motion_baked=True,
            root_removal_rule='Use provided root_xy/root_yaw once; preserve vertical body motion. Do not also remove source root.',
            suggested_motion='trajectory_with_estimated_retiming' if default_move else 'play_original_clip_at_fixed_location',
            approximate_placement_available=True,contact_verified=False,anti_skating_verified=False,
            metric_scale_measured=False,loopable=False,loop_interval=None,default_playback='once',
            consumer='playback.compose returns world_from_asset and sample_frame. Native runtime adds asset_from_native once; inverse camera adapter preserves SH/covariance.',
            caveats=['Approximate floor offset may retain floating or penetration.', 'Speed is an estimate, not measured scene travel.',
                     'No pose retargeting; moving a standing clip does not create walking.', 'No implicit looping or extrapolation.',
                     'Original strict diagnostic failures remain recorded and do not block approximate use.'],created=time.time())
        insertion=read(package/'insertion.json');insertion['practical_playback']=record;save(package/'insertion.json',insertion)
        for name in ('manifest.json','motion_metadata.json'):
            data=read(package/name);data['approximate_playback']='insertion.json#/practical_playback';save(package/name,data)
        note='\n近似插入与播放：读取 insertion.json 的 practical_playback 及 placement_playback.npz；human/playback.py 的 sample() 返回 world_from_asset 和连续帧号。尺度为模型估计，接地/步速可有残差，默认单次播放。不要重复去除源根运动或套旧UP_FIX。\n'
        readme=package/'README.txt';old=readme.read_text() if readme.exists() else ''
        if '近似插入与播放：' not in old:readme.write_text(old+note)
        return validate_estimate(package)


def validate_estimate(package):
    package=Path(package);r=read(package/'insertion.json')['practical_playback']
    assert r['array_sha256']==sha(package/'placement_playback.npz')
    assert r['placement_sha256']==sha(package/'placement.json')
    assert r['source_motion_sha256']==sha(package/'placement_motion.npz')
    assert r['asset_sha256']==read(package/'placement.json')['asset_sha256']
    assert not r['contact_verified'] and not r['anti_skating_verified'] and not r['loopable']
    with np.load(package/'placement_playback.npz',allow_pickle=False) as z:
        assert all(np.isfinite(z[k]).all() for k in z.files)
        times=z['frame_timestamps']
        M,frames=compose(times,z['root_xy'],z['root_yaw'],z['asset_to_standard'],z['ground_offset'],times,np.zeros((len(times),3)),np.zeros(len(times)))
        np.testing.assert_allclose(frames,np.arange(len(times)),atol=1e-8)
        assert np.isfinite(M).all()
    return dict(identity=package.name,state='approximate_playback_numeric_readback_passed',frame_count=len(times),metadata_sha256=sha(package/'insertion.json'))


async def backfill(ids):
    semaphore=asyncio.Semaphore(2)
    async def one(identity):
        async with semaphore:
            try:
                env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                process=await asyncio.create_subprocess_exec(sys.executable,str(Path(__file__).resolve()),identity,'--one',
                        env=env,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
                out,error=await process.communicate()
                if process.returncode:raise RuntimeError(error.decode()[-2000:])
                result=json.loads(out)
            except Exception as error:result=dict(identity=identity,state='failed',error=str(error))
            print(result,flush=True);return result
    results=await asyncio.gather(*map(one,ids))
    completed=[p.parent for p in ASSET_ROOT.glob('H*/package_complete.json') if read(p).get('state')=='complete_pending_visual_review']
    current=[]
    for package in sorted(completed):
        try:current.append(validate_estimate(package))
        except Exception as error:current.append(dict(identity=package.name,state='missing_or_invalid',error=str(error)))
    save(ROOT/'reports/approximate_playback_backfill.json',dict(results=current,last_updated_ids=ids,
         last_update_results=results,finished=time.time(),scope='Current complete package readback; does not recount a partial update as the whole backfill'))
    if any(r['state']=='failed' for r in results):raise RuntimeError('Some approximate sidecars need review')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('identities',nargs='*');parser.add_argument('--all-complete',action='store_true');parser.add_argument('--one',action='store_true');args=parser.parse_args()
    ids=sorted(p.parent.name for p in ASSET_ROOT.glob('H*/package_complete.json') if read(p).get('state')=='complete_pending_visual_review') if args.all_complete else args.identities
    if not ids:parser.error('Provide identities or --all-complete')
    if args.one:
        assert len(ids)==1
        print(json.dumps(build_estimates(ASSET_ROOT/ids[0])),flush=True)
    else:asyncio.run(backfill(ids))
