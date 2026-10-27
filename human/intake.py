"""nuScenes intake: rank tracked people, prepare calibrated G0 inputs, keep provenance.

This adapter does not modify the fixed generator or reconstruction objectives.
Interpolated annotation visibility is a selection hint, never a quality label.
"""
import os
import argparse
import asyncio
from collections import Counter, defaultdict
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from common import ROOT, CODE_ROOT, read, save, sha

DATA = Path(os.environ.get('NUSCENES_ROOT', 'data/nuscenes')).resolve()


def process_running(pid, process_root=Path('/proc')):
    """An unreaped zombie has exited; its /proc directory can remain present."""
    try:
        status=(process_root/str(pid)/'status').read_text()
    except (FileNotFoundError,ProcessLookupError):
        return False
    except PermissionError:
        return True  # Lack of visibility is not evidence of process exit.
    state=next(line.split()[1] for line in status.splitlines() if line.startswith('State:'))
    return state not in ('Z','X','x')


def rigid(record):
    q = np.asarray(record['rotation'])
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    m[:3, 3] = record['translation']
    return m


def corners(record):
    # nuScenes size is width, length, height; local X is length.
    w, l, h = record['size']
    signs = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    m = rigid(record)
    return (signs * np.array([l,w,h]) / 2) @ m[:3,:3].T + m[:3,3]


def dense_observations(keys, images, sensors, poses):
    times=np.array([r['sample_timestamp_s'] for r in keys]); origin=times[0];relative=times-origin
    translations=np.array([r['object_world']['translation'] for r in keys])
    sizes=np.array([r['object_world']['size'] for r in keys])
    quats=np.array([r['object_world']['rotation'] for r in keys])
    rotations=Slerp(relative,Rotation.from_quat(quats[:,[1,2,3,0]]))
    token=keys[0]['sample_data_token'];rows=[]
    while token:
        image=images[token];t=image['timestamp']/1e6;token=image['next']
        if t<times[0]:continue
        if t>times[-1]:break
        path=DATA/image['filename']
        if not path.is_file():raise ValueError('Missing actual exposure')
        if rows and t-rows[-1]['timestamp_s']>.15:raise ValueError('Camera gap')
        right=min(int(np.searchsorted(times,t,side='right')),len(keys)-1);left=max(0,right-1)
        frac=(t-times[left])/(times[right]-times[left])
        xyz=(1-frac)*translations[left]+frac*translations[right]
        size=(1-frac)*sizes[left]+frac*sizes[right]
        q=rotations(t-origin).as_quat();box=corners(dict(translation=xyz,size=size,rotation=q[[3,0,1,2]]))
        cs=sensors[image['calibrated_sensor_token']];c2w=rigid(poses[image['ego_pose_token']])@rigid(cs);K=np.array(cs['camera_intrinsic'])
        pts=(box-c2w[:3,3])@c2w[:3,:3]
        if pts[:,2].min()<=.1:raise ValueError('Behind camera')
        uv=pts@K.T;uv=uv[:,:2]/uv[:,2:];lo=uv.min(0);hi=uv.max(0)
        if lo.min()<0 or hi[0]>image['width'] or hi[1]>image['height']:raise ValueError('Clipped dense box')
        rows.append(dict(frame_index=len(rows),timestamp_s=t,sample_data_token=image['token'],
            image_path=str(path),width=image['width'],height=image['height'],bbox_xyxy=np.r_[lo,hi].tolist(),
            camera_to_world=c2w.tolist(),K=K.tolist(),annotation_tokens=[keys[left]['annotation_token'],keys[right]['annotation_token']],
            annotation_fraction=float(frac),visibility_hint=keys[left if frac<.5 else right]['visibility']))
        if rows[-1]['timestamp_s']-rows[0]['timestamp_s']>=2.4:break
    if len(rows)<25 or rows[-1]['timestamp_s']-rows[0]['timestamp_s']<2.4:raise ValueError('Insufficient actual exposure coverage')
    return rows


def rank(limit=100, reserve=False, destination=None, excluded=()):
    destination=Path(destination) if destination else ROOT/('reserve.json' if reserve else 'selection.json')
    assert not destination.exists(), 'Selection already frozen'
    ROOT.mkdir(parents=True,exist_ok=True)
    save(ROOT/'intake_status.json',dict(state='loading_local_metadata',time=time.time()))
    original = DATA/'v1.0-trainval'
    samples = {r['token']:dict(r,data={}) for r in read(original/'sample.json')}
    scenes = {r['token']:r for r in read(original/'scene.json')}
    sensors = {r['token']:r for r in read(original/'calibrated_sensor.json')}
    camera_sensors = {r['token']:r['channel'] for r in read(original/'sensor.json') if r['modality']=='camera'}
    camera_calibs = {k:camera_sensors[r['sensor_token']] for k,r in sensors.items() if r['sensor_token'] in camera_sensors}
    poses = {r['token']:r for r in read(original/'ego_pose.json')}
    images = {r['token']:r for r in read(original/'sample_data.json') if r['calibrated_sensor_token'] in camera_calibs}
    for tok,image in images.items():
        if image['is_key_frame']:
            samples[image['sample_token']]['data'][camera_calibs[image['calibrated_sensor_token']]]=tok
    categories={r['token']:r['name'] for r in read(original/'category.json')}
    instances={r['token']:categories[r['category_token']] for r in read(original/'instance.json')}
    person_categories={'human.pedestrian.adult','human.pedestrian.child','human.pedestrian.construction_worker','human.pedestrian.police_officer'}
    people=[dict(r,category_name=instances[r['instance_token']]) for r in read(original/'sample_annotation.json')
            if instances[r['instance_token']] in person_categories]
    tracks = defaultdict(list)
    cameras = {}
    for ann in people:
        sample = samples[ann['sample_token']]
        world = corners(ann)
        for channel, tok in sample['data'].items():
            if tok not in images: continue
            image = images[tok]
            if tok not in cameras:
                cs, ep = sensors[image['calibrated_sensor_token']], poses[image['ego_pose_token']]
                cameras[tok] = (rigid(ep) @ rigid(cs), np.array(cs['camera_intrinsic']), (DATA/image['filename']).is_file())
            c2w, K, exists = cameras[tok]
            if not exists: continue
            p = (world-c2w[:3,3]) @ c2w[:3,:3]
            if p[:,2].min() <= .1: continue
            q = p @ K.T; uv = q[:,:2]/q[:,2:]
            lo, hi = uv.min(0), uv.max(0); w,h=hi-lo
            if h < 100 or w < 20: continue
            edge = bool(lo.min()<4 or hi[0]>image['width']-4 or hi[1]>image['height']-4)
            if lo[0]<-w*.1 or hi[0]>image['width']+w*.1 or lo[1]<-h*.05 or hi[1]>image['height']+h*.05:continue
            tracks[(ann['instance_token'],channel)].append(dict(
                timestamp_s=image['timestamp']/1e6, sample_timestamp_s=sample['timestamp']/1e6,
                sample_data_token=tok, annotation_token=ann['token'], scene=scenes[sample['scene_token']]['name'],
                image_path=str(DATA/image['filename']), width=image['width'],height=image['height'],
                bbox_xyxy=np.r_[lo,hi].tolist(),visibility=int(ann.get('visibility_token') or 0),edge=edge,
                camera_to_world=c2w.tolist(),K=K.tolist(),category=ann['category_name'],object_world=ann))
    candidates=[]
    for (instance,channel), rows in tracks.items():
        rows=sorted({r['timestamp_s']:r for r in rows}.values(),key=lambda x:x['timestamp_s'])
        best=None
        # Seven annotated keyframes cover ~3s. Dense real camera exposures are resolved only for shortlisted windows.
        for start in range(0,len(rows)-6):
            window=rows[start:start+7];dt=np.diff([r['timestamp_s'] for r in window])
            if not ((dt>.35)&(dt<.65)).all():continue
            span=window[-1]['timestamp_s']-window[0]['timestamp_s']
            if span<2.4:continue
            heights=np.array([r['bbox_xyxy'][3]-r['bbox_xyxy'][1] for r in window])
            visible=np.mean([r['visibility']>=3 for r in window]);edge=np.mean([r['edge'] for r in window])
            if np.median(heights)<140 or visible<.8 or edge>.10:continue
            score=float(np.median(np.minimum(heights,650))/650+.3*visible-.5*edge-.1*np.std(heights)/np.mean(heights))
            if best is None or score>best['rank_score']:
                best=dict(instance_token=instance,camera=channel,scene=window[0]['scene'],
                    identity_key='nuscenes:'+instance,rank_score=score,height_median_px=float(np.median(heights)),
                    visibility_hint_fraction=float(visible),edge_fraction=float(edge),frames=61,
                    keyframes=window)
        if best:candidates.append(best)
    candidates.sort(key=lambda r:(-r['rank_score'],r['identity_key'],r['camera']))
    previous=(read(ROOT/'selection.json')['items'] if reserve else [])+list(excluded)
    previous={r['id']:r for r in previous}.values()
    selected=[];seen={r['instance_token'] for r in previous};per_scene=Counter()
    # An explicit expansion also respects the scene cap across earlier pools.
    if excluded:per_scene.update(r['scene'] for r in previous)
    offset=max((int(r['id'][2:]) for r in previous),default=0)
    for row in candidates:
        if row['instance_token'] in seen or per_scene[row['scene']]>=5:continue
        try:
            observations = dense_observations(row['keyframes'], images, sensors, poses)
        except ValueError:
            continue
        row=dict(row,id=f'HN{offset+len(selected)+1:04d}',observations=observations,native_frames=len(observations))
        selected.append(row);seen.add(row['instance_token']);per_scene[row['scene']]+=1
        if len(selected)==limit:break
    save(destination,dict(dataset='nuscenes',time=time.time(),target_count=limit,
        items=selected,demo_ids=[r['id'] for r in selected[:2]],quality_approved=False,
        available_instance_camera_candidates=len(candidates),unique_eligible_instances=len({r['instance_token'] for r in candidates}),
        scenes=dict(per_scene),input_tables=str(original),
        rule=dict(native_frames='actual exposure brackets covering 2.4s',output_frames=61,fps=25,min_median_height_px=140,
                  min_visibility_3plus_fraction=.8,max_border_fraction=.1,max_per_scene=5),
        limitations='Original v1.0-trainval annotations and sensor records, filtered to locally present images. Between-keyframe boxes are geometric interpolations, not additional ground truth. Projected 3D boxes identify tracks but are not precise person masks. Not full nuScenes coverage or quality acceptance.'))
    save(ROOT/'intake_status.json',dict(state='ranked_candidates_not_quality_approved',selected=len(selected),
        unique_eligible_instances=len({r['instance_token'] for r in candidates}),time=time.time()))
    print(f'Selected {len(selected)} tracks from {len(candidates)} instance-camera candidates',flush=True)


def selected(identity):
    return next(r for r in read(ROOT/'selection.json')['items'] if r['id']==identity)


def clip_root(identity):
    return ROOT/'inputs'/identity/'clip0'


def native(identity):
    from PIL import Image
    import cv2
    from prepare import encode
    row=selected(identity);obs=row['observations'];out=clip_root(identity)
    out.mkdir(parents=True,exist_ok=True);(out/'native').mkdir(exist_ok=False)
    boxes=np.array([r['bbox_xyxy'] for r in obs]);extent=(boxes[:,2:]-boxes[:,:2]).max(0)*1.3
    unit=int(np.ceil(max(extent[0]/11,extent[1]/20)/2)*2);w,h=11*unit,20*unit
    if w>obs[0]['width'] or h>obs[0]['height']:raise ValueError('Native padded crop exceeds original image; select another track')
    records=[];Ks=[];Rs=[];maps=[];c2ws=[];rgb=[]
    for i,(r,box) in enumerate(zip(obs,boxes,strict=True)):
        path=Path(r['image_path']);pixels=Image.open(path).convert('RGB')
        assert pixels.size==(r['width'],r['height'])
        lo=np.rint((box[:2]+box[2:]-[w,h])/2).astype(int)
        lo=np.maximum(0,np.minimum(lo,[pixels.width-w,pixels.height-h]));hi=lo+[w,h]
        A=np.eye(3);A[:2,2]=-lo;K=A@np.array(r['K']);b=box-np.tile(lo,2)
        assert min(b)>=0 and b[2]<=w and b[3]<=h
        image=pixels.crop((*lo,*hi));target=out/'native'/f'{i:05d}.png';image.save(target);rgb.append(image)
        c2w=np.array(r['camera_to_world']);Ks.append(K);Rs.append(c2w[:3,:3].T);maps.append(A);c2ws.append(c2w)
        records.append(dict(native_index=i,frame_index=i,raw_source_path=str(path),raw_source_sha256=sha(path),
            path=str(target),sha256=sha(target),image_center_timestamp_s=r['timestamp_s'],
            frame_timestamp_micros=round(r['timestamp_s']*1e6),crop_xyxy=[*lo.tolist(),*hi.tolist()],
            undistorted_to_input=A.tolist(),K_input=K.tolist(),bbox_input_xyxy=b.tolist(),
            interpolated=False,enhanced=False,sample_data_token=r['sample_data_token']))
    np.savez_compressed(out/'native_cameras.npz',K_input=Ks,R_world_to_cv_camera=Rs,
        image_center_cv_camera_to_world=c2ws,undistorted_to_input=maps,image_center_timestamps_s=[r['timestamp_s'] for r in obs])
    save(out/'native.json',dict(state='native_calibrated_crop_ready',dataset='nuscenes',size=[w,h],frames=records,
        scene=row['scene'],track_id=row['instance_token'],camera_index=row['camera'],
        camera_npz_sha256=sha(out/'native_cameras.npz'),camera_source='Original v1.0-trainval sample_data/ego_pose/calibrated_sensor',
        image_geometry='Dataset camera pixels and K, translation crop only, no guessed distortion correction',
        timing='Original sensor timestamps; box center/size linear and rotation Slerp between annotated keyframes',
        world_pose_verified=False))
    encode(out/'observed_native.mp4',rgb,fps=12,timestamps=[r['timestamp_s'] for r in obs])
    save(out.parent/'candidate.json',row)


def interpolate(identity):
    import sys,torch
    from torch.nn import functional as F
    from PIL import Image
    from prepare import encode
    from common import RESOURCES,gpu_identity
    gpu_identity();out=clip_root(identity);meta=read(out/'native.json');records=meta['frames']
    dest=out/'inputs_rife';dest.mkdir(exist_ok=False)
    sys.path.insert(0,str(CODE_ROOT/'vendor/rife'))
    from train_log.IFNet_HDv3 import IFNet
    model=IFNet().eval().cuda();weights=torch.load(RESOURCES['rife_weights'],map_location='cpu',weights_only=True)
    weights={k.removeprefix('module.'):v for k,v in weights.items()};model.load_state_dict({k:v for k,v in weights.items() if not k.startswith(('teacher.','caltime.'))},strict=True)
    images=[np.array(Image.open(r['path']).convert('RGB')) for r in records];h,w=images[0].shape[:2]
    tensors=[F.pad(torch.from_numpy(a[:,:,::-1].copy()).permute(2,0,1).float()[None]/255,(0,(-w)%64,0,(-h)%64)) for a in images]
    with np.load(out/'native_cameras.npz') as z:cam={k:z[k] for k in z.files}
    rel=cam['image_center_timestamps_s']-cam['image_center_timestamps_s'][0];times=np.arange(61)/25
    assert rel[-1]>=times[-1] and (np.diff(rel)>0).all()
    result=[];mapping=[];affines=[];intrinsics=[]
    with torch.inference_mode():
        for slot,t in enumerate(times):
            right=int(np.searchsorted(rel,t,side='left'))
            if abs(rel[right]-t)<1e-8:left=right;fraction=0.
            else:left=right-1;fraction=float((t-rel[left])/(rel[right]-rel[left]))
            if left==right:im=images[left]
            else:
                pred=model(torch.cat((tensors[left],tensors[right]),1).cuda(),torch.tensor(fraction,device='cuda').reshape(1,1,1,1),[16,8,4,2,1])[2][-1]
                im=pred[0,:,:h,:w].clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()[:,:,::-1].copy()
            result.append(Image.fromarray(im));A=(1-fraction)*cam['undistorted_to_input'][left]+fraction*cam['undistorted_to_input'][right];K=(1-fraction)*cam['K_input'][left]+fraction*cam['K_input'][right]
            affines.append(A);intrinsics.append(K)
            mapping.append(dict(slot=slot,video_seconds=float(t),source_relative_exposure_seconds=float(t),
                left_native_index=left,right_native_index=right,left_source_frame=left,right_source_frame=right,
                interpolation_fraction=fraction,interpolated=left!=right,source_path=records[left]['raw_source_path'],
                crop_xyxy=[0,0,w,h],bbox_input_xyxy=((1-fraction)*np.array(records[left]['bbox_input_xyxy'])+fraction*np.array(records[right]['bbox_input_xyxy'])).tolist(),
                undistorted_to_input=A.tolist(),K_input=K.tolist()))
    encode(dest/'main.mp4',result)
    rotations=Slerp(rel,Rotation.from_matrix(cam['R_world_to_cv_camera']))(times).as_matrix()
    np.savez_compressed(dest/'cameras.npz',K_input=intrinsics,R_world_to_cv_camera=rotations,undistorted_to_input=affines,
        source_relative_exposure_seconds=times,video_seconds=times)
    save(dest/'frame_mapping.json',dict(dataset='nuscenes',frames=mapping,native_frames=len(records),output_frames=61,
        video_sha256=sha(dest/'main.mp4'),camera_npz_sha256=sha(dest/'cameras.npz'),native_source_sha256=sha(out/'native.json'),
        RIFE_weights_sha256=sha(RESOURCES['rife_weights']),source_speed_ratio=1.,speedup=False,endpoint_padding=False,
        source_timestamps='Original camera exposures; RGB interpolation and SO3/crop-K resampling use identical brackets'))


def segment(identity):
    import sys,torch,av,shutil
    from PIL import Image
    from prepare import encode
    from common import RESOURCES,gpu_identity
    gpu_identity();out=clip_root(identity);source=out/'inputs_rife';seg=out/'segmentation'
    frames_dir=seg/'frames';frames_dir.mkdir(parents=True,exist_ok=True)
    rgb=[]
    with av.open(str(source/'main.mp4')) as con:
        con.streams.video[0].thread_count=1
        for i,f in enumerate(con.decode(video=0)):
            im=f.to_image();rgb.append(np.array(im));im.save(frames_dir/f'{i:05d}.jpg',quality=95,subsampling=0)
    assert len(rgb)==61
    sys.path.insert(0,str(CODE_ROOT/'vendor/segmentation'))
    from sam2.build_sam import build_sam2_video_predictor
    from vendor.sam_transfer_guard import install_transfer_guard
    mapping=read(source/'frame_mapping.json');box=np.array(mapping['frames'][0]['bbox_input_xyxy'],np.float32)
    cached=seg/'target_masks.npz'
    if cached.exists():
        masks=np.load(cached,allow_pickle=False)['masks']
    else:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',RESOURCES['sam2_weights'],device='cuda',apply_postprocessing=False)
            install_transfer_guard(predictor);state=predictor.init_state(video_path=str(frames_dir),offload_video_to_cpu=True,offload_state_to_cpu=True)
            # Same BiRefNet -> SAM2 mask initialization used by the fixed
            # downstream RGBA preparation. Preserve bags in the foreground seed.
            from common import FRONTEND
            sys.path.insert(0,str(FRONTEND))
            from fdanyone.foreground import predict_foreground_masks
            from fdanyone.assets import resolve_foreground_model
            # The pinned BiRefNet owns FP16 weights/input. Do not inherit
            # SAM2's BF16 autocast into its BatchNorm operations.
            with torch.autocast('cuda', enabled=False):
                seed=predict_foreground_masks((rgb[0],),resolve_foreground_model(FRONTEND/'models'),'cuda:0')[0]
            from scipy.ndimage import label
            labels_cc,n=label(seed>127)
            xa,ya,xb,yb=np.rint(box).astype(int)
            scores=np.bincount(labels_cc[max(0,ya):yb,max(0,xa):xb].ravel(),minlength=n+1);scores[0]=0
            assert n and scores.max()>100,'No foreground component for tracked person'
            chosen=labels_cc==int(scores.argmax())
            np.savez_compressed(seg/'initial_mask.npz',alpha=seed,selected_component=chosen)
            predictor.add_new_mask(state,frame_idx=0,obj_id=1,mask=chosen)
            masks=[]
            for i,ids,logits in predictor.propagate_in_video(state):
                assert ids==[1] and i==len(masks) and torch.isfinite(logits).all()
                masks.append(logits[0,0].sigmoid().float().cpu().numpy().__mul__(255).round().astype(np.uint8))
        masks=np.array(masks)
        np.savez_compressed(seg/'target_masks.npz',masks=masks)
        save(seg/'area_check.json',dict(areas=(masks>127).sum((1,2)).tolist()))
        assert len(masks)==61 and ((masks>127).sum((1,2))>=100).all(), 'Empty tracked mask; inspect area_check.json'
    assert len(masks)==61 and ((masks>127).sum((1,2))>=100).all(), 'Empty tracked mask; inspect area_check.json'
    dest=out/'inputs';dest.mkdir(exist_ok=True)
    assert not (dest/'frame_mapping.json').exists(), 'Completed composition should be recovered, not overwritten'
    if not (dest/'cameras.npz').exists():(dest/'cameras.npz').symlink_to(source/'cameras.npz')
    composited=[Image.fromarray(np.rint(im*(a[:,:,None]/255)+255*(1-a[:,:,None]/255)).clip(0,255).astype(np.uint8)) for im,a in zip(rgb,masks,strict=True)]
    encode(dest/'main.mp4',composited);mapping['video_sha256']=sha(dest/'main.mp4');mapping['background_compositing']={'method':'straight alpha over white','mask_source':'segmentation/target_masks.npz'}
    save(dest/'frame_mapping.json',mapping)
    save(seg/'manifest.json',dict(frames=61,box=box.tolist(),initialization='BiRefNet connected target foreground -> SAM2; same mask-init as backend; input-only adapter repair',accessory_completeness='pending_input_check',mask_sha256=sha(seg/'target_masks.npz')))
    shutil.rmtree(frames_dir)


def register(identity):
    import fcntl
    row=selected(identity);p=clip_root(identity);condition=p.parent/'conditions/preflight'
    spec=dict(prepared_frames=61,raw_input=str(p/'inputs/main.mp4'),conditioning=str(condition/'conditioning'),
        motion=str(condition/'motion'),root_motion=str(condition/'root_motion.npz'),source_dataset='nuscenes',
        manifest=str(ROOT/'data'/identity/'fdanyone.json'),config=str(ROOT/'config'/f'{identity}.json'),
        selection=dict(row,package=str(p.parent),source=str(p/'observed_native.mp4'),pose=str(p/'motion_pair/projection_comparison.mp4')),
        input_review=dict(allow_generation=False,accessory='pending',input_sha256=sha(p/'inputs/main.mp4')))
    (ROOT/'config').mkdir(exist_ok=True)
    with (ROOT/'config/registry.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        registry=read(ROOT/'config/cases.json') if (ROOT/'config/cases.json').exists() else {}
        registry[identity]=spec;save(ROOT/'config/cases.json',registry)
    from prepare import inspect_input
    inspect_input(identity)


async def prepare_cases(cases,gpus):
    from common import worker
    from batch import memory_free
    import fcntl
    assert cases and gpus and len(set(gpus))==len(gpus)
    async def one(identity,gpu):
        (ROOT/'jobs').mkdir(exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True)
        with (ROOT/'jobs'/f'intake_{identity}.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                while memory_free(gpu)<24000:await asyncio.sleep(30)
                stages=[('native','intake','4danyone'),('interpolate','intake','4danyone'),('segment','intake','4danyone-sam2'),('motion','intake_motion','4danyone'),('condition','intake_motion','4danyone'),('register','intake','4danyone')]
                for stage,module,env in stages:
                    receipt=ROOT/'reports'/f'{identity}_intake_{stage}.json'
                    if receipt.exists() and read(receipt)['state']=='completed':continue
                    save(ROOT/'jobs'/f'intake_{identity}.json',dict(state='running',stage=stage,gpu=gpu,time=time.time()))
                    await worker(module,[stage,identity],gpu,library=env,log_name=f'{identity}_intake_{stage}.log')
                    save(receipt,dict(state='completed',stage=stage,time=time.time()))
                save(ROOT/'jobs'/f'intake_{identity}.json',dict(state='input_ready_pending_pilot_review',time=time.time()))
            except Exception as error:
                save(ROOT/'jobs'/f'intake_{identity}.json',dict(state='failed',error=str(error),time=time.time()));raise
    pending=list(cases);errors=[]
    async def consume(gpu):
        import shutil
        while pending:
            if memory_free(gpu)<24000 or shutil.disk_usage(ROOT).free<12*2**30:
                save(ROOT/'jobs'/f'intake_worker_gpu{gpu}.json',dict(state='waiting_resources',pending=len(pending),time=time.time()))
                await asyncio.sleep(30);continue
            identity=pending.pop(0)
            save(ROOT/'jobs'/f'intake_worker_gpu{gpu}.json',dict(state='running',identity=identity,time=time.time()))
            try:await one(identity,gpu)
            except Exception as error:errors.append(dict(identity=identity,error=str(error)))
        save(ROOT/'jobs'/f'intake_worker_gpu{gpu}.json',dict(state='queue_drained',time=time.time()))
    await asyncio.gather(*(consume(gpu) for gpu in gpus))
    save(ROOT/'intake_queue_exit.json',dict(errors=errors,time=time.time(),cases=cases,
        state='completed_with_rejections' if errors else 'completed_inputs_pending_review'))


def instance_check(identity):
    """Count people in the actual masked input; VLM alone can miss merged people."""
    import av
    from ultralytics import YOLO
    from common import FRONTEND, gpu_identity
    gpu_identity()
    p=clip_root(identity);source=p/'inputs/main.mp4';target=ROOT/'review'/f'{identity}_instances.json'
    slots=[0,10,20,30,40,50,60];images=[]
    with av.open(str(source)) as video:
        video.streams.video[0].thread_count=1
        for i,frame in enumerate(video.decode(video=0)):
            if i in slots:images.append(frame.to_image())
    assert len(images)==len(slots)
    weights=FRONTEND/'third_party/GVHMR/inputs/checkpoints/yolo/yolov8x.pt'
    model=YOLO(str(weights));predictions=model.predict(images,classes=[0],conf=.35,imgsz=640,device=0,verbose=False,batch=2)
    rows=[]
    for slot,pred in zip(slots,predictions,strict=True):
        boxes=pred.boxes.xyxy.cpu().numpy();scores=pred.boxes.conf.cpu().numpy()
        h,w=pred.orig_shape;area=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
        keep=area>h*w*.015
        rows.append(dict(frame=slot,count=int(keep.sum()),boxes=boxes[keep].tolist(),confidence=scores[keep].tolist()))
    with np.load(p/'segmentation/target_masks.npz',allow_pickle=False) as z:
        masks=z['masks'];area=(masks>127).sum((1,2));opaque=(masks>230).sum((1,2))/np.maximum(area,1)
    save(target,dict(identity=identity,source_sha256=sha(source),mask_sha256=sha(p/'segmentation/target_masks.npz'),
        rows=rows,one_person_every_sample=all(r['count']==1 for r in rows),
        first_opaque_fraction=float(opaque[0]),median_opaque_fraction=float(np.median(opaque)),
        opacity_admission=bool(opaque[0]>=.65 and np.median(opaque)>=.7),
        weights=str(weights),weights_sha256=sha(weights),thresholds=dict(confidence=.35,min_image_area=.015,first_opaque=.65,median_opaque=.7),
        scope='Engineering input screen with seven frames, not ground truth or final asset quality certification'))


async def check_registered(watch=False):
    from common import worker
    sem=asyncio.Semaphore(1)
    async def one(identity):
        if (ROOT/'reports'/f'{identity}_input_retired.json').exists():return
        p=ROOT/'review'/f'{identity}_instances.json';source=clip_root(identity)/'inputs/main.mp4'
        if p.exists() and read(p)['source_sha256']==sha(source):return
        async with sem:
            await worker('intake',['instance',identity],6,library='4danyone',log_name=f'{identity}_instance_check.log')
    while True:
        registry=read(ROOT/'config/cases.json') if (ROOT/'config/cases.json').exists() else {}
        results=await asyncio.gather(*(one(i) for i in registry),return_exceptions=True)
        save(ROOT/'INSTANCE_CHECK_STATUS.json',dict(registered=len(registry),errors=[str(x) for x in results if isinstance(x,BaseException)],time=time.time()))
        if not watch or (ROOT/'intake_queue_exit.json').exists():break
        await asyncio.sleep(30)


INPUT_REVIEW_PROMPT = '''Review a tracked-person preprocessing contact sheet. Rows are times, columns are original observation, actual white-background model input, and skeleton comparison. Only judge the MIDDLE column for segmentation: multiple people in the original observation or repeated panels are NOT themselves errors. Check that the intended person is retained, no other person's torso/head is carried into the white input, and an observed backpack/handbag is not cut out. Minor blur or uncertain fingers/feet alone are acceptable for exploratory generation. Do not infer missing hidden anatomy from occlusion. Return JSON only with exactly: identity_consistent (boolean), extra_person_present (boolean), major_body_missing (boolean), carried_item_lost (boolean or null), foreground_background_leak (boolean), action (standing/walking/running/gesture/mixed/unknown), reason (string). This is approximate triage, not final asset certification.'''


ATTACHMENT_PROMPT = '''Inspect the three-time contact sheet: LEFT is observed crop, MIDDLE is actual segmented white-background model input, RIGHT is pose comparison. Determine whether the TARGET person handles a LARGE object whose main body is outside the MIDDLE image or has been cut away, leaving truncated handles, poles, rails or other fragments. Examples include a shopping cart, stroller, bicycle, large sign or long object extending beyond the crop. A hand gripping handles that run off the image is positive evidence even when the cart itself is outside the crop. Do not confuse poles/vehicles in the LEFT background with retained foreground in the MIDDLE. Whole backpacks and ordinary handbags are allowed; a thin body-worn strap alone is not a large object. Minor blur, fingers, feet, shadows and ordinary partial body occlusion are not this defect. Do not remove an item to pretend the person never interacted with it. Return JSON only with exactly: large_handled_object (boolean or null), incomplete_in_model_input (boolean or null), evidence_slots (list drawn from 0,30,60), reason (string). If no LARGE handled object is present, including a person with only a whole backpack or normal handbag, explicitly return false for BOTH flags and [] for evidence_slots; do NOT use null for absence. Use null only when image evidence is insufficient to decide whether such an object is present/cut. Inspect all three rows. This is a conservative input screen, not final quality certification.'''


def attachment_decision(value):
    expected = {'large_handled_object', 'incomplete_in_model_input', 'evidence_slots', 'reason'}
    if set(value) != expected:
        raise ValueError('Invalid attachment response fields')
    if any(value[k] is not None and type(value[k]) is not bool
           for k in ('large_handled_object', 'incomplete_in_model_input')):
        raise ValueError('Attachment flags must be boolean or null')
    slots = value['evidence_slots']
    if not isinstance(slots, list) or any(type(i) is not int or i not in (0,30,60) for i in slots):
        raise ValueError('Invalid attachment evidence slots')
    if not isinstance(value['reason'], str) or not value['reason'].strip():
        raise ValueError('Attachment evidence reason required')
    if value['large_handled_object'] is False:
        return 'allow'
    if value['large_handled_object'] is True and value['incomplete_in_model_input'] is False:
        return 'allow'
    if value['large_handled_object'] is True and value['incomplete_in_model_input'] is True and slots:
        return 'reject'
    return 'needs_review'


async def screen_attachment(identity, client=None):
    """Opt-in input guard, repeated before diffusion even with an old controller."""
    import base64, fcntl, hashlib
    from common import case
    policy_path = ROOT/'ATTACHMENT_POLICY.json'
    exclusion = ROOT/'review'/f'{identity}_generation_exclusion.json'
    if exclusion.exists() and read(exclusion).get('excluded'):
        return dict(decision='reject', allow_generation=False, basis=str(exclusion))
    if not policy_path.exists() or read(policy_path).get('mode') != 'enforce':
        return dict(decision='not_enabled', allow_generation=True)
    if client is None:
        from openai import AsyncOpenAI
        async with AsyncOpenAI(base_url='http://127.0.0.1:8082/v1', api_key='local-only',
                               timeout=180, max_retries=0) as connection:
            return await screen_attachment(identity, connection)
    contact = ROOT/'review'/f'{identity}.jpg'
    meta = read(ROOT/'review'/f'{identity}.json')
    signature = dict(contact_sha256=sha(contact), policy_sha256=sha(policy_path),
                     input_sha256=sha(case(identity)['raw_input']),
                     prompt_sha256=hashlib.sha256(ATTACHMENT_PROMPT.encode()).hexdigest())
    assert signature['contact_sha256'] == meta['image_sha256']
    assert all(sha(s['path']) == s['sha256'] for s in meta['sources'])
    report = ROOT/'review'/f'{identity}_attachment_extent.json'
    with (ROOT/'review'/f'{identity}_attachment.lock').open('a+') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(.2)
        if report.exists() and read(report).get('signature') == signature:
            return read(report)
        response = await client.chat.completions.create(model='Qwen/Qwen3-VL-8B-Instruct',
            messages=[dict(role='system',content=ATTACHMENT_PROMPT),dict(role='user',content=[
                dict(type='text',text=identity+'; slots 0,30,60; inspect truncated handled objects in middle column'),
                dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+
                     base64.b64encode(contact.read_bytes()).decode()))])],
            response_format={'type':'json_object'},temperature=0,seed=42,max_tokens=1000)
        if response.choices[0].finish_reason != 'stop':
            raise ValueError('Incomplete attachment response')
        value = __import__('json').loads(response.choices[0].message.content)
        decision = attachment_decision(value)
        result = dict(identity=identity,decision=decision,allow_generation=decision=='allow',
                      vision=value,signature=signature,time=time.time(),
                      scope='Sampled large-object crop check; ordinary whole bags retained; no final quality certification')
        save(report,result)
        return result


async def review_inputs(watch=False):
    import base64,fcntl
    from openai import AsyncOpenAI
    from frontend import temporal_diagnosis
    from common import case
    semaphore=asyncio.Semaphore(3)
    async with AsyncOpenAI(base_url='http://127.0.0.1:8082/v1',api_key='local-only',timeout=180,max_retries=0) as client:
        async def one(identity):
            async with semaphore:
                if (ROOT/'reports'/f'{identity}_input_retired.json').exists():return
                report=ROOT/'review'/f'{identity}_admission.json'
                contact=ROOT/'review'/f'{identity}.jpg';meta=read(ROOT/'review'/f'{identity}.json')
                instance_path=ROOT/'review'/f'{identity}_instances.json'
                if not instance_path.exists():return
                # Admission evidence belongs to the original packet. An admitted
                # object's later verified temporal subset must not invalidate it.
                instances=read(instance_path);assert instances['source_sha256']==sha(registry_snapshot[identity]['raw_input'])
                signature=dict(contact_sha256=sha(contact),instance_check_sha256=sha(instance_path),source_sha256={s['path']:sha(s['path']) for s in meta['sources']})
                assert signature['contact_sha256']==meta['image_sha256']
                assert all(signature['source_sha256'][s['path']]==s['sha256'] for s in meta['sources'])
                attachment = await screen_attachment(identity, client)
                if report.exists() and read(report).get('signature')==signature:
                    if not attachment['allow_generation']:
                        prior=read(report);prior.update(allow_exploratory_generation=False,attachment=attachment)
                        save(report,prior)
                        with (ROOT/'config/registry.lock').open('a+') as lock:
                            fcntl.flock(lock,fcntl.LOCK_EX);registry=read(ROOT/'config/cases.json')
                            # Old live controllers already hold these queued IDs.
                            # The per-child guard stops them without crashing a
                            # controller while it refreshes its pending list.
                            scheduled=read(ROOT/'PRODUCTION_INPUTS.json')['ids'] if (ROOT/'PRODUCTION_INPUTS.json').exists() else []
                            if identity not in scheduled:
                                registry[identity]['input_review']['allow_generation']=False
                            registry[identity]['input_review']['attachment']=attachment
                            save(ROOT/'config/cases.json',registry)
                    return
                temporal=temporal_diagnosis(identity);save(ROOT/'reports'/f'{identity}_temporal_before_generation.json',temporal)
                value=None
                if temporal['decision']!='skip' and instances['one_person_every_sample'] and instances['opacity_admission']:
                    response=await client.chat.completions.create(model='Qwen/Qwen3-VL-8B-Instruct',
                        messages=[dict(role='system',content=INPUT_REVIEW_PROMPT),dict(role='user',content=[
                            dict(type='text',text=identity+'; compare source and middle-column input at0/30/60; note bags'),
                            dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(contact.read_bytes()).decode()))])],
                        response_format={'type':'json_object'},temperature=0,seed=42,max_tokens=1200)
                    assert response.choices[0].finish_reason=='stop'
                    value=__import__('json').loads(response.choices[0].message.content)
                    expected={'identity_consistent','extra_person_present','major_body_missing','carried_item_lost','foreground_background_leak','action','reason'}
                    assert set(value)==expected
                    assert all(type(value[k]) is bool for k in ('identity_consistent','extra_person_present','major_body_missing','foreground_background_leak'))
                    assert value['carried_item_lost'] is None or type(value['carried_item_lost']) is bool
                    assert value['action'] in ('standing','walking','running','gesture','mixed','unknown') and isinstance(value['reason'],str)
                admitted=bool(attachment['allow_generation'] and value and value['identity_consistent'] and not any(value[k] for k in ('extra_person_present','major_body_missing','foreground_background_leak')) and value['carried_item_lost'] is not True)
                save(report,dict(identity=identity,signature=signature,temporal=temporal['decision'],
                    vision=value,attachment=attachment,instance_screen=instances,allow_exploratory_generation=admitted,quality_certified=False,time=time.time(),
                    scope='Automated source/segmentation/pose triage; sampled VLM can miss defects. Not final asset or metric/ground/loop acceptance.',prompt_sha256=__import__('hashlib').sha256(INPUT_REVIEW_PROMPT.encode()).hexdigest()))
                with (ROOT/'config/registry.lock').open('a+') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX);registry=read(ROOT/'config/cases.json');spec=registry[identity]
                    # Preserve an actual direct pilot review; the automated record
                    # remains separate evidence, not a fabricated human review.
                    if not spec['input_review'].get('review_method','').startswith('direct_'):
                        spec['input_review']=dict(allow_generation=admitted,review_method='automatic_input_triage',
                            input_sha256=sha(spec['raw_input']),reviewed_slots=meta['frames'],accessory=value['carried_item_lost'] if value else 'not_checked',
                            evidence=str(report),quality_certified=False)
                        save(ROOT/'config/cases.json',registry)
        while True:
            registry_snapshot=read(ROOT/'config/cases.json') if (ROOT/'config/cases.json').exists() else {}
            ready=[i for i in registry_snapshot if (ROOT/'review'/f'{i}.jpg').is_file()]
            results=await asyncio.gather(*(one(i) for i in ready),return_exceptions=True)
            reports=[read(p) for p in (ROOT/'review').glob('HN*_admission.json')]
            save(ROOT/'INPUT_PROGRESS.json',dict(prepared=len(registry_snapshot),available_input_packages=len(ready),
                retired_inputs=len(list((ROOT/'reports').glob('HN*_input_retired.json'))),automatically_admitted=sum(p['allow_exploratory_generation'] for p in reports),
                reviewed=len(reports),errors=[str(r) for r in results if isinstance(r,BaseException)],time=time.time()))
            if not watch or (ROOT/'intake_queue_exit.json').exists():break
            await asyncio.sleep(30)


def admit_prepared():
    """Append current, screened inputs without changing already scheduled identities."""
    from common import case
    production = read(ROOT/'PRODUCTION_INPUTS.json')
    registry = read(ROOT/'config/cases.json')
    retained=[]
    for identity in production['ids']:
        exclusion=ROOT/'review'/f'{identity}_generation_exclusion.json'
        extent=ROOT/'review'/f'{identity}_attachment_extent.json'
        explicit=exclusion.exists() and read(exclusion).get('excluded')
        withheld=extent.exists() and not read(extent).get('allow_generation',False)
        started=(ROOT/'outputs'/identity/'generation').exists()
        if explicit or (withheld and not started):
            production.setdefault('withdrawn',[]).append(dict(identity=identity,
                reason='Explicit exclusion' if explicit else 'Large-object input check withheld before generation',time=time.time()))
        else:retained.append(identity)
    production['ids']=retained
    for item in read(ROOT/'selection.json')['items']:
        identity = item['id']
        if (ROOT/'reports'/f'{identity}_input_retired.json').exists():continue
        exclusion=ROOT/'review'/f'{identity}_generation_exclusion.json'
        if exclusion.exists() and read(exclusion).get('excluded'):
            continue
        if len(production['ids']) >= production['target_count']:
            break
        if identity in production['ids'] or identity not in registry:
            continue
        report_path = ROOT/'review'/f'{identity}_admission.json'
        if not report_path.exists():
            continue
        report = read(report_path)
        spec = case(identity)
        policy=ROOT/'ATTACHMENT_POLICY.json'
        if policy.exists() and read(policy).get('mode')=='enforce':
            extent=ROOT/'review'/f'{identity}_attachment_extent.json'
            if not extent.exists():continue
            evidence=read(extent)
            if not evidence['allow_generation'] or evidence['signature']['policy_sha256']!=sha(policy):continue
            if evidence['signature']['input_sha256']!=sha(spec['raw_input']):continue
            if evidence['signature'].get('prompt_sha256')!=__import__('hashlib').sha256(ATTACHMENT_PROMPT.encode()).hexdigest():continue
        signature = report.get('signature', {})
        if not (report['allow_exploratory_generation'] and spec['input_review']['allow_generation']):
            continue
        if signature.get('instance_check_sha256') != sha(ROOT/'review'/f'{identity}_instances.json'):
            continue
        if signature.get('contact_sha256') != sha(ROOT/'review'/f'{identity}.jpg'):
            continue
        if any(sha(p) != h for p, h in signature['source_sha256'].items()):
            continue
        assert sha(spec['raw_input']) == spec['input_review']['input_sha256']
        production['ids'].append(identity)
    production['input_selection_complete'] = len(production['ids']) == production['target_count']
    production['admission_scope'] = 'Screened inputs for exploratory generation, not quality-approved final assets'
    production['updated_at'] = time.time()
    save(ROOT/'PRODUCTION_INPUTS.json', production)
    return production


async def fill_production(gpus, initial_pids):
    """Finish initial intake, then replenish in bounded waves until target admission."""
    import fcntl
    with (ROOT/'jobs/intake_fill.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while any(process_running(pid) for pid in initial_pids):
            save(ROOT/'FILL_STATUS.json', dict(state='waiting_initial_input_workers', time=time.time()))
            await asyncio.sleep(30)
        while True:
            await check_registered()
            await review_inputs()
            progress = read(ROOT/'INPUT_PROGRESS.json')
            checks = read(ROOT/'INSTANCE_CHECK_STATUS.json')
            if progress['errors'] or checks['errors']:
                # Failed/missing evidence is never admitted. A single API error
                # should not block independently checked people; next wave
                # rechecks missing evidence using the same frozen protocol.
                save(ROOT/'reports/input_check_errors.json', dict(
                    review_errors=progress['errors'], instance_errors=checks['errors'],
                    policy='Unverified inputs withheld; continue valid inputs and retry missing evidence next wave',
                    time=time.time()))
            production = admit_prepared()
            if (ROOT/'REJECTED_INPUT_CLEANUP.json').exists() and read(ROOT/'REJECTED_INPUT_CLEANUP.json').get('enabled'):
                retire_rejected_inputs()
            if production['input_selection_complete']:
                save(ROOT/'FILL_STATUS.json', dict(state='target_inputs_admitted',
                     admitted=len(production['ids']), target=production['target_count'], time=time.time()))
                return
            selection = read(ROOT/'selection.json')
            prepared = {r['id'] for r in selection['items']}
            registry = read(ROOT/'config/cases.json')
            unfinished = [r['id'] for r in selection['items'] if r['id'] not in registry
                          and (not (ROOT/'jobs'/f"intake_{r['id']}.json").exists()
                               or read(ROOT/'jobs'/f"intake_{r['id']}.json")['state'] != 'failed')]
            if unfinished:
                await prepare_cases(unfinished, gpus)
                continue
            remaining = [r for r in read(ROOT/'reserve.json')['items'] if r['id'] not in prepared]
            if not remaining:
                save(ROOT/'FILL_STATUS.json', dict(state='reserve_exhausted',
                     admitted=len(production['ids']), target=production['target_count'], time=time.time()))
                return
            need = production['target_count'] - len(production['ids'])
            wave = remaining[:min(64, max(16, need * 2))]
            selection['items'].extend(wave)
            selection.setdefault('reserve_waves', []).append(dict(ids=[r['id'] for r in wave], time=time.time()))
            save(ROOT/'selection.json', selection)
            save(ROOT/'FILL_STATUS.json', dict(state='preparing_reserve_wave',
                 admitted=len(production['ids']), target=production['target_count'],
                 wave=[r['id'] for r in wave], gpus=gpus, time=time.time()))
            await prepare_cases([r['id'] for r in wave], gpus)


def retire_rejected_inputs():
    """Remove only unused rejected input payloads; keep text evidence and sources."""
    import fcntl, shutil
    from contextlib import ExitStack
    from common import ASSET_ROOT
    scheduled=set(read(ROOT/'PRODUCTION_INPUTS.json')['ids'])
    cleaned=[]
    for row in read(ROOT/'selection.json')['items']:
        identity=row['id'];receipt=ROOT/'reports'/f'{identity}_input_retired.json'
        if identity in scheduled or receipt.exists():continue
        if (ASSET_ROOT/identity).exists() or (ROOT/'outputs'/identity).exists():continue
        job=ROOT/'jobs'/f'intake_{identity}.json'
        admission=ROOT/'review'/f'{identity}_admission.json'
        extent=ROOT/'review'/f'{identity}_attachment_extent.json'
        job_record=read(job) if job.exists() else {}
        review=read(admission) if admission.exists() else {}
        attachment=read(extent) if extent.exists() else {}
        technical_failure=job_record.get('state')=='failed'
        visual_rejection=review.get('allow_exploratory_generation') is False
        # Unknown/API error is not a failed-quality decision.
        if attachment.get('decision')=='needs_review':continue
        if not (technical_failure or visual_rejection or attachment.get('decision')=='reject'):continue
        with ExitStack() as guards:
            try:
                for name in (f'intake_{identity}.lock',f'{identity}.lock'):
                    lock=guards.enter_context((ROOT/'jobs'/name).open('a+'))
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            # Recheck after acquiring the same locks used by the workers.
            if identity in read(ROOT/'PRODUCTION_INPUTS.json')['ids']:continue
            if (ASSET_ROOT/identity).exists() or (ROOT/'outputs'/identity).exists():continue
            record=dict(identity=identity,state='cleanup_started',time=time.time(),
                reason='technical_preparation_failure' if technical_failure else 'input_screen_rejected',
                prior_job=job_record,admission=review,attachment=attachment,
                source_identity_key=row['identity_key'],source_records='selection.json',removed=[])
            save(receipt,record)
            save(ROOT/'review'/f'{identity}_generation_exclusion.json',dict(excluded=True,
                reason=record['reason'],retirement_record=str(receipt),time=time.time()))
            for path in (ROOT/'inputs'/identity,ROOT/'data'/identity):
                if not path.exists():continue
                assert path.resolve()==path and path.name==identity and not path.is_symlink()
                files=[p for p in path.rglob('*') if p.is_file() and not p.is_symlink()]
                record['removed'].append(dict(path=str(path),files=len(files),bytes=sum(p.stat().st_size for p in files)))
                shutil.rmtree(path)
            contact=ROOT/'review'/f'{identity}.jpg'
            if contact.exists():
                record['removed'].append(dict(path=str(contact),bytes=contact.stat().st_size,sha256=sha(contact)))
                contact.unlink()
            record.update(state='retired',bytes_removed=sum(p['bytes'] for p in record['removed']))
            save(receipt,record);cleaned.append(dict(identity=identity,bytes_removed=record['bytes_removed']))
    return cleaned


def expansion_quality_signature():
    """Freeze the actual screen implementation/configs, not an advertised label."""
    import hashlib, inspect
    from frontend import classify_temporal_signals
    functions=[rank,instance_check,attachment_decision,classify_temporal_signals]
    return dict(functions={f.__name__:hashlib.sha256(inspect.getsource(f).encode()).hexdigest() for f in functions},
        prompts={name:hashlib.sha256(value.encode()).hexdigest() for name,value in
                 [('input',INPUT_REVIEW_PROMPT),('attachment',ATTACHMENT_PROMPT)]},
        files={name:sha(ROOT/name) for name in ['TEMPORAL_POLICY.json','ATTACHMENT_POLICY.json','config/recipe.json','config/export.json']})


def raise_target(target):
    import fcntl
    with (ROOT/'jobs/replenish.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        production=read(ROOT/'PRODUCTION_INPUTS.json')
        assert len(production['ids'])<=production['target_count']<=target<=300
        production.setdefault('target_history',[]).append(dict(previous=production['target_count'],target=target,
            basis='User authorized 200–300 only at unchanged quality standards; stop if suitable candidates exhausted',time=time.time()))
        production.update(target_count=target,input_selection_complete=len(production['ids'])>=target,updated_at=time.time())
        production.pop('input_shortfall',None);production.pop('completion_reason',None)
        save(ROOT/'PRODUCTION_INPUTS.json',production)
    return production


async def expand_production(target,limit,gpus,initial_pids):
    """One finite remaining pool; stop short instead of relaxing screening."""
    import fcntl,os
    with (ROOT/'jobs/expansion.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record_path=ROOT/'EXPANSION.json';staging=ROOT/'reports/reserve_expansion_pool.json'
        policy=expansion_quality_signature()
        if record_path.exists():
            record=read(record_path);assert record['quality_signature']==policy
            assert record['target']==target and record['candidate_limit']==limit
        else:
            record=dict(state='ranking_remaining_candidates',target=target,candidate_limit=limit,
                quality_signature=policy,time=time.time(),pid=os.getpid(),
                quality_policy='Same thresholds; no relaxed occlusion, pose, segmentation, bag or frame-completeness gates',
                stopping_policy='Stop at target or exhaust the finite admissible pool; smaller totals are acceptable')
            save(record_path,record)
        if not record.get('reserve_merged'):
            previous=read(ROOT/'reserve.json');previous_hash=sha(ROOT/'reserve.json')
            if not staging.exists():rank(limit,reserve=True,destination=staging,excluded=previous['items'])
            extension=read(staging)
            record.update(state='waiting_current_intake',ranked_additional=len(extension['items']))
            save(record_path,record)
            while any(process_running(pid) for pid in initial_pids):await asyncio.sleep(30)
            assert expansion_quality_signature()==policy
            assert sha(ROOT/'reserve.json')==previous_hash,'Reserve changed while ranking; inspect before merging'
            merged=previous['items']+extension['items']
            assert len({r['id'] for r in merged})==len(merged)
            assert len({r['instance_token'] for r in merged})==len(merged)
            assert previous['rule']==extension['rule'],'Expansion may not lower selection thresholds'
            previous['items']=merged
            previous.setdefault('expansions',[]).append(dict(count=len(extension['items']),time=time.time(),
                pool_sha256=sha(staging),scope='New unique identities, same filters; scene cap includes prior pools'))
            save(ROOT/'reserve.json',previous)
            record.update(reserve_merged=True,additional_pool_sha256=sha(staging),state='expanded_input_preparation')
            save(record_path,record);staging.unlink()
        raise_target(target)
        save(ROOT/'REJECTED_INPUT_CLEANUP.json',dict(enabled=True,time=time.time(),
            scope='Unscheduled, unlocked rejected input payloads only; source dataset and text provenance retained',
            user_authorization='筛选不过关的记录下来后删除有关输出；不降低质量门槛'))
        retire_rejected_inputs()
        await fill_production(gpus,[])
        assert expansion_quality_signature()==policy
        production=read(ROOT/'PRODUCTION_INPUTS.json')
        reached=len(production['ids'])>=target
        if not reached:
            production.update(input_selection_complete=True,input_shortfall=target-len(production['ids']),
                completion_reason='Quality-constrained finite pool exhausted; screening stopped without relaxing rules')
            save(ROOT/'PRODUCTION_INPUTS.json',production)
        record.update(state='target_inputs_admitted' if reached else 'stopped_at_available_quality',
            admitted=len(production['ids']),finished_at=time.time())
        save(record_path,record)
        save(ROOT/'FILL_STATUS.json',dict(state=record['state'],admitted=record['admitted'],target=target,time=time.time()))


async def maintain_inputs(gpus, initial_pids):
    """Finish the target after the current filler; permit one bounded reserve extension."""
    while any(process_running(pid) for pid in initial_pids):
        await asyncio.sleep(30)
    policy=ROOT/'ATTACHMENT_POLICY.json'
    if policy.exists() and read(policy).get('mode')=='enforce':
        await review_inputs()
        admit_prepared()
    production = read(ROOT/'PRODUCTION_INPUTS.json')
    if len(production['ids']) >= production['target_count']:
        return
    await fill_production(gpus, [])
    production = read(ROOT/'PRODUCTION_INPUTS.json')
    if len(production['ids']) >= production['target_count']:
        return
    extension = ROOT/'reports/additional_reserve.json'
    if not extension.exists():
        previous = ROOT/'reports/initial_reserve_exhausted.json'
        assert not previous.exists(), 'Inspect interrupted reserve extension before resuming'
        (ROOT/'reserve.json').replace(previous)
        rank(150, reserve=True)
        save(extension, dict(previous_reserve=str(previous), new_reserve_sha256=sha(ROOT/'reserve.json'),
             count=len(read(ROOT/'reserve.json')['items']), time=time.time(),
             scope='One additional ranked pool; unchanged filters and no repeated identities'))
        await fill_production(gpus, [])
    production = read(ROOT/'PRODUCTION_INPUTS.json')
    if len(production['ids']) < production['target_count']:
        production.update(input_selection_complete=True,
                          input_shortfall=production['target_count']-len(production['ids']),
                          completion_reason='Bounded ranked reserves exhausted; no weaker inputs admitted')
        save(ROOT/'PRODUCTION_INPUTS.json', production)
        save(ROOT/'FILL_STATUS.json', dict(state='selection_finished_with_shortfall',
             admitted=len(production['ids']), target=production['target_count'], time=time.time()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['rank','rank-reserve','native','interpolate','segment','register','prepare','review','instance','checks','fill','admit','maintain','attachment','expand','cleanup'])
    parser.add_argument('identity',nargs='?')
    parser.add_argument('--limit',type=int,default=100)
    parser.add_argument('--target',type=int,default=300)
    parser.add_argument('--cases',nargs='+');parser.add_argument('--gpus',nargs='+',type=int)
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--exclude',nargs='*',default=[])
    parser.add_argument('--wait-pids',nargs='*',type=int,default=[])
    args=parser.parse_args()
    if args.stage in ('rank','rank-reserve'):rank(args.limit,reserve=args.stage=='rank-reserve')
    elif args.stage=='attachment':
        result=asyncio.run(screen_attachment(args.identity))
        print(__import__('json').dumps(result,ensure_ascii=False),flush=True)
        if not result['allow_generation']:raise RuntimeError(f'Attachment screening withheld {args.identity}')
    elif args.stage=='review':asyncio.run(review_inputs(args.watch))
    elif args.stage=='checks':asyncio.run(check_registered(args.watch))
    elif args.stage=='instance':instance_check(args.identity)
    elif args.stage=='admit':admit_prepared()
    elif args.stage=='fill':asyncio.run(fill_production(args.gpus,args.wait_pids))
    elif args.stage=='maintain':asyncio.run(maintain_inputs(args.gpus,args.wait_pids))
    elif args.stage=='expand':asyncio.run(expand_production(args.target,args.limit,args.gpus,args.wait_pids))
    elif args.stage=='cleanup':print(__import__('json').dumps(retire_rejected_inputs(),ensure_ascii=False))
    elif args.stage=='prepare':asyncio.run(prepare_cases([r['id'] for r in read(ROOT/'selection.json')['items'] if r['id'] not in args.exclude] if args.all else args.cases,args.gpus))
    else:globals()[args.stage](args.identity)

if __name__=='__main__':main()
