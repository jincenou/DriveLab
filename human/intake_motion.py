"""Calibrated GVHMR and fixed root-local 24-view conditioning for dataset intake.

Ported from the frozen G0 implementation; source RGB, camera rotations and crop K
are explicit inputs. World translation is not measured by GVHMR.
"""
import sys, json
from pathlib import Path
from fractions import Fraction
import numpy as np
from PIL import Image, ImageDraw
from common import ROOT, CODE_ROOT, FRONTEND as REPO, read, save, sha, gpu_identity
from prepare import encode as encode_video

def encode(path, images, fps=25):
    assert fps == 25
    encode_video(path, images)


def run_root(identity):
    return ROOT/"inputs"/identity/"clip0"

"""Explicit horizontal-root coordinate change before skeleton projection."""
from dataclasses import replace

def separate_horizontal_root(geometry):
    root=np.asarray(geometry.joints_world[:,0],dtype=np.float64)
    delta=root-root[:1];delta[:,1]=0
    transform=np.repeat(np.eye(4)[None],len(root),axis=0);transform[:,:3,3]=delta
    def local(points):return (np.asarray(points,dtype=np.float64)-delta[:,None,:]).astype(np.float32)
    updated=replace(geometry,vertices_world=local(geometry.vertices_world),joints_world=local(geometry.joints_world),
        keypoints_world=local(geometry.keypoints_world),mesh_world=None if geometry.mesh_world is None else local(geometry.mesh_world))
    # This original constant transform is provenance, not the new local map.
    # local_from_motion(t) is stored explicitly and must be used by consumers.
    inverse=np.linalg.inv(transform)
    maps={'local_to_canonical':transform,'canonical_to_local':inverse,
          'motion_to_local':inverse@np.asarray(geometry.motion_world_to_canonical_world)[None]}
    checks={}
    for name in ('vertices_world','joints_world','keypoints_world','mesh_world'):
        original=getattr(geometry,name);modified=getattr(updated,name)
        if original is None:continue
        checks[name+'_recomposition_max_abs']=float(np.max(np.abs(modified+delta[:,None,:]-original)))
        checks[name+'_vertical_max_abs']=float(np.max(np.abs(modified[...,1]-original[...,1])))
    vectors=geometry.joints_world-geometry.joints_world[:,:1]
    local_vectors=updated.joints_world-updated.joints_world[:,:1]
    checks['root_relative_joint_vectors_max_abs']=float(np.max(np.abs(vectors-local_vectors)))
    checks['horizontal_root_residual_max_abs']=float(np.max(np.abs(updated.joints_world[:,0][:,[0,2]]-updated.joints_world[0,0,[0,2]])))
    checks['horizontal_root_path_removed']=float(np.linalg.norm(np.diff(delta,axis=0),axis=1).sum())
    assert max(v for k,v in checks.items() if k.endswith('_max_abs'))<2e-5,checks
    assert np.array_equal(updated.keypoints_incam,geometry.keypoints_incam)
    return updated,maps,checks


def source_camera_maps(original,local,local_to_canonical):
    # Infer the rigid relation between the SAME SMPL body in two stored spaces.
    # This consistency check does not recover measured dataset-world calibration.
    a=np.asarray(original.keypoints_world,dtype=np.float64);b=np.asarray(original.keypoints_incam,dtype=np.float64)
    ac=a.mean(1,keepdims=True);bc=b.mean(1,keepdims=True)
    u,_,vt=np.linalg.svd(np.swapaxes(a-ac,1,2)@(b-bc))
    fix=np.repeat(np.eye(3)[None],len(a),axis=0);fix[:,2,2]=np.linalg.det(np.swapaxes(vt,1,2)@np.swapaxes(u,1,2))
    rot=np.swapaxes(vt,1,2)@fix@np.swapaxes(u,1,2)
    trans=bc[:,0]-np.einsum('tij,tj->ti',rot,ac[:,0])
    canonical_to_incam=np.repeat(np.eye(4)[None],len(a),axis=0);canonical_to_incam[:,:3,:3]=rot;canonical_to_incam[:,:3,3]=trans
    local_to_incam=canonical_to_incam@local_to_canonical
    projected=np.einsum('tij,tkj->tki',local_to_incam[:,:3,:3],local.keypoints_world)+local_to_incam[:,None,:3,3]
    error=float(np.max(np.abs(projected-b)))
    if error>=1e-4:raise ValueError(f'Global/incam SMPL geometry is not a consistent rigid pair: max error {error}')
    return local_to_incam,error

"""Numerically stable evaluation of the unchanged published vertex-regressor weights."""
import inspect,hashlib

def stable_geometry(original):
 source=inspect.getsource(original)
 for space in ['global','incam']:
  before=f'(vertices_{space}[:, support] * weights[None, :, :, None]).sum(dim=2)'
  after=f'(vertices_{space}[:, support].double() * weights.double()[None, :, :, None]).sum(dim=2).to(vertices_{space}.dtype)'
  assert before in source
  source=source.replace(before,after)
 scope=dict(original.__globals__);exec(compile(source,'<random8 stable vertex regression>','exec'),scope);function=scope[original.__name__];function.random8_precision_source_sha256=hashlib.sha256(source.encode()).hexdigest();return function

COCO_NAMES=('nose','left-eye','right-eye','left-ear','right-ear','left-shoulder','right-shoulder',
    'left-elbow','right-elbow','left-wrist','right-wrist','left-hip','right-hip','left-knee','right-knee','left-ankle','right-ankle')


def estimate(identity):
    ROOT=run_root(identity); COUNT=61
    sys.path.insert(0,str(REPO))
    import torch
    from fdanyone.motion import gvhmr
    from fdanyone.motion.result import MotionResult,SMPL_PARAMETER_NAMES
    from fdanyone.video import decode_canonical_clip,write_gvhmr_video
    from fdanyone.skeleton.pipeline import _body_geometry
    from fdanyone.skeleton.keypoints import KEYPOINT_NAMES,LINKS
    from fdanyone.assets import resolve_regressor
    MotionResult.validate.__defaults__=(COUNT,)
    torch.set_num_threads(4);device=gpu_identity()
    root,revision=gvhmr.validate_gvhmr(REPO/'third_party/GVHMR')
    mapping=json.loads((ROOT/'inputs/frame_mapping.json').read_text())
    if sha(ROOT/'inputs/main.mp4')!=mapping['video_sha256'] or sha(ROOT/'inputs/cameras.npz')!=mapping['camera_npz_sha256']:
        raise ValueError('Input or camera data changed')
    out=ROOT/'motion_pair';out.mkdir(exist_ok=False);save(out/'device.json',device)
    clip=decode_canonical_clip(ROOT/'inputs/main.mp4',num_frames=COUNT,start_time=0,fps=Fraction(25))
    clip.write_metadata(out/'canonical_clip.json');video=write_gvhmr_video(clip,out/'canonical_clip.mp4')
    camera=np.load(ROOT/'inputs/cameras.npz');boxes=torch.tensor([x['bbox_input_xyxy'] for x in mapping['frames']],dtype=torch.float32)
    results={}
    with gvhmr.gvhmr_imports(root),gvhmr._legacy_checkpoint_loading():
        gvhmr.install_pytorch3d_compat()
        # Register optional imports before importing preproc.__init__. We supply
        # measured camera rotations; SimpleVO remains unavailable and is not called.
        gvhmr._register_inference_store()
        import hydra
        from hydra import compose,initialize_config_module
        from omegaconf import open_dict
        from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy,estimate_K
        from hmr4d.utils.geo_transform import compute_cam_angvel
        from hmr4d.utils.preproc.vitpose import VitPoseExtractor
        from hmr4d.utils.preproc.vitfeat_extractor import Extractor
        from hmr4d.utils.net_utils import detach_to_cpu
        from hmr4d.utils.video_io_utils import get_video_reader
        with initialize_config_module(version_base='1.3',config_module='hmr4d.configs'):
            cfg=compose(config_name='demo',overrides=[gvhmr.hydra_override('video_name','calibrated'),
                'static_cam=false','verbose=false','use_dpvo=false',gvhmr.hydra_override('output_root',out/'runtime')])
        with open_dict(cfg):cfg.video_path=str(video)
        Path(cfg.preprocess_dir).mkdir(parents=True,exist_ok=False)
        gvhmr._verify_gvhmr_decode(clip,video,get_video_reader)
        bbx_xys=get_bbx_xys_from_xyxy(boxes,base_enlarge=1.2).float()
        torch.save({'bbx_xyxy':boxes,'bbx_xys':bbx_xys},cfg.paths.bbx)
        # Dataset identity boxes are explicit inputs; no tracker substitution.
        extractor=VitPoseExtractor();keypoints=extractor.extract(str(video),bbx_xys);del extractor
        torch.save(keypoints,cfg.paths.vitpose)
        extractor=Extractor();features=extractor.extract_video_features(str(video),bbx_xys);del extractor
        torch.save(features,cfg.paths.vit_features);torch.cuda.empty_cache()
        if tuple(keypoints.shape)!=(COUNT,17,3):raise ValueError('Unexpected detector keypoint format')
        model=hydra.utils.instantiate(cfg.model,_recursive_=False)
        checkpoint=torch.load(cfg.ckpt_path,map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['state_dict'],strict=True);del checkpoint
        model=model.eval().cuda()
        shared={'length':torch.tensor(COUNT),'bbx_xys':bbx_xys,'kp2d':keypoints,'f_imgseq':features}
        variants={'calibrated_moving':{'K_fullimg':torch.from_numpy(camera['K_input']).float(),'cam_angvel':compute_cam_angvel(torch.from_numpy(camera['R_world_to_cv_camera']).float())}}
        for name,calibration in variants.items():
            if not all(torch.isfinite(v).all() for v in calibration.values()):raise ValueError('Invalid camera tensors')
            with torch.inference_mode():prediction=detach_to_cpu(model.predict({**shared,**calibration},static_cam=name!='calibrated_moving'))
            motion=MotionResult(gvhmr_revision=revision,fps=clip.fps,
                frame_timestamps_sec=tuple(float(f.canonical_timestamp) for f in clip.frames),
                source_frame_indices=tuple(f.source_index for f in clip.frames),source_pts=tuple(f.source_pts for f in clip.frames),
                source_size_bytes=clip.source_size_bytes,source_mtime_ns=clip.source_mtime_ns,
                image_height=clip.height,image_width=clip.width,
                smpl_params_global={k:prediction['smpl_params_global'][k] for k in SMPL_PARAMETER_NAMES},
                smpl_params_incam={k:prediction['smpl_params_incam'][k] for k in SMPL_PARAMETER_NAMES},
                K_fullimg=prediction['K_fullimg'],observed_keypoints_2d=keypoints.detach().cpu())
            motion.validate_against_clip(clip);motion.save(out/name);results[name]=motion
            save(out/name/'calibration.json',{'static_cam':name!='calibrated_moving','strict_checkpoint_load':True,
                'motion_coordinate_world':'gvhmr_gravity_aligned_y_up; not registered nuScenes world',
                'camera_translation_consumed':False,'camera_npz':str(ROOT/'inputs/cameras.npz') if name!='static_estimated' else None,
                'source_video_sha256':mapping['video_sha256'],'known_target_boxes':True,'tracker_used':False})
        del model,features,shared;torch.cuda.empty_cache()
    projected={};metrics={};observed=keypoints.detach().cpu().numpy();coco_ids=[KEYPOINT_NAMES.index(n) for n in COCO_NAMES]
    for name,motion in results.items():
        geometry=_body_geometry(motion,resolve_regressor(None,model_dir=REPO/'models'),root,'cuda:0')
        points=geometry.keypoints_incam;K=motion.K_fullimg.numpy();xyz=np.einsum('tij,tnj->tni',K,points)
        if not np.isfinite(xyz).all() or (points[...,2]<=0).any():raise ValueError('Nonfinite or behind-camera body projection')
        uv=xyz[...,:2]/xyz[...,2:];projected[name]=uv
        distance=np.linalg.norm(uv[:,coco_ids]-observed[...,:2],axis=-1)
        trusted=observed[...,2]>.3;heights=boxes.numpy()[:,3]-boxes.numpy()[:,1]
        if not trusted.any():raise ValueError('No confident source detections')
        normalized=distance/heights[:,None]
        metrics[name]={'median_detector_reprojection_px':float(np.median(distance[trusted])),
            'p95_detector_reprojection_px':float(np.percentile(distance[trusted],95)),
            'median_detector_reprojection_over_bbox_height':float(np.median(normalized[trusted])),
            'p95_detector_reprojection_over_bbox_height':float(np.percentile(normalized[trusted],95)),
            'confident_points_per_frame_min_median':[int(trusted.sum(1).min()),float(np.median(trusted.sum(1)))],
            'global_root_span_xyz':np.ptp(motion.smpl_params_global['transl'].numpy(),axis=0).tolist(),
            'diagnostic_only':'Error versus detected COCO17 points, not ground-truth body pose; manual overlay required'}
        np.savez_compressed(out/name/'projection.npz',keypoints_uv=uv,observed_coco17=observed,
            confidence_mask=trusted,detector_distance_px=distance,source_boxes=boxes.numpy())
    panels=[]
    for frame in range(COUNT):
        images=[]
        for name in ('source','calibrated_moving'):
            im=Image.fromarray(clip.frames[frame].rgb.copy());draw=ImageDraw.Draw(im)
            if name!='source':
                uv=projected[name][frame]
                for _,first,second,_,major in LINKS:
                    if major:draw.line([tuple(uv[first]),tuple(uv[second])],fill='lime',width=2)
                for u,v,confidence in observed[frame]:
                    if confidence>.3:draw.ellipse((u-2,v-2,u+2,v+2),fill='red')
            draw.rectangle((0,0,clip.width,22),fill='black');draw.text((4,4),f'{name} f{frame}',fill='white');images.append(im)
        sheet=Image.new('RGB',(clip.width*2,clip.height))
        for col,im in enumerate(images):sheet.paste(im,(col*clip.width,0))
        panels.append(sheet)
    encode(out/'projection_comparison.mp4',panels,25)
    import av
    with av.open(str(out/'projection_comparison.mp4')) as container:
        decoded=list(container.decode(video=0))
        if len(decoded)!=COUNT or any(f.width!=clip.width*2 or f.height!=clip.height for f in decoded):raise ValueError('Incomplete projection video')
    save(out/'comparison.json',{'state':'calibrated_motion_inferred_projection_review_pending','device':device,
        'gvhmr_revision':revision,'input_sha256':mapping['video_sha256'],'metrics':metrics,
        'protocol':['Per-frame dataset camera K with translation crop',
            'Actual exposure-aligned camera rotation increments',
            'GVHMR moving-camera postprocessing; no static-camera comparison run'],
        'video_sha256':sha(out/'projection_comparison.mp4'),'video_frames_decoded':COUNT,
        'color_legend':{'green':'regressed SMPL-X skeleton projection','red':'ViTPose detections'},
        'generation_started':False,'source_projection_approved':False,
        'limitations':['SAM foreground source with dataset-identity boxes; inference diagnostic only',
            'Camera translation is not consumed by GVHMR and world metric trajectory is not verified',
            'Single exposure-center pose approximates rolling shutter; human fitting accuracy remains to be inspected']})
    print(json.dumps({'state':'motion_pair_ready','metrics':metrics}),flush=True)



def condition(identity):
    from dataclasses import replace
    import torch
    from fdanyone.skeleton import pipeline as skeleton
    from fdanyone.motion.result import MotionResult
    from fdanyone.video import decode_canonical_clip
    from fdanyone.assets import resolve_regressor,resolve_foreground_model
    from fdanyone.views import resolve_view_plan
    ORIGINAL=run_root(identity);ROOT=ORIGINAL.parent/'conditions';COUNT=61;VIEWS=24
    MotionResult.validate.__defaults__=(COUNT,)
    variant='calibrated_moving'; motion_dir=ORIGINAL/'motion_pair'/variant
    assert (motion_dir/'projection.npz').is_file()
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'inputs').mkdir(exist_ok=True);(ROOT/'preflight').mkdir(exist_ok=True);(ROOT/'research').mkdir(exist_ok=True)
    (ROOT/'inputs/main.mp4').symlink_to(ORIGINAL/'inputs/main.mp4')
    (ROOT/'preflight/motion').symlink_to(motion_dir,target_is_directory=True)
    sys.path.insert(0,str(REPO));device=gpu_identity()
    clip=decode_canonical_clip(ROOT/'inputs/main.mp4',num_frames=COUNT,start_time=0,fps=Fraction(25))
    motion=MotionResult.load(ROOT/'preflight/motion');motion.validate_against_clip(clip)
    original_geometry=stable_geometry(skeleton._body_geometry);original_framing=skeleton.solve_clip_framing
    original_masks=skeleton.predict_foreground_masks;record={'preprocessing_protocol':'random8-frontend-v1.2','regression_precision':'same weights,float64 products/sum then originalfloat32','geometry_function_sha256':original_geometry.random8_precision_source_sha256};geometry_cache={}
    def geometry(*args,**kwargs):
        g=original_geometry(*args,**kwargs);local,maps,checks=separate_horizontal_root(g)
        source_map,error=source_camera_maps(g,local,maps['local_to_canonical'])
        maps['local_to_smpl_incam']=source_map
        np.savez_compressed(ROOT/'preflight/root_motion.npz',**maps,frame_times=np.asarray(motion.frame_timestamps_sec))
        geometry_cache.update(original=g,local=local)
        record.update(root_checks=checks,source_rigid_recomposition_error=error)
        return local
    def masks(*args,**kwargs):
        result=original_masks(*args,**kwargs);np.savez_compressed(ROOT/'preflight/source_masks.npz',masks=result);return result
    def framing(points,names,profile,camera_factory,aspect_ratio):
        base=original_framing(points,names,profile,camera_factory,aspect_ratio)
        def inside(radius):
            for c in camera_factory(radius,base.target_height):
                crop=skeleton.center_crop(c.image_height,c.image_width,1280,704)
                k=np.asarray(c.K).copy();k[0,0]=k[1,1]=base.focal_normalized*c.image_height
                c=skeleton._cropped_camera(replace(c,K=tuple(map(tuple,k))),crop)
                w=np.asarray(c.world_to_camera);p=points@w[:3,:3].T+w[:3,3]
                q=p@np.asarray(c.K).T;uv=q[...,:2]/q[...,2:]
                # 1e-5 interior guard prevents the binary-search boundary from
                # rounding outside the independently checked 6% margin.
                if not ((p[...,2]>0)&(uv[...,0]>=.06001*c.image_width)&(uv[...,0]<=.93999*c.image_width)&(uv[...,1]>=.06001*c.image_height)&(uv[...,1]<=.93999*c.image_height)).all():return False
            return True
        low=.1*base.radius;high=base.radius
        while not inside(high):
            high*=1.15
            if high>2*base.radius:raise RuntimeError('Root-local branch still needs excessive camera radius')
        for _ in range(24):
            mid=(low+high)/2
            if inside(mid):high=mid
            else:low=mid
        record['framing']={'initial_radius':base.radius,'local_all_time_radius':high,'margin_fraction':.06}
        return replace(base,radius=high,radius_solve=replace(base.radius_solve,radius=high,bound=None,limiting_constraint='root_local_all_time_joint_bounds'))
    skeleton._body_geometry=geometry;skeleton.solve_clip_framing=framing;skeleton.predict_foreground_masks=masks
    target=ROOT/'preflight/conditioning'
    skeleton.build_skeleton_conditioning(motion=motion,clip=clip,regressor_path=resolve_regressor(None,model_dir=REPO/'models'),
        foreground_model_path=resolve_foreground_model(REPO/'models'),gvhmr_root=REPO/'third_party/GVHMR',output_dir=target,device='cuda:0',
        view_plan=resolve_view_plan(views_per_layer=VIEWS,layer_pitches=[15],start_yaw=0,yaw_span=360,enable_rcp=VIEWS==24,enable_tcr=True))
    payload=json.loads((target/'cameras.json').read_text());meta=json.loads((target/'metadata.json').read_text())
    payload['original_motion_world_to_canonical_world']=payload.pop('motion_world_to_canonical_world')
    payload['world_frame']={**payload['world_frame'],'name':'root_local_human_world','root_translation':'per-frame horizontal displacement stored separately'}
    payload['motion_world_to_local_world_file']='../root_motion.npz';save(target/'cameras.json',payload)
    points=geometry_cache['local'].keypoints_world;coverage=[]
    for c in payload['cameras']:
        w=np.asarray(c['world_to_camera']);p=points@w[:3,:3].T+w[:3,3];q=p@np.asarray(c['K']).T;uv=q[...,:2]/q[...,2:]
        good=(p[...,2]>0)&(uv[...,0]>=.06*c['image_width'])&(uv[...,0]<=.94*c['image_width'])&(uv[...,1]>=.06*c['image_height'])&(uv[...,1]<=.94*c['image_height'])
        coverage.append({'view':c['camera_id'],'all_joints_all_frames_inside':bool(good.all()),'median_joint_height_fraction':float(np.median(np.ptp(uv[...,1],axis=1))/c['image_height'])})
    crop=meta['source_crop'];source_masks=np.load(ROOT/'preflight/source_masks.npz')['masks']>12.75
    retained=source_masks[:,crop['top']:crop['top']+crop['height'],crop['left']:crop['left']+crop['width']].sum((1,2))/np.maximum(source_masks.sum((1,2)),1)
    assert all(c['all_joints_all_frames_inside'] for c in coverage) and retained.min()>=.95,(coverage,retained.min())
    record.update(state='geometry_and_source_retention_passed_generation_not_requested',source_experiment=str(ORIGINAL),
        source_video_sha256=sha(ORIGINAL/'inputs/main.mp4'),root_motion_sha256=sha(ROOT/'preflight/root_motion.npz'),coverage=coverage,
        source_motion_hashes={name:sha(motion_dir/name) for name in ('motion.json','motion.safetensors')},
        source_mask_retained_min=float(retained.min()),source_motion_reused_without_modification=True,motion_variant=variant,motion_source=str(motion_dir),world_frame='root_local_human_world',
        limitations=['Experimental transformation; generated appearance/motion still need review','Camera translation is not consumed; root trajectory is estimated and not nuScenes world registered','Not arbitrary SMPL-X retargeting'])
    save(ROOT/'research/root_local.json',record)
    save(ROOT/'preflight/review.json',{'allow_generation':False,'conditioning_prepared':True,'scope':f'{VIEWS}-view skeleton conditions only; RGB generation not requested','input_visual_review':'partial; see pilot review','evidence':'../research/root_local.json'})
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['motion','condition']);p.add_argument('identity');args=p.parse_args()
    sys.path.insert(0,str(REPO))
    (estimate if args.stage=='motion' else condition)(args.identity)
