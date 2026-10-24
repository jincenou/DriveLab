"""Identity and coverage checks independent of detector/model quality."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from asset_extraction.vision import (
    _appearance, _bbox_iou, _cross_camera_tracklets, _mask_quality, _nms, _source_record,
    _tracklets, collect_frame_images, _motion_boxes, ImageCache,
    run_scene_asset_export, class_for_label,
    _supplement_vehicle_views, FrameImage, _stitch_static_tracks,
)
from nuscenes.quality_filter import filter_manifest, _same_source_identity, _same_frame_mask, _cross_frame_match, _static_source_match, _same_foreground_across_categories
from nuscenes.prepare_handoff import prepare
from nuscenes.prepare_routes import build_routes
from asset_extraction.real_views import select_real_views


def observation(frame, x=0, category='car', camera='front', color=None):
    row = dict(frame_id=frame, camera_name=camera, category=category,
               bbox_xyxy=[x, 0, x+20, 20], image_path=f'{camera}_{frame}.jpg', score=.9)
    if color is not None:
        row['appearance'] = color
    return row


def calibrated_track(camera, camera_x, point_x=0., color=None):
    track = []
    for frame in range(3):
        transform = np.eye(4)
        transform[0, 3] = camera_x + .5 * frame
        x = 100 * (point_x-transform[0, 3])/10 + 100
        row = observation(frame, x-10, category='cone', camera=camera, color=color or [1.,0.])
        row.update(bbox_xyxy=[x-10,90,x+10,110], timestamp_us=frame*500_000,
                   camera_intrinsic=[[100,0,100],[0,100,100],[0,0,1]], camera_to_world=transform.tolist())
        track.append(row)
    return track


class TrackingTests(unittest.TestCase):
    def test_detection_replay_requires_matching_extraction_settings(self):
        from dataclasses import asdict
        from asset_extraction.config import PipelineConfig
        from asset_extraction.pipeline import _scene_export_config
        with TemporaryDirectory() as temp:
            root=Path(temp)
            cfg=PipelineConfig('rigid',root,root,'replay',grounded_sam2_root=root,
                               dataset_root=root,detection_cache_root=root/'old')
            cached=root/'old/scenes/scene/logs/detections.json';cached.parent.mkdir(parents=True)
            cached.write_text('[]')
            signature=root/'old/input_signature.json'
            signature.write_text(json.dumps(dict(config=asdict(cfg)),default=str))
            export=_scene_export_config(cfg,'scene',root/'scene.json',root/'new')
            self.assertEqual(export.detection_cache,cached)
            cfg.thresholds.box=.4
            with self.assertRaisesRegex(ValueError,'Detection cache settings differ'):
                _scene_export_config(cfg,'scene',root/'scene.json',root/'new')

    def test_intermediate_refs_require_foreground_flow_and_sam_agreement(self):
        import cv2
        with TemporaryDirectory() as temp:
            root=Path(temp);paths=[]
            texture=np.random.default_rng(8).integers(20,200,(120,140,3),dtype='uint8')
            texture=cv2.GaussianBlur(texture,(3,3),.5)
            for i in range(4):
                rgb=np.full((240,320,3),255,dtype='uint8');rgb[60:180,90+i*5:230+i*5]=texture
                path=root/f'{i}.png';Image.fromarray(rgb).save(path);paths.append(str(path))
            mask=np.zeros((240,320),bool);mask[60:180,90:230]=True
            anchor=dict(observation(0),image_path=paths[0],bbox_xyxy=[90,60,230,180],
                tracklet_id='t1',quality=.9,pre_score=.9,real_view_eligible=True,_mask=mask)
            frames=[FrameImage(paths[0],'front',0),FrameImage(paths[3],'front',1,motion_paths=paths[1:3])]
            cfg=SimpleNamespace(vehicle_sam_candidates=6,real_views_max=4,min_mask_area_ratio=.0005,
                max_mask_area_ratio=.8,min_largest_component_ratio=.6,max_edge_touch_ratio=.35)
            class Segments:
                wrong=False
                def segment_loaded_image(self,rgb,boxes):
                    foreground=np.any(rgb<240,axis=-1) if not self.wrong else np.zeros(rgb.shape[:2],bool)
                    return [SimpleNamespace(mask=foreground,sam_score=.95) for _ in boxes]
            adapter=Segments()
            kept,attempted=_supplement_vehicle_views([anchor],[anchor],[anchor],frames,ImageCache(8),adapter,cfg)
            self.assertEqual(len(kept),2)
            self.assertEqual({r['image_path'] for r in kept},set(paths[1:3]))
            self.assertTrue(all(r['propagated_mask_iou']>.9 for r in kept))
            self.assertEqual(anchor['image_path'],paths[0])
            self.assertTrue(all(0<r['frame_id']<1 and r['camera_to_world'] is None for r in kept))
            adapter.wrong=True
            self.assertEqual(_supplement_vehicle_views([anchor],[anchor],[anchor],frames,ImageCache(8),adapter,cfg)[0],[])

    def test_masked_cross_frame_identity_requires_distributed_unique_features(self):
        import cv2
        with TemporaryDirectory() as temp:
            root=Path(temp)
            texture=np.random.default_rng(19).integers(0,255,(240,280,3),dtype='uint8')
            texture=cv2.GaussianBlur(texture,(3,3),.6)
            rgba=np.dstack([texture,np.full((240,280),255,dtype='uint8')])
            def record(name,frame,pixels):
                path=root/f'{name}.png';Image.fromarray(pixels).save(path)
                return dict(real_images=[dict(rgba_path=str(path),camera_name='front',frame_index=frame,
                    source_image=f'{frame}.jpg',bbox_xyxy=[10,20,290,260])],source=dict(frame_records=[]))
            a=record('a',0,rgba)
            b=record('b',1,cv2.warpAffine(rgba,np.float32([[1,0,5],[0,1,7]]),(280,240)))
            result=_cross_frame_match(a,b)
            self.assertIsNotNone(result)
            self.assertGreater(result['inliers'],20)
            # Similar size/color alone never authorizes a match.
            c=record('c',1,np.full_like(rgba,255))
            self.assertIsNone(_cross_frame_match(a,c))
            b['source']['frame_records']=[dict(source_image='shared.jpg',bbox_xyxy=[50,0,70,20])]
            a['source']['frame_records']=[dict(source_image='shared.jpg',bbox_xyxy=[0,0,20,20])]
            self.assertIsNone(_cross_frame_match(a,b))

    def test_new_props_and_two_wheel_categories_are_distinct(self):
        allowed=['vehicle','trash_can','hydrant','bicycle','motorcycle']
        for phrase,category in [('garbage bin','trash_can'),('fire hydrant','hydrant'),
                                ('bicycle','bicycle'),('motorbike','motorcycle')]:
            self.assertEqual(class_for_label(phrase,allowed,'rigid'),category)

    def test_coverage_cap_and_class_aware_nms(self):
        rows = [observation(0, i*30) for i in range(8)]
        rows += [dict(rows[0],score=.8), dict(rows[0],category='traffic_light')]
        self.assertEqual(len(_nms(rows,.45,0)),9)
        self.assertEqual(len(_nms(rows,.45,3)),3)

    def test_frame_assignment_is_one_to_one(self):
        rows = [observation(0,0),observation(0,4),observation(1,1),observation(1,5)]
        rows[0]['motion_boxes']={'1':rows[2]['bbox_xyxy']}
        rows[1]['motion_boxes']={'1':rows[3]['bbox_xyxy']}
        tracks = _tracklets(rows)
        self.assertEqual(sorted(map(len,tracks)),[2,2])
        self.assertTrue(all(len({r['frame_id'] for r in t})==len(t) for t in tracks))

    def test_short_gap_motion_and_color(self):
        # Measured motion supports a short gap; extrapolated overlap alone cannot.
        rows=[observation(0,0),observation(1,8),observation(3,24)]
        rows[0]['motion_boxes']={'1':rows[1]['bbox_xyxy']}
        rows[1]['motion_boxes']={'3':rows[2]['bbox_xyxy']}
        track = _tracklets(rows,max_gap=2)
        self.assertEqual(list(map(len,track)),[3])
        self.assertEqual(len(_tracklets([observation(0),observation(4)],max_gap=3)),2)
        self.assertEqual(len(_tracklets([observation(0,color=[1.,0.]),
                                        observation(1,color=[0.,1.])])),2)
        self.assertEqual(len(_tracklets([observation(0),observation(1,camera='back')])),2)

    def test_vehicle_overlap_without_measured_motion_is_not_identity(self):
        # Recorded boxes from six mixed source tracks in scene-0103. The visible
        # vehicles differ, while old IoU/color association joined each group.
        boxes = [
            [[747,377,1599,898],[369,405,1398,898],[139,411,949,842]],
            [[844,471,1600,866],[1118,436,1599,838],[881,430,1599,869]],
            [[332,466,1004,856],[90,418,952,823],[28,423,740,790]],
            [[0,473,406,714],[0,441,389,769],[0,445,274,723]],
            [[370,474,562,607],[234,455,552,632],[32,450,504,667]],
            [[618,418,715,488],[615,403,743,495],[644,401,777,486]],
        ]
        for group in boxes:
            rows=[dict(observation(i,color=[1.,0.]),bbox_xyxy=box) for i,box in enumerate(group)]
            self.assertEqual(len(_tracklets(rows)),len(rows))
        # Nor may a poor motion fit authorize an otherwise overlapping pair.
        a,b=observation(0),observation(1)
        a['motion_boxes']={'1':[100,0,120,20]}
        self.assertEqual(len(_tracklets([a,b])),2)

    def test_image_triplet_keeps_texture_identity_when_sparse_flow_is_absent(self):
        import cv2
        texture = cv2.resize(np.random.default_rng(5).integers(0,256,(16,32,3),dtype=np.uint8),
                             (120,60),interpolation=cv2.INTER_NEAREST)
        images, rows = {}, []
        for frame in range(5):
            x = 10+4*frame
            image = np.full((100,180,3),128,np.uint8)
            image[20:80,x:x+120] = texture
            row = dict(observation(frame,color=[1.,0.]),bbox_xyxy=[x,20,x+120,80])
            images[row['image_path']] = image
            rows.append(row)
        cache = SimpleNamespace(get_rgb=lambda name: images[name])
        tracks = _tracklets(rows,image_cache=cache)
        self.assertEqual([len(track) for track in tracks],[5])
        # Two isolated observations do not provide the required temporal proof.
        self.assertEqual(len(_tracklets([dict(r) for r in rows[:2]],image_cache=cache)),2)
        # Similar color and geometry with different body texture must stay apart.
        for frame,row in enumerate(rows):
            x = 10+4*frame
            images[row['image_path']][20:80,x:x+120] = cv2.resize(
                np.random.default_rng(30+frame).integers(0,256,(16,32,3),dtype=np.uint8),
                (120,60),interpolation=cv2.INTER_NEAREST)
        self.assertEqual(len(_tracklets([dict(r) for r in rows],image_cache=cache)),5)

    def test_calibrated_static_motion_without_image_overlap(self):
        rows=[]
        for frame in range(3):
            for object_x in ((0.,4.) if frame%2==0 else (4.,0.)):
                transform=np.eye(4);transform[0,3]=frame*2.
                x=100+(object_x-frame*2)*10
                row=observation(frame,category='traffic_light',color=[1.,0.])
                row.update(bbox_xyxy=[x-5,95,x+5,105],object_x=object_x,
                    camera_intrinsic=[[100,0,100],[0,100,100],[0,0,1]],camera_to_world=transform.tolist())
                rows.append(row)
        tracks=_tracklets(rows)
        self.assertEqual(sorted(map(len,tracks)),[3,3])
        self.assertTrue(all(len({r['object_x'] for r in track})==1 for track in tracks))
        # Two rays alone cannot authorize an identity merge.
        self.assertEqual(len(_tracklets(rows[:4])),4)

    def test_nearby_identical_lights_with_occlusion_and_pixel_noise(self):
        rng=np.random.default_rng(27);track_count=0
        for _ in range(100):
            points=np.column_stack([rng.uniform(-5,5,4),rng.uniform(-3,-1,4),rng.uniform(25,45,4)])
            rows=[]
            for frame in range(8):
                angle=frame*.018
                rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
                transform=np.eye(4);transform[:3,:3]=rotation;transform[2,3]=frame*2
                for identity in rng.permutation(4):
                    if rng.random()<.15:
                        continue
                    xyz=rotation.T@(points[identity]-transform[:3,3])
                    x=800+900*xyz[0]/xyz[2]+rng.normal(0,1)
                    y=450+900*xyz[1]/xyz[2]+rng.normal(0,1)
                    w,h=900*.28/xyz[2],900*.75/xyz[2]
                    row=observation(frame,category='traffic_light',color=[1.,0.])
                    row.update(object_id=int(identity),bbox_xyxy=[x-w/2,y-h/2,x+w/2,y+h/2],
                        camera_intrinsic=[[900,0,800],[0,900,450],[0,0,1]],camera_to_world=transform.tolist())
                    rows.append(row)
            tracks=_tracklets(rows);track_count+=len(tracks)
            self.assertTrue(all(len({r['object_id'] for r in t})==1 for t in tracks))
        self.assertLess(track_count,1400)  # Reject the trivial all-singletons implementation.

    def test_static_identity_survives_long_gap_without_merging_covisible_heads(self):
        anchor = calibrated_track('front',0)
        returning = calibrated_track('front',6)
        for row in returning:
            row['frame_id'] += 12
        neighbour = calibrated_track('front',6,point_x=4)
        for row in neighbour:
            row['frame_id'] += 12
        tracks = _stitch_static_tracks([anchor,returning,neighbour],max_gap=3)
        self.assertEqual(sorted(map(len,tracks)),[3,6])
        self.assertTrue(all(len({r['frame_id'] for r in t})==len(t) for t in tracks))

    def test_verified_category_static_dedup_requires_full_geometry(self):
        def source(rows,offset=0):
            return dict(source=dict(camera_name='front',frame_records=[dict(r,
                frame_index=r['frame_id']+offset,source_image=f"{r['frame_id']+offset}.jpg") for r in rows]))
        a=source(calibrated_track('front',0))
        b=source(calibrated_track('front',6),12)
        self.assertIsNotNone(_static_source_match(a,b,'traffic_light'))
        self.assertIsNone(_static_source_match(a,b,'car'))
        self.assertIsNone(_static_source_match(a,source(calibrated_track('front',6,point_x=4),12),'traffic_light'))
        self.assertIsNone(_static_source_match(source(calibrated_track('front',0)[:2]),
                                              source(calibrated_track('front',6)[:2],12),'traffic_light'))

    def test_nonvehicle_one_edge_cut_is_not_diluted_by_other_sides(self):
        cfg=SimpleNamespace(min_mask_area_ratio=.0005,max_mask_area_ratio=.8,
                            min_largest_component_ratio=.6,max_edge_touch_ratio=.35)
        row=dict(observation(0,category='barrier'),pre_score=.9)
        mask=np.zeros((100,300),bool);mask[30:60,40:299]=True
        self.assertEqual(_mask_quality(row,mask,1.,mask.shape+(3,),cfg)[2],
                         'mask_truncated_at_image_edge')

    def test_sparse_flow_motion_and_post_classification_identity(self):
        import cv2
        with TemporaryDirectory() as temp:
            root=Path(temp);rows=[]
            texture=np.random.default_rng(5).integers(0,255,(60,60,3),dtype='uint8')
            for frame in range(2):
                rgb=np.full((160,200,3),100,np.uint8);x=20+frame*35
                rgb[40:100,x:x+60]=texture
                path=root/f'{frame}.png';Image.fromarray(rgb).save(path)
                row=observation(frame);row.update(image_path=str(path),bbox_xyxy=[x,40,x+60,100])
                rows.append(row)
            cv2.setNumThreads(2)
            _motion_boxes(rows,ImageCache(2),3)
            self.assertGreater(_bbox_iou(rows[0]['motion_boxes']['1'],rows[1]['bbox_xyxy']),.8)
        def source(offset):
            return {'source':{'frame_records':[dict(source_image=f'{i}.jpg',bbox_xyxy=[offset,0,offset+20,20]) for i in range(3)]}}
        self.assertTrue(_same_source_identity(source(0),source(1)))
        self.assertFalse(_same_source_identity(source(0),source(30)))

    def test_calibrated_cross_camera_repeated_evidence(self):
        a,b = calibrated_track('front',-1),calibrated_track('left',1)
        tracks,count = _cross_camera_tracklets([a,b])
        self.assertEqual((len(tracks),count),(1,1))
        self.assertEqual(len(tracks[0]),6)
        # A single coincident view or a similar color without calibration is insufficient.
        self.assertEqual(_cross_camera_tracklets([a[:1],b[:1]])[1],0)
        self.assertEqual(_cross_camera_tracklets([[observation(i) for i in range(3)],b])[1],0)
        self.assertEqual(_cross_camera_tracklets([a,calibrated_track('left',1,color=[0.,1.])])[1],0)
        # Ambiguous repeated objects in one camera must not both merge into another.
        self.assertEqual(_cross_camera_tracklets([a,b,[dict(r) for r in b]])[1],0)
        # Geometrically inconsistent objects fail even with the same color.
        self.assertEqual(_cross_camera_tracklets([a,calibrated_track('left',1,point_x=4)])[1],0)

    def test_frame_adapter_and_provenance(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            frame=root/'frame.png';Image.new('RGB',(20,20)).save(frame)
            observation=calibrated_track('handheld',0)[0]
            index=root/'scene.json'
            index.write_text(json.dumps({'frames':[dict(observation,path=str(frame))]}))
            loaded=collect_frame_images(index)
            self.assertEqual(loaded[0].camera_name,'handheld')
            self.assertEqual(loaded[0].camera_to_world,observation['camera_to_world'])
            observation.update(image_path=str(frame),pre_score=.9,quality=.9,tracklet_id='t1')
            source=_source_record(SimpleNamespace(scene_id='clip'),'obj',observation,[observation])
            self.assertEqual(source['source']['timestamp_us'],0)
            self.assertEqual(source['source']['frame_records'][0]['camera_name'],'handheld')

    def test_stride_preserves_intermediate_motion_images(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);rows=[]
            for i in range(5):
                frame=root/f'frame{i}.png';Image.new('RGB',(20,20)).save(frame)
                rows.append(dict(path=str(frame),frame_id=i,camera_name='handheld',
                                 motion_paths=[] if i==0 else [f'sweep{i}.png']))
            index=root/'scene.json';index.write_text(json.dumps({'frames':rows}))
            selected=collect_frame_images(index,frame_stride=2)
            self.assertEqual([r.frame_id for r in selected],[0,2,4])
            self.assertEqual(selected[1].motion_paths,['sweep1.png',rows[1]['path'],'sweep2.png'])
            self.assertEqual(selected[2].motion_paths,['sweep3.png',rows[3]['path'],'sweep4.png'])

    def test_mask_extent_and_truncation(self):
        row=observation(0);row['pre_score']=.9
        cfg=SimpleNamespace(min_mask_area_ratio=.0005,max_mask_area_ratio=.8,
                            min_largest_component_ratio=.6,max_edge_touch_ratio=.35)
        mask=np.zeros((100,100),bool);mask[:50,:50]=True
        self.assertTrue(_mask_quality(row,mask,1.,(100,100,3),cfg)[0])
        self.assertFalse(row['real_view_eligible'])
        self.assertEqual(_mask_quality(dict(row,category='cone'),mask,1.,(100,100,3),cfg)[2],
                         'mask_truncated_at_image_edge')
        mask[:]=True
        self.assertEqual(_mask_quality(row,mask,1.,(100,100,3),cfg)[2],'mask_too_large')
        image=np.zeros((100,100,3),np.uint8);image[20:40,20:40]=[10,50,90]
        self.assertAlmostEqual(sum(_appearance(image,[20,20,40,40])),1.)

    def test_frame_edge_excluded_from_extra_references_but_limited_anchor_retained(self):
        cfg=SimpleNamespace(min_mask_area_ratio=.0005,max_mask_area_ratio=.8,
                            min_largest_component_ratio=.6,max_edge_touch_ratio=.35)
        full=observation(0);full['pre_score']=.9
        cut=observation(1);cut['pre_score']=.9
        mask=np.zeros((100,100),bool);mask[20:70,20:80]=True
        _,full['quality'],_=_mask_quality(full,mask,1.,mask.shape+(3,),cfg)
        # A long cut at x=1 used to evade contact tests on column zero.
        mask[:]=False;mask[20:70,1:60]=True
        _,cut['quality'],_=_mask_quality(cut,mask,1.,mask.shape+(3,),cfg)
        self.assertTrue(full['real_view_eligible']);self.assertFalse(cut['real_view_eligible'])
        selected,policy=select_real_views([full,cut],full)
        self.assertEqual(selected,[full])
        self.assertEqual(policy['excluded_observations'][0]['reason'],'truncated_at_source_frame_edge')
        selected,policy=select_real_views([cut],cut)
        self.assertEqual(selected,[cut]);self.assertTrue(policy['limited_anchor_truncated'])
        self.assertEqual(policy['status'],'single_view_limited')

    def test_same_appearance_is_not_identity(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            pixels=np.random.default_rng(4).integers(0,200,(80,80,3),dtype='uint8')
            paths=[root/f'{i}.png' for i in range(2)]
            for path in paths:Image.fromarray(pixels).save(path)
            rows=[dict(scene_id='scene',obj_id=str(i),category='cone',image_path=str(p)) for i,p in enumerate(paths)]
            rows.append(dict(rows[0],obj_id='duplicate'))
            manifest=root/'manifest.json';manifest.write_text(json.dumps(rows))
            output=root/'out.json';reject=root/'reject.jsonl'
            result=filter_manifest(manifest,output,reject)
            self.assertEqual((result['kept'],result['rejected']),(2,1))
            self.assertEqual(json.loads(reject.read_text())['reason'],'repeated_source_image')

    def test_same_frame_dedup_requires_original_mask_support(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);refs=[]
            for i in range(3):
                pixels=np.zeros((100,100,4),np.uint8)
                pixels[10:90,10:90,3]=255
                if i==1:pixels[10,10:20,3]=0
                if i==2:pixels[10:90,50:90,3]=0
                path=root/f'{i}.png';Image.fromarray(pixels).save(path)
                refs.append(dict(source_image='same.jpg',bbox_xyxy=[10,10,90,90],
                    crop_box=[0,0,100,100],rgba_path=str(path),sha256=str(i)))
            self.assertTrue(_same_frame_mask(refs[0],refs[1]))
            self.assertFalse(_same_frame_mask(refs[0],refs[2]))
            self.assertFalse(_same_frame_mask(refs[0],dict(refs[0],source_image='next.jpg')))
            self.assertFalse(_same_frame_mask(refs[0],dict(refs[0],bbox_xyxy=[50,10,130,90])))
            self.assertTrue(_same_frame_mask(refs[0],dict(refs[0],bbox_xyxy=[0,0,100,100])))
            self.assertFalse(_same_frame_mask(refs[0],dict(refs[0],bbox_xyxy=[30,10,110,90],crop_box=[20,0,120,100])))

    def test_cross_category_same_pixels_keep_larger_usable_source(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);rows=[];records=[]
            rgba=np.ones((100,100,4),np.uint8)*255;rgba[:,:,3]=0
            rgba[10:90,10:90,:3]=np.random.default_rng(23).integers(0,200,(80,80,3),dtype='uint8')
            rgba[10:90,10:90,3]=255
            for index,category in enumerate(['bus','truck','car']):
                path=root/f'{index}.png';Image.fromarray(rgba).save(path)
                box=[10,10,90,90] if index<2 else [30,10,110,90]
                ref=dict(source_image='shared.jpg',bbox_xyxy=box,crop_box=[0,0,100,100] if index<2 else [20,0,120,100],rgba_path=str(path),sha256='same-pixels')
                records.append(dict(asset_id=str(index),image_path=str(path),
                    quality=dict(mask_quality=.9 if index==0 else .8,real_view_eligible=True),
                    source=dict(frame_records=[dict(source_image='shared.jpg',bbox_xyxy=box)],
                        native_foreground=dict(pixels=6400 if index!=1 else 25600)),real_images=[ref]))
                rows.append(dict(scene_id='scene',obj_id=str(index),category=category,image_path=str(path),prompts=['keep exact prompt']))
            source=root/'scenes/scene/logs/export_items_full.json';source.parent.mkdir(parents=True);source.write_text(json.dumps(records))
            manifest=root/'manifest.json';manifest.write_text(json.dumps(rows));output=root/'out.json';reject=root/'reject.jsonl'
            result=filter_manifest(manifest,output,reject,run_dir=root)
            kept=json.loads(output.read_text())
            self.assertEqual(result['kept'],2)
            self.assertEqual({r['obj_id'] for r in kept},{'1','2'})
            self.assertEqual(kept[0],rows[1])
            self.assertEqual(json.loads(reject.read_text())['evidence']['method'],'same_frame_foreground_across_categories')

    def test_shared_mask_cannot_override_distinct_covisible_tracks(self):
        a=dict(source=dict(frame_records=[dict(source_image='later.jpg',bbox_xyxy=[0,0,20,20])]),real_images=[{}])
        b=dict(source=dict(frame_records=[dict(source_image='later.jpg',bbox_xyxy=[30,0,50,20])]),real_images=[{}])
        with patch('nuscenes.quality_filter._same_frame_mask',return_value=True):
            self.assertFalse(_same_source_identity(a,b))
            self.assertFalse(_same_foreground_across_categories(a,b))

    def test_duplicate_provenance_preserves_anchor_without_transitive_merges(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);rows=[];records=[]
            pixels=np.random.default_rng(42).integers(0,200,(80,80,3),dtype='uint8')
            for i in range(3):
                oid=str(i);path=root/f'{i}.png';Image.fromarray(pixels).save(path)
                rows.append(dict(scene_id='scene',obj_id=oid,category='car',image_path=str(path)))
                records.append(dict(asset_id=oid,image_path=str(path),
                    quality=dict(mask_quality=1.-i*.1,real_view_eligible=True),
                    source=dict(frame_records=[],source_image='same.jpg'),
                    real_images=[dict(source_image='same.jpg',sha256=oid)]))
            source=root/'scenes/scene/logs/export_items_full.json'
            source.parent.mkdir(parents=True);source.write_text(json.dumps(records))
            manifest=root/'manifest.json';manifest.write_text(json.dumps(rows))
            # B matches A and C, but C does not directly match the fixed keeper A.
            adjacent=lambda a,b:abs(int(a['asset_id'])-int(b['asset_id']))==1
            output=root/'out.json';reject=root/'reject.jsonl'
            with patch('nuscenes.quality_filter._same_source_identity',side_effect=adjacent):
                result=filter_manifest(manifest,output,reject,run_dir=root)
            self.assertEqual((result['kept'],result['rejected']),(2,1))
            self.assertEqual([r['obj_id'] for r in json.loads(output.read_text())],['0','2'])
            saved=json.loads(source.read_text())
            self.assertEqual(saved[0]['real_images'],records[0]['real_images'])
            alias=saved[0]['source']['merged_from'][0]
            self.assertEqual(alias['asset_id'],'1')
            self.assertEqual(alias['real_images'],records[1]['real_images'])
            self.assertEqual(len(alias['image_sha256']),64)
            self.assertNotIn('merged_from',saved[2]['source'])

    def test_pixel_duplicates_do_not_consume_real_view_slots(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);rows=[]
            for frame in (0,1,2,3,100):
                rgb=np.full((60,80,3),255,np.uint8)
                rgb[10:50,10:70]=[40 + (frame % 100)*30,60,80]
                path=root/f'{frame}.png';Image.fromarray(rgb).save(path)
                row=observation(frame)
                row.update(image_path=str(path),bbox_xyxy=[10,10,70,50],pre_score=.9,
                    quality=.9,tracklet_id='t1',_mask=np.any(rgb<245,axis=2))
                rows.append(row)
            cache=SimpleNamespace(get_rgb=lambda p:np.asarray(Image.open(p).convert('RGB')))
            extracted=dict(cache=cache,selected=[rows[0]],tracklet_rows={'t1':rows},
                segmented=rows,rejects=[],frames=rows,raw_rows=rows,tracklets=[rows],sam_rows=rows)
            cfg=SimpleNamespace(scene_id='scene',output_dir=root/'output',export_max_size=768,
                jpg_quality=92,real_views_max=4,real_view_quality_ratio=.7)
            with patch('asset_extraction.vision._extract_scene_assets',return_value=extracted):
                run_scene_asset_export(cfg)
            source=json.loads((cfg.output_dir/'logs/export_items_full.json').read_text())[0]
            self.assertEqual(len(source['real_images']),4)
            self.assertEqual({r['frame_index'] for r in source['real_images']},{0,1,2,3})

    def test_nonvehicle_reclassified_as_vehicle_retains_sam_evidence(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            rgb=np.full((60,80,3),255,np.uint8);rgb[10:50,10:70]=[40,60,80]
            path=root/'frame.png';Image.fromarray(rgb).save(path)
            row=observation(0,category='barrier')
            row.update(image_path=str(path),bbox_xyxy=[10,10,70,50],pre_score=.9,
                quality=.9,tracklet_id='t1',_mask=np.any(rgb<245,axis=2))
            cache=SimpleNamespace(get_rgb=lambda p:np.asarray(Image.open(p).convert('RGB')))
            extracted=dict(cache=cache,selected=[row],tracklet_rows={'t1':[row]},
                segmented=[row],rejects=[],frames=[row],raw_rows=[row],tracklets=[[row]],sam_rows=[row])
            cfg=SimpleNamespace(scene_id='scene',output_dir=root/'scenes/scene',export_max_size=768,
                jpg_quality=92,real_views_max=4,real_view_quality_ratio=.7)
            with patch('asset_extraction.vision._extract_scene_assets',return_value=extracted):
                exported=run_scene_asset_export(cfg)
            manifest=exported['items'];manifest[0].update(category='car',prompts=[
                'Generate a front-view image. Preserve identity.', 'Generate a side view. Preserve identity.'])
            source=root/'manifest.json';source.write_text(json.dumps(manifest))
            handoff=root/'handoff.json';prepare(source,handoff)
            routes=root/'routes.json';routes.write_text(json.dumps(build_routes(handoff,root)))
            route=json.loads(routes.read_text())['assets'][0]
            self.assertEqual(route['route'],'H')
            self.assertEqual(route['real_view_selection']['status'],'single_view_limited')
            self.assertEqual(len(route['real_images']),1)
            with Image.open(route['real_images'][0]['rgba_path']) as image:
                self.assertEqual(image.mode,'RGBA')
                self.assertEqual(image.getchannel('A').getextrema(),(0,255))
            manifest[0].update(category='barrier',prompts=['Generate a barrier.'])
            source.write_text(json.dumps(manifest));prepare(source,handoff);routes.write_text(json.dumps(build_routes(handoff,root)))
            self.assertEqual(json.loads(routes.read_text())['assets'][0]['real_images'],[])


if __name__=='__main__':unittest.main()
