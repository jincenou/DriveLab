"""Regression checks for sparse evidence, identity coverage and unchanged handoff."""
import hashlib
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np
from PIL import Image
from asset_extraction.real_views import select_real_views
from asset_extraction.vision import _select_for_sam, run_scene_asset_export
from nuscenes.prepare_handoff import prepare
from nuscenes.prepare_routes import build_routes

def row(i, quality=1.0):
    return dict(frame_id=i, camera_name='front', image_path=f'{i}.png',
                quality=quality, pre_score=quality, category='vehicle')

class SelectionTests(unittest.TestCase):
    def test_sparse_evidence_and_quality(self):
        for n in range(1,7):
            rows=[row(i) for i in range(n)]
            selected,audit=select_real_views(rows,rows[0])
            self.assertEqual(len(selected),min(n,4))
            self.assertIs(selected[0],rows[0])
            self.assertEqual(audit['status']=='preferred',n>=3)
        rows=[row(0),row(1,.95),row(2,.1),row(3,.2)]
        self.assertEqual(len(select_real_views(rows,rows[0])[0]),2)

    def test_duplicate_observation_not_counted(self):
        a=row(0); b=dict(a)
        selected,_=select_real_views([a,b,row(1)],a)
        self.assertEqual(len(selected),2)

    def test_budget_first_covers_identities(self):
        tracks=[[row(i) for i in range(10)] for _ in range(5)]
        candidates=_select_for_sam(tracks,1,8,6)
        self.assertEqual(len(candidates),30)
        self.assertEqual(len({r['tracklet_id'] for r in candidates[:5]}),5)
        # Identity budgeting cannot consume the selected vehicles' view budget.
        self.assertEqual(len({r['tracklet_id'] for r in _select_for_sam(tracks,1,3,6)}),3)
        self.assertEqual(len(_select_for_sam(tracks,1,3,6)),18)

    def test_export_routes_and_contract(self):
        with TemporaryDirectory() as temp:
            root=Path(temp); scene=root/'scenes/scene-0001'; rows=[]
            for i in range(4):
                rgb=np.full((80,120,3),255,np.uint8)
                rgb[20:60,20+i:100]=[40+i*25,80,120]
                path=root/f'frame{i}.png';Image.fromarray(rgb).save(path)
                r=row(i,1-i*.02)
                r.update(image_path=str(path),bbox_xyxy=[20,20,100,60],score=.9,
                         tracklet_id='t000001',tracklet_len=4,sam_score=.99,
                         _mask=np.any(rgb<245,axis=2),num_components=1)
                rows.append(r)
            cache=SimpleNamespace(get_rgb=lambda p: np.asarray(Image.open(p).convert('RGB')))
            extracted=dict(cache=cache,selected=[rows[0]],tracklet_rows={'t000001':rows},
                segmented=rows,rejects=[],frames=list(range(4)),raw_rows=rows,
                tracklets=[rows],sam_rows=rows)
            config=SimpleNamespace(scene_id='scene-0001',output_dir=scene,
                export_max_size=768,jpg_quality=92,real_views_max=4,real_view_quality_ratio=.7)
            with patch('asset_extraction.vision._extract_scene_assets',return_value=extracted):
                result=run_scene_asset_export(config)
            manifest=result['items']
            self.assertEqual(set(manifest[0]),{'scene_id','obj_id','category','image_path','prompts'})
            # Stand-in finalized Qwen result: this checks integration, not model quality.
            manifest[0]['category']='car'
            manifest[0]['prompts']=['Generate a clear front-view image. Preserve identity.',
                                    'Generate a clear side-view image. Preserve identity.']
            source=root/'manifest.json';source.write_text(json.dumps(manifest))
            handoff=root/'handoff/flux_input.json';prepare(source,handoff)
            before=handoff.read_bytes()
            out=root/'handoff/asset_routes.json';out.write_text(json.dumps(build_routes(handoff,root)))
            self.assertEqual(before,handoff.read_bytes())
            route=json.loads(out.read_text())['assets'][0]
            self.assertEqual(route['route'],'H');self.assertEqual(len(route['real_images']),4)
            self.assertEqual(len(route['flux_output_keys']),2)
            for r in route['real_images']:
                p=Path(r['rgba_path'])
                with Image.open(p) as image:self.assertEqual(image.mode,'RGBA')
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),r['sha256'])
            # A modified real image cannot be silently accepted by handoff.
            Path(route['real_images'][0]['rgba_path']).write_bytes(b'changed')
            with self.assertRaises(ValueError):build_routes(handoff,root)
            # Nonvehicles retain the B0 route and do not require real multiviews.
            manifest[0]['category']='barrier';manifest[0]['prompts']=['Complete this barrier.']
            source.write_text(json.dumps(manifest));prepare(source,handoff)
            out.write_text(json.dumps(build_routes(handoff,root)))
            route=json.loads(out.read_text())['assets'][0]
            self.assertEqual(route['route'],'B0');self.assertEqual(route['real_images'],[])

class ReferenceAdmissionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from asset_extraction.config import PipelineConfig
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = PipelineConfig('rigid', Path('.'), self.root, 'test')
        self.config.vlm.concurrency = 2
        self.good = dict(target_is_vehicle=True, occlusion='mild', visible_structure='complete',
                         detail='usable', mask_alignment='good', reason='Full target is usable')

    def reference(self, name, native_size=(160, 90), display_size=None):
        w, h = display_size or native_size
        image = Image.new('RGBA', (w+20, h+20), (255, 255, 255, 0))
        image.paste((80, 90, 100, 255), (10, 10, 10+w, 10+h))
        path = self.root / (name+'.png'); image.save(path)
        raw = self.root / (name+'.jpg');image.convert('RGB').save(raw)
        return dict(rgba_path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_image=str(raw), bbox_xyxy=[10, 10, 10+w, 10+h],
            crop_box=[0, 0, (w+20)*native_size[0]/w, (h+20)*native_size[1]/h],
            real_view_eligible=True, source_frame_edge={'contact_fraction':0})

    def route(self, refs, uid='car'):
        return dict(obj_id=uid, route='H', category='car', real_images=refs, real_view_selection={'real_count':len(refs)})

    async def test_tiny_or_upscaled_images_become_zero_without_model_call(self):
        from asset_extraction.real_views import review_references, reference_pixels
        small = self.reference('small', (42, 31), display_size=(420, 310))
        self.assertFalse(reference_pixels(small)['accepted'])
        with patch('asset_extraction.vlm.AsyncOpenAIClient.call_json',new_callable=AsyncMock) as call:
            rows = await review_references([self.route([small])], self.config, self.root/'review.json')
        call.assert_not_awaited()
        self.assertEqual(rows[0]['real_images'], [])
        self.assertEqual(rows[0]['route'], 'H')
        self.assertEqual(rows[0]['real_view_selection']['status'], 'flux_only')

    async def test_reject_occlusion_blur_and_mask_error_independently(self):
        from asset_extraction.real_views import review_references
        reviews = [self.good, dict(self.good,occlusion='severe'),
                   dict(self.good,detail='blurred'),dict(self.good,mask_alignment='bad')]
        refs=[self.reference(str(i)) for i in range(4)]
        # Distinct source context prevents one mocked response serving all inputs.
        for i,r in enumerate(refs):
            Image.new('RGB',(200,120),(i*20,30,40)).save(r['source_image'])
        with patch('asset_extraction.vlm.AsyncOpenAIClient.call_json',new_callable=AsyncMock,side_effect=reviews):
            rows=await review_references([self.route(refs)],self.config,self.root/'review.json')
        self.assertEqual(len(rows[0]['real_images']),1)
        self.assertEqual(len(rows[0]['real_view_selection']['reference_reviews']),4)
        self.assertEqual(rows[0]['real_images'][0]['rgba_path'],refs[0]['rgba_path'])

    async def test_bounded_requests_and_cache(self):
        from asset_extraction.real_views import review_references
        active=peak=0
        async def call(*args,**kwargs):
            nonlocal active,peak
            active+=1;peak=max(peak,active)
            await asyncio.sleep(.01)
            active-=1
            return self.good
        routes=[self.route([self.reference(str(i),(160+i,90))],str(i)) for i in range(5)]
        with patch('asset_extraction.vlm.AsyncOpenAIClient.call_json',side_effect=call) as mock:
            first=await review_references(routes,self.config,self.root/'review.json')
            self.assertEqual(peak,2);self.assertEqual(mock.call_count,5)
            second=await review_references(routes,self.config,self.root/'review.json')
            self.assertEqual(first,second);self.assertEqual(mock.call_count,5)

    async def test_api_failure_is_not_converted_to_zero_real(self):
        from asset_extraction.real_views import review_references
        with patch('asset_extraction.vlm.AsyncOpenAIClient.call_json',new_callable=AsyncMock,side_effect=RuntimeError('offline')):
            with self.assertRaisesRegex(RuntimeError,'quality requests failed'):
                await review_references([self.route([self.reference('large')])],self.config,self.root/'review.json')
        self.assertEqual(len(json.loads((self.root/'review.json').read_text())['failures']),1)


class AsyncClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from asset_extraction.config import PipelineConfig
        from asset_extraction.vlm import AsyncOpenAIClient
        self.config = PipelineConfig('rigid', Path('.'), Path('/tmp'), 'test')
        self.client = AsyncOpenAIClient(self.config, None)

    async def asyncTearDown(self):
        await self.client.close()

    async def test_bounded_concurrency_order_and_json_contract(self):
        from asset_extraction.vlm import RIGID_RESPONSE_SCHEMA
        active = peak = 0
        async def create(**kwargs):
            nonlocal active, peak
            self.assertEqual(kwargs['response_format'], {'type':'json_schema',
                'json_schema': {'name':'asset_response', 'strict':True, 'schema':RIGID_RESPONSE_SCHEMA}})
            index = int(kwargs['messages'][0]['content'])
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(.005 * (4-index%3))
                payload = dict(category='car',subtype=str(index),color='white',vehicle_view='front_side',
                               shape_description='sedan',visible_markings='not visible')
                return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
                    message=SimpleNamespace(content=json.dumps(payload)))])
            finally:
                active -= 1
        with patch.object(self.client.clients[0].chat.completions,'create',side_effect=create):
            results = await self.client.call_many_json(
                [[dict(role='user',content=str(i))] for i in range(8)],
                concurrency=3,response_schema=RIGID_RESPONSE_SCHEMA)
        self.assertEqual(peak,3)
        self.assertEqual(active,0)
        self.assertEqual([r['subtype'] for r in results],list(map(str,range(8))))

    async def test_reject_invalid_json_schema_and_incomplete_response(self):
        from asset_extraction.vlm import RIGID_RESPONSE_SCHEMA
        from jsonschema import ValidationError
        for content, finish, error in [('{}','stop',ValidationError),
                ('[]','stop',ValueError), ('```json\n{}\n```','stop',ValueError),
                ('','stop',ValueError), ('{}','length',ValueError)]:
            response = SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish,
                message=SimpleNamespace(content=content))])
            with patch.object(self.client.clients[0].chat.completions,'create',new_callable=AsyncMock,return_value=response) as create:
                with self.assertRaises(error) as raised:
                    await self.client.call_json([],response_schema=RIGID_RESPONSE_SCHEMA)
                if error is ValidationError:
                    self.assertEqual(raised.exception.response, json.loads(content))
                create.assert_awaited_once()

    async def test_failed_request_cancels_and_drains_siblings(self):
        active = 0
        async def call(messages, **kwargs):
            nonlocal active
            active += 1
            try:
                if messages[0]['content']=='bad':
                    await asyncio.sleep(.01)
                    raise RuntimeError('server failed')
                await asyncio.sleep(10)
            finally:
                active -= 1
        with patch.object(self.client,'call_json',side_effect=call):
            with self.assertRaisesRegex(RuntimeError,'server failed'):
                await self.client.call_many_json([[dict(content=x)] for x in ['slow','bad','queued']],concurrency=2)
        self.assertEqual(active,0)

    async def test_async_scene_slots_and_failures(self):
        import sys
        from asset_extraction.pipeline import SceneScheduler, parse_args
        active = peak = 0
        async def launch(*args, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            scene = args[0]
            async def wait():
                nonlocal active
                await asyncio.sleep(.01)
                active -= 1
                return 1 if scene=='scene-2' else 0
            return SimpleNamespace(wait=wait)
        with TemporaryDirectory() as temp:
            self.config.gpu_ids=[4,5]
            scheduler=SceneScheduler(self.config,Path('config'),[Path(f'scene-{i}') for i in range(5)],Path(temp),None)
            command=scheduler._command('scene-0',Path('scene-0.json'),Path(temp),4)
            with patch.object(sys,'argv',['pipeline',*command[3:]]):
                parsed=parse_args()
                self.assertTrue(parsed.scene_worker)
                self.assertEqual(parsed.gpu,'4')
            with patch.object(scheduler,'_command',side_effect=lambda scene,*args:[scene]), \
                 patch.object(scheduler,'_env',return_value={}), \
                 patch('asyncio.create_subprocess_exec',side_effect=launch):
                results=await scheduler._run()
            self.assertEqual(peak,2)
            self.assertEqual(len(results),5)
            self.assertEqual([r['scene_id'] for r in results if r['status']=='failed'],['scene-2'])

    async def test_independent_schema_failure_preserves_other_queries(self):
        async def call(messages, **kwargs):
            await asyncio.sleep(.001)
            if messages[0]['content'] == 'bad':
                raise ValueError('invalid vehicle view')
            return {'identity': messages[0]['content']}
        with patch.object(self.client, 'call_json', side_effect=call):
            result = await self.client.call_many_json(
                [[dict(content=x)] for x in ['first','bad','third']], concurrency=2, collect_errors=True)
        self.assertEqual(result[0], {'identity':'first'})
        self.assertIsInstance(result[1], ValueError)
        self.assertEqual(result[2], {'identity':'third'})

    async def test_mask_errors_and_unknown_heading_are_not_hidden(self):
        from asset_extraction.vision import _clean_mask
        from asset_extraction.vlm import RigidPromptBuilder
        with patch('cv2.connectedComponentsWithStats',side_effect=RuntimeError('mask failure')):
            with self.assertRaisesRegex(RuntimeError,'mask failure'):
                _clean_mask(np.ones((8,8),dtype='uint8'),[0,0,8,8])
        with self.assertRaises(ValueError):
            RigidPromptBuilder().build({'category':'car'},{'vehicle_view':'unknown'})

class PromptContractTests(unittest.TestCase):
    def test_traffic_signal_and_identity_reach_final_prompt(self):
        from asset_extraction.vlm import build_diffusion_prompts, RIGID_RESPONSE_SCHEMA
        from jsonschema import validate
        attributes = dict(category='traffic_light', subtype='pedestrian signal', color='black',
                          shape_description='one rectangular housing on a left-side support',
                          visible_markings='illuminated red hand', vehicle_view='unknown')
        validate(attributes, RIGID_RESPONSE_SCHEMA)
        prompt, = build_diffusion_prompts({'category':'pole'}, attributes)
        self.assertIn('exact object in the reference image', prompt)
        self.assertIn(attributes['shape_description'], prompt)
        self.assertIn('illuminated red hand', prompt)
        self.assertIn('plain white background', prompt)
        self.assertLess(len(prompt.split()), 80)
        for category in ('trash_can', 'hydrant', 'bicycle', 'motorcycle'):
            attributes['category'] = category
            validate(attributes, RIGID_RESPONSE_SCHEMA)
            self.assertEqual(len(build_diffusion_prompts({'category':category}, attributes)), 1)

    def test_vehicle_views_preserve_only_observed_details(self):
        from asset_extraction.vlm import build_diffusion_prompts
        attributes = dict(category='van', color='blue', shape_description='split rear doors and square rear windows',
                          visible_markings='unknown', vehicle_view='rear_side')
        rear, side = build_diffusion_prompts({'category':'van'}, attributes)
        self.assertIn('rear-view', rear)
        self.assertIn('side-view', side)
        for prompt in (rear, side):
            self.assertIn(attributes['shape_description'], prompt)
            self.assertIn('blue', prompt)
            self.assertNotIn('grille', prompt)
            self.assertNotIn('headlamp', prompt)
            self.assertNotIn('unknown', prompt)
            self.assertLess(len(prompt.split()), 70)

    def test_unstriped_surface_does_not_gain_category_template_details(self):
        from asset_extraction.vlm import build_diffusion_prompts
        prompt, = build_diffusion_prompts({'category':'cone'}, dict(category='cone', color='yellow',
            shape_description='solid yellow surface with a round base', visible_markings='not visible'))
        self.assertIn('solid yellow surface with a round base', prompt)
        self.assertNotIn('stripe', prompt)
        self.assertNotIn('reflective', prompt)
        self.assertNotIn('square', prompt)
        self.assertNotIn('Do not', prompt)

class FrameAdapterTests(unittest.TestCase):
    def test_resume_rejects_changed_frontend_parameters(self):
        from asset_extraction.config import PipelineConfig
        from asset_extraction.pipeline import ImagePipeline
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = PipelineConfig('rigid', root, root/'out', 'first')
            pipeline = ImagePipeline(config, root/'config.yaml')
            pipeline._claim_run()
            config.gpu_ids = [4, 5]
            pipeline._claim_run()
            signature = (pipeline.run_dir/'input_signature.json').read_bytes()
            config.thresholds.box = .5
            with self.assertRaisesRegex(ValueError, 'new RUN_NAME'):
                pipeline._claim_run()
            self.assertEqual(signature, (pipeline.run_dir/'input_signature.json').read_bytes())

    def test_generic_frames_without_dataset_annotations(self):
        from asset_extraction.config import PipelineConfig, validate_config
        from asset_extraction.pipeline import _discover_scene_dirs
        with TemporaryDirectory() as temp:
            root=Path(temp); inputs=root/'input'; inputs.mkdir()
            Image.new('RGB',(32,24),'white').save(inputs/'frame.png')
            Image.new('RGB',(32,24),'gray').save(inputs/'intermediate.png')
            payload=dict(scene_id='clip',frames=[dict(path='frame.png',camera_name='webcam',frame_id=0,timestamp_us=100,
                                                     motion_paths=['intermediate.png'])])
            index=inputs/'clip.json'; index.write_text(json.dumps(payload))
            cfg=PipelineConfig('rigid',root,root/'out','test',dataset='frames',dataset_root=inputs)
            validate_config(cfg)
            paths=_discover_scene_dirs(cfg,root/'run')
            exported=json.loads(paths[0].read_text())
            self.assertEqual(exported['frames'][0]['path'],str(inputs/'frame.png'))
            self.assertEqual(exported['frames'][0]['motion_paths'],[str(inputs/'intermediate.png')])
            payload['frames']*=2;index.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,'Duplicate'):_discover_scene_dirs(cfg,root/'run')

    def test_nuscenes_camera_pose_without_object_annotations(self):
        from asset_extraction.config import PipelineConfig
        from asset_extraction.pipeline import _prepare_nuscenes_scene_dirs
        with TemporaryDirectory() as temp:
            root=Path(temp);meta=root/'v1.0-test';meta.mkdir()
            image=root/'samples/CAM_FRONT/frame.jpg';image.parent.mkdir(parents=True)
            Image.new('RGB',(32,24),'white').save(image)
            Image.new('RGB',(32,24),'white').save(image.parent/'next.jpg')
            intermediate=root/'sweeps/CAM_FRONT/middle.jpg';intermediate.parent.mkdir(parents=True)
            Image.new('RGB',(32,24),'gray').save(intermediate)
            tables={
                'scene':[dict(token='scene',name='scene-1',nbr_samples=2)],
                'sample':[dict(token='sample',scene_token='scene',timestamp=100),dict(token='sample2',scene_token='scene',timestamp=200)],
                'sample_data':[
                    dict(is_key_frame=True,filename='samples/CAM_FRONT/frame.jpg',sample_token='sample',timestamp=101,calibrated_sensor_token='calib',ego_pose_token='pose'),
                    dict(is_key_frame=True,filename='samples/CAM_FRONT/next.jpg',sample_token='sample2',timestamp=201,calibrated_sensor_token='calib',ego_pose_token='pose'),
                    dict(is_key_frame=False,filename='sweeps/CAM_FRONT/middle.jpg',sample_token='sample2',timestamp=150)],
                'calibrated_sensor':[dict(token='calib',rotation=[1,0,0,0],translation=[1,2,3],camera_intrinsic=[[20,0,16],[0,20,12],[0,0,1]])],
                'ego_pose':[dict(token='pose',rotation=[1,0,0,0],translation=[10,0,0])],
            }
            for name,rows in tables.items():(meta/f'{name}.json').write_text(json.dumps(rows))
            cfg=PipelineConfig('rigid',root,root/'out','test',dataset_root=root,dataset_version='v1.0-test',camera_channels=['CAM_FRONT'],scene_ids=['scene-1'])
            paths=_prepare_nuscenes_scene_dirs(cfg,root/'run')
            frames=json.loads(paths[0].read_text())['frames'];frame=frames[0]
            np.testing.assert_allclose(np.asarray(frame['camera_to_world'])[:3,3],[11,2,3])
            self.assertEqual(frame['timestamp_us'],101)
            self.assertEqual(frame['motion_paths'],[])
            self.assertEqual(frames[1]['motion_paths'],[str(intermediate)])
            self.assertFalse((meta/'sample_annotation.json').exists())

if __name__=='__main__':unittest.main()
