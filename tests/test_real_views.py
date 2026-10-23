"""Regression checks for sparse evidence, identity coverage and unchanged handoff."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from asset_extraction.real_views import select_real_views
from asset_extraction.vision import _select_for_sam, run_scene_asset_export
from nuscenes.prepare_handoff import prepare
from nuscenes.prepare_routes import prepare_routes

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
        self.assertEqual(len(candidates),8)
        self.assertEqual(len({r['tracklet_id'] for r in candidates[:5]}),5)
        # A low budget must never spend six masks on one object before others.
        self.assertEqual(len({r['tracklet_id'] for r in _select_for_sam(tracks,1,3,6)}),3)

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
            out=root/'handoff/asset_routes.json';prepare_routes(handoff,root,out)
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
            with self.assertRaises(ValueError):prepare_routes(handoff,root,out)
            # Nonvehicles retain the B0 route and do not require real multiviews.
            manifest[0]['category']='barrier';manifest[0]['prompts']=['Complete this barrier.']
            source.write_text(json.dumps(manifest));prepare(source,handoff)
            prepare_routes(handoff,root,out)
            route=json.loads(out.read_text())['assets'][0]
            self.assertEqual(route['route'],'B0');self.assertEqual(route['real_images'],[])

if __name__=='__main__':unittest.main()
