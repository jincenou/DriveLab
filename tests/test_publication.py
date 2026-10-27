import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
import yaml

from asset_extraction.bank import publish
from asset_extraction.generation import sha256
from asset_extraction.io_utils import write_json, read_json
from asset_extraction.run import execute


class PublicationTests(unittest.TestCase):
    def test_idempotent_packages_keep_other_scenes_and_detect_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source.png';source.write_bytes(b'source')
            payload=root/'raw.ply';payload.write_bytes(b'validated upstream payload')
            def state(scene):
                return {'assets':{'obj':dict(status='keep',entry=dict(
                    uid='obj',category='car',route='H',status='keep',scale={},
                    canonical={'canonical_path':str(payload),'sha256':sha256(payload),'size_xyz':[1,1,1]},
                    provenance={'scene_id':scene,'source_image':str(source),'real_images':[],'source':{}},
                    quality={'rendering':{}}))}}
            bank=root/'bank'
            first=publish(state('one'),bank,dataset='video',scene_id='one')
            publish(state('two'),bank,dataset='video',scene_id='two')
            publish(state('one'),bank,dataset='video',scene_id='one')
            self.assertEqual(read_json(bank/'manifests/asset_library_rigid.json')['total'],2)
            package=bank/first[0]['package_path']
            self.assertEqual(read_json(package/'asset.json')['canonical']['canonical_path'],'asset.ply')
            self.assertTrue((package/'refs/source.png').is_file())
            (package/'asset.ply').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'changed'):
                publish(state('one'),bank,dataset='video',scene_id='one')


class EmptySceneTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_filtered_scene_finishes_without_model_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);frames=root/'frames';frames.mkdir()
            from PIL import Image
            Image.new('RGB',(32,32),'white').save(root/'frame.png')
            write_json(frames/'scene.json',dict(scene_id='scene',frames=[dict(
                path=str(root/'frame.png'),camera_name='front',frame_id=0,timestamp_us=0)]))
            config=root/'frontend.yaml'
            config.write_text(yaml.safe_dump(dict(mode='rigid',workspace=str(root),output_root=str(root),
                run_name='test',dataset='frames',dataset_root=str(frames),camera_channels=['front'],
                gpu_ids=[0],runtime={'python':sys.executable})))
            generation=write_json(root/'models.json',{'seed':42})
            run=root/'run';handoff=run/'frontend/runs/scene/handoff'
            write_json(handoff/'flux_input.json',[]);write_json(handoff/'asset_routes.json',{'assets':[]})
            spec=dict(source={'type':'frames','path':str(frames)},frontend_config=str(config),
                generation_config=str(generation),run_dir=str(run),library_dir=str(root/'bank'),
                renderer_python=sys.executable,gpus=[0])
            with patch.dict('os.environ', {}), patch('asset_extraction.run.generate',new_callable=AsyncMock) as generate:
                result=await execute(spec)
                self.assertEqual(result['scenes'][0]['queries'],0)
                generate.assert_not_called()
