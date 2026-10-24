"""Generation contract, resumability and bounded async worker scheduling."""
import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from asset_extraction import generation as gen
from asset_extraction.io_utils import write_json


class GenerationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'trellis/pipelines').mkdir(parents=True)
        for name in ['flux.py', 'gen_gs.py', 'trellis/pipelines/trellis_image_to_3d.py', 'birefnet-general-lite.onnx', 'source.png', 'real.png']:
            (self.root / name).write_text(name)
        self.config = {key: str(self.root) for key in gen.PATH_KEYS}
        self.config.update(seed=42, segmentation_model=str(self.root / 'birefnet-general-lite.onnx'))
        rows = [dict(scene_id='scene', obj_id=f'obj{i}', category=category,
                     image_path=str(self.root / 'source.png'), prompts=['unchanged exact prompt'])
                for i, category in enumerate(['car', 'traffic_sign', 'car', 'barrier'])]
        self.handoff = write_json(self.root / 'flux_input.json', rows)
        self.routes = write_json(self.root / 'routes.json', dict(flux_input_sha256=gen.sha256(self.handoff), assets=[
            dict(obj_id=row['obj_id'], scene_id='scene', category=row['category'],
                 route='H' if row['category'] == 'car' else 'B0',
                 real_images=[dict(rgba_path=str(self.root / 'real.png'), sha256=gen.sha256(self.root / 'real.png'))] if row['category'] == 'car' else [],
                 real_view_selection=None, source={'frame_id': 0}) for row in rows]))

    def bundle(self):
        return gen.prepare_jobs(self.handoff, self.routes, self.config)

    def test_exact_manifest_and_real_hash_validation(self):
        before = self.handoff.read_bytes()
        jobs = self.bundle()
        self.assertEqual(['H', 'B0', 'H', 'B0'], [r['route'] for r in jobs['assets']])
        self.assertEqual(['unchanged exact prompt'], jobs['assets'][0]['prompts'])
        self.assertEqual(before, self.handoff.read_bytes())
        (self.root / 'real.png').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Changed or duplicated'):
            self.bundle()

    def test_route_and_duplicate_evidence_rejected(self):
        routes = gen.read_json(self.routes)
        routes['assets'][0]['real_images'] *= 2
        write_json(self.routes, routes)
        with self.assertRaisesRegex(ValueError, 'duplicated'):
            self.bundle()
        routes['assets'][0]['real_images'].pop()
        routes['assets'][0]['route'] = 'B0'
        write_json(self.routes, routes)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.bundle()

    def test_zero_real_preserves_vehicle_backend_and_query(self):
        routes = gen.read_json(self.routes)
        routes['assets'][0]['real_images'] = []
        write_json(self.routes, routes)
        jobs = self.bundle()['assets']
        self.assertEqual(len(jobs), 4)
        self.assertEqual(jobs[0]['route'], 'H')
        self.assertEqual(jobs[0]['input_mode'], 'flux_only')
        self.assertEqual(jobs[0]['prompts'], ['unchanged exact prompt'])
        routes['schema_version'] = 2
        write_json(self.routes, routes)
        with self.assertRaisesRegex(ValueError, 'Unapproved'):
            self.bundle()

    async def test_concurrency_resume_and_failure_denominator(self):
        active = peak = 0
        calls = []
        async def worker(stage, ids, gpu, jobs_path, accept):
            nonlocal active, peak
            active += 1; peak = max(peak, active); calls.append((stage, gpu, tuple(ids)))
            await asyncio.sleep(.01)
            for index in ids:
                result = dict(item_index=index, stage=stage, status='succeeded')
                output = jobs_path.parent / f'{stage}_{index}.bin'
                output.write_text('output')
                if stage == 'flux':
                    result['images'] = [dict(path=str(output), sha256=gen.sha256(output))]
                elif index == 3:
                    result.update(status='failed', error='injected decoder error')
                else:
                    result.update(raw_path=str(output), raw_sha256=gen.sha256(output), raw_up_axis='z', gaussian_count=123)
                accept(result)
            active -= 1
        bundle = self.bundle()
        with patch.object(gen, '_launch', worker):
            state = await gen.generate(bundle, self.root / 'work', [2, 4])
            self.assertEqual(2, peak)
            self.assertEqual(dict(queries=4, generated=3), state['counts'])
            self.assertEqual('failed', state['assets'][3]['status'])
            calls.clear()
            await gen.generate(bundle, self.root / 'work', [2, 4])
            self.assertEqual([('B0', 2, (3,))], calls)
            Path(state['assets'][0]['raw_path']).write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'changed'):
                await gen.generate(bundle, self.root / 'work', [2, 4])

    def test_opacity_serialization_preserves_finite_extreme_logits(self):
        import numpy as np
        import torch
        from plyfile import PlyData, PlyElement
        class Asset:
            get_xyz = torch.zeros(2, 3)
            _opacity = torch.tensor([[50.0], [-100.0]])
            opacity_bias = torch.tensor(-2.0)
            def save_ply(self, path):
                vertices = np.zeros(2, dtype=[('x', 'f4'), ('opacity', 'f4')])
                vertices['opacity'] = [np.inf, -np.inf]
                PlyData([PlyElement.describe(vertices, 'vertex')]).write(path)
        target = self.root / 'raw.ply'
        gen.save_gaussian(Asset(), target)
        np.testing.assert_array_equal(PlyData.read(target)['vertex']['opacity'], [48.0, -102.0])
        invalid = Asset()
        invalid._opacity = torch.tensor([[float('nan')], [1.0]])
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            gen.save_gaussian(invalid, target)

    async def test_resume_rejects_config_change(self):
        work = self.root / 'work'
        work.mkdir()
        write_json(work / 'generation.json', {'signature': 'different', 'assets': []})
        with self.assertRaisesRegex(ValueError, 'Inputs/config/code changed'):
            await gen.generate(self.bundle(), work, [0])


if __name__ == '__main__':
    unittest.main()
