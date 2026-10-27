"""Portable Gaussian export, strict metadata, and original-loader integration."""
from copy import deepcopy
import base64
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from asset_extraction.export import (FIELDS, canonicalize_raw, export_library, inspect_ply, _verdict,
                                     _audit_sheet, _vlm_messages, AUDIT_SCHEMA, _validate_sources, _sha)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'raw.ply'
        vertex = np.zeros(512, dtype=[(name, '<f4') for name in FIELDS])
        xyz = np.random.default_rng(42).normal(size=(512, 3)) * [2, .7, .5]
        for i, name in enumerate(['x', 'y', 'z']):
            vertex[name] = xyz[:, i]
        vertex['rot_0'] = 1
        for name in ['scale_0', 'scale_1', 'scale_2']:
            vertex[name] = -4
        PlyData([PlyElement.describe(vertex, 'vertex')], text=False).write(str(self.source))

    def entry(self, uid='car_001', category='car', status='review', heading=None):
        canonical = canonicalize_raw(self.source, self.root / f'{uid}.ply',
                                    route='H', category=category, heading=heading)
        return dict(uid=uid, category=category, status=status, quality_score=7.,
                    source_path=str(self.source), canonical=canonical,
                    provenance={'source_obj_id': uid, 'dataset': 'test'},
                    scale={'scale_confidence': 'low'}, label={'sub_type': 'sedan'})

    def test_unreviewed_heading_stays_review_and_unique_payload(self):
        row = self.entry()
        duplicate = deepcopy(row)
        duplicate['uid'] = 'car_002'
        output = self.root / 'export'
        result = export_library([row, duplicate], output)
        published = json.loads(result.read_text())['assets']
        self.assertEqual(len(list(output.rglob('*.ply'))), 1)
        self.assertEqual(len(list(output.rglob('*.*'))), 3)
        self.assertEqual(published[0]['status'], 'review')
        self.assertIsNone(published[0]['canonical']['heading'])
        self.assertEqual(published[0]['scale']['scale_confidence'], 'low')
        self.assertEqual(published[0]['source_path'], published[1]['source_path'])
        self.assertTrue((output / published[0]['source_path']).is_file())
        self.assertEqual(row['canonical']['canonical_path'], str(self.root / 'car_001.ply'))
        with self.assertRaises(FileExistsError):
            export_library([row], output)

    def test_keep_requires_resolved_heading_and_rejects_wrong_bounds(self):
        row = self.entry(status='keep')
        with self.assertRaisesRegex(ValueError, 'resolved canonical front'):
            export_library([row], self.root / 'invalid')
        self.assertFalse((self.root / 'invalid').exists())
        row['status'] = 'review'
        row['canonical']['size_xyz'][0] *= 2
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            export_library([row], self.root / 'invalid_size')
        self.assertFalse((self.root / 'invalid_size').exists())

    def test_original_loader_vehicle_and_nonvehicle(self):
        car = self.entry(status='keep', heading={'front': 'B', 'confidence': .95,
                                                'source': 'controlled test orientation'})
        car['scale'] = dict(est_length_m=4.5, est_width_m=1.8, est_height_m=1.5,
                            scale_confidence='medium')
        car['color_saliency'] = 'dark'
        cone = self.entry(uid='cone_001', category='cone', status='keep')
        cone['scale'] = dict(est_length_m=.5, est_width_m=.5, est_height_m=.7,
                             scale_confidence='medium')
        pending = self.entry(uid='car_pending')
        output = self.root / 'library'
        manifest = export_library([car, cone, pending], output)
        import os
        legacy = os.environ.get('LEGACY_TRAJECTORY_ROOT', '/mnt/wmj/workspace4/pipeline/traj_pipeline')
        if not Path(legacy).is_dir():
            self.skipTest('Optional legacy trajectory consumer is not installed')
        if legacy not in sys.path:
            sys.path.insert(0, legacy)
        from corner_case.assets import AssetLibrary, bind_rigid
        from corner_case.planning.objects import AssetQuery
        from corner_case.taxonomy import Appearance
        library = AssetLibrary(manifest, output)
        self.assertEqual({row.uid for row in library.entries}, {'car_001', 'cone_001'})
        self.assertEqual([row.uid for row in library.candidates(AssetQuery('car',appearance=Appearance.DARK))], ['car_001'])
        for entry in library.entries:
            bound = bind_rigid(entry, np.random.default_rng(0))
            self.assertTrue(bound.asset_path.is_file())
            self.assertGreater(bound.scale, 0)
            self.assertGreater(min(bound.size_lwh_m), 0)
            self.assertEqual(inspect_ply(bound.asset_path)['gaussian_count'], 512)

    def test_publication_cannot_bypass_scale_admission(self):
        row=self.entry(category='cone',status='keep')
        for scale in [dict(scale_confidence='low',est_length_m=.4,est_width_m=.4,est_height_m=.8),
                      dict(scale_confidence='medium',est_length_m=.4,est_width_m=0,est_height_m=.8)]:
            row['scale']=scale
            with self.assertRaisesRegex(ValueError,'trusted metric-scale'):
                export_library([row],self.root/'invalid_scale')
            self.assertFalse((self.root/'invalid_scale').exists())

    def test_backend_axis_and_finite_attribute_validation(self):
        canonical = canonicalize_raw(self.source, self.root / 'b0.ply', route='B0', category='cone')
        self.assertEqual(canonical['raw_up'], '-Y')
        low = np.asarray(canonical['validation']['robust_bounds'])[0]
        self.assertAlmostEqual(low[2], 0, places=5)
        vertex = PlyData.read(str(self.source))['vertex'].data.copy()
        vertex['opacity'][0] = np.inf
        PlyData([PlyElement.describe(vertex, 'vertex')], text=False).write(str(self.source))
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            canonicalize_raw(self.source, self.root / 'invalid.ply', route='H', category='car')
        self.assertFalse((self.root / 'invalid.ply').exists())

    def test_quality_verdict_requires_evidence_and_trusted_scale(self):
        audit = dict(keep=True, category_match=True, identity_consistent=True,
                     symmetry_360_ok=False, quality_score=7, scale=dict(scale_confidence='medium'))
        rendering = dict(all_16_nonempty=True)
        heading = dict(front='A', confidence=.8)
        self.assertEqual(_verdict(audit, rendering, heading, True), 'keep')
        self.assertEqual(_verdict(audit, rendering, None, False), 'keep')
        self.assertEqual(_verdict(audit, rendering, None, True), 'review')
        self.assertEqual(_verdict(audit, rendering, dict(front='unknown', confidence=.8), True), 'review')
        self.assertEqual(_verdict(audit, rendering, dict(front='B', confidence=.6), True), 'review')
        low = deepcopy(audit); low['scale']['scale_confidence'] = 'low'
        self.assertEqual(_verdict(low, rendering, heading, True), 'review')
        self.assertEqual(_verdict(audit, dict(all_16_nonempty=False), heading, True), 'reject')
        mismatch = deepcopy(audit); mismatch['identity_consistent'] = False
        self.assertEqual(_verdict(mismatch, rendering, heading, True), 'reject')

    def test_signal_face_rotation_requires_independent_confirmation(self):
        heading = dict(front='+X', azimuth_deg=270., view_index=12, confidence=.95,
                       source='controlled semantic face evidence')
        signal = self.entry(uid='signal', category='traffic_light', status='keep', heading=heading)
        canonical = signal['canonical']
        # The observed -Y signal face, unlike a vehicle long axis, must become +X.
        np.testing.assert_allclose(np.asarray(canonical['rotation']) @ [0, -1, 0], [1, 0, 0], atol=1e-8)
        self.assertFalse(canonical['heading_verified'])
        with self.assertRaisesRegex(ValueError, 'resolved canonical front'):
            export_library([signal], self.root / 'unverified_signal')
        audit = dict(keep=True, category_match=True, identity_consistent=True,
                     quality_score=7, scale=dict(scale_confidence='medium'))
        rendered = dict(all_16_nonempty=True)
        self.assertEqual(_verdict(audit, rendered, heading, True, semantic_face=True), 'review')
        for check in [dict(front='unknown', confidence=0), dict(front='B', confidence=.99),
                      dict(front='A', confidence=.6)]:
            heading['verification'] = check
            self.assertEqual(_verdict(audit, rendered, heading, True, semantic_face=True), 'review')
        heading['verification'] = dict(front='A', confidence=.9)
        self.assertEqual(_verdict(audit, rendered, heading, True, semantic_face=True), 'keep')
        confirmed = self.entry(uid='signal_confirmed', category='traffic_light', status='keep', heading=heading)
        self.assertTrue(confirmed['canonical']['heading_verified'])
        confirmed['scale'] = dict(est_length_m=.4, est_width_m=.4, est_height_m=3.5, scale_confidence='medium')
        output = self.root / 'signal_library'
        manifest = export_library([confirmed], output)
        import os
        legacy = os.environ.get('LEGACY_TRAJECTORY_ROOT', '/mnt/wmj/workspace4/pipeline/traj_pipeline')
        if not Path(legacy).is_dir():
            self.skipTest('Optional legacy trajectory consumer is not installed')
        if legacy not in sys.path:
            sys.path.insert(0, legacy)
        from corner_case.assets import AssetLibrary, bind_rigid
        library = AssetLibrary(manifest, output)
        self.assertEqual(len(library.entries), 1)
        bound = bind_rigid(library.entries[0], np.random.default_rng(0))
        jitter = np.random.default_rng(0).uniform(.94, 1.06)
        self.assertAlmostEqual(bound.size_lwh_m[2], 3.5*jitter)
        self.assertTrue(bound.asset_path.is_file())
        # Semantic orientation does not weaken the identity/quality gate.
        audit['identity_consistent'] = False
        self.assertEqual(_verdict(audit, rendered, heading, True, semantic_face=True), 'reject')

    def test_signal_without_semantic_face_cannot_be_admitted(self):
        signal = self.entry(uid='signal_unknown', category='traffic_light', status='keep')
        self.assertIsNone(signal['canonical']['heading'])
        self.assertFalse(signal['canonical']['heading_verified'])
        with self.assertRaisesRegex(ValueError, 'resolved canonical front'):
            export_library([signal], self.root / 'signal_unknown_library')
        for angle in [-1, 360, 12., float('nan')]:
            with self.assertRaisesRegex(ValueError, 'semantic face azimuth'):
                canonicalize_raw(self.source, self.root/'bad_signal.ply', route='B0', category='traffic_light',
                    heading=dict(front='+X', azimuth_deg=angle, confidence=.95, source='test'))

    def test_horizontal_props_bind_by_the_long_axis_not_the_thin_axis(self):
        from itertools import product
        from asset_extraction.export import LONG_AXIS_CATEGORIES
        import os
        legacy = os.environ.get('LEGACY_TRAJECTORY_ROOT', '/mnt/wmj/workspace4/pipeline/traj_pipeline')
        if not Path(legacy).is_dir():
            self.skipTest('Optional legacy trajectory consumer is not installed')
        if legacy not in sys.path:
            sys.path.insert(0, legacy)
        from corner_case.assets import AssetLibrary, bind_rigid, HEIGHT_SCALED_CATEGORIES
        self.assertTrue(LONG_AXIS_CATEGORIES.isdisjoint(HEIGHT_SCALED_CATEGORIES))
        vertex = PlyData.read(str(self.source), mmap=False)['vertex'].data.copy()
        xyz = np.tile(np.array(list(product([-.15, .15], [-1., 1.], [-.5, .5]))), (64, 1))
        for index, field in enumerate(['x', 'y', 'z']):
            vertex[field] = xyz[:, index]
        PlyData([PlyElement.describe(vertex, 'vertex')], text=False).write(str(self.source))
        entries = []
        for category in sorted(LONG_AXIS_CATEGORIES):
            entry = self.entry(uid=category, category=category, status='keep')
            np.testing.assert_allclose(entry['canonical']['size_xyz'], [2., .3, 1.], atol=1e-6)
            self.assertIsNone(entry['canonical']['heading'])
            self.assertFalse(entry['canonical']['heading_verified'])
            entry['scale'] = dict(est_length_m=2., est_width_m=.3, est_height_m=1., scale_confidence='medium')
            entries.append(entry)
        # Height-scaled props retain their native horizontal axes.
        cone = self.entry(uid='height_control', category='cone', status='keep')
        np.testing.assert_allclose(cone['canonical']['size_xyz'], [.3, 2., 1.], atol=1e-6)
        cone['scale'] = dict(est_length_m=2., est_width_m=.3, est_height_m=1., scale_confidence='medium')
        entries.append(cone)
        output = self.root / 'long_axis_library'
        manifest = export_library(entries, output)
        jitter = np.random.default_rng(0).uniform(.94, 1.06)
        for entry in AssetLibrary(manifest, output).entries:
            bound = bind_rigid(entry, np.random.default_rng(0))
            expected = [.3, 2., 1.] if entry.category == 'cone' else [2., .3, 1.]
            np.testing.assert_allclose(bound.size_lwh_m, np.array(expected)*jitter, atol=1e-6)
            self.assertLess(max(bound.size_lwh_m), 2.2)

    def test_source_render_audit_uses_one_labeled_image(self):
        source = self.root / 'source.png'
        renders = self.root / 'renders.png'
        # Transparent source must be composited on white in its separate panel.
        reference = Image.new('RGBA', (80, 40), (0, 0, 0, 0))
        reference.paste((255, 0, 0, 255), (20, 10, 60, 30))
        reference.save(source)
        Image.new('RGB', (384, 408), (0, 0, 255)).save(renders)
        path = _audit_sheet(source, renders, self.root / 'audit.jpg')
        messages = _vlm_messages('LEFT SOURCE, RIGHT RENDERS', path, AUDIT_SCHEMA)
        images = [item for item in messages[0]['content'] if item['type'] == 'image_url']
        self.assertEqual(len(images), 1)
        decoded = Image.open(BytesIO(base64.b64decode(images[0]['image_url']['url'].split(',', 1)[1])))
        self.assertEqual(decoded.size, (496, 432))
        # The 80px source is enlarged to the 96px panel, rather than left tiny
        # when the combined sheet is later resized by the image encoder.
        self.assertGreater(decoded.getpixel((25, 48))[0], 180)
        self.assertLess(decoded.getpixel((25, 48))[1], 80)
        self.assertGreater(decoded.getpixel((48, 44))[0], 240)  # red source center
        self.assertLess(decoded.getpixel((48, 44))[2], 20)
        self.assertGreater(decoded.getpixel((300, 200))[2], 240)  # blue render panel
        self.assertGreater(min(decoded.getpixel((15, 25))), 240)  # white source transparency
        self.assertLess(np.asarray(decoded)[0:20, :96].min(), 40)  # SOURCE label
        self.assertLess(np.asarray(decoded)[0:20, 112:].min(), 40)  # RENDERS label

    def test_curation_rechecks_source_and_real_reference_hashes(self):
        source, real = self.root/'source.png', self.root/'real.png'
        source.write_bytes(b'source'); real.write_bytes(b'real')
        row = dict(obj_id='car_001', image_path=str(source), image_sha256=_sha(source),
                   real_images=[dict(rgba_path=str(real), sha256=_sha(real))])
        _validate_sources(row)
        source.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Source image changed'):
            _validate_sources(row)
        source.write_bytes(b'source'); real.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Real reference changed'):
            _validate_sources(row)

    def test_vehicle_identity_variation_retention_preserves_other_gates(self):
        from asset_extraction.export import _identity_variation_only
        audit = dict(keep=True, category_match=True, identity_consistent=False,
                     quality_score=8, scale=dict(scale_confidence='medium'))
        render = dict(all_16_nonempty=True)
        heading = dict(front='A', confidence=.95)
        self.assertTrue(_identity_variation_only(audit, render))
        self.assertEqual(_verdict(audit, render, heading, True, allow_identity_variation=True), 'keep')
        self.assertFalse(audit['identity_consistent'])
        self.assertEqual(_verdict(audit, render, None, True, allow_identity_variation=True), 'review')
        low = dict(audit, scale=dict(scale_confidence='low'))
        self.assertEqual(_verdict(low, render, heading, True, allow_identity_variation=True), 'review')
        for changed in [dict(audit, keep=False), dict(audit, category_match=False), dict(audit, quality_score=4)]:
            self.assertFalse(_identity_variation_only(changed, render))
            self.assertEqual(_verdict(changed, render, heading, True, allow_identity_variation=True), 'reject')
        empty = dict(all_16_nonempty=False)
        self.assertFalse(_identity_variation_only(audit, empty))
        self.assertEqual(_verdict(audit, empty, heading, True, allow_identity_variation=True), 'reject')
        # Non-vehicle callers still use the strict default.
        self.assertEqual(_verdict(audit, render, heading, True), 'reject')

    def test_compact_audit_field_limits(self):
        from jsonschema import Draft202012Validator, ValidationError
        properties = AUDIT_SCHEMA['properties']
        defects = Draft202012Validator(properties['defects'])
        defects.validate(['a'*160]*5)
        for value in [['a']*6, ['a'*161]]:
            with self.assertRaises(ValidationError):
                defects.validate(value)
        with self.assertRaises(ValidationError):
            Draft202012Validator(properties['usable_as']).validate(['role']*6)

    def test_metric_dimensions_allow_decimals_but_reject_zero(self):
        from jsonschema import Draft202012Validator, ValidationError
        from asset_extraction.export import SCALE_SCHEMA
        validator = Draft202012Validator(SCALE_SCHEMA)
        row = dict(object_desc='road cone', est_length_m=.35, est_width_m=.35,
                   est_height_m=.8, scale_confidence='medium')
        validator.validate(row)
        for field in ['est_length_m', 'est_width_m', 'est_height_m']:
            for value in [0, -.1, .0001]:
                with self.assertRaises(ValidationError):
                    validator.validate(dict(row, **{field: value}))

    def test_frontend_context_is_one_image_with_red_target_box(self):
        from asset_extraction.vlm import rigid_vlm_messages
        source, foreground = self.root/'scene.jpg', self.root/'foreground.png'
        Image.new('RGB', (800, 600), (100, 100, 100)).save(source)
        Image.new('RGBA', (80, 100), (0, 255, 0, 255)).save(foreground)
        before = foreground.read_bytes()
        context = dict(source_image=str(source), bbox_xyxy=[350, 250, 450, 350])
        messages = rigid_vlm_messages(foreground, 'trash_can', source_context=context)
        images = [item for item in messages[0]['content'] if item['type']=='image_url']
        self.assertEqual(len(images), 1)
        image = Image.open(BytesIO(base64.b64decode(images[0]['image_url']['url'].split(',', 1)[1])))
        self.assertEqual(image.size, (1280, 664))
        pixels = np.asarray(image)
        red = (pixels[...,0]>180) & (pixels[...,1]<80) & (pixels[...,2]<80)
        self.assertGreater(red[:,640:].sum(), 500)
        self.assertEqual(red[:,:640].sum(), 0)
        self.assertGreater(image.getpixel((320,344))[1], 240)
        self.assertIn('category=unknown', messages[0]['content'][0]['text'])
        self.assertEqual(foreground.read_bytes(), before)
        for bbox in ([1,2,0,4], [0,0,801,600], [0,0,float('nan'),4], [1,2,3]):
            with self.assertRaises(ValueError):
                rigid_vlm_messages(foreground, 'trash_can', source_context=dict(context,bbox_xyxy=bbox))


if __name__ == '__main__':
    unittest.main()
