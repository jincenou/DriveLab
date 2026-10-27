"""Dataset geometry and shared-library regressions; no model inference required."""
import copy
import tempfile
import unittest
import asyncio
import types
from unittest.mock import patch, AsyncMock
from pathlib import Path
import numpy as np
from common import asset_uid, merge_catalogue, save, read, sha
from intake import (corners, rigid, dense_observations, admit_prepared, review_inputs,
                    maintain_inputs, attachment_decision, screen_attachment,
                    retire_rejected_inputs, raise_target, expand_production, process_running)


class IntakeContractTests(unittest.TestCase):
    def test_completed_zombie_does_not_block_queue_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'123';p.mkdir()
            (p/'status').write_text('Name:\tpython\nState:\tZ (zombie)\n')
            self.assertFalse(process_running(123,root))
            (p/'status').write_text('Name:\tpython\nState:\tS (sleeping)\n')
            self.assertTrue(process_running(123,root))
            self.assertFalse(process_running(456,root))

    def test_audit_freeze_respects_dataset_registry_in_shared_asset_library(self):
        from audit import freeze
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ',{'HUMAN_RUN_ROOT':tmp}):
            root=Path(tmp);assets=root/'assets';out=root/'audit'
            save(root/'config/cases.json',{'HN0004':{}})
            for identity in ['H0001','HN0004']:
                package=assets/identity
                save(package/'package_complete.json',dict(state='complete_pending_visual_review'))
                (package/'demo.mp4').write_bytes(b'video')
            freeze(out,assets)
            self.assertEqual([r['identity'] for r in read(out/'protocol.json')['packages']],['HN0004'])

    def test_incremental_audit_selects_requested_complete_packages_only(self):
        from audit import freeze
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'HUMAN_RUN_ROOT':tmp}):
            root=Path(tmp);assets=root/'assets'
            save(root/'config/cases.json',{'HN0004':{},'HN0005':{}})
            for identity in ['H0001','HN0004','HN0005']:
                package=assets/identity
                save(package/'package_complete.json',dict(state='complete_pending_visual_review'))
                (package/'demo.mp4').write_bytes(b'video')
            freeze(root/'selected',assets,['HN0005'])
            self.assertEqual([r['identity'] for r in read(root/'selected/protocol.json')['packages']],['HN0005'])
            with self.assertRaises(AssertionError):freeze(root/'foreign',assets,['H0001'])
            with self.assertRaises(AssertionError):freeze(root/'missing',assets,['HN9999'])
            with self.assertRaises(AssertionError):freeze(root/'duplicate',assets,['HN0004','HN0004'])

    def test_audit_watch_reuses_matching_evidence_and_does_not_approve_or_repeat_errors(self):
        import audit
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'HUMAN_RUN_ROOT':tmp}):
            root=Path(tmp);assets=root/'assets';previous=root/'old';out=root/'watch'
            save(root/'config/cases.json',{'HN0004':{},'HN0005':{},'HN0006':{},'HN0007':{}})
            save(root/'PRODUCTION_INPUTS.json',dict(ids=['HN0004','HN0005','HN0006','HN0007'],input_selection_complete=True))
            for identity in ['HN0004','HN0005','HN0006']:
                package=assets/identity
                save(package/'package_complete.json',dict(state='complete_pending_visual_review'))
                (package/'demo.mp4').write_bytes(identity.encode())
            audit.freeze(previous,assets,['HN0004'])
            def evidence(directory,identities,failed=()):
                for identity in identities:
                    if identity in failed:
                        save(directory/'errors'/f'{identity}.json',dict(error='temporary service error'))
                        continue
                    contact=directory/'contacts'/f'{identity}.jpg';contact.parent.mkdir(exist_ok=True)
                    contact.write_bytes(b'contact')
                    save(directory/'technical'/f'{identity}.json',dict(source_sha256=sha(assets/identity/'demo.mp4'),contact_sha256=sha(contact)))
                    save(directory/'vlm'/f'{identity}.json',dict(contact_sha256=sha(contact),result=dict(triage='no_gross_issue_observed')))
                save(directory/'status.json',dict(state='completed_with_errors' if failed else 'completed_requires_visual_crosscheck',
                    cases={i:'failed' if i in failed else 'completed_requires_visual_crosscheck' for i in identities}))
            evidence(previous,['HN0004'])
            before={str(p):p.read_bytes() for p in assets.rglob('*') if p.is_file()}
            live=[True];called=[]
            async def run_new(directory):
                identities=[r['identity'] for r in read(directory/'protocol.json')['packages']]
                called.append(identities)
                evidence(directory,identities,failed=['HN0006'])
                live[0]=False
            with patch.object(audit,'run',side_effect=run_new),patch.object(audit,'watched_process',side_effect=lambda *a:'start' if live[0] else None):
                asyncio.run(audit.watch(out,assets,[123],[previous],30))
            self.assertEqual(called,[['HN0005','HN0006']])
            state=read(out/'status.json')
            self.assertEqual(state['failed'],['HN0006'])
            self.assertEqual(state['production_ids_not_published'],['HN0007'])
            self.assertTrue(all(not r['quality_approved'] for r in state['cases']))
            self.assertFalse(state['library_modified'])
            self.assertEqual(before,{str(p):p.read_bytes() for p in assets.rglob('*') if p.is_file()})
            (assets/'HN0004/demo.mp4').write_bytes(b'changed')
            self.assertEqual(audit.prior_attempts([previous],assets),{})
            protocol=read(previous/'protocol.json');protocol['prompt']='different protocol';save(previous/'protocol.json',protocol)
            with self.assertRaises(AssertionError):audit.prior_attempts([previous],assets)

    def test_retirement_keeps_queued_published_unknown_and_live_inputs(self):
        import fcntl
        with tempfile.TemporaryDirectory() as tmp, patch('intake.ROOT',Path(tmp)), \
             patch('common.ASSET_ROOT',Path(tmp)/'published'):
            root=Path(tmp);ids=[f'HN{i:04d}' for i in range(1,7)]
            save(root/'PRODUCTION_INPUTS.json',dict(ids=[ids[1]],target_count=300))
            save(root/'selection.json',dict(items=[dict(id=i,identity_key=i) for i in ids]))
            for i in ids:
                path=root/'inputs'/i;path.mkdir(parents=True);(path/'pixels.bin').write_bytes(b'pixels')
                save(root/'jobs'/f'intake_{i}.json',dict(state='failed',error='test'))
            (root/'published'/ids[2]).mkdir(parents=True)
            save(root/'review'/f'{ids[3]}_attachment_extent.json',dict(decision='needs_review'))
            (root/'outputs'/ids[4]).mkdir(parents=True)
            with (root/'jobs'/f'{ids[5]}.lock').open('a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                result=retire_rejected_inputs()
            self.assertEqual([r['identity'] for r in result],[ids[0]])
            self.assertFalse((root/'inputs'/ids[0]).exists())
            self.assertTrue(all((root/'inputs'/i).exists() for i in ids[1:]))
            receipt=read(root/'reports'/f'{ids[0]}_input_retired.json')
            self.assertEqual(receipt['reason'],'technical_preparation_failure')
            self.assertTrue(read(root/'review'/f'{ids[0]}_generation_exclusion.json')['excluded'])

    def test_target_expansion_preserves_existing_entries_and_has_a_ceiling(self):
        with tempfile.TemporaryDirectory() as tmp, patch('intake.ROOT',Path(tmp)):
            root=Path(tmp);(root/'jobs').mkdir()
            save(root/'PRODUCTION_INPUTS.json',dict(ids=['HN0004'],target_count=100,input_selection_complete=True))
            result=raise_target(300)
            self.assertEqual(result['ids'],['HN0004'])
            self.assertFalse(result['input_selection_complete'])
            with self.assertRaises(AssertionError):raise_target(301)

    def test_expansion_stops_short_without_reducing_quality_rules(self):
        with tempfile.TemporaryDirectory() as tmp, patch('intake.ROOT',Path(tmp)), \
             patch('intake.expansion_quality_signature',return_value={'frozen':'same'}), \
             patch('intake.fill_production',new_callable=AsyncMock), patch('intake.retire_rejected_inputs'):
            root=Path(tmp);(root/'jobs').mkdir()
            save(root/'PRODUCTION_INPUTS.json',dict(ids=['HN0004'],target_count=100))
            save(root/'reserve.json',dict(items=[dict(id='HN0010',instance_token='old')],rule={'min_height':140}))
            def fake_rank(*args,**kwargs):
                save(kwargs['destination'],dict(items=[dict(id='HN0011',instance_token='new')],rule={'min_height':140}))
            with patch('intake.rank',side_effect=fake_rank):
                asyncio.run(expand_production(300,800,[4,7],[]))
            result=read(root/'PRODUCTION_INPUTS.json')
            self.assertEqual(result['ids'],['HN0004'])
            self.assertEqual(result['input_shortfall'],299)
            self.assertTrue(result['input_selection_complete'])
            self.assertEqual(read(root/'EXPANSION.json')['state'],'stopped_at_available_quality')
            self.assertEqual(read(root/'reserve.json')['rule'],{'min_height':140})

    def test_large_cropped_object_is_withheld_but_whole_bags_are_allowed(self):
        value=dict(large_handled_object=True,incomplete_in_model_input=True,
                   evidence_slots=[0,30,60],reason='Cart handles leave the input crop')
        self.assertEqual(attachment_decision(value),'reject')
        self.assertEqual(attachment_decision(dict(value,large_handled_object=False)),'allow')
        self.assertEqual(attachment_decision(dict(value,incomplete_in_model_input=False)),'allow')
        self.assertEqual(attachment_decision(dict(value,evidence_slots=[])),'needs_review')
        self.assertEqual(attachment_decision(dict(value,large_handled_object=None)),'needs_review')
        with self.assertRaises(ValueError):attachment_decision(dict(value,large_handled_object='false'))

    def test_absent_attachment_policy_keeps_existing_dataset_behavior(self):
        with tempfile.TemporaryDirectory() as tmp, patch('intake.ROOT',Path(tmp)):
            result=asyncio.run(screen_attachment('H0001'))
            self.assertEqual(result,dict(decision='not_enabled',allow_generation=True))

    def test_explicit_rejection_cannot_be_bypassed_by_disabled_policy(self):
        with tempfile.TemporaryDirectory() as tmp, patch('intake.ROOT',Path(tmp)):
            save(Path(tmp)/'review/HN0026_generation_exclusion.json',dict(excluded=True))
            self.assertFalse(asyncio.run(screen_attachment('HN0026'))['allow_generation'])

    def test_attachment_api_failure_does_not_write_an_allow_receipt(self):
        with tempfile.TemporaryDirectory() as tmp, patch('intake.ROOT',Path(tmp)), patch('common.ROOT',Path(tmp)):
            root=Path(tmp);source=root/'input.mp4';source.write_bytes(b'input')
            contact=root/'review/HN0026.jpg';contact.parent.mkdir();contact.write_bytes(b'contact')
            save(root/'config/cases.json',{'HN0026':dict(raw_input=str(source))})
            save(root/'ATTACHMENT_POLICY.json',dict(mode='enforce'))
            save(root/'review/HN0026.json',dict(image_sha256=sha(contact),sources=[dict(path=str(source),sha256=sha(source))]))
            client=types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(
                create=AsyncMock(side_effect=ConnectionError('offline')))))
            with self.assertRaises(ConnectionError):asyncio.run(screen_attachment('HN0026',client))
            self.assertFalse((root/'review/HN0026_attachment_extent.json').exists())

    def test_nuscenes_size_and_camera_roundtrip(self):
        object_box=dict(translation=[1,2,8],rotation=[1,0,0,0],size=[2,4,6])
        vertices=corners(object_box)
        np.testing.assert_allclose(np.ptp(vertices,axis=0),[4,2,6])
        camera=rigid(dict(translation=[2,-1,3],rotation=[2**-.5,0,0,2**-.5]))
        incam=(vertices-camera[:3,3])@camera[:3,:3]
        np.testing.assert_allclose(incam@camera[:3,:3].T+camera[:3,3],vertices,atol=1e-12)

    def test_dense_frames_follow_actual_camera_exposures(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'original.jpg';path.touch()
            keys=[];images={};poses={}
            for i in range(7):
                keys.append(dict(sample_timestamp_s=100+i*.5,sample_data_token='s0',annotation_token=f'a{i}',visibility=4,
                    object_world=dict(translation=[i*.1,0,8],size=[.5,.5,1.8],rotation=[1,0,0,0])))
            for i in range(33):
                images[f's{i}']=dict(token=f's{i}',timestamp=round((100+i/12)*1e6),next=f's{i+1}' if i<32 else '',filename=str(path),
                    calibrated_sensor_token='cal',ego_pose_token=f'p{i}',width=1600,height=900)
                poses[f'p{i}']=dict(rotation=[1,0,0,0],translation=[0,0,0])
            sensors={'cal':dict(rotation=[1,0,0,0],translation=[0,0,0],camera_intrinsic=[[700,0,800],[0,700,450],[0,0,1]])}
            result=dense_observations(keys,images,sensors,poses)
            self.assertEqual(len(result),30)
            self.assertEqual(result[1]['timestamp_s'],images['s1']['timestamp']/1e6)
            self.assertEqual(result[1]['sample_data_token'],'s1')
            self.assertGreater(result[-1]['timestamp_s']-result[0]['timestamp_s'],2.4)
            self.assertEqual(result[1]['annotation_tokens'],['a0','a1'])
            images['s1']['next']='s5'
            with self.assertRaisesRegex(ValueError,'Camera gap'):dense_observations(keys,images,sensors,poses)

    def test_other_dataset_catalogue_entries_survive_unchanged(self):
        old=[dict(global_uid=asset_uid('H0001','waymo'),package_path='H0001',custom={'quality':'unverified'}),
             dict(global_uid=asset_uid('HN0003','nuscenes'),package_path='HN0003',version=0)]
        before=copy.deepcopy(old)
        current=[dict(global_uid=asset_uid('HN0003','nuscenes'),package_path='HN0003',version=1)]
        result=merge_catalogue(old,current,[current[0]['global_uid']])
        self.assertEqual(result[0],before[0]);self.assertEqual(old,before)
        self.assertEqual(result[1]['version'],1)
        collision=[dict(global_uid=asset_uid('HN0003','nuscenes'),package_path='H0001')]
        with self.assertRaisesRegex(ValueError,'collide'):merge_catalogue(old,collision,[current[0]['global_uid']])

    def test_admission_checks_current_evidence_and_never_exceeds_target(self):
        with tempfile.TemporaryDirectory() as tmp, patch('intake.ROOT', Path(tmp)), patch('common.ROOT', Path(tmp)):
            root = Path(tmp)
            save(root/'PRODUCTION_INPUTS.json', dict(ids=['HN0004'], target_count=2))
            save(root/'selection.json', dict(items=[dict(id=i) for i in ['HN0004','HN0018','HN0019']]))
            registry = {}
            for identity in ['HN0018','HN0019']:
                source = root/f'{identity}.mp4'; source.write_bytes(b'test-source')
                image = root/'review'/f'{identity}.jpg'; image.parent.mkdir(exist_ok=True); image.write_bytes(b'contact')
                instance = root/'review'/f'{identity}_instances.json'; save(instance, dict(one_person_every_sample=True))
                registry[identity] = dict(raw_input=str(source), input_review=dict(allow_generation=True,input_sha256=sha(source)))
                save(root/'review'/f'{identity}_admission.json', dict(allow_exploratory_generation=True,
                    signature=dict(instance_check_sha256=sha(instance),contact_sha256=sha(image),source_sha256={str(source):sha(source)})))
            save(root/'config/cases.json', registry)
            # A changed contact invalidates prior approval even if allow=true.
            (root/'review/HN0018.jpg').write_bytes(b'changed')
            result = admit_prepared()
            self.assertEqual(result['ids'], ['HN0004','HN0019'])
            self.assertTrue(result['input_selection_complete'])
            self.assertEqual(admit_prepared()['ids'], result['ids'])

    def test_explicit_withdrawal_is_not_readmitted_and_target_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp, patch('intake.ROOT',Path(tmp)), patch('common.ROOT',Path(tmp)):
            root=Path(tmp)
            save(root/'PRODUCTION_INPUTS.json',dict(ids=['HN0026'],target_count=100))
            save(root/'selection.json',dict(items=[dict(id='HN0026')]))
            save(root/'config/cases.json',{'HN0026':dict(input_review=dict(allow_generation=True))})
            save(root/'review/HN0026_generation_exclusion.json',dict(excluded=True))
            result=admit_prepared()
            self.assertEqual(result['ids'],[])
            self.assertEqual(result['target_count'],100)
            self.assertFalse(result['input_selection_complete'])

    def test_native_video_preserves_exposure_timing_and_pixels(self):
        import av
        from PIL import Image
        from prepare import encode
        images=[Image.new('RGB',(32,32),(i*40,12,90)) for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'native.mp4'
            encode(target,images,fps=12,timestamps=[100,100.083333,100.166667])
            with av.open(str(target)) as video:
                frames=list(video.decode(video=0))
            self.assertEqual(len(frames),3)
            np.testing.assert_allclose([float(f.pts*f.time_base) for f in frames],[0,1/12,2/12],atol=1e-4)
            for source, decoded in zip(images,frames,strict=True):
                np.testing.assert_array_equal(np.asarray(source),np.asarray(decoded.to_image()))

    def test_cached_review_uses_original_registry_without_network_or_shadowing(self):
        class Client:
            def __init__(self, **kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
        with tempfile.TemporaryDirectory() as tmp, patch('intake.ROOT', Path(tmp)), \
             patch('common.ROOT', Path(tmp)), patch.dict('sys.modules', {'openai': types.SimpleNamespace(AsyncOpenAI=Client)}):
            root=Path(tmp);identity='HN0018';source=root/'source.mp4';source.write_bytes(b'source')
            contact=root/'review'/f'{identity}.jpg';contact.parent.mkdir();contact.write_bytes(b'contact')
            instance=root/'review'/f'{identity}_instances.json';save(instance,dict(source_sha256=sha(source)))
            meta=dict(image_sha256=sha(contact),sources=[dict(path=str(source),sha256=sha(source))],frames=[0,30,60])
            save(root/'review'/f'{identity}.json',meta)
            save(root/'config/cases.json',{identity:dict(raw_input=str(source))})
            save(root/'review'/f'{identity}_admission.json',dict(allow_exploratory_generation=True,
                signature=dict(contact_sha256=sha(contact),instance_check_sha256=sha(instance),source_sha256={str(source):sha(source)})))
            with patch('frontend.temporal_diagnosis',side_effect=AssertionError('Cached review must not rerun pose diagnosis')):
                asyncio.run(review_inputs())
            self.assertEqual(read(root/'INPUT_PROGRESS.json')['errors'],[])
            self.assertEqual(read(root/'INPUT_PROGRESS.json')['automatically_admitted'],1)

    def test_completed_input_target_does_not_start_another_wave(self):
        with tempfile.TemporaryDirectory() as tmp, patch('intake.ROOT',Path(tmp)), \
             patch('intake.fill_production',new_callable=AsyncMock) as fill:
            save(Path(tmp)/'PRODUCTION_INPUTS.json',dict(ids=['HN0001'],target_count=1))
            asyncio.run(maintain_inputs([4,7],[]))
            fill.assert_not_awaited()

    def test_empty_reserves_report_shortfall_without_weakening_admission(self):
        with tempfile.TemporaryDirectory() as tmp, patch('intake.ROOT',Path(tmp)), \
             patch('intake.fill_production',new_callable=AsyncMock):
            root=Path(tmp);(root/'reports').mkdir()
            save(root/'PRODUCTION_INPUTS.json',dict(ids=['HN0001'],target_count=2,input_selection_complete=False))
            save(root/'reserve.json',dict(items=[]))
            with patch('intake.rank',side_effect=lambda *args,**kwargs:save(root/'reserve.json',dict(items=[]))):
                asyncio.run(maintain_inputs([4,7],[]))
            result=read(root/'PRODUCTION_INPUTS.json')
            self.assertEqual(result['ids'],['HN0001'])
            self.assertEqual(result['input_shortfall'],1)
            self.assertEqual(read(root/'FILL_STATUS.json')['state'],'selection_finished_with_shortfall')

if __name__=='__main__':unittest.main()
