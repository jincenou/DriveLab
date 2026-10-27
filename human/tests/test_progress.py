import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import batch


class ProgressTests(unittest.TestCase):
    def test_snapshot_requires_delivery_evidence_and_distinguishes_live_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=base/'work';assets=base/'assets';proc=base/'proc'
            proc.mkdir()
            def put(path,data):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(data))
            ids=['H0001','H0002','H0003','H0004','H0005']
            put(root/'PRODUCTION_INPUTS.json',dict(ids=ids,target_count=300))
            put(root/'jobs/H0001.json',dict(state='failed_or_needs_review'))
            put(root/'jobs/H0002.json',dict(state='running'))
            put(root/'jobs/H0003.json',dict(state='running'))
            for identity in ('H0001','H0005'):
                put(assets/identity/'package_complete.json',dict(state='complete_pending_visual_review'))
                put(assets/identity/'runtime_replay.json',dict(state='published_runtime_replay_passed'))
                put(assets/identity/'presentation.json',dict(state='five_column_video_verified'))
                put(assets/identity/'insertion.json',dict(skip_insertion=identity=='H0001'))
            pid=proc/'123';pid.mkdir()
            (pid/'cmdline').write_bytes(b'python\0'+str(batch.CODE_ROOT/'reconstruction.py').encode()+b'\0native\0H0002\0')
            (pid/'environ').write_bytes(('HUMAN_RUN_ROOT='+str(root)).encode()+b'\0')
            monitor=proc/'124';monitor.mkdir()
            (monitor/'cmdline').write_bytes(b'python\0'+str(batch.CODE_ROOT/'run.py').encode()+b'\0batch\0progress\0--watch\0')
            (monitor/'environ').write_bytes(('HUMAN_RUN_ROOT='+str(root)).encode()+b'\0')
            def validate(package):
                if package.name=='H0005':raise AssertionError('Sidecar hash mismatch')
                return {}
            before={str(p):p.read_bytes()for p in assets.rglob('*')if p.is_file()}
            with patch.object(batch,'ROOT',root),patch.object(batch,'ASSET_ROOT',assets),patch('playback.validate_estimate',side_effect=validate):
                result=batch.progress_snapshot(proc)
            states={x['identity']:x['state']for x in result['assets']}
            self.assertEqual(states,dict(H0001='technical_complete_visual_review_pending',
                H0002='active_process_confirmed',H0003='job_state_unverified_process_not_visible',
                H0004='not_started',H0005='published_evidence_needs_review'))
            self.assertEqual(result['insertion_excluded'],1)
            self.assertEqual([p['pid'] for p in result['visible_processes']],[123])
            self.assertEqual(sum(result['counts'].values()),len(ids))
            self.assertEqual(before,{str(p):p.read_bytes()for p in assets.rglob('*')if p.is_file()})
            self.assertEqual(json.loads((root/'jobs/H0001.json').read_text())['state'],'failed_or_needs_review')


if __name__=='__main__':unittest.main()
