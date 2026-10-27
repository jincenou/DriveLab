"""Pause new admissions without interrupting an admitted asset's export."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import batch


class ProductionPauseTests(unittest.TestCase):
    def test_device_handoff_requires_whole_asset_and_all_workers_drained(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(batch, 'ROOT', Path(directory)):
            root=Path(directory)
            plan=dict(drain_ids=['HN0001'],previous_gpus=[0,1],gpus=[2,3,4,5,6,7])
            policy=dict(paused=True,drain_ids=plan['drain_ids'],next_gpus=plan['gpus'])
            common.save(root/'PRODUCTION_PAUSE.json',policy)
            common.save(root/'jobs/HN0001.json',dict(state='running',stage='reconstruction_A'))
            common.save(root/'jobs/worker_gpu0.json',dict(state='running'))
            common.save(root/'jobs/worker_gpu1.json',dict(state='paused_pending_asset_metadata_validation'))
            self.assertFalse(batch.handoff_ready(plan))
            common.save(root/'jobs/HN0001.json',dict(state='replayed_visual_review_pending'))
            self.assertFalse(batch.handoff_ready(plan))
            common.save(root/'jobs/worker_gpu0.json',dict(state='paused_pending_asset_metadata_validation'))
            self.assertTrue(batch.handoff_ready(plan))
            policy['paused']=False
            common.save(root/'PRODUCTION_PAUSE.json',policy)
            with self.assertRaisesRegex(RuntimeError,'policy changed'):
                batch.handoff_ready(plan)

    def test_handoff_retries_only_explicitly_prepared_unchanged_failures(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(batch,'ROOT',Path(directory)), \
             patch.object(batch,'verified_package',side_effect=lambda i:i=='HN0001'):
            root=Path(directory);ids=['HN0001','HN0002','HN0003','HN0004','HN0005']
            common.save(root/'PRODUCTION_INPUTS.json',dict(ids=ids))
            for identity in ['HN0002','HN0003','HN0004']:
                common.save(root/'jobs'/f'{identity}.json',dict(state='failed_or_needs_review',error='CUDA OOM'))
            plan=dict(cases=ids,prepared_failures={i:common.sha(root/'jobs'/f'{i}.json')
                                                for i in ['HN0002','HN0003']})
            common.save(root/'jobs/HN0003.json',dict(state='failed_or_needs_review',error='New failure'))
            ready,withheld=batch.handoff_cases(plan)
            self.assertEqual(ready,['HN0002','HN0005'])
            self.assertEqual([x['identity'] for x in withheld],['HN0003','HN0004'])
            common.save(root/'jobs/HN0005.json',dict(state='running'))
            with self.assertRaisesRegex(RuntimeError,'unexpected job state'):
                batch.handoff_cases(plan)

    def test_drain_ids_and_future_queue(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(common, "ROOT", Path(directory)):
            self.assertFalse(common.production_paused("H0001"))
            common.save(common.ROOT / "PRODUCTION_PAUSE.json", {"paused": True, "drain_ids": ["H0001"]})
            self.assertFalse(common.production_paused("H0001"))
            self.assertTrue(common.production_paused("H0002"))
            self.assertTrue(common.production_paused())

    def test_old_controller_child_waits_then_resumes(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(common, "ROOT", Path(directory)), \
             patch.object(sys, "argv", [str(common.CODE_ROOT / "frontend.py"), "generate", "H0002"]), \
             patch.dict(common.os.environ, {"HUMAN_GPU_UUID": "GPU-test"}):
            common.save(common.ROOT / "PRODUCTION_PAUSE.json", {"paused": True, "drain_ids": ["H0001"]})
            def release(_):
                common.save(common.ROOT / "PRODUCTION_PAUSE.json", {"paused": False})
            with patch.object(common.time, "sleep", side_effect=release) as sleep:
                common.wait_for_production_permission()
                self.assertEqual(sleep.call_count, 1)
            receipt = next((common.ROOT / "jobs").glob("pause_wait_*.json"))
            self.assertEqual(common.read(receipt)["state"], "resumed")

    def test_admitted_child_does_not_wait(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(common, "ROOT", Path(directory)), \
             patch.object(sys, "argv", [str(common.CODE_ROOT / "batch.py"), "package", "H0001"]), \
             patch.dict(common.os.environ, {"HUMAN_GPU_UUID": "GPU-test"}):
            common.save(common.ROOT / "PRODUCTION_PAUSE.json", {"paused": True, "drain_ids": ["H0001"]})
            with patch.object(common.time, "sleep") as sleep:
                common.wait_for_production_permission()
                sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
