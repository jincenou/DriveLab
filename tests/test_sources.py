"""Exercise real video decoding, source identity and cross-controller GPU exclusion."""
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch
import unittest
import cv2
import numpy as np

from asset_extraction.sources import video, scene_key
from asset_extraction.generation import GpuPool
from asset_extraction.vision import _select_for_sam


class SourcesTests(unittest.TestCase):
    def test_video_timestamps_and_motion_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'clip.avi'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,64))
            for i in range(20):
                writer.write(np.full((64,64,3),i*10,np.uint8))
            writer.release()
            index=json.loads(video(source,root/'prepared').read_text())
            self.assertEqual(len(index['frames']),4)
            self.assertEqual([r['timestamp_us'] for r in index['frames']],[0,500000,1000000,1500000])
            self.assertEqual(len(index['frames'][1]['motion_paths']),4)
            self.assertFalse(index['calibration_available'])
            self.assertTrue(all(Path(r['path']).is_file() for r in index['frames']))
            old=scene_key(source)
            with source.open('ab') as f:f.write(b'changed')
            self.assertNotEqual(old,scene_key(source))

    def test_zero_budget_keeps_all_identity_queues(self):
        tracks=[[dict(category='cone',pre_score=1.,frame_id=i,bbox_xyxy=[0,0,40,50])] for i in range(400)]
        self.assertEqual(len(_select_for_sam(tracks,1,0)),400)
        self.assertEqual(len(_select_for_sam(tracks,1,24)),24)


class LeaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_pools_cannot_claim_same_gpu_and_release_reopens_slot(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict('os.environ',DRIVELAB_GPU_LOCK_DIR=temp), \
             patch('asset_extraction.generation.gpu_free_mib',AsyncMock(return_value={0:80000})):
            first=GpuPool([0],poll_seconds=.01,timeout_seconds=.03)
            second=GpuPool([0],poll_seconds=.01,timeout_seconds=.03)
            self.assertEqual(await first.acquire('H'),0)
            with self.assertRaises(TimeoutError):await second.acquire('H')
            first.release(0)
            self.assertEqual(await second.acquire('render'),0)
            second.release(0)


if __name__=='__main__':
    unittest.main()
