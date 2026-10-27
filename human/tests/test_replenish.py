import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import batch
from common import save,sha


class ReplenishTests(unittest.TestCase):
    def test_swap_preserves_history_and_rejects_unsafe_admission(self):
        for fault in (None, 'running', 'published', 'unreviewed', 'changed_input', 'duplicate', 'child_live'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'work';assets=Path(tmp)/'assets'
                (root/'jobs').mkdir(parents=True);assets.mkdir()
                ids=[f'H{i:04d}' for i in range(1,301)]
                save(root/'PRODUCTION_INPUTS.json',dict(ids=ids,target_count=300,replacement_history=[]))
                save(root/'jobs/H0001.json',dict(state='running' if fault=='running' else 'failed_or_needs_review',error='retained'))
                save(root/'TEMPORAL_POLICY.json',dict(version='unchanged'))
                if fault=='published':save(assets/'H0001/package_complete.json',dict(state='complete_pending_visual_review'))
                source=root/'source';source.write_bytes(b'original RGB')
                motion=root/'motion';motion.write_bytes(b'frozen motion')
                contact=root/'review/H0301.jpg';contact.parent.mkdir();contact.write_bytes(b'contact evidence')
                save(root/'review/H0301.json',dict(frames=[0,30,60],image_sha256=sha(contact),sources=[dict(path=str(source),sha256=sha(source))]))
                spec=dict(selection=dict(identity_key='H0002' if fault=='duplicate' else 'new-track'),raw_input=str(source),
                          input_review=dict(allow_generation=fault!='unreviewed',reviewed_slots=[0,30,60],input_sha256=sha(source)))
                plan=root/'plan.json'
                save(plan,dict(approved=True,temporal_policy_sha256=sha(root/'TEMPORAL_POLICY.json'),pairs=[dict(old='H0001',new='H0301',reason='reviewed',input_sha256=sha(source),temporal_decision='keep_full',temporal_source_sha256={str(motion):sha(motion)})]))
                if fault=='changed_input':source.write_bytes(b'changed RGB')
                original=(root/'PRODUCTION_INPUTS.json').read_bytes();job=(root/'jobs/H0001.json').read_bytes()
                library={str(p):p.read_bytes() for p in assets.rglob('*') if p.is_file()}
                def get_case(identity):
                    return spec if identity=='H0301' else dict(selection=dict(identity_key=identity))
                with patch.object(batch,'ROOT',root),patch.object(batch,'ASSET_ROOT',assets),patch.object(batch,'case',side_effect=get_case),patch.object(batch,'active_child',return_value=[123] if fault=='child_live' else []):
                    if fault:
                        with self.assertRaises(AssertionError):batch.replenish_inputs(plan)
                        self.assertEqual(original,(root/'PRODUCTION_INPUTS.json').read_bytes())
                    else:
                        result=batch.replenish_inputs(plan)
                        self.assertEqual(result['replaced'],1)
                        p=json.loads((root/'PRODUCTION_INPUTS.json').read_text())
                        self.assertEqual(len(p['ids']),300);self.assertEqual(p['ids'][0],'H0301')
                        self.assertEqual(p['replacement_history'][0]['pairs'][0]['old'],'H0001')
                self.assertEqual(job,(root/'jobs/H0001.json').read_bytes())
                self.assertEqual(library,{str(p):p.read_bytes() for p in assets.rglob('*') if p.is_file()})


if __name__=='__main__':unittest.main()
