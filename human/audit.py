"""Read-only published asset audit; evidence stays outside the asset library."""
import argparse
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parent
from common import PYTHONS, ASSET_ROOT
PYTHON = str(PYTHONS['4danyone'])
VLM_PYTHON = str(PYTHONS['vllm_vlm'])
PROMPT = '''Review ONE human asset from a contact sheet. Rows are chronological
sampled video frames; columns are (1) real dataset observation, (2) estimated
skeleton overlay, (3) generated 4DAnyone view00, (4) reconstructed view00,
(5) reconstructed view08. Compare column 3 to 4 for backend damage; column 5
has a different viewpoint, so do not require pixel or facing-direction equality.
The real camera moves. Screen motion is NOT physical speed or walking direction.
Check gross limb loss/deformation, body truncation, generated occluding planes,
large white/black artifacts, identity/clothing color changes, and carried objects
deleted or whitened. Only claim an object belongs to the target when visible.
Dark source clothing is not automatically a blackening failure. Some source
occlusions can be plausibly completed; do not reject on occlusion alone.
Small faces/hands and blur must not be described as verified detailed defects.
No visible floor in reconstruction: ground contact, metric speed, no-slip and
seamless looping cannot be certified. Sparse frames cannot establish absence of
temporal flicker. Do not invent unsampled events, exact measurements, or repairs.
Return one JSON object with exactly these keys:
triage: "severe_candidate" | "review_needed" | "no_gross_issue_observed" | "uncertain";
action: "standing" | "walking" | "running" | "gesture" | "mixed" | "unknown";
issues: list of {type: "limb_loss"|"deformation"|"occluding_plane"|"color_change"|
"accessory_loss"|"source_truncation"|"other", stage: "input"|"generation"|
"reconstruction"|"uncertain", frames: list of supplied frame IDs,
description: string, severity: "major"|"minor"|"uncertain"};
evidence: string; uncertainties: list of strings.
Use Chinese explanations. This is unverified model triage, never asset acceptance.'''


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.writing')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def published_packages(assets):
    """Only this run's completed deliveries, never another dataset's library."""
    run_root = Path(os.environ['HUMAN_RUN_ROOT']).resolve()
    registered = set(read(run_root/'config/cases.json'))
    return sorted(p.parent for p in assets.glob('H*/package_complete.json')
                  if p.parent.name in registered
                  and read(p).get('state') == 'complete_pending_visual_review')


def freeze(out, assets, identities=None):
    assert not (out / 'protocol.json').exists(), 'Frozen audit already exists'
    run_root = Path(os.environ['HUMAN_RUN_ROOT']).resolve()
    assert (run_root/'config/cases.json').is_file()
    packages = published_packages(assets)
    if identities is not None:
        requested = set(identities)
        assert len(requested) == len(identities), 'Duplicate audit identity'
        packages = [p for p in packages if p.name in requested]
        assert {p.name for p in packages} == requested, 'Requested package not complete or not owned'
    assert packages
    save(out / 'protocol.json', dict(created_at=time.time(), assets=str(assets),run_root=str(run_root),
        packages=[dict(identity=p.name, marker_sha256=digest(p/'package_complete.json'),
                       demo_sha256=digest(p/'demo.mp4')) for p in packages],
        code_sha256=digest(__file__), prompt=PROMPT, sample_rule='five uniform integer frames',
        technical_concurrency=3, vlm_concurrency=2,
        model='Qwen/Qwen3-VL-8B-Instruct', base_url='http://127.0.0.1:8082/v1',
        output_scope='Read-only library audit, no fresh GPU replay; VLM is unverified sparse triage',
        response_format={'type':'json_object'}))


def technical(out, identity):
    import av
    import numpy as np
    from PIL import Image, ImageDraw
    from batch import verified_package
    from common import ROOT, ASSET_ROOT
    protocol = read(out/'protocol.json')
    assert ROOT == Path(protocol['run_root']) and ASSET_ROOT == Path(protocol['assets'])
    row = next(x for x in protocol['packages'] if x['identity'] == identity)
    package = Path(protocol['assets'])/identity
    assert digest(package/'package_complete.json') == row['marker_sha256']
    assert verified_package(identity), 'Published integrity or coordinate contract failed'
    assert digest(package/'demo.mp4') == row['demo_sha256']
    meta = read(package/'presentation.json')
    assert meta['sha256'] == row['demo_sha256']
    expected = meta['frame_count']
    slots = np.unique(np.linspace(0, expected-1, 5, dtype=int)).tolist()
    images = {}
    observed = []
    with av.open(str(package/'demo.mp4')) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        for index, frame in enumerate(container.decode(video=0)):
            assert [frame.width, frame.height] == meta['size']
            assert abs(float(frame.pts*frame.time_base) - index/meta['fps']) < 1e-5
            observed.append(index)
            if index in slots:
                images[index] = frame.to_image()
    assert len(observed) == expected and sorted(images) == slots
    sheet = Image.new('RGB', (1100, 440*len(slots)), (55,55,55))
    draw = ImageDraw.Draw(sheet)
    for row_index, slot in enumerate(slots):
        image = images[slot]
        image.thumbnail((1100,420))
        sheet.paste(image,(0,row_index*440+20))
        draw.text((5,row_index*440+3),f'{identity} frame {slot}: source | skeleton | generated00 | GS00 | GS08',fill='white')
    sheet.save(out/'contacts'/f'{identity}.jpg', quality=90)
    placement = read(package/'placement.json')
    insertion = read(package/'insertion.json')
    practical = insertion['practical_playback']
    quality = read(package/'quality.json')
    assert digest(package/'package_complete.json') == row['marker_sha256'], 'Asset changed during audit'
    save(out/'technical'/f'{identity}.json',dict(identity=identity,state='integrity_and_decode_passed',
        frame_count=expected,sampled_frames=slots,source_sha256=row['demo_sha256'],
        contact_sha256=digest(out/'contacts'/f'{identity}.jpg'),
        reconstruction_metrics=quality['mean'], metrics_reference='4DAnyone generated reference, not real novel-view truth',
        action=placement['action'],scale=placement['scale'],
        motion_review_group=placement['motion_review_group'],
        ground_verified=placement['feet']['ground_verified'],contact_verified=placement['feet']['contact_verified'],
        loopable=placement['playback']['loopable'],skip_insertion=insertion.get('skip_insertion',False),
        approximate_support_height_model_m=[practical.get('support_height_after_p05_model_m'), practical.get('support_height_after_p95_model_m')],
        scope='Fresh payload hashes, saved replay record consistency, coordinate/sidecar contract, full video decoding. Not a new rendering or visual acceptance.',finished_at=time.time()))


def validate_vlm(value, frames):
    assert set(value)=={'triage','action','issues','evidence','uncertainties'}
    assert value['triage'] in {'severe_candidate','review_needed','no_gross_issue_observed','uncertain'}
    assert value['action'] in {'standing','walking','running','gesture','mixed','unknown'}
    assert isinstance(value['evidence'],str)
    assert isinstance(value['uncertainties'],list) and all(isinstance(s,str) for s in value['uncertainties'])
    assert isinstance(value['issues'],list)
    for issue in value['issues']:
        assert set(issue)=={'type','stage','frames','description','severity'}
        assert issue['type'] in {'limb_loss','deformation','occluding_plane','color_change','accessory_loss','source_truncation','other'}
        assert issue['stage'] in {'input','generation','reconstruction','uncertain'}
        assert issue['severity'] in {'major','minor','uncertain'}
        assert isinstance(issue['description'],str) and isinstance(issue['frames'],list)
        assert issue['frames'] and all(type(f) is int and f in frames for f in issue['frames'])


async def vlm(out, identity):
    from openai import AsyncOpenAI
    protocol=read(out/'protocol.json');meta=read(out/'technical'/f'{identity}.json')
    path=out/'contacts'/f'{identity}.jpg'
    assert digest(path)==meta['contact_sha256']
    parts=[dict(type='text',text=json.dumps(dict(identity=identity,frames=meta['sampled_frames']))),
        dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode()))]
    async with AsyncOpenAI(base_url=protocol['base_url'],api_key='local-only',timeout=180,max_retries=0) as client:
        response=await client.chat.completions.create(model=protocol['model'],
            messages=[dict(role='system',content=protocol['prompt']),dict(role='user',content=parts)],
            response_format={'type':'json_object'},temperature=0,seed=42,max_tokens=1800)
    save(out/'vlm_raw'/f'{identity}.json',dict(response=response.model_dump(),contact_sha256=meta['contact_sha256']))
    assert response.choices[0].finish_reason=='stop', response.choices[0].finish_reason
    value=json.loads(response.choices[0].message.content);validate_vlm(value,meta['sampled_frames'])
    save(out/'vlm'/f'{identity}.json',dict(identity=identity,state='unverified_model_triage',result=value,
        sampled_frames=meta['sampled_frames'],contact_sha256=meta['contact_sha256'],finished_at=time.time()))


async def run(out):
    protocol=read(out/'protocol.json')
    assert protocol['code_sha256']==digest(__file__), 'Audit code differs from frozen protocol'
    cpu=asyncio.Semaphore(protocol['technical_concurrency']);vision=asyncio.Semaphore(protocol['vlm_concurrency'])
    state=dict(started_at=time.time(),state='running',total=len(protocol['packages']),cases={},library_modified=False)
    def update():save(out/'status.json',state)
    update()
    async def child(mode, identity, python):
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        proc=await asyncio.create_subprocess_exec(python,__file__,mode,'--output',str(out),'--identity',identity,
            env=env,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        stdout,stderr=await proc.communicate()
        if proc.returncode:raise RuntimeError(stderr.decode()[-2500:] or stdout.decode()[-2500:])
    async def one(row):
        identity=row['identity']
        try:
            async with cpu:
                state['cases'][identity]='technical_running';update()
                if not (out/'technical'/f'{identity}.json').exists():await child('technical',identity,PYTHON)
            async with vision:
                state['cases'][identity]='vlm_running';update()
                if not (out/'vlm'/f'{identity}.json').exists():await child('vlm',identity,VLM_PYTHON)
            state['cases'][identity]='completed_requires_visual_crosscheck'
        except Exception as error:
            state['cases'][identity]='failed';save(out/'errors'/f'{identity}.json',dict(error=str(error),time=time.time()))
        update()
    await asyncio.gather(*(one(row) for row in protocol['packages']))
    state.update(state='completed_with_errors' if 'failed' in state['cases'].values() else 'completed_requires_visual_crosscheck',finished_at=time.time());update()


def prior_attempts(directories, assets):
    """Bind reuse to the same package, prompt and sampling; retain failed attempts."""
    attempts = {}
    run_root = Path(os.environ['HUMAN_RUN_ROOT']).resolve()
    for directory in directories:
        protocol = read(directory/'protocol.json')
        assert Path(protocol['assets']).resolve() == assets.resolve()
        assert Path(protocol['run_root']).resolve() == run_root
        assert protocol['prompt'] == PROMPT and protocol['sample_rule'] == 'five uniform integer frames'
        state = read(directory/'status.json')
        assert state['state'] in ('completed_with_errors', 'completed_requires_visual_crosscheck'), 'Audit still active; do not duplicate it'
        for row in protocol['packages']:
            package = assets/row['identity']
            if not (package/'package_complete.json').exists():
                continue
            if digest(package/'package_complete.json') != row['marker_sha256'] or digest(package/'demo.mp4') != row['demo_sha256']:
                continue
            result = state['cases'][row['identity']]
            if result == 'completed_requires_visual_crosscheck':
                technical_record = read(directory/'technical'/f"{row['identity']}.json")
                vision = read(directory/'vlm'/f"{row['identity']}.json")
                assert technical_record['source_sha256'] == row['demo_sha256']
                assert technical_record['contact_sha256'] == vision['contact_sha256']
                assert digest(directory/'contacts'/f"{row['identity']}.jpg") == vision['contact_sha256']
            else:
                assert result == 'failed'
                read(directory/'errors'/f"{row['identity']}.json")
            attempts[row['identity']] = dict(directory=str(directory), state=result, **row)
    return attempts


def watched_process(pid, expected_start=None):
    """PID identity and kernel state, not stale status files, determine liveness."""
    from intake import process_running
    if not process_running(pid):
        return None
    try:
        start = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None
    return start if expected_start is None or start == expected_start else None


async def watch(out, assets, pids, reuse, interval=60):
    """Incremental read-only audits after publishing; no acceptance or deletion."""
    import fcntl
    assert pids and interval >= 30
    out.mkdir(parents=True, exist_ok=True)
    with (out/'watch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = out/'watch_protocol.json'
        if binding.exists():
            fixed = read(binding)
            assert fixed['code_sha256'] == digest(__file__)
            assert fixed['pids'] == pids and fixed['assets'] == str(assets.resolve())
            assert fixed['reuse'] == [str(p.resolve()) for p in reuse]
        else:
            fixed = dict(pids=pids, starts={str(p):watched_process(p) for p in pids},
                assets=str(assets.resolve()), reuse=[str(p.resolve()) for p in reuse],
                code_sha256=digest(__file__), created_at=time.time())
            assert all(fixed['starts'].values()), 'Confirm all producers live before starting watch'
            save(binding, fixed)
        while True:
            directories = reuse + sorted(out.glob('batch_*/'))
            attempted = prior_attempts(directories, assets)
            packages = published_packages(assets)
            pending = [p.name for p in packages if p.name not in attempted]
            live = [p for p in pids if fixed['starts'][str(p)] and watched_process(p, fixed['starts'][str(p)])]
            rows = []
            for package in packages:
                identity = package.name
                attempt = attempted.get(identity)
                triage = None
                if attempt and attempt['state'] == 'completed_requires_visual_crosscheck':
                    triage = read(Path(attempt['directory'])/'vlm'/f'{identity}.json')['result']['triage']
                review_path = package/'visual_review.json'
                review = read(review_path) if review_path.exists() else {}
                rows.append(dict(identity=identity, audit=attempt, model_triage=triage,
                    recorded_visual_flags=review.get('quality_flags', []),
                    quality_approved=review.get('final_asset_quality_certified', False)
                                     and not review.get('severe_failure', False)))
            state = dict(state='auditing_new_deliveries' if pending else 'waiting_producers' if live else 'producers_stopped_audit_finished',
                pid=os.getpid(), updated_at=time.time(), live_producer_pids=live,
                published=len(packages), attempted=len(attempted), pending=pending,
                failed=[i for i,r in attempted.items() if r['state']=='failed'], cases=rows,
                library_modified=False, scope='Sparse model triage and delivery integrity; no automatic quality acceptance or removal')
            if not live:
                production = read(Path(os.environ['HUMAN_RUN_ROOT'])/'PRODUCTION_INPUTS.json')
                state['production_ids_not_published'] = sorted(set(production['ids']) - {p.name for p in packages})
                state['input_selection_complete'] = production.get('input_selection_complete', False)
            save(out/'status.json', state)
            if pending:
                batch = out/f'batch_{len(list(out.glob("batch_*/")))+1:04d}'
                for name in ['contacts','technical','vlm','vlm_raw','errors']:
                    (batch/name).mkdir(parents=True)
                freeze(batch, assets, pending[:16])
                await run(batch)
            elif live:
                await asyncio.sleep(interval)
            else:
                return


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','technical','vlm','run','watch'])
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--identity')
    parser.add_argument('--assets',default=ASSET_ROOT,type=Path)
    parser.add_argument('--cases', nargs='+');parser.add_argument('--wait-pids', nargs='+', type=int)
    parser.add_argument('--reuse', nargs='*', type=Path, default=[])
    parser.add_argument('--interval', type=int, default=60)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.mode!='watch':
        for name in ['contacts','technical','vlm','vlm_raw','errors']:(args.output/name).mkdir(exist_ok=True)
    if args.mode=='watch':asyncio.run(watch(args.output,args.assets,args.wait_pids,args.reuse,args.interval))
    elif args.mode=='freeze':freeze(args.output,args.assets,args.cases)
    elif args.mode=='technical':technical(args.output,args.identity)
    elif args.mode=='vlm':asyncio.run(vlm(args.output,args.identity))
    else:asyncio.run(run(args.output))
