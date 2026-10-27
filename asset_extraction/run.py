"""Generic scene/video → FLUX → H/B0 → quality audit → rigid asset bank."""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import sys

import yaml
from .config import load_config
from .generation import GpuPool, digest, generate, prepare_jobs, sha256
from .io_utils import read_json, write_json
from .sources import download

ROOT = Path(__file__).resolve().parents[1]


def expand(value):
    if isinstance(value, dict):
        return {k:expand(v) for k,v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, str):
        result = os.path.expandvars(os.path.expanduser(value))
        if '$' in result:
            raise ValueError('Unresolved environment variable: ' + value)
        return result
    return value


def path(value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (ROOT/p).resolve()


async def command(argv, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('ab') as stream:
        process = await asyncio.create_subprocess_exec(*map(str,argv), cwd=ROOT,
            stdout=stream, stderr=asyncio.subprocess.STDOUT)
        try:
            code = await process.wait()
        finally:
            if process.returncode is None:
                process.terminate()
                await process.wait()
    if code:
        raise RuntimeError(f'Command failed ({code}); see {log}')


async def execute(spec, *, prepare_only=False, handoff_only=False):
    from .pipeline import _discover_scene_dirs
    from .export import curate
    from .bank import publish, retire_rejected
    from nuscenes.prepare_handoff import prepare
    from nuscenes.quality_filter import filter_manifest
    from nuscenes.prepare_routes import prepare_routes

    allowed = {'source','frontend_config','generation_config','run_dir','library_dir','gpus',
               'renderer_python','chunk_size','delete_rejected','vlm_endpoint'}
    if set(spec)-allowed:
        raise ValueError('Unknown pipeline keys: '+str(sorted(set(spec)-allowed)))
    run = path(spec['run_dir'])
    run.mkdir(parents=True, exist_ok=True)
    gpus = spec.get('gpus',[0])
    if not gpus or len(set(gpus))!=len(gpus):
        raise ValueError('Provide distinct GPU indices')
    size = spec.get('chunk_size',24)
    if not isinstance(size,int) or size<1:
        raise ValueError('chunk_size must be positive (a resource window, not an asset target)')
    os.environ['VLM_RESPONSE_FORMAT']='json_object'
    if spec.get('vlm_endpoint'):
        os.environ['VLM_BASE_URL']=spec['vlm_endpoint']
    os.environ.setdefault('DRIVELAB_GPU_LOCK_DIR',str(path(spec['library_dir'])/'.gpu_locks'))
    snapshot = run/'configuration.json'
    frozen = dict(spec=spec, frontend_sha256=sha256(path(spec['frontend_config'])),
                  generation_sha256=sha256(path(spec['generation_config'])),
                  code={str(p.relative_to(ROOT)):sha256(p) for folder in ('asset_extraction','nuscenes')
                        for p in sorted((ROOT/folder).glob('*.py'))})
    if snapshot.exists() and read_json(snapshot)!=frozen:
        raise ValueError('Run configuration changed; choose a new run_dir')
    write_json(snapshot,frozen)
    source = spec['source']
    kind = source['type']
    if kind not in {'nuscenes','frames','waymo','video'}:
        raise ValueError('source.type must be nuscenes, frames, waymo, or video')
    frontend = yaml.safe_load(path(spec['frontend_config']).read_text())
    frontend['dataset_root'] = source['path'] if kind in {'nuscenes','frames'} else '.'
    frontend = expand(frontend)
    frontend['dataset'] = kind if kind in {'nuscenes','frames'} else 'frames'
    frontend.update(workspace=str(ROOT),output_root=str(run),run_name='source',
        gpu_ids=gpus,forbidden_gpu_ids=[],scene_workers_per_gpu=1,resume=True,
        scene_ids=source.get('scene_ids',[]),limit_scenes=None)
    if spec.get('vlm_endpoint'):
        frontend['vlm']['endpoint']=spec['vlm_endpoint']
    if kind in {'video','waymo'}:
        prepared = run/'inputs/prepared.json'
        if not prepared.exists():
            location = source['path']
            if kind=='video' and location.startswith(('http://','https://')):
                inputs=[await download(location,run/'downloads')]
            else:
                location=path(location)
                inputs=sorted(location.glob('*.tfrecord')) if location.is_dir() and kind=='waymo' else [location]
            if not inputs or any(not p.is_file() for p in inputs):
                raise ValueError('No valid input video / TFRecord files')
            # Decode one file at a time to bound CPU and disk pressure; model jobs use async GPU slots.
            for item in inputs:
                args=[source.get('python',sys.executable),'-m','asset_extraction.sources',
                    '--type',kind,'--input',item,'--output',run/'inputs']
                args += ['--sample-hz',str(source.get('sample_hz',2))] if kind=='video' else ['--stride',str(source.get('stride',5))]
                await command(args,run/'logs/prepare.log')
            write_json(prepared,dict(inputs=[dict(path=str(p),sha256=sha256(p)) for p in inputs]))
        else:
            for item in read_json(prepared)['inputs']:
                if sha256(item['path']) != item['sha256']:
                    raise ValueError('Prepared source changed')
        frontend.update(dataset='frames',dataset_root=str(run/'inputs/indexes'),frame_stride=1,
            camera_channels=['front'] if kind=='video' else ['front','front_left','front_right','side_left','side_right'])
    else:
        frontend.update(dataset=kind,dataset_root=str(path(source['path'])))
        if kind=='frames':
            frontend['camera_channels']=source.get('cameras',['front'])
    config_path=run/'frontend.yaml'
    config_path.write_text(yaml.safe_dump(frontend,sort_keys=False))
    config=load_config(config_path)
    scenes=_discover_scene_dirs(config,run/'discovery')
    if not scenes:
        raise ValueError('No complete scenes selected')
    write_json(run/'scenes.json',dict(scenes=[str(p) for p in scenes]))
    if prepare_only:
        return dict(status='prepared',scenes=len(scenes))
    generation_config=expand(read_json(path(spec['generation_config'])))
    for key,value in generation_config.items():
        if key!='seed':
            generation_config[key]=str(path(value))
    statuses=[]
    for scene in scenes:
        scene_dir=run/'scenes'/scene.stem
        scene_dir.mkdir(parents=True,exist_ok=True)
        done=scene_dir/'complete.json'
        if done.exists():
            statuses.append(read_json(done))
            continue
        scene_cfg=dict(frontend,dataset='frames',dataset_root=str(scene.parent),
            scene_ids=[scene.stem],run_name=scene.stem,output_root=str(run/'frontend'))
        scene_cfg['camera_channels']=sorted({f['camera_name'] for f in read_json(scene)['frames']})
        scene_path=scene_dir/'frontend.yaml'
        scene_path.write_text(yaml.safe_dump(scene_cfg,sort_keys=False))
        current=load_config(scene_path)
        extracted=run/'frontend/runs'/scene.stem
        handoff=extracted/'handoff/flux_input.json'
        routes=extracted/'handoff/asset_routes.json'
        if not routes.exists():
            pool=GpuPool(gpus)
            gpu=await pool.acquire('frontend')
            try:
                await command([current.runtime.python,'-m','asset_extraction.pipeline','--config',scene_path,'--gpu',str(gpu)],scene_dir/'frontend.log')
            finally:
                pool.release(gpu)
            filtered=extracted/'merged/manifest.filtered.json'
            filter_manifest(extracted/'merged/manifest.json',filtered,extracted/'logs/filter_rejects.jsonl',run_dir=extracted)
            prepare(filtered,handoff)
            await prepare_routes(handoff,extracted,routes,current)
        rows=read_json(handoff)
        if handoff_only:
            statuses.append(dict(scene_id=scene.stem,status='handoff',queries=len(rows)))
            continue
        routing=read_json(routes)
        results=[]
        for start in range(0,len(rows),size):
            chunk=scene_dir/f'chunk_{start//size:04d}'
            receipt=chunk/'complete.json'
            if receipt.exists():
                results.append(read_json(receipt))
                continue
            chosen=rows[start:start+size]
            ids={r['obj_id'] for r in chosen}
            write_json(chunk/'flux_input.json',chosen)
            subset=dict(routing,assets=[r for r in routing['assets'] if r['obj_id'] in ids],
                flux_input=str(chunk/'flux_input.json'),flux_input_sha256=sha256(chunk/'flux_input.json'))
            write_json(chunk/'asset_routes.json',subset)
            await generate(prepare_jobs(chunk/'flux_input.json',chunk/'asset_routes.json',generation_config),chunk/'generation',gpus)
            ledger=chunk/'curation/curation.json'
            state=read_json(ledger) if ledger.exists() else {}
            if 'counts' not in state:
                state=await curate(chunk/'generation/generation.json',chunk/'library',current,
                    renderer_python=str(path(spec['renderer_python'])),gpus=gpus,work_dir=chunk/'curation')
            publish(state,path(spec['library_dir']),dataset=kind,scene_id=scene.stem)
            result=dict(queries=len(chosen),counts=state['counts'],chunk=str(chunk))
            # Receipt makes completed chunks resumable even after rejected outputs are retired.
            write_json(receipt,result)
            if spec.get('delete_rejected',True):
                retire_rejected(state,chunk)
            results.append(result)
        count=Counter()
        for result in results:
            count.update(result['counts'])
        result=dict(scene_id=scene.stem,status='completed',queries=len(rows),counts=dict(count))
        write_json(done,result)
        statuses.append(result)
        write_json(run/'progress.json',dict(scenes=statuses))
    result=dict(status='handoff' if handoff_only else 'completed',scenes=statuses)
    write_json(run/'status.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--source',choices=['nuscenes','waymo','frames','video'])
    parser.add_argument('--input')
    parser.add_argument('--scene-ids',nargs='+')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only',action='store_true')
    mode.add_argument('--handoff-only',action='store_true')
    args=parser.parse_args()
    spec=yaml.safe_load(args.config.read_text())
    if args.source:
        if args.source != spec['source']['type']:
            spec['source'].pop('scene_ids',None)
        spec['source']['type']=args.source
    if args.input:spec['source']['path']=args.input
    if args.scene_ids:spec['source']['scene_ids']=args.scene_ids
    spec=expand(spec)
    run=path(spec['run_dir']);run.mkdir(parents=True,exist_ok=True)
    with (run/'.run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This run already has an active controller')
        print(json.dumps(asyncio.run(execute(spec,prepare_only=args.prepare_only,handoff_only=args.handoff_only)),indent=2))


if __name__=='__main__':
    main()
