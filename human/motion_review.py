"""Optional local VLM evidence audit; never alters assets or approves placement."""
import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from common import ASSET_ROOT, ROOT, CODE_ROOT, read, save, sha

import os
from common import PYTHONS
MODEL=Path(os.environ.get('QWEN_MODEL', 'models/Qwen3-VL-8B-Instruct'))
PROMPT='''Audit one target human's motion using the supplied chronological images.
The scope explicitly says real observed input or two reconstructed views. Do not
confuse reconstructed evidence with real observations. The real input has moving
camera/crop: screen displacement is NOT world motion or walking direction.
Judge visible limb articulation. For rendered views, the background is flat and
there is no visible ground plane: do NOT invent exact contact or ground height.
Left/right mean the PERSON'S anatomical left/right, never screen left/right.
Occluded, blurred, cropped, ambiguous feet, or ambiguous side identity must be
unknown, not silently completed. Never infer meters, exact speed, world trajectory,
or force/contact certainty. Samples are sparse: no claim of no flicker between them.
Return JSON with exactly these keys:
action: standing|walking|running|turning|stepping|gesture|mixed|other|unknown;
direction: forward|backward|sideways|in_place|unknown;
confidence: number 0..1, subjective and uncalibrated;
evidence: list of {frame:int, observation:string}, only sampled frame IDs;
feet: list, one per sampled frame, each {frame:int, left_visible:bool,
 right_visible:bool, support:left|right|both|neither|unknown,
 left_xy:[x,y] or null, right_xy:[x,y] or null, reason:string}.
Foot coordinates are normalized 0..1 within the LEFT panel (or the sole image).
Only report a coordinate if anatomical identity and the foot are visible.
Support is a visual HYPOTHESIS only; unknown is expected without evidence.
issues: list of {start_frame:int,end_frame:int,type:occlusion|limb_deformation|
 floating|sliding|color_artifact|identity_ambiguity|other,observation:string};
uncertainties: list of strings; next_check: string.
Do not say this asset passed ground, no-slip, or loop validation. JSON only.'''


def validate_result(value, frames):
    required={'action','direction','confidence','evidence','feet','issues','uncertainties','next_check'}
    assert set(value)==required
    assert value['action'] in {'standing','walking','running','turning','stepping','gesture','mixed','other','unknown'}
    assert value['direction'] in {'forward','backward','sideways','in_place','unknown'}
    assert isinstance(value['confidence'],(float,int)) and 0<=value['confidence']<=1
    assert [p['frame'] for p in value['feet']]==frames
    for p in value['feet']:
        assert set(p)=={'frame','left_visible','right_visible','support','left_xy','right_xy','reason'}
        assert p['support'] in {'left','right','both','neither','unknown'}
        assert isinstance(p['reason'],str)
        for side in ('left','right'):
            assert isinstance(p[side+'_visible'],bool)
            xy=p[side+'_xy']
            assert xy is None or p[side+'_visible'] and len(xy)==2 and all(isinstance(v,(int,float)) and 0<=v<=1 for v in xy)
    for event in value['evidence']:
        assert set(event)=={'frame','observation'} and event['frame'] in frames and isinstance(event['observation'],str)
    for event in value['issues']:
        assert set(event)=={'start_frame','end_frame','type','observation'}
        assert event['start_frame'] in frames and event['end_frame'] in frames and event['start_frame']<=event['end_frame']
        assert event['type'] in {'occlusion','limb_deformation','floating','sliding','color_artifact','identity_ambiguity','other'}
        assert isinstance(event['observation'],str)
    assert isinstance(value['uncertainties'],list) and all(isinstance(x,str) for x in value['uncertainties'])
    assert isinstance(value['next_check'],str)


def prepare(identity,scope,center=None):
    import cv2
    path=ASSET_ROOT/identity/'demo.mp4';cap=cv2.VideoCapture(str(path))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    assert count>0 and fps>0
    frames=np.unique(np.linspace(0,count-1,13,dtype=int)).tolist() if center is None else np.clip([center-4,center,center+4],0,count-1).tolist()
    assert len(set(frames))==len(frames)
    parts=[dict(type='text',text=json.dumps(dict(identity=identity,scope=scope,sampled_frames=frames,
           fps=fps,panels='real source observation' if scope=='observed' else ('reconstructed view00 only' if center is not None else 'reconstructed view00 | reconstructed view08'))))]
    hashes=[]
    try:
        for frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();assert ok and bgr.shape[:2]==(672,1760)
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            crop=rgb[32:672,:352] if scope=='observed' else rgb[32:672,1056:(1408 if center is not None else 1760)]
            image=Image.fromarray(crop);stream=io.BytesIO();image.save(stream,format='JPEG',quality=92)
            raw=stream.getvalue();hashes.append(dict(frame=frame,sha256=hashlib.sha256(raw).hexdigest()))
            parts.extend([dict(type='text',text=f'frame={frame}; elapsed_seconds={frame/fps:.3f}'),
                dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(raw).decode()))])
    finally:cap.release()
    return parts,dict(identity=identity,scope=scope,frames=frames,fps=fps,video_sha256=sha(path),image_hashes=hashes)


def crosscheck(identity,scope,value):
    """Keep raw VLM claims; flag unsupported support and coordinate mismatch."""
    contradictions=[p['frame'] for p in value['feet'] if not p['left_visible'] and not p['right_visible'] and p['support']!='unknown']
    result=dict(state='unverified_semantic_hints_only',support_asserted_without_visible_feet=contradictions,
                contact_verified=False,loop_verified=False,metric_speed_available=False)
    if scope!='reconstructed':return result
    package=ASSET_ROOT/identity;meta=read(package/'placement.json');cameras=read(package/'native_cameras.json')
    record=next(c for c in cameras['cameras'] if c['view']==0);K=np.asarray(record['K'])
    with np.load(package/'placement_motion.npz',allow_pickle=False) as motion,np.load(package/'asset_4dgs.npz',allow_pickle=False) as asset:
        into=np.linalg.inv(np.asarray(record['camtoworld']))@np.linalg.inv(motion['asset_to_standard_model_meters']@asset['asset_from_native'])
        errors=[];whole_panel_errors=[]
        for foot in value['feet']:
            for side in ('left','right'):
                xy=foot[side+'_xy']
                if xy is None or not meta['motion']['associations'][side+'-foot']['numerically_supported']:continue
                p=motion[side+'-foot_proxy_position'][foot['frame']];v=into[:3,:3]@p+into[:3,3]
                if v[2]<=0:continue
                projected=K@v;projected=projected[:2]/projected[2]
                claimed=np.asarray(xy)*[cameras['width'],cameras['height']]
                errors.append(float(np.linalg.norm(projected-claimed)))
                # Diagnostic alternative only: never silently reinterpret coordinates.
                whole_panel_errors.append(float(np.linalg.norm(projected-claimed*[2,1])))
        result['projection_diagnostic']=dict(comparisons=len(errors),
            mean_pixel_error=float(np.mean(errors)) if errors else None,
            p95_pixel_error=float(np.quantile(errors,.95)) if errors else None,
            alternative_entire_composite_mean_pixel_error=float(np.mean(whole_panel_errors)) if errors else None,
            reference='Existing Gaussian foot proxies, not ground truth; unsupported associations excluded',
            coordinates_reinterpreted=False)
    result['ground_visibility']='Flat reconstruction background; no observed floor or contact truth'
    return result


async def main(args):
    from openai import AsyncOpenAI
    output=ROOT/'reports'/('vlm_motion_windows' if args.window_probe else 'vlm_motion_trial');output.mkdir(parents=True,exist_ok=True)
    state=dict(state='starting',started=time.time(),cases={},automatic_asset_changes=False)
    def update():save(output/'status.json',state)
    jobs=([(i,s,None) for i in args.identities for s in ('observed','reconstructed')] if not args.window_probe else
          [('H0107',s,c) for s in ('observed','reconstructed') for c in (10,40,70,100)]+[('H0001','reconstructed',60),('H0144','observed',30)])
    protocol=dict(jobs=[list(j) for j in jobs],samples=3 if args.window_probe else 13,concurrency=2,
                  model=str(MODEL),model_config_sha256=sha(MODEL/'config.json'),prompt=PROMPT,
                  role='Uncalibrated semantic hints; no metric/contact/trajectory/loop approval',
                  response_format={'type':'json_object'},code_sha256=sha(Path(__file__)))
    if (output/'protocol.json').exists():
        assert read(output/'protocol.json')==protocol,'Refuse to overwrite a different frozen VLM protocol'
    else:
        save(output/'protocol.json',protocol)
        (output/'source_at_launch.py').write_bytes(Path(__file__).read_bytes())
    update();client=AsyncOpenAI(base_url=f'http://127.0.0.1:{args.port}/v1',api_key='local-only',timeout=300,max_retries=0)
    # Explicit numeric index, after checking GPU UUID; required by vLLM 0.11 NVML.
    check=await asyncio.create_subprocess_exec('nvidia-smi',f'--id={args.gpu}','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE)
    checked,_=await check.communicate();assert check.returncode==0
    uuid,free=checked.decode().strip().split(',');assert uuid.startswith('GPU-') and int(free)>27000
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
             PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_USE_FLASHINFER_SAMPLER='0')
    env.pop('PYTHONPATH',None)
    argv=[str(PYTHONS['vllm_vlm']),'-m','vllm.entrypoints.openai.api_server',
          '--model',str(MODEL),'--served-model-name','human-motion-review','--host','127.0.0.1','--port',str(args.port),
          '--dtype','bfloat16','--max-model-len','12288','--max-num-seqs','2','--gpu-memory-utilization','0.48','--enforce-eager',
          '--limit-mm-per-prompt','{"image":13,"video":0}','--mm-processor-kwargs','{"max_pixels":200704}']
    semaphore=asyncio.Semaphore(2)
    async def review(identity,scope,center):
        key=identity+'_'+scope+(f'_{center:03d}' if center is not None else '')
        async with semaphore:
            if (output/(key+'.json')).exists():raise FileExistsError(key)
            parts,context=await asyncio.to_thread(prepare,identity,scope,center)
            state['cases'][key]=dict(state='running');update()
            try:
                response=await client.chat.completions.create(model='human-motion-review',messages=[dict(role='system',content=PROMPT),dict(role='user',content=parts)],
                            response_format={'type':'json_object'},temperature=0,seed=42,max_tokens=3500)
                raw=dict(context=context,response=response.model_dump(),protocol_sha256=sha(output/'protocol.json'))
                save(output/(key+'_raw.json'),raw)
                assert response.choices[0].finish_reason=='stop',response.choices[0].finish_reason
                value=json.loads(response.choices[0].message.content);validate_result(value,context['frames'])
                save(output/(key+'.json'),dict(context=context,result=value,state='model_hypothesis_requires_evidence_check'))
                save(output/(key+'_crosscheck.json'),await asyncio.to_thread(crosscheck,identity,scope,value))
                state['cases'][key]=dict(state='completed_unverified',action=value['action'],confidence=value['confidence'])
            except Exception as error:
                state['cases'][key]=dict(state='failed',error=str(error));raise
            finally:update()
    with (output/'server.log').open('x') as log:
        server=await asyncio.create_subprocess_exec(*argv,env=env,stdout=log,stderr=log)
        state.update(state='loading_model',server_pid=server.pid,gpu=args.gpu,gpu_uuid=uuid);update()
        try:
            deadline=time.monotonic()+600
            while True:
                if server.returncode is not None:raise RuntimeError(f'Server exited {server.returncode}')
                try:
                    models=await client.models.list()
                    assert any(m.id=='human-motion-review' for m in models.data);break
                except Exception:
                    if time.monotonic()>deadline:raise
                    await asyncio.sleep(3)
            state['state']='reviewing';update()
            results=await asyncio.gather(*(review(*job) for job in jobs),return_exceptions=True)
            state['state']='completed_with_failures' if any(isinstance(r,BaseException) for r in results) else 'completed_requires_crosscheck'
        except Exception as error:state.update(state='failed',error=str(error))
        finally:
            await client.close()
            if server.returncode is None:
                server.terminate()
                try:await asyncio.wait_for(server.wait(),45)
                except asyncio.TimeoutError:server.kill();await server.wait()
            state.update(finished=time.time(),server_exit_code=server.returncode,server_stopped=True);update()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('identities',nargs='+')
    parser.add_argument('--gpu',type=int,required=True);parser.add_argument('--port',type=int,default=18542)
    parser.add_argument('--window-probe',action='store_true',help='Fixed diagnostic single-view windows; no production use')
    asyncio.run(main(parser.parse_args()))
