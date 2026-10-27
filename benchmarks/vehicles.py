"""Matched-source PSNR/LPIPS and single-asset rasterizer throughput.

Consumes a frozen registry with PLY hashes, original RGBA and optional native
matched camera/cuboid metadata. No view search, training, or mesh conversion.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.writing')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def normalized(rgba):
    import numpy as np
    from PIL import Image
    rgba=rgba.convert('RGBA');a=np.array(rgba.getchannel('A'))>127
    ys,xs=np.where(a)
    if not len(xs):raise ValueError('Empty foreground')
    x0,x1,y0,y1=int(xs.min()),int(xs.max()+1),int(ys.min()),int(ys.max()+1)
    side=int(np.ceil(max(x1-x0,y1-y0)*1.1));left=int(np.floor((x0+x1-side)/2));top=int(np.floor((y0+y1-side)/2))
    rgb=Image.new('RGB',rgba.size,'white');rgb.paste(rgba,mask=Image.fromarray(a.astype('uint8')*255))
    canvas=Image.new('RGB',(side,side),'white');canvas.paste(rgb,(-left,-top))
    return canvas.resize((512,512),Image.Resampling.BICUBIC)


def render(args):
    import numpy as np
    import torch
    from gsplat import rasterization
    from plyfile import PlyData
    from PIL import Image
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    registry=read(args.registry)
    protocol=dict(registry_sha256=sha(args.registry),script_sha256=sha(__file__),
        resolution=[512,512],batch_size=1,views=16,warmup_frames=32,timed_frames=128,
        timing='CUDA events around 128 batch=1 renders, plus synchronized wall time; excludes loading, disk I/O and metric models',
        vram='Per-process PyTorch peak allocated/reserved during rasterization, including resident asset tensors; not global nvidia-smi usage',
        fps_aggregation='Total timed frames / sum of CUDA elapsed seconds; median per-asset FPS also reported',
        gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,device=os.environ.get('CUDA_VISIBLE_DEVICES'),
        quality='Original segmented source versus native matched view; independent foreground centering, isotropic size normalization, white background and bicubic512; no best-view selection',
        PSNR='Per-asset RGB [0,1] whole-square 10log10(1/MSE), then arithmetic asset mean',
        LPIPS='AlexNet learned v0.1 calibration, RGB[-1,1], float32 evaluation',
        scope='Source-conditioned appearance diagnostic, not heldout full-scene reconstruction PSNR',
        hardware_sharing='Other jobs may share this GPU; results are not a controlled baseline comparison',created=time.time())
    write(args.output/'protocol.json',protocol)
    def tensor(x):return torch.as_tensor(x,device='cuda',dtype=torch.float32)
    for row in registry['assets']:
        key=hashlib.sha256((row['dataset']+':'+row['obj_id']).encode()).hexdigest()[:20]
        folder=args.output/'assets'/key;dest=folder/'render.json'
        signature=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
        previous = read(dest) if dest.exists() else None
        if previous:
            if previous['signature']!=signature:raise ValueError('Cached registry changed')
            if previous['status']=='completed':continue
        folder.mkdir(parents=True,exist_ok=True)
        result=dict(signature=signature,dataset=row['dataset'],obj_id=row['obj_id'],category=row['category'])
        if previous:result['previous_failure']=previous
        try:
            if sha(row['ply'])!=row['ply_sha256']:raise ValueError('PLY changed')
            torch.cuda.empty_cache()
            baseline=torch.cuda.memory_allocated()
            vertex=PlyData.read(row['ply'])['vertex']
            def fields(keys):return np.stack([np.asarray(vertex[k],dtype=np.float32) for k in keys],axis=-1)
            xyz=fields(['x','y','z']);lo,hi=np.quantile(xyz,[.005,.995],axis=0)
            quats=tensor(fields(['rot_0','rot_1','rot_2','rot_3']));quats/=quats.norm(dim=1,keepdim=True)
            arrays=[tensor(xyz),quats,tensor(np.exp(fields(['scale_0','scale_1','scale_2']))),
                tensor(1/(1+np.exp(-np.asarray(vertex['opacity'],np.float32)))),
                tensor(np.clip(.5+.28209479177387814*fields(['f_dc_0','f_dc_1','f_dc_2']),0,1))]
            center=(lo+hi)/2;radius=np.linalg.norm(hi-lo)/2;distance=radius*2.2/.78
            focal=.5*512*distance/radius*.78;k=tensor([[focal,0,256],[0,focal,256],[0,0,1]])[None]
            views=[]
            for i in range(16):
                az=i*np.pi/8;el=np.deg2rad(12);direction=np.array([np.cos(az)*np.cos(el),np.sin(az)*np.cos(el),np.sin(el)])
                forward=-direction;right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
                rotation=np.stack([right,np.cross(forward,right),forward]);v=np.eye(4);v[:3,:3]=rotation;v[:3,3]=-rotation@(center+distance*direction)
                views.append(tensor(v)[None])
            background=torch.ones((1,3),device='cuda')
            def raster(view,intrinsic=k,width=512,height=512,scale=1.):
                return rasterization(arrays[0] if scale==1 else arrays[0]*scale, arrays[1],
                    arrays[2] if scale==1 else arrays[2]*scale,*arrays[3:],view,intrinsic,width,height,
                    backgrounds=background,packed=False)
            with torch.inference_mode():
                for i in range(32):
                    rendered=raster(views[i%16]);del rendered
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                wall=time.perf_counter();start.record()
                for i in range(128):
                    rendered=raster(views[i%16]);del rendered
                end.record();end.synchronize();wall=time.perf_counter()-wall
                seconds=start.elapsed_time(end)/1000
                result.update(status='completed',gaussians=len(xyz),timed_frames=128,seconds=seconds,
                    fps=128/seconds,wall_fps=128/wall,peak_allocated_MiB=(torch.cuda.max_memory_allocated()-baseline)/2**20,
                    peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
                annotation=row.get('matched_annotation')
                if annotation:
                    if sha(row['source_rgba'])!=row['source_sha256']:raise ValueError('Source changed')
                    source=Image.open(row['source_rgba']).convert('RGBA');w,h=source.size
                    scale=annotation['size_lwh'][0]/(hi[0]-lo[0]);ground=np.eye(4);ground[2,3]=-annotation['size_lwh'][2]/2
                    v=np.linalg.inv(annotation['camera_to_world'])@np.asarray(annotation['object_to_world'])@ground
                    x0,y0,x1,y1=row['source_crop_box'];crop=np.array([[w/(x1-x0),0,-x0*w/(x1-x0)],[0,h/(y1-y0),-y0*h/(y1-y0)],[0,0,1]])
                    intrinsic=crop@np.asarray(annotation['camera_intrinsic'])
                    rgb,alpha,_=raster(tensor(v)[None],tensor(intrinsic)[None],w,h,scale)
                    rgba=torch.cat([rgb[0,:,:,:3].clamp(0,1),alpha[0].clamp(0,1)],dim=-1)
                    target=Image.fromarray((rgba.cpu().numpy()*255).round().astype('uint8'),'RGBA')
                    normalized(source).save(folder/'source.png');normalized(target).save(folder/'matched.png')
                    del rgb, alpha, rgba, _
                    result['matched']=True
                else:
                    result['matched']=False;result['matched_exclusion']=row.get('matched_exclusion')
            del arrays,quats,views,k,background
        except Exception as exc:
            result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        write(dest,result)
        print(row['dataset'],row['obj_id'],result['status'],flush=True)


def pixels(args):
    import numpy as np
    import torch,lpips
    from PIL import Image
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    net=lpips.LPIPS(net='alex',version='0.1',verbose=False).cuda().eval()
    for folder in sorted((args.output/'assets').iterdir()):
        record=read(folder/'render.json');dest=folder/'pixels.json'
        if dest.exists() or record['status']!='completed' or not record.get('matched'):continue
        a=np.array(Image.open(folder/'source.png'),dtype=np.float64)/255
        b=np.array(Image.open(folder/'matched.png'),dtype=np.float64)/255
        mse=float(((a-b)**2).mean())
        def tensor(v):return torch.from_numpy((v*2-1).astype('float32')).permute(2,0,1)[None].cuda()
        with torch.inference_mode():perceptual=float(net(tensor(a),tensor(b)).item())
        write(dest,dict(PSNR_dB=-10*np.log10(mse) if mse else None,LPIPS=perceptual,
            exact_pixel_match=mse==0,source_sha256=sha(folder/'source.png'),render_sha256=sha(folder/'matched.png')))
        print('pixels',record['obj_id'],flush=True)


def summarize(args):
    rows=[]
    for folder in sorted((args.output/'assets').iterdir()):
        row=read(folder/'render.json')
        if (folder/'pixels.json').exists():row.update(read(folder/'pixels.json'))
        rows.append(row)
    summaries=[]
    for name in ['nuscenes','waymo','all']:
        chosen=[r for r in rows if name=='all' or r['dataset']==name]
        valid=[r for r in chosen if r['status']=='completed']
        quality=[r for r in valid if r.get('PSNR_dB') is not None]
        summaries.append(dict(dataset=name,assets=len(chosen),valid_renders=len(valid),matched_assets=len(quality),
            PSNR_dB=statistics.mean(r['PSNR_dB'] for r in quality) if quality else None,
            LPIPS=statistics.mean(r['LPIPS'] for r in quality) if quality else None,
            FPS=sum(r['timed_frames'] for r in valid)/sum(r['seconds'] for r in valid) if valid else None,
            median_FPS=statistics.median(r['fps'] for r in valid) if valid else None,
            peak_allocated_mean_MiB=statistics.mean(r['peak_allocated_MiB'] for r in valid) if valid else None,
            peak_allocated_max_MiB=max((r['peak_allocated_MiB'] for r in valid),default=None),
            peak_reserved_max_MiB=max((r['peak_reserved_MiB'] for r in valid),default=None),
            mean_gaussians=statistics.mean(r['gaussians'] for r in valid) if valid else None))
    write(args.output/'results.json',dict(protocol=read(args.output/'protocol.json'),summary=summaries,assets=rows))
    import csv
    with (args.output/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['render','pixels','summarize'])
    parser.add_argument('--registry',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    globals()[args.stage](args)
