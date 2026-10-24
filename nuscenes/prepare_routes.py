"""Write downstream routing/real evidence separately from the FLUX JSON contract."""
from __future__ import annotations
import argparse, asyncio, hashlib, json
from pathlib import Path
from asset_extraction.io_utils import write_json
from asset_extraction.real_views import VEHICLES, review_references
from .prepare_handoff import validate_rows

def build_routes(handoff: Path, run_dir: Path) -> dict:
    rows=json.loads(handoff.read_text()); validate_rows(rows)
    sources={}
    for path in sorted((run_dir/'scenes').glob('*/logs/export_items_full.json')):
        for record in json.loads(path.read_text()):
            if record['asset_id'] in sources:
                raise ValueError('Duplicate source identity: '+record['asset_id'])
            sources[record['asset_id']]=record
    routes=[]
    for row in rows:
        oid=row['obj_id']; source=sources.get(oid)
        if source is None:
            raise ValueError('Missing provenance for '+oid)
        vehicle=row['category'].lower() in VEHICLES
        real=source.get('real_images',[]) if vehicle else []
        if vehicle and not 0<=len(real)<=4:
            raise ValueError(f'{oid}: expected 0..4 real images')
        for image in real:
            path=Path(image['rgba_path'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=image['sha256']:
                raise ValueError('Real reference changed: '+str(path))
        routes.append(dict(obj_id=oid,scene_id=row.get('scene_id'),category=row['category'],
            route='H' if vehicle else 'B0',real_images=real,
            real_view_selection=source.get('real_view_selection') if vehicle else None,
            flux_output_keys=[f'{oid}_{i}.png' for i in range(len(row['prompts']))],
            attributes=source.get('vlm'), source=source['source']))
    result=dict(schema_version=1,flux_input=str(handoff.resolve()),
        flux_input_sha256=hashlib.sha256(handoff.read_bytes()).hexdigest(),assets=routes)
    return result


async def prepare_routes(handoff: Path, run_dir: Path, output: Path, config) -> dict:
    result = build_routes(handoff, run_dir)
    result['assets'] = await review_references(result['assets'], config, run_dir/'reports/reference_quality.json')
    result['schema_version'] = 2
    write_json(output, result)
    routes = result['assets']
    return dict(assets=len(routes),H=sum(r['route']=='H' for r in routes),
                B0=sum(r['route']=='B0' for r in routes),output=str(output.resolve()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--handoff',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/nuscenes.yaml');a=p.parse_args()
    from asset_extraction.config import load_config
    print(json.dumps(asyncio.run(prepare_routes(a.handoff,a.run_dir,a.output,load_config(a.config))),indent=2))
