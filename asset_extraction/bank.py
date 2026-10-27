"""Publish portable, categorized rigid packages and merge their catalogue atomically."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .generation import sha256
from .io_utils import read_json, write_json


def publish(state, library: Path, *, dataset: str, scene_id: str):
    library.mkdir(parents=True, exist_ok=True)
    added = []
    for result in state['assets'].values():
        if result['status'] not in {'keep', 'review'} or 'entry' not in result:
            continue
        row = deepcopy(result['entry'])
        uid = f'{dataset}:{scene_id}:{row["uid"]}'
        token = hashlib.sha256(uid.encode()).hexdigest()[:16]
        group = Path('rigid') / ('_review' if row['status']=='review' else '') / row['category']
        relative = group / f'{row["route"]}_{dataset}_{token}'
        destination = library / relative
        row.update(uid=uid, global_uid=uid, dataset=dataset)
        row['provenance']['scene_id'] = scene_id
        signature = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        if destination.exists():
            receipt = read_json(destination / 'package.json')
            if receipt['signature'] != signature:
                raise ValueError(f'Existing package differs: {uid}; use a separate library for changed recipes')
            if any(sha256(destination/name)!=digest for name,digest in receipt['files'].items()):
                raise ValueError(f'Published package changed: {uid}')
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.publish-', dir=destination.parent) as temp:
                stage = Path(temp) / 'asset'
                stage.mkdir()
                mapping = {}
                def include(source, target):
                    if not source:
                        return
                    source = Path(source)
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    target = stage / target
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    if sha256(source) != sha256(target):
                        raise ValueError('Source changed during publication')
                    mapping[str(source)] = str(target.relative_to(stage))
                include(row['canonical']['canonical_path'], 'asset.ply')
                include(row['provenance']['source_image'], 'refs/source.png')
                for i, image in enumerate(row['provenance']['real_images']):
                    include(image['rgba_path'], f'refs/real_{i}.png')
                quality = row['quality']
                include(quality.get('audit_image'), 'previews/source_and_renders.jpg')
                for key in ('evidence','heading_evidence'):
                    include(quality['rendering'].get(key), f'previews/{key}.jpg')
                def local(value):
                    if isinstance(value, dict):
                        return {k:local(v) for k,v in value.items()}
                    if isinstance(value, list):
                        return [local(v) for v in value]
                    return mapping.get(value,value) if isinstance(value,str) else value
                portable = local(row)
                portable['source_path'] = 'asset.ply'
                write_json(stage/'asset.json', portable)
                write_json(stage/'source_camera.json', row['provenance']['source'])
                write_json(stage/'placement.json', dict(
                    coordinate_system='right-handed, +Z up; heading-sensitive accepted assets face +X',
                    canonical_transform_already_applied=True, canonical_bounds_size=row['canonical']['size_xyz'],
                    estimated_metric_dimensions=row['scale'],
                    metric_scale_applied=False, target_scene=None, trajectory=None,
                    note='Apply one isotropic scale and a target-scene pose. Dimensions are estimates; source camera is provenance, not an insertion pose.'))
                write_json(stage/'package.json', dict(uid=uid, signature=signature,
                    files={str(p.relative_to(stage)):sha256(p) for p in stage.rglob('*') if p.is_file()}))
                stage.rename(destination)
        row['source_path'] = str(relative/'asset.ply')
        row['canonical']['canonical_path'] = str(relative/'asset.ply')
        row['package_path'] = str(relative)
        added.append(row)
    with (library/'.rigid.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = library/'manifests/asset_library_rigid.json'
        previous = read_json(path)['assets'] if path.exists() else []
        index = {r['uid']:r for r in previous}
        for row in added:
            if row['uid'] in index and index[row['uid']]['canonical']['sha256'] != row['canonical']['sha256']:
                raise ValueError('Refusing to overwrite an existing asset identity')
            index[row['uid']] = row
        assets = sorted(index.values(), key=lambda r:r['uid'])
        write_json(path, dict(schema_version=2,total=len(assets),
            counts=dict(Counter(r['status'] for r in assets)), assets=assets))
    return added


def retire_rejected(state, chunk: Path):
    """Archive verdicts before removing only this chunk's rejected output folders."""
    rejected = {uid:r for uid,r in state['assets'].items() if r['status']=='reject'}
    if not rejected:
        return
    generation = read_json(chunk/'generation/generation.json')
    targets = []
    for row in generation['assets']:
        if row['obj_id'] in rejected:
            for stage in ('generation','curation'):
                path = chunk/stage/f'asset_{row["item_index"]:05d}'
                if path.is_dir() and not path.is_symlink():
                    targets.append(path)
    manifest = chunk/'library/manifests/asset_library_rigid.json'
    payloads = []
    if manifest.exists():
        local = read_json(manifest)
        retained = [r for r in local['assets'] if r['uid'] not in rejected]
        protected = {r['canonical']['canonical_path'] for r in retained}
        for row in local['assets']:
            relative = row['canonical']['canonical_path']
            target = chunk/'library'/relative
            if row['uid'] in rejected and relative not in protected and target.is_file():
                if not target.resolve().is_relative_to((chunk/'library').resolve()):
                    raise ValueError('Refusing to retire a payload outside this chunk')
                payloads.append(target)
    audit = dict(assets=rejected, files={str(p.relative_to(chunk)):sha256(p)
        for folder in targets for p in folder.rglob('*') if p.is_file()},
        library_payloads={str(p.relative_to(chunk)):sha256(p) for p in payloads})
    write_json(chunk/'rejections.json', audit)
    with (chunk/'rejections.json').open('rb') as stream:
        os.fsync(stream.fileno())
    for target in targets:
        shutil.rmtree(target)
    for target in set(payloads):
        target.unlink()
    if manifest.exists():
        local.update(assets=retained,total=len(retained),counts=dict(Counter(r['category'] for r in retained)))
        write_json(manifest,local)
