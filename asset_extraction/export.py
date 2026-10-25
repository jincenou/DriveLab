"""Canonical rigid Gaussians and a portable, content-addressed asset library.

Generation/inspection evidence stays in the work directory. The final directory
contains only unique PLY payloads and two central manifests. Publication never
changes a quality verdict or estimates an object's metric size.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np
from plyfile import PlyData

from .real_views import VEHICLES
from .io_utils import read_json, write_json

LEGACY_ROOT = Path('/mnt/wmj/workspace4/pipeline')
FACING_CATEGORIES = {'traffic_light'}
LONG_AXIS_CATEGORIES = {'barrier', 'fallen_tree', 'road_debris'}
FIELDS = ('x', 'y', 'z', 'opacity', 'f_dc_0', 'f_dc_1', 'f_dc_2',
          'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3')


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_ply(path: str | Path) -> dict:
    """Validate the SH0/log-scale/logit-opacity/wxyz Gaussian contract on CPU."""
    vertex = PlyData.read(str(path))['vertex'].data
    missing = set(FIELDS) - set(vertex.dtype.names)
    if missing or not len(vertex):
        raise ValueError(f'Empty or incomplete Gaussian PLY: {sorted(missing)}')
    values = np.column_stack([vertex[name] for name in FIELDS])
    if not np.isfinite(values).all():
        raise ValueError('Gaussian attributes contain nonfinite values')
    rest = [name for name in vertex.dtype.names if name.startswith('f_rest_')]
    if any(np.any(vertex[name] != 0) for name in rest):
        raise ValueError('Directional SH rotation is not supported by this exporter')
    norms = np.linalg.norm(values[:, 10:14], axis=1)
    with np.errstate(over='ignore', under='ignore'):
        scales = np.exp(values[:, 7:10])
    if np.any(norms < 1e-8) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('Degenerate Gaussian rotations or scales')
    if not np.any(values[:, 3] > np.log(.01 / .99)):
        raise ValueError('No visible Gaussian opacity')
    low, high = np.quantile(values[:, :3], [.005, .995], axis=0)
    if np.any(high - low <= 1e-6):
        raise ValueError('Degenerate Gaussian bounds')
    return dict(gaussian_count=len(vertex), robust_bounds=[low.tolist(), high.tolist()],
                scale_min_max=[float(scales.min()), float(scales.max())],
                quaternion_norm_min_max=[float(norms.min()), float(norms.max())])


def _needs_heading(category):
    return category in VEHICLES or category == 'traffic_sign' or category in FACING_CATEGORIES


def _verified_heading(heading, *, semantic_face=False):
    if not heading or heading.get('front') not in {'A', 'B', '+X'}:
        return False
    confidence = heading.get('confidence', 0)
    if not isinstance(confidence, (int, float)) or not .7 <= confidence <= 1:
        return False
    if not semantic_face:
        return True
    check = heading.get('verification') or {}
    return (heading.get('front') == '+X' and heading.get('azimuth_deg') in np.arange(16)*22.5
            and check.get('front') == 'A' and .7 <= check.get('confidence', 0) <= 1)


def canonicalize_raw(source: str | Path, target: str | Path, *, route: str,
                     category: str, heading: dict | None = None,
                     legacy_root: str | Path = LEGACY_ROOT) -> dict:
    """Use the original Gaussian transform/grounding, with explicit backend up.

    For vehicles and signs, `heading` is the reviewed A/B end decision after
    PCA alignment (A = +x, B = -x), with confidence in [0,1] and its source.
    Traffic lights use a semantic face azimuth from the 16-view sheet, then
    require a separate +x face verification before admission.
    Omitting it produces a usable review payload with unresolved front, never
    a falsely verified orientation. Metric scale is not baked into the PLY.
    """
    source, target = Path(source), Path(target)
    if route not in {'H', 'B0'}:
        raise ValueError(f'Unknown reconstruction route: {route}')
    if target.exists():
        raise FileExistsError(target)
    inspect_ply(source)
    root = str(Path(legacy_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from curate.core import load_rigid
    from curate.stages import UP_FIX, FLIP_Z, _export_ply, _grounding, _yaw_rotation

    gaussians = load_rigid(source)
    rotation = np.eye(3) if route == 'H' else UP_FIX.copy()
    if category in FACING_CATEGORIES:
        if heading is not None:
            if (heading.get('front') != '+X' or heading.get('azimuth_deg') not in np.arange(16)*22.5
                    or not isinstance(heading.get('confidence'), (int, float))
                    or not 0 <= heading['confidence'] <= 1 or not heading.get('source')):
                raise ValueError('Signal heading needs a semantic face azimuth, confidence and source')
            yaw = np.deg2rad(-heading['azimuth_deg'])
            c, s = np.cos(yaw), np.sin(yaw)
            rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ rotation
    elif _needs_heading(category):
        mode = 'long_axis' if category in VEHICLES else 'normal_axis'
        rotation = _yaw_rotation(gaussians, mode, up_rotation=rotation) @ rotation
        if heading is not None:
            if (heading.get('front') not in {'A', 'B'}
                    or not isinstance(heading.get('confidence'), (int, float))
                    or not 0 <= heading['confidence'] <= 1
                    or not heading.get('source')):
                raise ValueError('Heading needs A/B, confidence in [0,1] and source')
            if heading['front'] == 'B':
                rotation = FLIP_Z @ rotation
    elif category in LONG_AXIS_CATEGORIES:
        if heading is not None:
            raise ValueError('Geometric long-axis alignment does not establish a semantic front')
        rotation = _yaw_rotation(gaussians, 'long_axis', up_rotation=rotation) @ rotation
    elif heading is not None:
        raise ValueError('Heading decision supplied to a category without axis alignment')
    translation, size = _grounding(gaussians.xyz[0].astype(np.float64) @ rotation.T)
    _export_ply(source, target, rotation, translation)
    validation = inspect_ply(target)
    return dict(canonical_path=str(target.resolve()), rotation=rotation.tolist(),
                translation=translation.tolist(), size_xyz=size,
                heading=deepcopy(heading), heading_verified=_verified_heading(heading, semantic_face=category in FACING_CATEGORIES),
                up='+Z', raw_up='+Z' if route == 'H' else '-Y',
                units='normalized', metric_scale_baked=False,
                grounding='0.5th percentile of Gaussian centers at z=0',
                collision_footprint_xy=[[-size[0]/2, -size[1]/2], [size[0]/2, -size[1]/2],
                                        [size[0]/2, size[1]/2], [-size[0]/2, size[1]/2]],
                validation=validation)


def export_library(entries: list[dict], output_dir: str | Path, *,
                   pipeline_root: str | Path | None = None) -> Path:
    """Publish validated canonical entries; existing destinations are immutable.

    Input entries follow the old `asset_library*.json` format. Relative input
    paths require `pipeline_root`. Output paths are relative to `output_dir`;
    pass that directory as `pipeline_root` to the original AssetLibrary loader.
    Review/reject rows remain in the manifest and are excluded by that loader.
    """
    output = Path(output_dir).absolute()
    if output.exists():
        raise FileExistsError(output)
    if not entries:
        raise ValueError('Cannot publish an empty asset library')
    rows = deepcopy(entries)
    if len({row['uid'] for row in rows}) != len(rows):
        raise ValueError('Duplicate logical asset uid')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f'.{output.name}.', dir=output.parent) as temp:
        staging = Path(temp) / 'library'
        (staging / 'manifests').mkdir(parents=True)
        physical = {}
        for row in rows:
            category = row['category']
            if not re.fullmatch(r'[a-z][a-z0-9_]*', category) or category == 'human':
                raise ValueError(f'Invalid rigid category: {category}')
            if row['status'] not in {'keep', 'review', 'reject'}:
                raise ValueError(f"Unknown quality status: {row['status']}")
            if not isinstance(row.get('provenance'), dict) or not row['provenance']:
                raise ValueError(f"Missing provenance: {row['uid']}")
            canonical = row['canonical']
            size = np.asarray(canonical['size_xyz'], dtype=float)
            if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
                raise ValueError('Canonical size_xyz must contain three positive extents')
            if row['status'] == 'keep' and _needs_heading(category):
                if not _verified_heading(canonical.get('heading'), semantic_face=category in FACING_CATEGORIES):
                    raise ValueError('Kept heading-sensitive asset needs a resolved canonical front')
            if row['status'] == 'keep':
                scale = row.get('scale', {})
                dimensions = np.asarray([scale.get(key, 0) for key in
                    ('est_length_m','est_width_m','est_height_m')], dtype=float)
                if (scale.get('scale_confidence') not in {'medium','high'} or
                        not np.isfinite(dimensions).all() or np.any(dimensions <= 0)):
                    raise ValueError('Kept asset needs positive dimensions and a trusted metric-scale estimate')
            if row.get('color_saliency') in {'dark','neutral','bright'}:
                row.setdefault('label', {})['body_brightness'] = row['color_saliency']
            source = Path(canonical['canonical_path'])
            if not source.is_absolute():
                if pipeline_root is None:
                    raise ValueError('Relative input canonical_path requires pipeline_root')
                source = Path(pipeline_root) / source
            validation = inspect_ply(source)
            low, high = np.asarray(validation['robust_bounds'])
            if not np.allclose(high - low, size, rtol=.005, atol=.001):
                raise ValueError('Canonical size metadata disagrees with its PLY')
            if not np.allclose([(low[0]+high[0])/2, (low[1]+high[1])/2, low[2]], 0, atol=.001):
                raise ValueError('Canonical PLY must be XY-centered and grounded at z=0')
            sha = _sha(source)
            if sha not in physical:
                relative = Path('payloads/candidate/ply') / category / sha[:2] / f'{sha}.ply'
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                if _sha(target) != sha:
                    raise ValueError(f'PLY changed while publishing: {source}')
                physical[sha] = dict(canonical_path=relative.as_posix(), category=category,
                                     format='ply', sha256=sha, size=target.stat().st_size,
                                     tier='candidate')
            relative = physical[sha]['canonical_path']
            # Keep source lineage in metadata, not a second copy of raw Gaussians.
            row['provenance'].setdefault('raw_source_path', row.get('source_path'))
            row['source_path'] = relative
            canonical['canonical_path'] = relative
            canonical['sha256'] = sha
        manifest = staging / 'manifests/asset_library_rigid.json'
        manifest.write_text(json.dumps(dict(schema_version=2, total=len(rows),
                            counts=dict(Counter(row['category'] for row in rows)), assets=rows),
                            ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        (staging / 'manifests/global_asset_manifest.jsonl').write_text(
            ''.join(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n'
                    for _, value in sorted(physical.items())))
        staging.rename(output)
    return output / 'manifests/asset_library_rigid.json'


def render_evidence(source: str | Path, output_dir: str | Path, *, size: int = 384,
                    legacy_root: str | Path = LEGACY_ROOT) -> dict:
    """Batch-render 16 shared orbit cameras; retain only two evidence sheets.

    Input must already be z-up and grounded. The paired A/B sheet views the
    canonical +x/-x ends, for the heading decision. Alpha is inspected before
    any JPEG conversion; this is renderability evidence, not a quality verdict.
    """
    import torch
    from gsplat import rasterization
    from PIL import Image, ImageDraw

    source, output = Path(source), Path(output_dir)
    inspect_ply(source)
    root = str(Path(legacy_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from curate.core import load_rigid

    gs = load_rigid(source)
    low, high = np.quantile(gs.xyz[0], [.005, .995], axis=0)
    center = (low + high) / 2
    radius = np.linalg.norm(high - low) / 2
    distance = radius / .85 * 2.2
    focal = .5 * size * distance / radius * .85
    angles = np.arange(16) * np.pi / 8
    elevation = np.deg2rad(12)
    direction = np.column_stack([np.cos(angles) * np.cos(elevation),
                                 np.sin(angles) * np.cos(elevation),
                                 np.full(16, np.sin(elevation))])
    eye = center + distance * direction
    forward = -direction
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    rotation = np.stack([right, np.cross(forward, right), forward], axis=1)
    views = np.tile(np.eye(4), (16, 1, 1))
    views[:, :3, :3] = rotation
    views[:, :3, 3] = -np.einsum('bij,bj->bi', rotation, eye)
    intrinsics = np.tile([[focal, 0, size/2], [0, focal, size/2], [0, 0, 1]], (16, 1, 1))

    def tensor(array):
        return torch.as_tensor(array, device='cuda', dtype=torch.float32)

    quats = tensor(gs.rot[0])
    quats /= quats.norm(dim=-1, keepdim=True)
    with torch.inference_mode():
        rgb, alpha, _ = rasterization(
            tensor(gs.xyz[0]), quats, tensor(gs.scale[0]), tensor(gs.opacity), tensor(gs.color),
            tensor(views), tensor(intrinsics), size, size,
            backgrounds=torch.full((16, 3), .5, device='cuda'), packed=False)
    if not torch.isfinite(rgb).all() or not torch.isfinite(alpha).all():
        raise ValueError('Rendered RGB/alpha contains nonfinite values')
    foreground = (alpha[..., 0] > .01).sum(dim=(1, 2)).cpu().tolist()
    frames = (rgb[..., :3].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    output.mkdir(parents=True, exist_ok=True)
    sheet = Image.new('RGB', (size*4, (size+24)*4), '#cccccc')
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(frames):
        x, y = index % 4 * size, index // 4 * (size+24)
        sheet.paste(Image.fromarray(frame), (x, y+24))
        draw.text((x+6, y+5), f'{index*22.5:g} deg', fill='black')
    sheet.save(output / 'evidence.jpg', quality=92)
    ends = Image.new('RGB', (size*2, size+24), '#cccccc')
    ends.paste(Image.fromarray(frames[0]), (0, 24))
    ends.paste(Image.fromarray(frames[8]), (size, 24))
    draw = ImageDraw.Draw(ends)
    draw.text((6, 5), 'A: +X', fill='black')
    draw.text((size+6, 5), 'B: -X', fill='black')
    ends.save(output / 'heading.jpg', quality=92)
    audit = dict(ply_sha256=_sha(source), resolution=size, azimuths_deg=(angles*180/np.pi).tolist(),
                 elevation_deg=12, foreground_alpha_threshold=.01,
                 foreground_pixels=foreground, all_16_nonempty=all(count > 0 for count in foreground),
                 evidence=str((output / 'evidence.jpg').resolve()),
                 heading_evidence=str((output / 'heading.jpg').resolve()),
                 loader='curate.core.load_rigid', renderer='gsplat.rasterization')
    (output / 'render.json').write_text(json.dumps(audit, indent=2) + '\n')
    return audit


def _schema(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


HEADING_SCHEMA = _schema(dict(front={'enum': ['A', 'B', 'unknown']},
                             confidence=dict(type='number', minimum=0, maximum=1)))
FACE_SCHEMA = _schema(dict(view_index=dict(type='integer', minimum=-1, maximum=15),
                          confidence=dict(type='number', minimum=0, maximum=1)))
SCALE_SCHEMA = _schema(dict(object_desc=dict(type='string', maxLength=160),
    # The deployed constrained decoder truncates sub-meter decimals with an
    # exclusive zero bound. An inclusive 1 mm bound keeps dimensions positive.
    est_length_m=dict(type='number', minimum=.001),
    est_width_m=dict(type='number', minimum=.001),
    est_height_m=dict(type='number', minimum=.001),
    scale_confidence={'enum': ['high', 'medium', 'low']}))
AUDIT_SCHEMA = _schema(dict(keep=dict(type='boolean'), quality_score=dict(type='number', minimum=0, maximum=10),
    defects=dict(type='array', maxItems=5, items=dict(type='string', maxLength=160)), category_match=dict(type='boolean'),
    identity_consistent=dict(type='boolean'), symmetry_360_ok=dict(type='boolean'),
    pose_state={'enum': ['upright', 'fallen', 'tilted', 'leaning']},
    ground_contact={'enum': ['flat_base', 'wheels', 'irregular', 'needs_burial']},
    sub_type=dict(type='string', maxLength=80), color_saliency={'enum': ['dark', 'neutral', 'bright']},
    usable_as=dict(type='array', maxItems=5, items=dict(type='string', maxLength=80)), scale=SCALE_SCHEMA))


def _identity_variation_only(audit, rendering):
    """An identity deviation alone does not make an otherwise usable vehicle broken."""
    return (not audit['identity_consistent'] and audit['keep'] and audit['category_match']
            and audit['quality_score'] >= 5 and rendering['all_16_nonempty'])


def _verdict(audit, rendering, heading, needs_heading, *, semantic_face=False,
             allow_identity_variation=False):
    broken = not all([rendering['all_16_nonempty'], audit['keep'], audit['category_match'],
                      audit['identity_consistent'] or allow_identity_variation, audit['quality_score'] >= 5])
    reliable = audit['scale']['scale_confidence'] in {'high', 'medium'} and (
        not needs_heading or _verified_heading(heading, semantic_face=semantic_face))
    return 'reject' if broken else 'keep' if reliable else 'review'


def _audit_sheet(source, renders, target):
    """One labeled image, compatible with a one-image VLM request budget."""
    from PIL import Image, ImageDraw

    with Image.open(renders) as image:
        orbit = image.convert('RGB')
    side = orbit.width // 4
    with Image.open(source) as image:
        reference = image.convert('RGBA')
    source_width, source_height = reference.size
    scale = side / max(reference.size)
    reference = reference.resize((max(1, round(reference.width*scale)),
                                  max(1, round(reference.height*scale))), Image.Resampling.LANCZOS)
    sheet = Image.new('RGB', (side + 16 + orbit.width, orbit.height + 24), 'white')
    x = (side - reference.width) // 2
    sheet.paste(reference, (x, 24), reference.getchannel('A'))
    sheet.paste(orbit, (side + 16, 24))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 5), f'SOURCE: real {source_width}x{source_height} px', fill='black')
    draw.text((side + 22, 5), 'RENDERS: 16 orbit views', fill='black')
    sheet.save(target, quality=95)
    return Path(target)


def _vlm_messages(prompt, image, schema):
    from .vlm import _image_data_url
    return [dict(role='user', content=[
        dict(type='text', text=prompt + '\nReturn a JSON object matching this schema exactly: ' + json.dumps(schema)),
        dict(type='image_url', image_url=dict(url=_image_data_url(Path(image))))])]


def _validate_sources(row):
    if _sha(Path(row['image_path'])) != row['image_sha256']:
        raise ValueError(f'Source image changed after generation: {row["obj_id"]}')
    if any(_sha(Path(image['rgba_path'])) != image['sha256'] for image in row['real_images']):
        raise ValueError(f'Real reference changed after generation: {row["obj_id"]}')


async def curate(generation_path, output_dir, config, *, renderer_python,
                 gpus, work_dir=None):
    """Concurrent JSON audits with GPU render slots, then minimal publication.

    Model responses and render checks are retained in one work ledger. A low
    confidence scale or unresolved required heading keeps an asset in review;
    it is never admitted through the old loader's category-size substitution.
    """
    from .vlm import AsyncOpenAIClient
    from .generation import GpuPool

    generation_path = Path(generation_path).resolve()
    generation = read_json(generation_path)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not gpus or len(gpus) != len(set(gpus)) or not Path(renderer_python).is_file():
        raise ValueError('Provide distinct GPU slots and an existing renderer Python')
    work = Path(work_dir).resolve() if work_dir else generation_path.parent / 'curation'
    work.mkdir(parents=True, exist_ok=True)
    state_path = work / 'curation.json'
    code_sha256 = _sha(Path(__file__))
    client_sha256 = _sha(Path(__file__).with_name('vlm.py'))
    signature = hashlib.sha256(json.dumps(dict(generation_sha256=_sha(generation_path),
        code_sha256=code_sha256, client_sha256=client_sha256,
        response_format=os.environ.get('VLM_RESPONSE_FORMAT', 'json_schema'),
        vlm=asdict(config.vlm)), sort_keys=True).encode()).hexdigest()
    state = read_json(state_path) if state_path.exists() else dict(signature=signature,
        code_sha256=code_sha256, client_sha256=client_sha256, assets={})
    if state['signature'] != signature:
        raise ValueError('Generation/code/VLM settings changed; use a new curation work directory')
    client = AsyncOpenAIClient(config, None)
    network = asyncio.Semaphore(config.vlm.concurrency)
    slots = GpuPool(gpus)
    root = Path(__file__).resolve().parents[1]
    legacy_root = generation['config']['legacy_root']

    async def ask(prompt, image, schema, index):
        async with network:
            return await client.call_json(_vlm_messages(prompt, image, schema),
                                          client_index=index, response_schema=schema)

    async def render(ply, directory):
        audit_path = directory / 'render.json'
        if audit_path.is_file():
            audit = read_json(audit_path)
            if audit['ply_sha256'] != _sha(ply):
                raise ValueError('Render cache no longer matches its Gaussian payload')
            if all(Path(audit[name]).is_file() for name in ['evidence', 'heading_evidence']):
                return audit
            raise FileNotFoundError('Render evidence referenced by cache is missing')
        gpu = await slots.acquire('render')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1',
                   PATH=str(Path(renderer_python).parent) + ':' + os.environ['PATH'],
                   CPATH='/home/wmj/miniconda3/envs/splatad/include',
                   TORCH_EXTENSIONS_DIR='/mnt/ojc/workplace1/reconviagen_demo/cache/torch_extensions', MAX_JOBS='2')
        process = None
        directory.mkdir(parents=True, exist_ok=True)
        try:
            with (directory / 'render.log').open('wb') as log:
                process = await asyncio.create_subprocess_exec(renderer_python, '-m', 'asset_extraction.export',
                    '--ply', str(ply), '--output', str(directory), '--legacy-root', str(legacy_root), cwd=root, env=env,
                    stdout=log, stderr=asyncio.subprocess.STDOUT)
                if await process.wait():
                    raise RuntimeError(f'Render failed: {directory / "render.log"}')
            return read_json(audit_path)
        finally:
            if process is not None and process.returncode is None:
                process.terminate()
                await process.wait()
            slots.release(gpu)

    async def one(row):
        _validate_sources(row)
        uid, index = row['obj_id'], row['item_index']
        if uid in state['assets']:
            result = state['assets'][uid]
            if 'entry' in result and _sha(Path(result['entry']['canonical']['canonical_path'])) != result['canonical_sha256']:
                raise ValueError(f'Curated payload changed: {uid}')
            return result
        if row['status'] != 'generated':
            result = dict(status='failed', error='Asset generation did not succeed')
            state['assets'][uid] = result
            write_json(state_path, state)
            return result
        directory = work / f'asset_{index:05d}'
        directory.mkdir(parents=True, exist_ok=True)
        try:
            source = Path(row['raw_path'])
            if _sha(source) != row['raw_sha256']:
                raise ValueError('Raw Gaussian payload changed after generation')
            axis_path = directory / 'axis.ply'
            # Uncommitted geometry needs a fresh work directory; incomplete
            # canonicalization is not advertised as resumable.
            if axis_path.exists():
                raise FileExistsError(f'Uncommitted canonicalization exists: {axis_path}')
            canonical = canonicalize_raw(source, axis_path, route=row['route'], category=row['category'],
                                         legacy_root=legacy_root)
            rendering = await render(axis_path, directory / 'axis_render')
            heading = None
            decisions = {}
            decision_path = directory / 'decisions.json'
            needs_heading = _needs_heading(row['category'])
            semantic_face = row['category'] in FACING_CATEGORIES
            if semantic_face:
                decision = await ask('The 4 by 4 contact sheet shows one traffic light at 16 azimuths. '
                    'Choose the view most directly facing the functional signal face: the colored lamps or pedestrian '
                    'symbol display, not the rear housing, side, pole, base or sticker. Rows run left to right, top '
                    'to bottom: view_index 0..15 correspond to 0, 22.5, 45, ...337.5 degrees. '
                    'Use view_index=-1 and low confidence if the signal face is unclear, missing, faces several '
                    'incompatible directions, or cannot be distinguished from the rear. Do not guess.',
                    rendering['evidence'], FACE_SCHEMA, index)
                decisions['semantic_face'] = decision
                write_json(decision_path, decisions)
                heading = dict(front='+X' if decision['view_index'] >= 0 else 'unknown', **decision,
                    source=f'{config.vlm.model}; face sheet SHA256={_sha(Path(rendering["evidence"]))}')
                if decision['view_index'] >= 0:
                    heading['azimuth_deg'] = decision['view_index'] * 22.5
                    canonical = canonicalize_raw(source, directory / 'canonical.ply', route=row['route'],
                        category=row['category'], heading=heading, legacy_root=legacy_root)
                    rendering = await render(Path(canonical['canonical_path']), directory / 'canonical_render')
                    heading['verification'] = await ask('These are opposite views of the rotated traffic light. '
                        'A looks from canonical +X and B from -X. Which view directly shows its functional colored '
                        'lamp or pedestrian symbol face? A side-on face does not qualify. Ignore stickers, rear '
                        'housing, pole and base. Use unknown if neither directly shows a clear signal face.',
                        rendering['heading_evidence'], HEADING_SCHEMA, index)
                    decisions['face_verification'] = heading['verification']
                    write_json(decision_path, decisions)
                    heading['verification_image_sha256'] = _sha(Path(rendering['heading_evidence']))
                    canonical['heading'] = deepcopy(heading)
                    canonical['heading_verified'] = _verified_heading(heading, semantic_face=True)
            elif needs_heading:
                kind = 'vehicle front with headlights/windshield/hood' if row['category'] in VEHICLES else 'readable sign face'
                heading = await ask(f'Two renders A and B show opposite ends of one {row["category"]}. '
                    f'Which shows the {kind}? Use unknown if unclear. Do not guess; confidence is 0..1.',
                    rendering['heading_evidence'], HEADING_SCHEMA, index)
                decisions['heading'] = heading
                write_json(decision_path, decisions)
                if heading['front'] != 'unknown':
                    heading.update(source=f'{config.vlm.model}; heading image SHA256={_sha(Path(rendering["heading_evidence"]))}')
                    canonical = canonicalize_raw(source, directory / 'canonical.ply', route=row['route'],
                        category=row['category'], heading=heading, legacy_root=legacy_root)
                    rendering = await render(Path(canonical['canonical_path']), directory / 'canonical_render')
            prompt = (f'Inspect a reconstructed 3D Gaussian rigid asset of category {row["category"]}. '
                'Return only a concise, strict JSON object. Give at most 5 defects, each at most 160 characters; '
                'state each distinct issue once without repeating explanations. usable_as has at most 5 short entries. '
                'Keep sub_type below 80 characters and object_desc below 160 characters. No narrative outside JSON. '
                'This is ONE labeled contact sheet: the LEFT SOURCE panel is the real source foreground; '
                'the RIGHT RENDERS panel contains 16 renders at 22.5 degree intervals. '
                'Evaluate source identity, category, full geometry, cross-view consistency, floating noise, holes, '
                'missing parts, severe texture corruption and deformations. Medium strictness: minor blur is acceptable. '
                'Explicitly compare any text, numbers, decals, stickers, logos, stripes and other conspicuous markings. '
                'New conspicuous markings unsupported by the real source are identity defects: set identity_consistent=false '
                'and describe the invented markings in defects. Do not treat an unreadable or low-resolution source as '
                'positive evidence that generated text or logos were present. Source dimensions are shown in the label; '
                'display enlargement adds no source detail. Do not invent a positive identity judgment from missing evidence. '
                'keep=false for clearly broken or implausible assets; quality_score 0..10. Report all defects. '
                'symmetry_360_ok describes rotational symmetry; asymmetry alone is not a quality defect. '
                'For signs, inspect plausible plate shape, content and an attached support. '
                'Describe insertion labels, ground contact and pose. Estimate physical size from recognized subtype; '
                'these are visual estimates, not measured sensor dimensions. length is the longest horizontal extent, '
                'width the other horizontal extent and height vertical. All three dimensions must be positive '
                'numbers in meters, never 0. Use low confidence for ambiguous subtype or uncertain size; '
                'uncertainty does not mean a zero-sized object. '
                'Do not inflate confidence to pass quality checks. usable_as is a list of suitable driving-scenario roles.')
            audit_image = _audit_sheet(row['image_path'], rendering['evidence'], directory / 'source_and_renders.jpg')
            audit = await ask(prompt, audit_image, AUDIT_SCHEMA, index)
            decisions['quality'] = audit
            write_json(decision_path, decisions)
            _validate_sources(row)
            vehicle_policy = row['category'] in VEHICLES
            status = _verdict(audit, rendering, heading, needs_heading, semantic_face=semantic_face,
                              allow_identity_variation=vehicle_policy)
            canonical['sha256'] = _sha(Path(canonical['canonical_path']))
            entry = dict(uid=uid, global_uid=f'{config.dataset}:{uid}', dataset=config.dataset,
                category=row['category'], **{'class': 'vehicle' if row['category'] in VEHICLES else row['category']},
                route=row['route'], source_path=str(source), status=status, quality_score=audit['quality_score'],
                defects=audit['defects'], category_match=audit['category_match'],
                pose_state=audit['pose_state'], ground_contact=audit['ground_contact'],
                symmetry_360_ok=audit['symmetry_360_ok'], color_saliency=audit['color_saliency'],
                label={key: audit[key] for key in ['sub_type', 'usable_as']}, scale=audit['scale'],
                canonical=canonical, provenance=dict(source_obj_id=uid, scene_id=row['scene_id'],
                    source_image=row['image_path'], source_image_sha256=row['image_sha256'], source=row['source'],
                    real_images=row['real_images'], real_view_selection=row['real_view_selection'],
                    prompts=row['prompts'], input_mode=row.get('input_mode', 'flux_plus_real' if row['real_images'] else 'flux_only'),
                    raw_sha256=row['raw_sha256'], generation_signature=generation['signature']),
                quality=dict(model=config.vlm.model, code_sha256=code_sha256, audit=audit, heading_decision=heading,
                             admission_policy='vehicle_identity_variation_retained_v1' if vehicle_policy else 'strict_source_identity_v1',
                             retained_with_identity_variation=bool(vehicle_policy and status in {'keep','review'}
                                 and _identity_variation_only(audit, rendering)),
                             response_format=os.environ.get('VLM_RESPONSE_FORMAT', 'json_schema'),
                             audit_image=str(audit_image), audit_image_sha256=_sha(audit_image),
                             rendering=rendering, scale_source='VLM subtype estimate; not a metric measurement'))
            if status == 'review':
                entry['reason'] = 'Required heading or metric-scale confidence is insufficient'
            result = dict(status=status, entry=entry, canonical_sha256=canonical['sha256'])
        except Exception as exc:
            result = dict(status='failed', error=f'{type(exc).__name__}: {exc}')
            if hasattr(exc, 'response'):
                result['response'] = exc.response
        state['assets'][uid] = result
        write_json(state_path, state)
        return result

    tasks = []
    try:
        await client.check_endpoints()
        tasks = [asyncio.create_task(one(row)) for row in generation['assets']]
        results = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()
    entries = [result['entry'] for result in results if 'entry' in result]
    state['counts'] = dict(queries=len(results), **dict(Counter(result['status'] for result in results)))
    if entries:
        state['library'] = str(export_library(entries, output))
    state['status'] = 'curated' if len(entries) == len(results) else 'incomplete'
    write_json(state_path, state)
    return state


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Render canonical rigid validation evidence')
    parser.add_argument('--ply', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=384)
    parser.add_argument('--legacy-root', type=Path, default=LEGACY_ROOT)
    parser.add_argument('--generation', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--renderer-python', default='/home/wmj/miniconda3/envs/sim/bin/python')
    parser.add_argument('--gpus', type=int, nargs='+', default=[7])
    parser.add_argument('--work', type=Path)
    args = parser.parse_args()
    if args.ply and not args.generation:
        print(json.dumps(render_evidence(args.ply, args.output, size=args.size, legacy_root=args.legacy_root)))
    elif args.generation and args.config and not args.ply:
        from .config import load_config
        result = asyncio.run(curate(args.generation, args.output, load_config(args.config),
            renderer_python=args.renderer_python, gpus=args.gpus, work_dir=args.work))
        print(json.dumps(dict(status=result['status'], counts=result['counts'])))
        if result['status'] == 'incomplete':
            raise SystemExit(1)
    else:
        parser.error('Use --ply for rendering, or --generation and --config for curation')
