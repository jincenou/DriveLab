"""Associated reference selection and async quality admission before reconstruction.

The FLUX anchor stays fixed. H receives only usable real evidence, including zero.
"""
from __future__ import annotations
import math
from typing import Any

VEHICLES = {"vehicle", "car", "truck", "bus", "van", "suv", "sedan", "trailer", "pickup"}

# Measured in the original image, before display enlargement. These are
# engineering admission thresholds, not a claimed optimum for reconstruction.
REFERENCE_POLICY = dict(version=1, min_long_side_px=96, min_short_side_px=48,
                        min_foreground_pixels=2048, max_edge_contact_fraction=.35)
REFERENCE_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'target_is_vehicle': {'type': 'boolean'},
        'occlusion': {'type': 'string', 'enum': ['none', 'mild', 'moderate', 'severe', 'unknown']},
        'visible_structure': {'type': 'string', 'enum': ['complete', 'partial', 'fragment', 'unknown']},
        'detail': {'type': 'string', 'enum': ['usable', 'blurred', 'insufficient', 'unknown']},
        'mask_alignment': {'type': 'string', 'enum': ['good', 'bad', 'uncertain']},
        'reason': {'type': 'string'},
    },
    'required': ['target_is_vehicle', 'occlusion', 'visible_structure', 'detail', 'mask_alignment', 'reason'],
}


def reference_pixels(reference: dict) -> dict:
    """Measure native foreground support; padding and upscaling cannot help."""
    import numpy as np
    from PIL import Image
    with Image.open(reference['rgba_path']) as image:
        if image.mode != 'RGBA':
            raise ValueError('Real evidence must retain its SAM alpha')
        mask = np.asarray(image.getchannel('A')) > 204
        width, height = image.size
    crop = reference['crop_box']
    sx, sy = (crop[2]-crop[0])/width, (crop[3]-crop[1])/height
    if min(sx, sy) <= 0:
        raise ValueError('Invalid native crop dimensions')
    ys, xs = np.nonzero(mask)
    sides = [(int(xs.max()-xs.min()+1)*sx), (int(ys.max()-ys.min()+1)*sy)] if len(xs) else [0., 0.]
    native_pixels = float(mask.sum()*sx*sy)
    edge = (reference.get('source_frame_edge') or {}).get('contact_fraction', 0.)
    reasons = []
    if max(sides) < REFERENCE_POLICY['min_long_side_px'] or min(sides) < REFERENCE_POLICY['min_short_side_px']:
        reasons.append('insufficient_native_foreground_dimensions')
    if native_pixels < REFERENCE_POLICY['min_foreground_pixels']:
        reasons.append('insufficient_native_foreground_pixels')
    if not reference.get('real_view_eligible', True) or edge > REFERENCE_POLICY['max_edge_contact_fraction']:
        reasons.append('truncated_at_source_frame_edge')
    return dict(native_foreground_wh=sides, native_foreground_pixels=native_pixels,
                stored_rgba_wh=[width, height], accepted=not reasons, reasons=reasons)


def reference_verdict(review: dict) -> bool:
    return (review['target_is_vehicle'] and review['occlusion'] in {'none', 'mild'}
            and review['visible_structure'] == 'complete' and review['detail'] == 'usable'
            and review['mask_alignment'] == 'good')


async def review_references(routes: list[dict], config, cache_path) -> list[dict]:
    """Review each vehicle reference once with bounded async JSON requests.

    The FLUX manifest is untouched. Zero surviving references is a supported
    flux-only ReconViaGen job. API failures remain explicit errors, not a
    judgment that the image was bad. Cached decisions are content-addressed.
    """
    import asyncio
    from copy import deepcopy
    import hashlib
    import json
    from pathlib import Path
    from .io_utils import read_json, write_json
    from .vlm import AsyncOpenAIClient, rigid_vlm_messages
    routes = deepcopy(routes)
    cache_path = Path(cache_path)
    cached = read_json(cache_path) if cache_path.exists() else {'decisions': {}}
    decisions = cached['decisions']
    client = AsyncOpenAIClient(config, None)
    semaphore = asyncio.Semaphore(config.vlm.concurrency)
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    client_hash = hashlib.sha256(Path(__file__).with_name('vlm.py').read_bytes()).hexdigest()
    failures = []

    async def one(route, reference):
        async with semaphore:
            rgba_hash = hashlib.sha256(Path(reference['rgba_path']).read_bytes()).hexdigest()
            if rgba_hash != reference['sha256']:
                raise ValueError('Real reference changed before review')
            pixels = reference_pixels(reference)
            source_hash = hashlib.sha256(Path(reference['source_image']).read_bytes()).hexdigest()
            key = hashlib.sha256(json.dumps(dict(rgba=rgba_hash, source=source_hash,
                bbox=reference['bbox_xyxy'], pixels=pixels, policy=REFERENCE_POLICY, code=code_hash,
                client=client_hash, model=config.vlm.model, temperature=config.vlm.temperature), sort_keys=True).encode()).hexdigest()
            if key not in decisions:
                decision = dict(reference_sha256=rgba_hash, source_sha256=source_hash, pixels=pixels,
                                accepted=False, review=None)
                if pixels['accepted']:
                    messages = rigid_vlm_messages(reference['rgba_path'], route['category'], source_context={
                        'source_image': reference['source_image'], 'bbox_xyxy': reference['bbox_xyxy']})
                    messages[0]['content'][0]['text'] = (
                        'Judge whether this REAL observation is suitable as reconstruction evidence. '
                        'LEFT is the SAM foreground; RIGHT is raw scene context with the target in a red box. '
                        f'Native foreground dimensions are {pixels["native_foreground_wh"]} pixels; '
                        'enlargement adds no information. Assess the target only, never a neighboring car. '
                        'target_is_vehicle means a car, van, bus, truck or trailer. '
                        'occlusion describes the target hidden by OTHER objects: none, mild (small local overlap), '
                        'moderate (a substantial body region missing), severe (most body missing), or unknown. '
                        'visible_structure=complete means the expected outline and major body regions for THIS '
                        'view are present. A frontal/rear view can be complete; do not require seeing the far side. '
                        'Use partial for missing major regions, fragment for only a small part. '
                        'detail=usable requires discernible body outline and major features such as windows/lights; '
                        'blurred for strong motion/defocus blur, insufficient for unresolved tiny/pixelated detail. '
                        'mask_alignment=good requires that the SAM foreground follows this target and does not '
                        'include a large occluder, cut away major body regions, or combine several objects. '
                        'A high SAM confidence does not prove completeness. Use unknown/uncertain where needed. '
                        'Return only the requested JSON fields; reason is one concise sentence. Do not assess '
                        'whether an image generator could hallucinate missing parts; assess the observed evidence.')
                    try:
                        decision['review'] = await client.call_json(messages, response_schema=REFERENCE_SCHEMA)
                        decision['accepted'] = reference_verdict(decision['review'])
                    except Exception as exc:
                        failures.append(dict(obj_id=route['obj_id'], reference_sha256=rgba_hash,
                                             error=f'{type(exc).__name__}: {exc}'))
                        return dict(**decision, error=f'{type(exc).__name__}: {exc}')
                decisions[key] = decision
                write_json(cache_path, dict(policy=REFERENCE_POLICY, code_sha256=code_hash,
                    client_sha256=client_hash, decisions=decisions, failures=failures))
            return dict(decisions[key], cache_key=key)

    async def asset(route):
        if route['route'] != 'H':
            return
        references = route['real_images']
        reviews = await asyncio.gather(*(one(route, reference) for reference in references))
        route['real_images'] = [dict(reference, quality_review=review) for reference, review in zip(references, reviews)
                               if review['accepted']]
        count = len(route['real_images'])
        route['real_view_selection'] = dict(real_count=count, preferred_count=[3, 4], maximum=4,
            status='preferred' if count >= 3 else ('two_view' if count == 2 else ('single_view_limited' if count else 'flux_only')),
            reason='use_only_quality_approved_observations', duplicate_padding=False,
            pre_review_selection=route.get('real_view_selection'), policy=REFERENCE_POLICY,
            reference_reviews=[dict(rgba_path=r['rgba_path'], **review) for r, review in zip(references, reviews)])

    tasks = []
    try:
        tasks = [asyncio.create_task(asset(route)) for route in routes]
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()
        write_json(cache_path, dict(policy=REFERENCE_POLICY, code_sha256=code_hash,
            client_sha256=client_hash, decisions=decisions, failures=failures))
    if failures:
        raise RuntimeError(f'{len(failures)} reference quality requests failed; see {cache_path}; no generation started')
    return routes

def observation_key(row: dict[str, Any]) -> tuple:
    if row.get("rgba_sha256"):
        return ("content", row["rgba_sha256"])
    return (row.get("camera_name", row.get("camera")), row.get("frame_id", row.get("frame_index")),
            row.get("image_path", row.get("path", row.get("id"))))

def diverse_order(rows: list[dict[str, Any]], anchor: dict[str, Any] | None = None,
                  score_key: str = "quality") -> list[dict[str, Any]]:
    if not rows:
        return []
    quality = lambda r: float(r.get(score_key, r.get("pre_score", 0.0)))
    anchor = anchor if anchor is not None else max(rows, key=quality)
    chosen = [anchor]
    unique = {observation_key(anchor)}
    remaining = []
    for row in sorted(rows, key=quality, reverse=True):
        key = observation_key(row)
        if key not in unique:
            unique.add(key); remaining.append(row)
    frames = [float(r.get('frame_id', r.get('frame_index', 0))) for r in rows]
    span = max(1.0, max(frames)-min(frames))
    max_quality = max(1e-9, max(map(quality, rows)))
    def distance(a, b):
        if a.get('view_azimuth_rad') is not None and b.get('view_azimuth_rad') is not None:
            angle = float(a['view_azimuth_rad'])-float(b['view_azimuth_rad'])
            return abs(math.atan2(math.sin(angle), math.cos(angle))) / math.pi
        # Temporal spacing is a proxy, explicitly not an estimated camera angle.
        return abs(float(a.get('frame_id', a.get('frame_index', 0)))-
                   float(b.get('frame_id', b.get('frame_index', 0))))/span
    while remaining:
        best = max(remaining, key=lambda r: (0.8*min(distance(r,s) for s in chosen)+
                                            0.2*quality(r)/max_quality, quality(r)))
        chosen.append(best)
        remaining = [r for r in remaining if observation_key(r) != observation_key(best)]
    return chosen

def select_real_views(rows: list[dict[str, Any]], anchor: dict[str, Any],
                      maximum: int = 4, quality_ratio: float = 0.70) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if maximum not in (2, 3, 4) or not 0 <= quality_ratio <= 1:
        raise ValueError('maximum must be 2..4; quality_ratio must be 0..1')
    threshold = float(anchor['quality'])*quality_ratio
    eligible = [r for r in rows if r.get('real_view_eligible',True) and float(r['quality']) >= threshold]
    limited_anchor = not anchor.get('real_view_eligible',True)
    selected = [anchor] if limited_anchor else diverse_order(eligible, anchor)[:maximum]
    excluded = [dict(source_image=r.get('image_path'), camera_name=r.get('camera_name'),
                     frame_index=r.get('frame_id',r.get('frame_index')),
                     reason='truncated_at_source_frame_edge', source_frame_edge=r.get('source_frame_edge'))
                for r in rows if r is not anchor and not r.get('real_view_eligible',True)]
    n = len(selected)
    return selected, dict(real_count=n, preferred_count=[3,4], maximum=maximum,
        quality_floor_relative_to_anchor=quality_ratio,
        status='preferred' if n>=3 else ('two_view' if n==2 else 'single_view_limited'),
        reason='only_truncated_source_observation_available' if limited_anchor else (
            'maximum_reached' if n==maximum else 'insufficient_distinct_quality_passed_observations'),
        limited_anchor_truncated=limited_anchor, excluded_observations=excluded,
        candidates_after_mask_qc=len(rows), candidates_after_relative_quality=len(eligible),
        candidates_after_frame_edge_check=sum(r.get('real_view_eligible',True) for r in rows),
        diversity='known azimuth if available, otherwise temporal spacing proxy; no inferred calibration',
        duplicate_padding=False)
