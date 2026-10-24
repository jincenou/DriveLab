#!/usr/bin/env python3
"""Moderate rigid-image quality filter used before the FLUX handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any
from asset_extraction.vision import _bbox_iou
from asset_extraction.real_views import VEHICLES, select_real_views
from asset_extraction.io_utils import write_json


@lru_cache(maxsize=128)
def _source_alpha(path: str, width: int, height: int):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.getchannel('A').resize((width,height),Image.Resampling.NEAREST)) > 127


def _same_frame_mask(a: dict, b: dict) -> bool:
    """Compare foreground support in the original RGB pixel coordinates."""
    import numpy as np

    if a['source_image'] != b['source_image'] or _bbox_iou(a['bbox_xyxy'],b['bbox_xyxy']) < .9:
        return False
    if a['sha256'] == b['sha256']:
        return True
    if 'crop_box' not in a or 'crop_box' not in b:
        return False
    x,y=a['crop_box'],b['crop_box']
    left=_source_alpha(a['rgba_path'],x[2]-x[0],x[3]-x[1])
    right=_source_alpha(b['rgba_path'],y[2]-y[0],y[3]-y[1])
    x1,y1,x2,y2=max(x[0],y[0]),max(x[1],y[1]),min(x[2],y[2]),min(x[3],y[3])
    if x2<=x1 or y2<=y1:
        return False
    intersection=np.logical_and(left[y1-x[1]:y2-x[1],x1-x[0]:x2-x[0]],
                                right[y1-y[1]:y2-y[1],x1-y[0]:x2-y[0]]).sum()
    union=int(left.sum()+right.sum()-intersection)
    return union>0 and float(intersection)/union >= .95


def _same_source_identity(a: dict, b: dict) -> bool:
    left = {r['source_image']: r['bbox_xyxy'] for r in a['source']['frame_records']}
    right = {r['source_image']: r['bbox_xyxy'] for r in b['source']['frame_records']}
    overlap = [_bbox_iou(left[key],right[key]) for key in left.keys() & right.keys()]
    if len(overlap) >= 3 and min(overlap) >= .8 and len(overlap) >= .8*min(len(left),len(right)):
        return True
    # Different SAM batches can differ slightly at edges of the same foreground.
    # Same source pixels and near-identical support establish this observation.
    return any(_same_frame_mask(x,y)
               for x in a.get('real_images',[]) for y in b.get('real_images',[]))


@lru_cache(maxsize=512)
def _masked_features(path: str):
    import cv2
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        rgba = np.asarray(image.convert('RGBA'))
    mask = cv2.erode((rgba[...,3] > 240).astype('uint8')*255, np.ones((3,3), 'uint8'))
    gray = cv2.cvtColor(rgba[...,:3], cv2.COLOR_RGB2GRAY)
    gray[mask == 0] = 255
    keys, descriptors = cv2.SIFT_create(nfeatures=800, contrastThreshold=.02).detectAndCompute(gray, mask)
    ys, xs = np.nonzero(mask)
    extent = np.maximum(1, [np.ptp(xs), np.ptp(ys)]) if len(xs) else np.ones(2)
    return keys, descriptors, extent


def _cross_frame_match(a: dict, b: dict) -> dict | None:
    """Conservative foreground evidence between fixed anchors, without a model."""
    import cv2
    import numpy as np

    if not a.get('real_images') or not b.get('real_images'):
        return None
    x, y = a['real_images'][0], b['real_images'][0]
    if any(key not in row for row in (x,y) for key in ('camera_name','frame_index','bbox_xyxy','rgba_path')):
        return None
    if x['camera_name'] != y['camera_name'] or not 0 < abs(x['frame_index']-y['frame_index']) <= 3:
        return None
    left = {r['source_image']: r['bbox_xyxy'] for r in a['source']['frame_records']}
    right = {r['source_image']: r['bbox_xyxy'] for r in b['source']['frame_records']}
    if any(_bbox_iou(left[key], right[key]) < .5 for key in left.keys() & right.keys()):
        return None  # Two objects visible separately in one frame are distinct.
    xb, yb = np.asarray(x['bbox_xyxy']), np.asarray(y['bbox_xyxy'])
    area_ratio = np.prod(xb[2:]-xb[:2]) / max(1, np.prod(yb[2:]-yb[:2]))
    distance = np.linalg.norm((xb[:2]+xb[2:]-yb[:2]-yb[2:])/2)
    if not .25 <= area_ratio <= 4 or distance >= 4*max(np.linalg.norm(xb[2:]-xb[:2]), np.linalg.norm(yb[2:]-yb[:2])):
        return None
    ak, ad, ae = _masked_features(x['rgba_path'])
    bk, bd, be = _masked_features(y['rgba_path'])
    if ad is None or bd is None or min(len(ad), len(bd)) < 10:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    backward = {(m.queryIdx, m.trainIdx) for m,n in matcher.knnMatch(bd,ad,k=2) if m.distance < .7*n.distance}
    good = [m for m,n in matcher.knnMatch(ad,bd,k=2) if m.distance < .7*n.distance and (m.trainIdx,m.queryIdx) in backward]
    if len(good) < 10:
        return None
    p, q = np.float32([ak[m.queryIdx].pt for m in good]), np.float32([bk[m.trainIdx].pt for m in good])
    affine, inliers = cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,
        ransacReprojThreshold=3, maxIters=2000, confidence=.99)
    if affine is None:
        return None
    keep = inliers.ravel().astype(bool)
    count, scale = int(keep.sum()), float(np.linalg.norm(affine[:,0]))
    if count < 10 or count/len(good) < .65 or not .5 <= scale <= 2:
        return None
    coverage, spans = [], []
    for points, extent in [(p[keep],ae), (q[keep],be)]:
        coverage.append(float(cv2.contourArea(cv2.convexHull(points)))/float(np.prod(extent)))
        spans.append((np.ptp(points,axis=0)/extent).tolist())
    if min(coverage) < .08 or any(w < .35 or h < .25 for w,h in spans):
        return None
    return dict(method='masked_sift_mutual_affine', inliers=count, inlier_fraction=count/len(good),
                scale=scale, foreground_hull_coverage=coverage, foreground_spans=spans,
                source_images=[x['source_image'],y['source_image']])


def image_metrics(path: Path) -> dict[str, Any]:
    import cv2
    import numpy as np
    from PIL import Image

    with Image.open(path).convert("RGB") as image:
        array = np.asarray(image)
    foreground = np.any(array < 245, axis=2)
    if not foreground.any():
        return {"bbox_w": 0, "bbox_h": 0, "bbox_area": 0, "foreground_area": 0,
                "blur": 0.0, "contrast": 0.0}
    ys, xs = np.where(foreground)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    crop = array[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if min(gray.shape) >= 3 else 0.0
    contrast = float(gray.std()) if gray.size else 0.0
    return {"bbox_w": x1 - x0, "bbox_h": y1 - y0, "bbox_area": (x1 - x0) * (y1 - y0),
            "foreground_area": int(foreground.sum()), "blur": blur, "contrast": contrast}


def filter_manifest(
    manifest: Path, output: Path, reject_log: Path, *, tiny_height: int = 32,
    tiny_area: int = 500, small_height: int = 40, small_area: int = 900,
    blur_threshold: float = 50.0, contrast_threshold: float = 20.0,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    _source_alpha.cache_clear()
    _masked_features.cache_clear()
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("manifest must be a JSON list")
    provenance, source_files = {}, {}
    if run_dir is not None:
        for path in sorted((run_dir/'scenes').glob('*/logs/export_items_full.json')):
            source_files[path] = json.loads(path.read_text())
            for source in source_files[path]:
                if source['asset_id'] in provenance:
                    raise ValueError('Repeated provenance identity: '+source['asset_id'])
                provenance[source['asset_id']] = source
        if any(row['obj_id'] not in provenance for row in rows):
            raise ValueError('Quality filtering requires provenance for every input identity')
    candidates: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        obj_id = str(item.get("obj_id") or "")
        path = Path(str(item.get("image_path") or "")).expanduser().resolve()
        if not obj_id or not path.is_file():
            rejects.append({"obj_id": obj_id, "image_path": str(path), "reason": "missing_or_invalid_image"})
            continue
        values = image_metrics(path)
        reason = None
        if values["bbox_h"] < tiny_height or values["bbox_area"] < tiny_area:
            reason = "rigid_tiny_object"
        elif values["blur"] < blur_threshold and values["contrast"] < contrast_threshold:
            reason = "rigid_blurry_low_contrast"
        elif (values["bbox_h"] < small_height or values["bbox_area"] < small_area) and (
            values["blur"] < blur_threshold or values["contrast"] < contrast_threshold
        ):
            reason = "rigid_small_low_quality"
        if reason:
            rejects.append({"scene_id": item.get("scene_id"), "obj_id": obj_id,
                            "category": item.get("category"), "image_path": str(path),
                            "reason": reason, **{k: values[k] for k in
                            ("bbox_w", "bbox_h", "bbox_area", "foreground_area", "blur", "contrast")}})
            continue
        item["image_path"] = str(path)
        candidates.append(item)

    # Qwen may reconcile different detector labels for the same physical object.
    # Match source observations, never just the appearance of two exported crops.
    if provenance:
        candidates.sort(key=lambda row: (
            provenance[row['obj_id']]['quality'].get('real_view_eligible',True),
            provenance[row['obj_id']]['quality']['mask_quality'],
            len(provenance[row['obj_id']].get('real_images',[]))), reverse=True)
    kept, identities, sources, merges, match_evidence = [], {}, {}, {}, {}
    for row in candidates:
        obj_id = str(row["obj_id"])
        if obj_id in identities and identities[obj_id] != row:
            raise ValueError(f"Conflicting manifest rows for obj_id {obj_id}")
        identities[obj_id] = row
        key = (str(row.get("scene_id")), str(row["image_path"]))
        if key in sources:
            rejects.append({"scene_id": row.get("scene_id"), "obj_id": obj_id,
                            "reason": "repeated_source_image", "duplicate_of": sources[key]})
        else:
            comparable = [other for other in kept if provenance and
                row['scene_id']==other['scene_id'] and row['category']==other['category']]
            matches = [(other,dict(method='source_observation_overlap')) for other in comparable
                if _same_source_identity(provenance[obj_id],provenance[other['obj_id']])]
            if not matches:
                matches = [(other,evidence) for other in comparable
                    if (evidence := _cross_frame_match(provenance[obj_id],provenance[other['obj_id']]))]
            duplicate = matches[0][0] if len(matches) == 1 else None
            if duplicate is not None:
                merges.setdefault(duplicate['obj_id'],[]).append(obj_id)
                match_evidence[obj_id] = matches[0][1]
                rejects.append({'scene_id':row['scene_id'],'obj_id':obj_id,
                    'reason':'duplicate_source_identity_after_category_verification',
                    'duplicate_of':duplicate['obj_id'], 'evidence':matches[0][1]})
                continue
            sources[key] = obj_id
            kept.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    reject_log.parent.mkdir(parents=True, exist_ok=True)
    # Decisions used fixed representatives, never growing/transitive clusters.
    # Only individually matched cross-frame anchors may supply extra real views.
    for keeper, aliases in merges.items():
        representative = provenance[keeper]
        references = list(representative.get('real_images',[]))
        evidence = {item['asset_id']: item for item in
                    representative['source'].get('merged_from',[])}
        for alias in aliases:
            item=provenance[alias]
            evidence[alias]=dict(asset_id=alias, image_path=item['image_path'],
                image_sha256=hashlib.sha256(Path(item['image_path']).read_bytes()).hexdigest(),
                detected_category=item.get('detected_category'), source=item['source'],
                real_images=item.get('real_images',[]),
                reason='post_category_source_identity_match', evidence=match_evidence[alias])
            if (match_evidence[alias]['method']=='masked_sift_mutual_affine' and
                    representative['vlm']['category'] in VEHICLES):
                candidate = item['real_images'][0]
                if not any(r['source_image']==candidate['source_image'] or r['sha256']==candidate['sha256'] for r in references):
                    references.append(candidate)
        representative['source']['merged_from']=list(evidence.values())
        if len(references) > len(representative.get('real_images',[])):
            # select_real_views expects a content key; do not add it to output rows.
            keyed = [dict(r,rgba_sha256=r['sha256']) for r in references]
            selected, selection = select_real_views(keyed,keyed[0],
                representative['real_view_selection']['maximum'],
                representative['real_view_selection']['quality_floor_relative_to_anchor'])
            representative['real_images'] = [{k:v for k,v in r.items() if k!='rgba_sha256'} for r in selected]
            representative['real_view_selection'] = dict(selection, extra_evidence='fixed_anchor_masked_sift')
    if merges:
        for path, records in source_files.items():
            write_json(path,records)
    output.write_text(json.dumps(kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reject_log.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rejects), encoding="utf-8")
    return {"input": len(rows), "kept": len(kept), "rejected": len(rejects),
            "output": str(output.resolve()), "reject_log": str(reject_log.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reject-log", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(filter_manifest(args.manifest, args.output, args.reject_log,
                                    run_dir=args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
