"""Lightweight, deterministic selection within an already associated identity.

No tracking, annotation requirement or model inference is introduced here.
The first image remains the FLUX anchor; extras are optional real evidence.
"""
from __future__ import annotations
import math
from typing import Any

VEHICLES = {"vehicle", "car", "truck", "bus", "van", "suv", "sedan", "trailer", "pickup"}

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
    eligible = [r for r in rows if float(r['quality']) >= threshold]
    selected = diverse_order(eligible, anchor)[:maximum]
    n = len(selected)
    return selected, dict(real_count=n, preferred_count=[3,4], maximum=maximum,
        quality_floor_relative_to_anchor=quality_ratio,
        status='preferred' if n>=3 else ('two_view_fallback' if n==2 else 'single_view_fallback'),
        reason='maximum_reached' if n==maximum else 'insufficient_distinct_quality_passed_observations',
        candidates_after_mask_qc=len(rows), candidates_after_relative_quality=len(eligible),
        diversity='known azimuth if available, otherwise temporal spacing proxy; no inferred calibration',
        duplicate_padding=False)
