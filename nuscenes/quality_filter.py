#!/usr/bin/env python3
"""Moderate rigid-image quality filter used before the FLUX handoff."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def image_metrics(path: Path) -> dict[str, Any]:
    import cv2
    import numpy as np
    from PIL import Image

    with Image.open(path).convert("RGB") as image:
        array = np.asarray(image)
    foreground = np.any(array < 245, axis=2)
    if not foreground.any():
        return {"bbox_w": 0, "bbox_h": 0, "bbox_area": 0, "foreground_area": 0,
                "blur": 0.0, "contrast": 0.0, "phash": 0}
    ys, xs = np.where(foreground)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    crop = array[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if min(gray.shape) >= 3 else 0.0
    contrast = float(gray.std()) if gray.size else 0.0
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype("float32")
    block = cv2.dct(resized)[:8, :8].flatten()
    median = float(np.median(block[1:]))
    phash = 0
    for bit in block > median:
        phash = (phash << 1) | int(bool(bit))
    return {"bbox_w": x1 - x0, "bbox_h": y1 - y0, "bbox_area": (x1 - x0) * (y1 - y0),
            "foreground_area": int(foreground.sum()), "blur": blur, "contrast": contrast,
            "phash": phash}


def filter_manifest(
    manifest: Path, output: Path, reject_log: Path, *, tiny_height: int = 32,
    tiny_area: int = 500, small_height: int = 40, small_area: int = 900,
    blur_threshold: float = 50.0, contrast_threshold: float = 20.0,
    dedupe_hamming: int = 6,
) -> dict[str, Any]:
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("manifest must be a JSON list")
    metrics: dict[str, dict[str, Any]] = {}
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
        metrics[obj_id] = values
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

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[(str(row.get("scene_id")), str(row.get("category")))].append(row)
    kept: list[dict[str, Any]] = []
    for key in sorted(groups):
        ranked = sorted(groups[key], key=lambda row: (
            metrics[str(row["obj_id"])]["bbox_area"], metrics[str(row["obj_id"])]["bbox_h"],
            metrics[str(row["obj_id"])]["blur"], metrics[str(row["obj_id"])]["contrast"],
            str(row["obj_id"])), reverse=True)
        accepted: list[tuple[str, int]] = []
        for row in ranked:
            obj_id, value = str(row["obj_id"]), int(metrics[str(row["obj_id"])]["phash"])
            duplicate = next(((other, (value ^ other_hash).bit_count()) for other, other_hash in accepted
                              if (value ^ other_hash).bit_count() <= dedupe_hamming), None)
            if duplicate:
                rejects.append({"scene_id": row.get("scene_id"), "obj_id": obj_id,
                                "category": row.get("category"), "image_path": row.get("image_path"),
                                "reason": "near_duplicate_in_scene_category",
                                "duplicate_of": duplicate[0], "hamming_distance": duplicate[1]})
            else:
                accepted.append((obj_id, value))
                kept.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    reject_log.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reject_log.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rejects), encoding="utf-8")
    return {"input": len(rows), "kept": len(kept), "rejected": len(rejects),
            "output": str(output.resolve()), "reject_log": str(reject_log.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reject-log", type=Path, required=True)
    parser.add_argument("--dedupe-hamming", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(filter_manifest(args.manifest, args.output, args.reject_log,
                                     dedupe_hamming=args.dedupe_hamming), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
