#!/usr/bin/env python3
"""Validate a pre-FLUX manifest and atomically create the FLUX input JSON."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

VEHICLES = {"car", "vehicle", "truck", "bus", "van", "trailer", "suv", "sedan"}


def validate_rows(rows: Any) -> dict[str, Any]:
    if not isinstance(rows, list):
        raise ValueError("handoff root must be a JSON list")
    identifiers: list[str] = []
    prompt_count = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} must be an object")
        obj_id, category = str(row.get("obj_id") or "").strip(), str(row.get("category") or "").strip()
        image_path, prompts = Path(str(row.get("image_path") or "")).expanduser(), row.get("prompts")
        if not obj_id or not category or not image_path.is_file():
            raise ValueError(f"row {index} lacks a valid obj_id/category/image_path")
        if not isinstance(prompts, list) or not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError(f"row {index} has invalid prompts")
        if category.lower() in VEHICLES:
            first = prompts[0].lower()
            if not ("front-view" in first or "rear-view" in first):
                raise ValueError(f"row {index} vehicle heading is unresolved")
            if not all("preserve" in prompt.lower() for prompt in prompts):
                raise ValueError(f"row {index} vehicle identity binding is missing")
        identifiers.append(obj_id)
        prompt_count += len(prompts)
    duplicates = [key for key, count in Counter(identifiers).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate obj_id values: {duplicates[:8]}")
    return {"items": len(rows), "prompts": prompt_count, "unique_obj_ids": len(identifiers)}


def prepare(source: Path, output: Path) -> dict[str, Any]:
    rows = json.loads(source.read_text(encoding="utf-8"))
    summary = validate_rows(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, output)
    return {"status": "ready_for_flux", "source": str(source.resolve()),
            "handoff": str(output.resolve()), **summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
