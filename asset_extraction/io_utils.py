"""JSON, logging and the unchanged FLUX manifest contract."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable


FINAL_MANIFEST_FIELDS = {"scene_id", "obj_id", "category", "image_path", "prompts"}


def setup_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(path.resolve())
        if not any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", None) == resolved
            for handler in logger.handlers
        ):
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger


def ensure_dir(path: str | Path) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target


def validate_manifest_item(item: Any) -> bool:
    if not isinstance(item, dict):
        raise ValueError("manifest item must be a dict")
    missing = FINAL_MANIFEST_FIELDS - set(item)
    extra = set(item) - FINAL_MANIFEST_FIELDS
    if missing:
        raise ValueError(f"manifest item missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"manifest item has forbidden fields: {sorted(extra)}")
    for key in ("scene_id", "obj_id", "category", "image_path"):
        if not isinstance(item.get(key), str) or not item[key].strip():
            raise ValueError(f"manifest item field {key} must be a non-empty string")
    prompts = item.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("manifest item prompts must be a non-empty list")
    if not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts):
        raise ValueError("manifest item prompts must contain only non-empty strings")
    return True


def write_manifest(path: str | Path, items: Iterable[dict[str, Any]]) -> Path:
    rows = list(items)
    for item in rows:
        validate_manifest_item(item)
    return write_json(path, rows)
