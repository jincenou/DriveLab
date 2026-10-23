"""I/O, logging, command execution, and manifest helpers."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Sequence


FINAL_MANIFEST_FIELDS = {"scene_id", "obj_id", "category", "image_path", "prompts"}
ACTION_MANIFEST_FIELDS = {"scene_id", "obj_id", "category", "video_path", "action", "prompts"}


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
    target.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def read_jsonl(path: str | Path) -> list[Any]:
    source = Path(path)
    if not source.is_file():
        return []
    rows: list[Any] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target


def append_jsonl(path: str | Path, row: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target


def safe_symlink_or_copy(src: str | Path, dst: str | Path) -> Path:
    source = Path(src)
    target = Path(dst)
    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    try:
        target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    return target


def run_command(
    cmd: Sequence[str],
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    logger: logging.Logger | None = None,
) -> subprocess.CompletedProcess[str]:
    if logger:
        logger.info("command: %s", " ".join(str(part) for part in cmd))
        logger.info("cwd: %s", cwd or os.getcwd())
    completed = subprocess.run(
        [str(part) for part in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if logger:
        if completed.stdout:
            logger.info("stdout:\n%s", completed.stdout.rstrip())
        if completed.stderr:
            logger.info("stderr:\n%s", completed.stderr.rstrip())
        logger.info("returncode: %s", completed.returncode)
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code "
            f"{completed.returncode}: {' '.join(str(part) for part in cmd)}\n"
            f"cwd={cwd or os.getcwd()}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def image_to_white_background(
    image_path: str | Path,
    mask_path: str | Path,
    output_path: str | Path,
) -> Path:
    try:
        from PIL import Image, ImageFilter
    except ImportError as exc:  # pragma: no cover - runtime dependency check.
        raise RuntimeError("Pillow is required for image export.") from exc
    image_source = Path(image_path)
    mask_source = Path(mask_path)
    if not image_source.is_file():
        raise FileNotFoundError(f"Image not found: {image_source}")
    if not mask_source.is_file():
        raise FileNotFoundError(f"Mask not found: {mask_source}")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_source) as image, Image.open(mask_source) as mask:
        rgb = image.convert("RGB")
        alpha = mask.convert("L").resize(rgb.size)
        alpha = alpha.point(lambda value: 255 if value > 0 else 0)
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=0.6))
        isolated = Image.new("RGB", rgb.size, (255, 255, 255))
        isolated.paste(rgb, (0, 0), alpha)
        isolated.save(target, format="JPEG", quality=95, optimize=True)
    return target


def save_jpg(image: Any, output_path: str | Path, quality: int = 95) -> Path:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - runtime dependency check.
        raise RuntimeError("Pillow is required for image export.") from exc
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        pil_image = Image.fromarray(image).convert("RGB")
    pil_image.save(target, format="JPEG", quality=quality, optimize=True)
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


def validate_action_manifest_item(item: Any) -> bool:
    if not isinstance(item, dict):
        raise ValueError("action manifest item must be a dict")
    missing = ACTION_MANIFEST_FIELDS - set(item)
    extra = set(item) - ACTION_MANIFEST_FIELDS
    if missing:
        raise ValueError(f"action manifest item missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"action manifest item has forbidden fields: {sorted(extra)}")
    for key in ("scene_id", "obj_id", "category", "video_path", "action"):
        if not isinstance(item.get(key), str) or not item[key].strip():
            raise ValueError(f"action manifest item field {key} must be a non-empty string")
    prompts = item.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("action manifest item prompts must be a non-empty list")
    if not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts):
        raise ValueError("action manifest item prompts must contain only non-empty strings")
    return True


def write_manifest(path: str | Path, items: Iterable[dict[str, Any]]) -> Path:
    rows = list(items)
    for item in rows:
        validate_manifest_item(item)
    return write_json(path, rows)
