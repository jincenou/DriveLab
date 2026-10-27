"""Legacy-compatible FLUX image preparation and inference, scheduled by generation.py."""
from __future__ import annotations
import contextlib
import os
from pathlib import Path
from typing import Any
from dataclasses import dataclass

@dataclass(frozen=True)
class FluxConfig:
    worker_count: int | None = None
    height: int = 1024
    width: int = 1024
    steps: int = 8
    guidance_scale: float = 1.0
    write_every: int = 16
    min_reference_size: int = 64
    max_aspect_ratio: float = 8.0

PIPE = None
CONFIG = FluxConfig()

def pad_extreme_aspect(image):
    """Center-pad the short side with white so the ratio fits the FLUX.2 8:1 limit."""
    from PIL import Image

    width, height = image.size
    if max(width, height) <= CONFIG.max_aspect_ratio * min(width, height):
        return image

    long_side = max(width, height)
    short_target = int(-(-long_side // CONFIG.max_aspect_ratio))  # ceil division
    new_w = max(width, short_target) if width < height else width
    new_h = max(height, short_target) if height <= width else height
    canvas = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    canvas.paste(image, ((new_w - width) // 2, (new_h - height) // 2))
    return canvas

def resize_reference_image(image):
    image = pad_extreme_aspect(image)
    width, height = image.size
    if width >= CONFIG.min_reference_size and height >= CONFIG.min_reference_size:
        return image

    scale = CONFIG.min_reference_size / min(width, height)
    new_size = (round(width * scale), round(height * scale))
    return image.resize(new_size)

def _generate(task: dict[str, Any]) -> list[str]:
    import torch
    from diffusers.utils import load_image

    if PIPE is None:
        raise RuntimeError("Worker model is not loaded")

    reference_image = resize_reference_image(load_image(task["image_path"]).convert("RGB"))
    output_paths = []
    for prompt_index, prompt in enumerate(task["prompts"]):
        output_path = Path(task["output_paths"][prompt_index])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            image = PIPE(
                prompt=prompt,
                image=reference_image,
                height=CONFIG.height,
                width=CONFIG.width,
                guidance_scale=CONFIG.guidance_scale,
                num_inference_steps=CONFIG.steps,
                generator=torch.Generator(device="cuda").manual_seed(task["item_index"] * 1000 + prompt_index),
            ).images[0]
        image.save(output_path)
        output_paths.append(str(output_path))
    return output_paths
