"""Qwen3-VL async JSON client and rigid-asset prompt builders."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import math
import mimetypes
from io import BytesIO
import os
import re
from pathlib import Path
from typing import Any, Callable

from .config import PipelineConfig

from openai import AsyncOpenAI
from jsonschema import Draft202012Validator, ValidationError


def _image_data_url(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"VLM image does not exist: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


class AsyncOpenAIClient:
    """AsyncOpenAI wrapper for strict JSON chat-completions calls."""

    def __init__(self, config: PipelineConfig, logger: Any):
        self.config = config
        self.logger = logger
        raw_endpoints = (
            os.environ.get("VLM_BASE_URLS")
            or os.environ.get("VLM_BASE_URL")
            or config.vlm.endpoint
        )
        self.endpoints = [
            endpoint.strip().rstrip("/")
            for endpoint in str(raw_endpoints).split(",")
            if endpoint.strip()
        ]
        if not self.endpoints:
            raise ValueError("At least one VLM endpoint is required")
        self.clients = [AsyncOpenAI(base_url=endpoint, api_key="EMPTY", timeout=180,
                                    max_retries=0) for endpoint in self.endpoints]

    async def close(self) -> None:
        await asyncio.gather(*(client.close() for client in self.clients))

    async def check_endpoints(self) -> list[dict[str, Any]]:
        timeout = float(os.environ.get("VLM_READY_TIMEOUT", "8"))

        async def check_one(endpoint, client):
            models = await asyncio.wait_for(client.models.list(), timeout=timeout)
            model_ids = [item.id for item in models.data]
            if self.config.vlm.model not in model_ids:
                raise ValueError(f"Configured model is not served by {endpoint}: {self.config.vlm.model}")
            return {"endpoint": endpoint, "status": "online", "models": model_ids}

        return await asyncio.gather(*(check_one(endpoint, client)
            for endpoint, client in zip(self.endpoints, self.clients)))

    async def call_json(
        self,
        messages: list[dict[str, Any]],
        *,
        client_index: int = 0,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        client = self.clients[client_index % len(self.clients)]
        extra_body: dict[str, Any] | None = None
        if os.environ.get("VLM_DISABLE_THINKING", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            # Qwen3 may otherwise spend the whole completion budget on hidden
            # reasoning and truncate the strict JSON response. Keep this
            # opt-in so existing extraction behavior is unchanged.
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        mode = os.environ.get('VLM_RESPONSE_FORMAT', 'json_schema')
        if mode not in {'json_schema', 'json_object'}:
            raise ValueError('VLM_RESPONSE_FORMAT must be json_schema or json_object')
        if response_schema is not None and mode == 'json_object':
            messages = [*messages, dict(role='user', content=
                'Return only a JSON object satisfying this required schema. Include every required field, '
                'use only the declared enums and types, and add no extra fields: ' + json.dumps(response_schema))]
        response_format = {"type": "json_object"} if response_schema is None or mode == 'json_object' else {
            "type": "json_schema", "json_schema": {
                "name": "asset_response", "strict": True, "schema": response_schema,
            },
        }
        completion = await client.chat.completions.create(
            model=self.config.vlm.model,
            messages=messages,
            temperature=self.config.vlm.temperature,
            max_tokens=self.config.vlm.max_tokens,
            response_format=response_format,
            extra_body=extra_body,
        )
        choice = completion.choices[0]
        if choice.finish_reason != "stop" or not choice.message.content:
            raise ValueError(f"Incomplete VLM JSON response: {choice.finish_reason}")
        result = parse_vlm_response(choice.message.content)
        if response_schema is not None:
            try:
                Draft202012Validator(response_schema).validate(result)
            except ValidationError as exc:
                exc.response = result
                raise
        return result

    async def call_many_json(
        self,
        message_batches: list[list[dict[str, Any]]],
        *,
        concurrency: int = 4,
        response_schema: dict[str, Any] | None = None,
        progress_callback: Callable[[int, dict[str, Any]], None] | None = None,
        collect_errors: bool = False,
    ) -> list[dict[str, Any] | Exception]:
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        semaphore = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()
        completed = 0

        async def run_one(index: int, messages: list[dict[str, Any]]) -> dict[str, Any]:
            nonlocal completed
            async with semaphore:
                result = await self.call_json(
                    messages,
                    client_index=index,
                    response_schema=response_schema,
                )
            async with lock:
                completed += 1
                if progress_callback:
                    progress_callback(completed, result)
            return result

        tasks = [asyncio.create_task(run_one(index, messages))
                 for index, messages in enumerate(message_batches)]
        try:
            return await asyncio.gather(*tasks, return_exceptions=collect_errors)
        finally:
            # Drain all tasks before the caller closes the shared SDK clients.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


RIGID_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": [
                "car",
                "truck",
                "bus",
                "van",
                "traffic_sign",
                "traffic_light",
                "cone",
                "barrier",
                "bollard",
                "pole",
                "fallen_tree",
                "road_debris",
                "trash_can",
                "hydrant",
                "bicycle",
                "motorcycle",
                "unknown",
            ],
        },
        "subtype": {"type": "string"},
        "color": {"type": "string"},
        "shape_description": {"type": "string"},
        "visible_markings": {"type": "string"},
        "vehicle_view": {
            "type": "string",
            "enum": ["front_side", "rear_side", "unknown"],
        },
    },
    "required": ["category", "subtype", "color", "shape_description", "visible_markings", "vehicle_view"],
    "additionalProperties": False,
}


def _token(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return re.sub(r"_+", "_", normalized).strip("_")


def _vehicle_end(value: Any) -> str | None:
    """Normalize a vehicle angle label to the visible front or rear end."""

    view = _token(value)
    if "rear" in view or "back" in view:
        return "rear"
    if "front" in view:
        return "front"
    return None


class RigidPromptBuilder:
    VEHICLE = {"vehicle", "car", "truck", "bus", "van", "trailer", "suv", "sedan"}
    CATEGORY_ALIASES = {
        "automobile": "car",
        "pickup": "truck",
        "pickup_truck": "truck",
        "traffic_cone": "cone",
        "road_barrier": "barrier",
        "roadblock": "barrier",
        "garbage_bin": "trash_can",
        "rubbish_bin": "trash_can",
        "fire_hydrant": "hydrant",
        "road_sign": "traffic_sign",
        "street_sign": "traffic_sign",
        "parking_sign": "traffic_sign",
        "roadside_sign": "traffic_sign",
        "sign": "traffic_sign",
        "downed_tree": "fallen_tree",
        "tree_branch_on_road": "fallen_branch",
        "branch_on_road": "fallen_branch",
        "large_road_obstacle": "large_obstacle",
        "road_obstacle": "large_obstacle",
        "unusual_rigid_obstacle": "extreme_rigid_obstacle",
        "broken_traffic_sign": "broken_sign",
        "damaged_traffic_sign": "broken_sign",
        "fallen_traffic_sign": "fallen_sign",
    }
    SUPPORTED = (set(RIGID_RESPONSE_SCHEMA["properties"]["category"]["enum"]) - {"unknown"}) | {
        "signpost", "traffic_light_pole", "traffic_camera", "fallen_branch", "large_obstacle",
        "broken_sign", "fallen_sign", "damaged_pole", "construction_debris", "extreme_rigid_obstacle",
        "lost_cargo", "overturned_vehicle", "damaged_vehicle", "construction_equipment",
    }

    @classmethod
    def normalize_category(cls, value: Any) -> str:
        category = _token(value)
        return cls.CATEGORY_ALIASES.get(category, category or "unknown")

    def build(
        self,
        candidate: dict[str, Any],
        vlm_result: dict[str, Any] | None = None,
    ) -> list[str]:
        result = vlm_result or {}
        category = self.normalize_category(result.get("category") or candidate.get("category"))
        if category not in self.SUPPORTED | self.VEHICLE:
            raise ValueError(f"Unsupported rigid prompt category: {category}")
        details = []
        for key in ("color", "shape_description", "visible_markings"):
            value = str(result.get(key) or "").strip().rstrip(".")
            if value.lower() not in {"", "unknown", "not visible", "none", "null"}:
                details.append(value)
        appearance = "; ".join(details) if details else "its original colors and structure"
        ending = f" on a plain white background, showing one complete object. Preserve the observed appearance: {appearance}."
        if category in self.VEHICLE:
            end = _vehicle_end(result.get("vehicle_view"))
            if end is None:
                raise ValueError("Vehicle prompts require an observed front or rear end")
            return [f"Create a {view}-view image of the exact vehicle in the reference image" + ending
                    for view in (end, "side")]
        return ["Isolate the exact object in the reference image" + ending]


def parse_vlm_response(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    cleaned = str(response or "").strip()
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("VLM response must be a JSON object")
    return payload


def build_diffusion_prompts(
    candidate: dict[str, Any],
    vlm_result: dict[str, Any] | None,
) -> list[str]:
    return RigidPromptBuilder().build(candidate, vlm_result)


def rigid_vlm_messages(image_path: str | Path, category: str,
                       source_context: dict | None = None) -> list[dict[str, Any]]:
    text = (
        "Return only one JSON object for this isolated rigid asset. "
        "Describe subtype, visible color and shape/proportions; use unknown when unsupported. "
        "shape_description must identify concrete visible instance details, not just repeat a generic category such as sedan or cone. "
        "Keep shape_description below 45 words, using positive descriptions of the observed appearance. "
        "For vehicles, describe the visible body silhouette, windows, doors, trim, wheels and the front or rear details "
        "actually visible in this view; an unseen end has unknown details. Do not guess brand/model. "
        "For traffic lights, state the number of visible signal heads, their illuminated color and their arrangement, "
        "report housing, arm and pole surface colors separately from the emitted light; the color field describes "
        "the physical hardware, not the illuminated signal. "
        "how they connect to any horizontal arm, and whether the vertical support is at the image-left, image-right "
        "or center. Use unknown for any layout or count that cannot be seen reliably. "
        "For cones, describe the visible surface color and pattern, band count (0 for a reliably solid surface), "
        "base shape and wear. Describe a zero-band surface positively as a solid observed color; "
        "use unknown if band count or surface pattern is unclear. "
        "For hydrants, separately describe cap/top color, body/base color, "
        "nozzle count/placement, color boundaries and worn paint. Count only elements reliably visible; otherwise state unknown. "
        "Use concise physical observations rather than idealized textbook appearance; do not invent missing details. "
        "visible_markings describes only clearly visible symbols, logos, lettering or patterns; "
        "use not visible when no marking is reliably legible. Never invent text or identity cues. "
        "Use this schema exactly: " + json.dumps(RIGID_RESPONSE_SCHEMA) + " "
        "For a vehicle, use front_side when its front end is visible and rear_side when its rear end is visible, "
        "including oblique side-front and side-rear views. Use direct headlights, grille, windshield, taillights, "
        "trunk, or tailgate cues. Use unknown for a strict profile where neither end is visible, or when the "
        "vehicle heading cannot be judged. Never guess an end. "
        "vehicle_view must be exactly front_side, rear_side, or unknown. For every non-vehicle "
        "category set vehicle_view to unknown. Never use side, front, rear, not applicable, or an empty string. "
        f"The detector category is {category!r}. Preserve the true visible object identity."
    )
    if source_context is None:
        return image_message(image_path, text)
    from PIL import Image, ImageDraw, ImageOps

    source = Path(source_context['source_image'])
    if not source.is_absolute() or not source.is_file():
        raise ValueError('source_context.source_image must be an existing absolute image path')
    bbox = source_context['bbox_xyxy']
    if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in bbox)):
        raise ValueError('source_context bbox_xyxy must contain four finite numbers')
    with Image.open(source) as image:
        raw = image.convert('RGB')
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= raw.width and 0 <= y1 < y2 <= raw.height):
        raise ValueError('source_context bbox_xyxy must be ordered and inside the source image')
    side = 4 * max(x2-x1, y2-y1)
    center_x, center_y = (x1+x2)/2, (y1+y2)/2
    crop = (max(0, math.floor(center_x-side/2)), max(0, math.floor(center_y-side/2)),
            min(raw.width, math.ceil(center_x+side/2)), min(raw.height, math.ceil(center_y+side/2)))
    context = raw.crop(crop)
    ImageDraw.Draw(context).rectangle((x1-crop[0], y1-crop[1], x2-crop[0], y2-crop[1]),
                                      outline='red', width=max(2, round(side/200)))
    with Image.open(image_path) as image:
        foreground = image.convert('RGBA')
    white = Image.new('RGBA', foreground.size, 'white')
    foreground = Image.alpha_composite(white, foreground).convert('RGB')
    sheet = Image.new('RGB', (1280, 664), 'white')
    for index, panel in enumerate([foreground, context]):
        resized = ImageOps.contain(panel, (640, 640), Image.Resampling.LANCZOS)
        sheet.paste(resized, (index*640+(640-resized.width)//2, 24+(640-resized.height)//2))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), 'LEFT: SAM foreground target', fill='black')
    draw.text((648, 6), 'RIGHT: raw context, target in red box', fill='black')
    buffer = BytesIO(); sheet.save(buffer, format='JPEG', quality=92)
    context_rule = (
        'This single image is a labeled pair: LEFT is the segmented target; RIGHT is its original scene context, '
        'with the target bounding box in red. Use the right panel ONLY to verify what the left target belongs to '
        'and whether it is an independent object. Do not describe neighboring objects as the target. '
        'A detached-looking traffic-light counterweight/base must not be called a trash can. '
        'If the left target is only a base, wheel, pole fragment or other component and does not support a complete '
        'instance of the claimed category, return category=unknown; do not infer the whole missing object into the crop. '
        'An occluded but still identifiable vehicle may be retained. Describe attributes from the left target itself. '
    )
    url = 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
    return [dict(role='user', content=[dict(type='text', text=context_rule+text),
                                      dict(type='image_url', image_url=dict(url=url))])]


def image_message(path: str | Path, text: str) -> list[dict[str, Any]]:
    source = Path(path)
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": _image_data_url(source)}},
            ],
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the configured Qwen OpenAI service.")
    parser.add_argument("--check-ready", action="store_true", required=True)
    parser.add_argument("--config", default="configs/nuscenes.yaml")
    args = parser.parse_args()
    from .config import load_config

    async def check():
        client = AsyncOpenAIClient(load_config(args.config), logging.getLogger(__name__))
        try:
            return await client.check_endpoints()
        finally:
            await client.close()

    print(json.dumps({"status": "ready", "endpoints": asyncio.run(check())},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
