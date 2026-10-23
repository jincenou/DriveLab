"""Qwen3-VL async JSON client and rigid-asset prompt builders."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Callable

from .config import PipelineConfig

from openai import AsyncOpenAI
from jsonschema import Draft202012Validator


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
        completion = await client.chat.completions.create(
            model=self.config.vlm.model,
            messages=messages,
            temperature=self.config.vlm.temperature,
            max_tokens=self.config.vlm.max_tokens,
            response_format={"type": "json_object"},
            extra_body=extra_body,
        )
        choice = completion.choices[0]
        if choice.finish_reason != "stop" or not choice.message.content:
            raise ValueError(f"Incomplete VLM JSON response: {choice.finish_reason}")
        result = parse_vlm_response(choice.message.content)
        if response_schema is not None:
            Draft202012Validator(response_schema).validate(result)
        return result

    async def call_many_json(
        self,
        message_batches: list[list[dict[str, Any]]],
        *,
        concurrency: int = 4,
        response_schema: dict[str, Any] | None = None,
        progress_callback: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
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
            return await asyncio.gather(*tasks)
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
                "unknown",
            ],
        },
        "subtype": {"type": "string"},
        "color": {"type": "string"},
        "vehicle_view": {
            "type": "string",
            "enum": ["front_side", "rear_side", "unknown"],
        },
    },
    "required": ["category", "subtype", "color", "vehicle_view"],
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
    SIGN_POLE = {
        "traffic_sign",
        "road_sign",
        "sign",
        "signpost",
        "pole",
        "traffic_light_pole",
        "street_sign",
        "parking_sign",
        "roadside_sign",
        "traffic_light",
    }
    CATEGORY_ALIASES = {
        "automobile": "car",
        "pickup": "truck",
        "pickup_truck": "truck",
        "traffic_cone": "cone",
        "road_barrier": "barrier",
        "roadblock": "barrier",
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
    EXTREME_RIGID = {
        "fallen_tree",
        "fallen_branch",
        "road_debris",
        "large_obstacle",
        "broken_sign",
        "fallen_sign",
        "damaged_pole",
        "construction_debris",
        "extreme_rigid_obstacle",
        "lost_cargo",
        "overturned_vehicle",
        "damaged_vehicle",
        "construction_equipment",
    }

    @classmethod
    def normalize_category(cls, value: Any) -> str:
        category = _token(value)
        return cls.CATEGORY_ALIASES.get(category, category or "unknown")

    def _vehicle_prompts(self, category: str, result: dict[str, Any] | None = None) -> list[str]:
        subtype = self.normalize_category((result or {}).get("subtype"))
        vehicle_type = subtype if subtype in self.VEHICLE and subtype != "vehicle" else category
        if vehicle_type == "vehicle":
            vehicle_type = "car"
        color = str((result or {}).get("color") or "").strip().lower()
        if color in {"", "none", "unknown", "null"}:
            color = ""
        name = f"{color} {vehicle_type}".strip()
        end = _vehicle_end((result or {}).get("vehicle_view"))
        if end is None:
            raise ValueError("Vehicle prompts require an observed front or rear end")
        primary = f"{end}-view"
        identity = (
            "Preserve its original body style, proportions, paint color, and visible markings; "
            "do not redesign or recolor it."
        )
        return [
            f"Generate a clear {primary} image of this {name}. {identity}",
            f"Generate a clear side-view image of this {name}. {identity}",
        ]

    def _sign_pole_prompt(self, category: str) -> list[str]:
        if category in {"traffic_sign", "road_sign", "street_sign", "sign"}:
            return [
                "Generate a clean, complete traffic sign asset with a clearly visible sign board and a long, slender, full-length vertical support pole extending downward to the ground. Preserve the complete pole length; do not make the pole short, cropped, or stump-like."
            ]
        if category == "signpost":
            return [
                "Generate a complete signpost asset with the sign board attached to a tall, thin, full-height support pole. The pole should be long, straight, and visibly extend below the sign toward the ground."
            ]
        if category in {"pole", "traffic_light_pole"}:
            return [
                "Generate a clean rigid roadside pole asset with a long, slender, full-length vertical structure. Preserve the complete pole from top to bottom and avoid short or truncated pole geometry."
            ]
        return [
            "Generate a complete pole-mounted roadside asset with a long, tall, full-length support pole, connected brackets, and an uncropped structure extending to the ground."
        ]

    def _extreme_prompt(self, category: str) -> list[str]:
        templates = {
            "fallen_tree": "Extract this as an extreme rigid road obstacle asset: a fallen tree lying on or near the road. Preserve the complete trunk and visible branches.",
            "fallen_branch": "Extract this as an extreme rigid road obstacle asset: a fallen branch or tree branch on the road. Preserve the complete branch shape and visible limbs.",
            "road_debris": "Extract this as an extreme rigid road obstacle asset: a large piece of road debris with complete shape, clean boundary, and no background clutter.",
            "large_obstacle": "Extract this as an extreme rigid road obstacle asset: a large unusual obstacle with complete rigid shape and a clean isolated boundary.",
            "broken_sign": "Extract this as an unusual rigid roadside asset: a broken traffic sign with the sign board and pole structure preserved.",
            "fallen_sign": "Extract this as an unusual rigid roadside asset: a damaged or fallen traffic sign with the sign board and pole structure preserved.",
            "damaged_pole": "Extract this as an unusual rigid roadside asset: a damaged pole with its full-length structure and visible base preserved.",
            "construction_debris": "Extract this as an extreme rigid road obstacle asset: construction debris with complete shape, clean boundary, and no background clutter.",
            "extreme_rigid_obstacle": "Extract this as an unusual rigid road obstacle asset with complete geometry, clean boundary, and no background clutter.",
            "lost_cargo": "Extract this as an extreme rigid road obstacle asset: lost cargo or a dropped box with complete shape and clean boundary.",
            "overturned_vehicle": "Reconstruct the overturned vehicle as a complete isolated rigid asset with clean geometry.",
            "damaged_vehicle": "Reconstruct the damaged vehicle or vehicle wreck as a complete isolated rigid asset.",
            "construction_equipment": "Reconstruct the construction equipment as a complete isolated rigid asset.",
        }
        return [templates[category]]

    def build(
        self,
        candidate: dict[str, Any],
        vlm_result: dict[str, Any] | None = None,
    ) -> list[str]:
        raw_category = candidate.get("category")
        result_category = (vlm_result or {}).get("category") or raw_category
        category = self.normalize_category(result_category)
        if category in self.VEHICLE:
            return self._vehicle_prompts(category, vlm_result)
        if category in self.SIGN_POLE:
            return self._sign_pole_prompt(category)
        if category in self.EXTREME_RIGID:
            return self._extreme_prompt(category)
        templates = {
            "traffic_light": "Generate a clean, complete traffic light asset with the signal head, support arm or pole, visible brackets, and full connected structure.",
            "cone": "Complete the cone shape and make the traffic cone clear, isolated, and uncropped.",
            "barrier": "Complete the roadside barrier and make it clear, isolated, and uncropped.",
            "bollard": "Complete the roadside bollard and make it clear, isolated, and uncropped.",
            "traffic_camera": "Complete the traffic camera support and make it clear, isolated, and uncropped.",
            "hydrant": "Complete the fire hydrant and make it clear, isolated, and uncropped.",
        }
        if category not in templates:
            raise ValueError(f"Unsupported rigid prompt category: {category}")
        return [templates[category]]


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


def rigid_vlm_messages(image_path: str | Path, category: str) -> list[dict[str, Any]]:
    text = (
        "Return only one JSON object for this isolated rigid nuScenes asset. "
        "Use this schema exactly: "
        '{"category":"car|truck|bus|van|traffic_sign|traffic_light|cone|barrier|bollard|pole|fallen_tree|road_debris|unknown",'
        '"subtype":"short object subtype","color":"dominant visible color or unknown",'
        '"vehicle_view":"front_side|rear_side|unknown"}. '
        "For a vehicle, use front_side when its front end is visible and rear_side when its rear end is visible, "
        "including oblique side-front and side-rear views. Use direct headlights, grille, windshield, taillights, "
        "trunk, or tailgate cues. Use unknown for a strict profile where neither end is visible, or when the "
        "vehicle heading cannot be judged. Never guess an end. "
        f"The detector category is {category!r}. Preserve the true visible object identity."
    )
    return image_message(image_path, text)


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
