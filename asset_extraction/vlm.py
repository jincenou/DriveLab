"""Qwen3-VL client, prompt builders, and final manifest item helpers."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import mimetypes
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

from .config import PipelineConfig
from .io_utils import append_jsonl, read_json, read_jsonl, validate_manifest_item, write_manifest
from .vision import Candidate

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - runtime dependency check.
    AsyncOpenAI = None  # type: ignore[assignment]


def _image_data_url(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"VLM image does not exist: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _endpoint_host_port(endpoint: str) -> tuple[str, int]:
    value = endpoint.strip().split("://", 1)[-1].split("/", 1)[0]
    if value.startswith("["):
        host, _, rest = value[1:].partition("]")
        port_text = rest[1:] if rest.startswith(":") else ""
    else:
        host, _, port_text = value.rpartition(":")
        if not host:
            host = value
            port_text = ""
    if port_text:
        return host, int(port_text)
    return host, 443 if endpoint.strip().startswith("https://") else 80


class AsyncOpenAIClient:
    """AsyncOpenAI wrapper for strict JSON chat-completions calls."""

    def __init__(self, config: PipelineConfig, logger: Any):
        self.config = config
        self.logger = logger
        self.bad_response_log = None
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
        if AsyncOpenAI is None:
            self.clients = []
        else:
            self.clients = [
                AsyncOpenAI(
                    base_url=endpoint,
                    api_key="EMPTY",
                    timeout=180,
                    max_retries=0,
                )
                for endpoint in self.endpoints
            ]

    async def close(self) -> None:
        await asyncio.gather(*(client.close() for client in self.clients))

    def check_ready(self) -> None:
        if AsyncOpenAI is None:
            raise RuntimeError("The openai package is required for Qwen3-VL/vLLM calls.")
        if not self.clients:
            raise RuntimeError("At least one VLM OpenAI-compatible endpoint is required.")

    async def check_endpoints(self) -> list[dict[str, Any]]:
        self.check_ready()
        timeout = float(os.environ.get("VLM_READY_TIMEOUT", "8"))

        async def check_one(index: int, endpoint: str) -> dict[str, Any]:
            host, port = _endpoint_host_port(endpoint)
            reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
            writer.close()
            await writer.wait_closed()
            client = AsyncOpenAI(base_url=endpoint, api_key="EMPTY", timeout=timeout, max_retries=0)
            try:
                models = await asyncio.wait_for(client.models.list(), timeout=timeout + 1)
                model_ids = [str(getattr(item, "id", "")) for item in getattr(models, "data", [])]
                return {
                    "endpoint": endpoint,
                    "status": "online",
                    "models": [model for model in model_ids if model],
                }
            finally:
                await client.close()

        return list(
            await asyncio.gather(
                *(check_one(index, endpoint) for index, endpoint in enumerate(self.endpoints))
            )
        )

    async def call_json(
        self,
        messages: list[dict[str, Any]],
        *,
        client_index: int = 0,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.check_ready()
        client = self.clients[client_index % len(self.clients)]
        response_format = (
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "asset_extraction_result",
                    "strict": True,
                    "schema": response_schema,
                },
            }
            if response_schema is not None
            else {"type": "json_object"}
        )
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
            response_format=response_format,
            extra_body=extra_body,
        )
        raw = completion.choices[0].message.content or "{}"
        return parse_vlm_response(raw)

    async def call_many_json(
        self,
        message_batches: list[list[dict[str, Any]]],
        *,
        concurrency: int = 4,
        response_schema: dict[str, Any] | None = None,
        progress_callback: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(max(1, int(concurrency)))
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

        return list(
            await asyncio.gather(
                *(run_one(index, messages) for index, messages in enumerate(message_batches))
            )
        )


VLMClient = AsyncOpenAIClient


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


VEHICLE_VIEW_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subtype": {
            "type": "string",
            "enum": ["car", "sedan", "suv", "truck", "bus", "van", "trailer", "unknown"],
        },
        "color": {"type": "string"},
        "vehicle_view": {
            "type": "string",
            "enum": ["front_side", "rear_side", "unknown"],
        },
    },
    "required": ["subtype", "color", "vehicle_view"],
    "additionalProperties": False,
}


class PromptBuilder:
    def build(
        self,
        candidate: Candidate | dict[str, Any],
        vlm_result: dict[str, Any] | None = None,
    ) -> list[str]:
        raise NotImplementedError


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


class RigidPromptBuilder(PromptBuilder):
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
        primary = "rear-view" if _vehicle_end((result or {}).get("vehicle_view")) == "rear" else "front-view"
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
        return [templates.get(category, templates["extreme_rigid_obstacle"])]

    def build(
        self,
        candidate: Candidate | dict[str, Any],
        vlm_result: dict[str, Any] | None = None,
    ) -> list[str]:
        raw_category = candidate.category if isinstance(candidate, Candidate) else candidate.get("category")
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
        return [templates.get(category, "Complete this rigid object and make it clear, isolated, and uncropped.")]


def parse_vlm_response(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    cleaned = str(response or "").strip()
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("VLM response must be a JSON object")
    return payload


def build_diffusion_prompts(
    candidate: Candidate | dict[str, Any],
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


def vehicle_view_vlm_messages(image_path: str | Path, category: str) -> list[dict[str, Any]]:
    """Request the visible vehicle end for an existing rigid manifest row."""

    text = (
        "Return only one JSON object for this isolated vehicle image using exactly this schema: "
        '{"subtype":"car|sedan|suv|truck|bus|van|trailer|unknown",'
        '"color":"short dominant paint color or unknown",'
        '"vehicle_view":"front_side|rear_side|unknown"}. '
        "Choose front_side when the front end is visible, including direct front and oblique side-front views. "
        "Choose rear_side when the rear end is visible, including direct rear and oblique side-rear views. "
        "Use unknown for a strict profile where neither the front face nor rear face is visible. Use headlights, "
        "grille, windshield, taillights, trunk, or tailgate cues for oblique views. Do not invent a different vehicle "
        "identity or paint color, and never guess an end. "
        f"The detector category is {category!r}."
    )
    return image_message(image_path, text)



def build_manifest_item(
    candidate: Candidate,
    image_path: str | Path,
    prompts: list[str],
) -> dict[str, Any]:
    item = {
        "scene_id": candidate.scene_id,
        "obj_id": candidate.obj_id,
        "category": candidate.category,
        "image_path": str(Path(image_path).resolve()),
        "prompts": prompts,
    }
    validate_manifest_item(item)
    return item


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


def _checkpoint_decisions(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    decisions: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        if not isinstance(row, dict) or not isinstance(row.get("decision"), dict):
            continue
        scene_id = str(row.get("scene_id") or "").strip()
        obj_id = str(row.get("obj_id") or "").strip()
        if scene_id and obj_id:
            decisions[(scene_id, obj_id)] = dict(row["decision"])
    return decisions


def _optional_vlm_text(value: Any) -> str | None:
    text = " ".join(str(value or "").strip().split())
    if text.lower() in {"", "none", "null", "unknown", "unclear", "n/a"}:
        return None
    return text


def _vehicle_manifest_rows(run_dir: Path) -> tuple[list[Path], dict[tuple[str, str], dict[str, Any]]]:
    manifest_paths = sorted((run_dir / "scenes").glob("*/diffusion_inputs/manifest.json"))
    if not manifest_paths:
        raise FileNotFoundError(f"No scene manifests found under {run_dir}")

    vehicles: dict[tuple[str, str], dict[str, Any]] = {}
    builder = RigidPromptBuilder()
    for manifest_path in manifest_paths:
        rows = read_json(manifest_path)
        if not isinstance(rows, list):
            raise ValueError(f"manifest must be a list: {manifest_path}")
        for row in rows:
            category = builder.normalize_category(row.get("category"))
            if category not in builder.VEHICLE:
                continue
            key = (str(row.get("scene_id") or ""), str(row.get("obj_id") or ""))
            if not all(key):
                raise ValueError(f"vehicle row has an invalid key in {manifest_path}: {row}")
            if key in vehicles:
                raise ValueError(f"duplicate vehicle key across scene manifests: {key}")
            vehicles[key] = dict(row)
    return manifest_paths, vehicles


def _write_refreshed_vehicle_prompts(
    manifest_path: Path,
    decisions: dict[tuple[str, str], dict[str, Any]],
) -> int:
    rows = read_json(manifest_path)
    if not isinstance(rows, list):
        raise ValueError(f"manifest must be a list: {manifest_path}")

    builder = RigidPromptBuilder()
    changed = 0
    for row in rows:
        category = builder.normalize_category(row.get("category"))
        if category not in builder.VEHICLE:
            validate_manifest_item(row)
            continue
        row_changed = False
        key = (str(row.get("scene_id") or ""), str(row.get("obj_id") or ""))
        decision = decisions.get(key)
        if decision is None:
            raise ValueError(f"missing VLM decision for vehicle {key} in {manifest_path}")
        decision_category = builder.normalize_category(decision.get("category"))
        if decision_category not in builder.VEHICLE and decision_category != "unknown":
            if row["category"] != decision_category:
                row["category"] = decision_category
                row_changed = True
            prompts = builder.build(row, decision)
        elif _vehicle_end(decision.get("vehicle_view")) is None:
            raise ValueError(f"missing front/rear decision for vehicle {key} in {manifest_path}")
        else:
            prompts = builder.build(row, decision)
        if row.get("prompts") != prompts:
            row["prompts"] = prompts
            row_changed = True
        if row_changed:
            changed += 1
        validate_manifest_item(row)

    if changed:
        temporary_path = manifest_path.with_name(f".{manifest_path.name}.tmp")
        write_manifest(temporary_path, rows)
        temporary_path.replace(manifest_path)
    return changed


async def refresh_rigid_manifest_prompts(
    config: PipelineConfig,
    run_dir: str | Path,
    *,
    concurrency: int,
    chunk_size: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Refresh vehicle prompts in an existing run without rerunning extraction."""

    run_path = Path(run_dir).expanduser().resolve()
    merged_manifest = run_path / "merged" / "manifest.json"
    if not merged_manifest.is_file():
        raise FileNotFoundError(merged_manifest)

    scene_manifests, vehicles = _vehicle_manifest_rows(run_path)
    checkpoint_path = run_path / "logs" / "vlm_checkpoint.jsonl"
    checkpoint = _checkpoint_decisions(checkpoint_path)
    decisions: dict[tuple[str, str], dict[str, Any]] = {}
    pending: list[tuple[tuple[str, str], dict[str, Any]]] = []
    for key, row in vehicles.items():
        decision = checkpoint.get(key)
        if decision is not None and _vehicle_end(decision.get("vehicle_view")) is not None:
            decisions[key] = decision
        else:
            pending.append((key, row))

    reused_count = len(decisions)
    inferred_count = 0
    logger.info(
        "vehicle prompt refresh: total=%d reused=%d pending=%d",
        len(vehicles),
        reused_count,
        len(pending),
    )

    if pending:
        config.vlm.max_tokens = min(int(config.vlm.max_tokens), 128)
        client = AsyncOpenAIClient(config, logger)
        try:
            unresolved: list[tuple[tuple[str, str], dict[str, Any]]] = []
            for start in range(0, len(pending), max(1, int(chunk_size))):
                chunk = pending[start : start + max(1, int(chunk_size))]
                results = await client.call_many_json(
                    [
                        vehicle_view_vlm_messages(row["image_path"], row["category"])
                        for _, row in chunk
                    ],
                    concurrency=max(1, int(concurrency)),
                    response_schema=VEHICLE_VIEW_RESPONSE_SCHEMA,
                )
                for (key, row), result in zip(chunk, results):
                    end = _vehicle_end(result.get("vehicle_view"))
                    if end is None:
                        unresolved.append((key, row))
                        continue
                    decision = {
                        "keep": True,
                        "category": row["category"],
                        "subtype": _optional_vlm_text(result.get("subtype")),
                        "color": _optional_vlm_text(result.get("color")),
                        "vehicle_view": f"{end}_side",
                        "vehicle_view_source": "vlm_strict_json",
                        "is_edge_case": False,
                        "edge_case_type": None,
                        "label_note": None,
                    }
                    decisions[key] = decision
                    checkpoint[key] = decision
                    append_jsonl(
                        checkpoint_path,
                        {
                            "scene_id": key[0],
                            "obj_id": key[1],
                            "input_category": row["category"],
                            "image_path": str(Path(row["image_path"]).resolve()),
                            "decision": decision,
                        },
                    )
                    inferred_count += 1
            pending = unresolved
        finally:
            await client.close()

    if pending:
        preview = ", ".join(f"{key[0]}/{key[1]}" for key, _ in pending[:10])
        raise RuntimeError(f"Unable to resolve {len(pending)} vehicle headings without guessing: {preview}")

    front_count = sum(_vehicle_end(item.get("vehicle_view")) == "front" for item in decisions.values())
    rear_count = sum(_vehicle_end(item.get("vehicle_view")) == "rear" for item in decisions.values())
    reclassified_count = 0
    changed_scene_rows = sum(
        _write_refreshed_vehicle_prompts(path, decisions) for path in scene_manifests
    )
    changed_merged_rows = _write_refreshed_vehicle_prompts(merged_manifest, decisions)
    merged_rows = read_json(merged_manifest)
    merged_vehicle_count = sum(
        RigidPromptBuilder.normalize_category(row.get("category")) in RigidPromptBuilder.VEHICLE
        for row in merged_rows
    )
    return {
        "run_dir": str(run_path),
        "scene_manifest_count": len(scene_manifests),
        "scene_vehicle_count": len(vehicles),
        "merged_vehicle_count": merged_vehicle_count,
        "reused_checkpoint_count": reused_count,
        "new_inference_count": inferred_count,
        "front_or_side_front_count": front_count,
        "rear_or_side_rear_count": rear_count,
        "reclassified_non_vehicle_count": reclassified_count,
        "changed_scene_rows": changed_scene_rows,
        "changed_merged_rows": changed_merged_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen-VL OpenAI-compatible client utilities.")
    parser.add_argument("--check-ready", action="store_true", help="Check configured VLM endpoints with AsyncOpenAI.")
    parser.add_argument(
        "--refresh-rigid-prompts",
        action="store_true",
        help="Refresh vehicle front/rear prompts in an existing rigid run.",
    )
    parser.add_argument("--config", default="configs/rigid.yaml", help="Pipeline config path.")
    parser.add_argument("--run-dir", help="Existing rigid run directory to refresh.")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args()
    if args.check_ready == args.refresh_rigid_prompts:
        parser.error("choose exactly one of --check-ready or --refresh-rigid-prompts")

    from .config import load_config

    logger = logging.getLogger("asset_extraction.vlm")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = load_config(args.config)
    if args.refresh_rigid_prompts:
        if not args.run_dir:
            parser.error("--run-dir is required with --refresh-rigid-prompts")
        summary = asyncio.run(
            refresh_rigid_manifest_prompts(
                config,
                args.run_dir,
                concurrency=args.concurrency,
                chunk_size=args.chunk_size,
                logger=logger,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    client = AsyncOpenAIClient(config, logger)

    async def run_check() -> None:
        try:
            endpoints = await client.check_endpoints()
        finally:
            await client.close()
        print(json.dumps({"status": "ready", "endpoints": endpoints}, ensure_ascii=False, indent=2))

    try:
        asyncio.run(run_check())
    except Exception as exc:
        print(
            json.dumps(
                {"status": "not_ready", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
