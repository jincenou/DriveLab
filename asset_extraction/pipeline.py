"""nuScenes rigid-asset extraction from raw six-camera scenes to a FLUX manifest."""

from __future__ import annotations

import argparse
from bisect import bisect_right
import asyncio
import hashlib
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from jsonschema import ValidationError

from .config import PipelineConfig, apply_cli_overrides, load_config, run_root, validate_config
from .io_utils import ensure_dir, read_json, setup_logger, validate_manifest_item, write_json, write_jsonl, write_manifest
from .vision import (
    SceneExportConfig,
    normalize_rigid_class,
    run_scene_asset_export,
)
from .vlm import (
    AsyncOpenAIClient,
    RIGID_RESPONSE_SCHEMA,
    _vehicle_end,
    RigidPromptBuilder,
    build_diffusion_prompts,
    rigid_vlm_messages,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _scene_id(scene_dir: Path) -> str:
    if scene_dir.is_file() and scene_dir.suffix == ".json":
        return scene_dir.stem
    split = scene_dir.parent.name
    return f"scene_{split}_{scene_dir.name}" if split in {"train", "val", "test"} else scene_dir.name


def _pose_matrix(row: dict[str, Any]):
    import numpy as np
    q = np.asarray(row["rotation"], dtype=float)
    t = np.asarray(row["translation"], dtype=float)
    if q.shape != (4,) or t.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(t).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("Invalid camera/ego pose")
    w, x, y, z = q / np.linalg.norm(q)
    matrix = np.eye(4)
    matrix[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                       [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                       [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    matrix[:3, 3] = t
    return matrix


def _prepare_nuscenes_scene_dirs(config: PipelineConfig, run_dir: Path) -> list[Path]:
    try:
        import ijson
    except ImportError as exc:
        raise RuntimeError("ijson is required to index nuScenes metadata efficiently") from exc

    if config.dataset_root is None:
        raise FileNotFoundError("dataset_root is required")
    metadata_root = config.dataset_root / config.dataset_version
    scene_rows = read_json(metadata_root / "scene.json")
    sample_rows = read_json(metadata_root / "sample.json")
    if not isinstance(scene_rows, list) or not isinstance(sample_rows, list):
        raise ValueError(f"Invalid nuScenes metadata under {metadata_root}")

    scenes_by_token = {str(row["token"]): row for row in scene_rows}
    scenes_by_name = {str(row["name"]): row for row in scene_rows}
    requested_names: set[str] = set()
    for value in config.scene_ids:
        token = str(value).strip()
        row = scenes_by_name.get(token) or scenes_by_token.get(token)
        if row is None:
            raise FileNotFoundError(f"nuScenes scene not found in metadata: {value}")
        requested_names.add(str(row["name"]))

    samples_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        scene = scenes_by_token.get(str(row.get("scene_token")))
        if scene is not None:
            samples_by_scene[str(scene["name"])].append(row)
    sample_lookup: dict[str, tuple[str, int]] = {}
    for scene_name, rows in samples_by_scene.items():
        for frame_id, row in enumerate(sorted(rows, key=lambda item: int(item["timestamp"]))):
            sample_lookup[str(row["token"])] = (scene_name, frame_id)

    if not config.camera_channels:
        raise ValueError("Explicit camera_channels are required")
    configured_channels = config.camera_channels
    channels = [str(channel).strip().upper() for channel in configured_channels]
    if len(channels) != len(set(channels)):
        raise ValueError(f"Duplicate nuScenes camera channels: {configured_channels}")
    invalid_channels = [channel for channel in channels if not channel.startswith("CAM_")]
    if invalid_channels:
        raise ValueError(f"Invalid nuScenes camera channels: {invalid_channels}")

    frames_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_data_path = metadata_root / "sample_data.json"
    with sample_data_path.open("rb") as handle:
        for row in ijson.items(handle, "item"):
            filename = str(row.get("filename") or "")
            if not row.get("is_key_frame") or not filename.startswith("samples/CAM_"):
                continue
            camera_name = Path(filename).parts[1]
            if camera_name not in channels:
                continue
            sample_info = sample_lookup.get(str(row.get("sample_token")))
            if sample_info is None:
                continue
            if requested_names and sample_info[0] not in requested_names:
                continue
            image_path = (config.dataset_root / filename).resolve()
            if not image_path.is_file():
                continue
            scene_name, frame_id = sample_info
            frames_by_scene[scene_name].append(
                {
                    "path": str(image_path),
                    "camera_name": camera_name.removeprefix("CAM_").lower(),
                    "frame_id": frame_id,
                    "sample_token": str(row["sample_token"]),
                    "timestamp_us": int(row["timestamp"]),
                    "calibrated_sensor_token": str(row["calibrated_sensor_token"]),
                    "ego_pose_token": str(row["ego_pose_token"]),
                }
            )

    complete_names: list[str] = []
    incomplete: list[dict[str, Any]] = []
    for scene_name in sorted(requested_names or scenes_by_name):
        scene = scenes_by_name[scene_name]
        frames = frames_by_scene.get(scene_name, [])
        camera_counts = Counter(str(row["camera_name"]).upper() for row in frames)
        expected = int(scene["nbr_samples"])
        expected_names = [channel.removeprefix("CAM_") for channel in channels]
        complete = all(camera_counts.get(name, 0) == expected for name in expected_names)
        if complete:
            complete_names.append(scene_name)
        else:
            incomplete.append(
                {
                    "scene_id": scene_name,
                    "expected_per_camera": expected,
                    "available_frames": len(frames),
                }
            )

    if requested_names:
        missing = sorted(requested_names - set(complete_names))
        if missing:
            raise FileNotFoundError(
                "nuScenes scenes are missing one or more requested camera streams: " + ", ".join(missing)
            )
        selected_names = [name for name in complete_names if name in requested_names]
    else:
        selected_names = complete_names
        if config.limit_scenes is not None:
            selected_names = selected_names[: config.limit_scenes]

    # Only sensor calibration and ego poses are read; object annotations are not required.
    calibration_path, ego_path = metadata_root / "calibrated_sensor.json", metadata_root / "ego_pose.json"
    if calibration_path.exists() != ego_path.exists():
        raise FileNotFoundError("Provide both calibrated_sensor.json and ego_pose.json, or neither")
    selected_frames = [frame for name in selected_names for frame in frames_by_scene[name]]
    # Intermediate RGB frames support cheap optical flow only, never extra detector calls.
    streams = defaultdict(list)
    for name in selected_names:
        for frame in frames_by_scene[name]:
            frame['motion_paths'] = []
            streams[(name, frame['camera_name'])].append(frame)
    stream_times = {}
    for key, frames in streams.items():
        frames.sort(key=lambda frame: frame['timestamp_us'])
        stream_times[key] = [frame['timestamp_us'] for frame in frames]
    motion_counts = dict(available=0, missing=0)
    with sample_data_path.open('rb') as handle:
        for row in ijson.items(handle, 'item'):
            filename = str(row.get('filename', ''))
            if row.get('is_key_frame') or not filename.startswith('sweeps/CAM_'):
                continue
            sample_info = sample_lookup.get(str(row.get('sample_token')))
            if sample_info is None:
                continue
            key = (sample_info[0], Path(filename).parts[1].removeprefix('CAM_').lower())
            if key not in streams:
                continue
            timestamp = int(row['timestamp'])
            position = bisect_right(stream_times[key], timestamp)
            if not 0 < position < len(streams[key]):
                continue
            image_path = (config.dataset_root / filename).resolve()
            if image_path.is_file():
                streams[key][position]['motion_paths'].append((timestamp, str(image_path)))
                motion_counts['available'] += 1
            else:
                motion_counts['missing'] += 1
    for frame in selected_frames:
        frame['motion_paths'] = [path for _, path in sorted(frame['motion_paths'])]
    if calibration_path.exists():
        calibrations = {row["token"]: row for row in read_json(calibration_path)}
        wanted_poses = {frame["ego_pose_token"] for frame in selected_frames}
        with ego_path.open("rb") as handle:
            poses = {row["token"]: row for row in ijson.items(handle, "item") if row["token"] in wanted_poses}
        for frame in selected_frames:
            sensor = calibrations[frame["calibrated_sensor_token"]]
            frame["camera_intrinsic"] = sensor["camera_intrinsic"]
            frame["camera_to_world"] = (_pose_matrix(poses[frame["ego_pose_token"]]) @ _pose_matrix(sensor)).tolist()

    index_dir = ensure_dir(run_dir / "debug" / "dataset_scenes")
    scene_paths: list[Path] = []
    for scene_name in selected_names:
        index_path = index_dir / f"{scene_name}.json"
        frames = sorted(
            frames_by_scene[scene_name],
            key=lambda row: (str(row["camera_name"]), int(row["frame_id"]), str(row["path"])),
        )
        write_json(
            index_path,
            {
                "dataset": "nuscenes",
                "dataset_root": str(config.dataset_root.resolve()),
                "dataset_version": config.dataset_version,
                "scene_id": scene_name,
                "scene_token": str(scenes_by_name[scene_name]["token"]),
                "camera_channels": channels,
                "frames": frames,
            },
        )
        scene_paths.append(index_path)

    write_json(
        run_dir / "reports" / "dataset_inventory.json",
        {
            "dataset": "nuscenes",
            "dataset_root": str(config.dataset_root.resolve()),
            "dataset_version": config.dataset_version,
            "dataset_split": config.dataset_split,
            "metadata_scene_count": len(scene_rows),
            "inventory_scope": "requested_scenes" if requested_names else "all_metadata_scenes",
            "metadata_sample_count": len(sample_rows),
            "complete_scene_count": len(complete_names),
            "incomplete_scene_count": len(incomplete),
            "selected_scene_count": len(selected_names),
            "camera_channels": channels,
            "frame_stride": config.frame_stride,
            "motion_only_intermediate_rgb": motion_counts,
            "incomplete_scene_examples": incomplete[:20],
        },
    )
    return scene_paths


def _discover_scene_dirs(config: PipelineConfig, run_dir: Path) -> list[Path]:
    if config.dataset == "nuscenes":
        return _prepare_nuscenes_scene_dirs(config, run_dir)
    paths = sorted(config.dataset_root.glob("*.json"))
    if config.scene_ids:
        by_id = {path.stem: path for path in paths}
        unknown = set(config.scene_ids) - set(by_id)
        if unknown:
            raise FileNotFoundError(f"Missing frame indexes: {sorted(unknown)}")
        paths = [by_id[name] for name in config.scene_ids]
    elif config.limit_scenes is not None:
        paths = paths[:config.limit_scenes]
    result = []
    for path in paths:
        payload = read_json(path)
        if not isinstance(payload, dict) or payload.get("scene_id") != path.stem or not isinstance(payload.get("frames"), list):
            raise ValueError(f"Invalid generic scene index: {path}")
        keys = set()
        for frame in payload["frames"]:
            key = (frame["camera_name"], frame["frame_id"])
            if key in keys:
                raise ValueError(f"Duplicate camera/frame identity in {path}: {key}")
            keys.add(key)
            source = Path(frame["path"]).expanduser()
            if not source.is_absolute():
                source = path.parent / source
            if not source.is_file():
                raise FileNotFoundError(source)
            frame["path"] = str(source.resolve())
            intermediates = frame.get('motion_paths', [])
            if not isinstance(intermediates, list) or len(set(intermediates)) != len(intermediates):
                raise ValueError(f'Expected unique ordered motion_paths in {path}')
            resolved = []
            for value in intermediates:
                intermediate = Path(value).expanduser()
                if not intermediate.is_absolute():
                    intermediate = path.parent / intermediate
                if not intermediate.is_file():
                    raise FileNotFoundError(intermediate)
                resolved.append(str(intermediate.resolve()))
            frame['motion_paths'] = resolved
        output = run_dir / "debug" / "dataset_scenes" / path.name
        write_json(output, payload)
        result.append(output)
    return result


def _write_scene_list(path: Path, scenes: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(_scene_id(scene) for scene in scenes) + ("\n" if scenes else ""),
        encoding="utf-8",
    )


def _classes(config: PipelineConfig) -> list[str]:
    normalized = [normalize_rigid_class(item) for item in config.classes]
    return list(dict.fromkeys(item for item in normalized if item and item != "unknown"))


def _scene_export_config(config: PipelineConfig, scene_id: str, scene_path: Path, scene_dir: Path) -> SceneExportConfig:
    if config.grounded_sam2_root is None:
        raise FileNotFoundError("grounded_sam2_root is required")
    cached = None
    if config.detection_cache_root is not None:
        previous = read_json(config.detection_cache_root / 'input_signature.json')['config']
        current = json.loads(json.dumps(asdict(config), default=str))
        keys = ('dataset', 'dataset_root', 'dataset_version', 'camera_channels', 'frame_stride',
                'grounded_sam2_root', 'classes', 'thresholds', 'max_detections_per_frame')
        qc_keys = ('min_bbox_height', 'min_bbox_width', 'max_occlusion_gap')
        if any(previous[key] != current[key] for key in keys) or any(
                previous['qc'][key] != current['qc'][key] for key in qc_keys):
            raise ValueError('Detection cache settings differ from the requested RGB extraction')
        cached = config.detection_cache_root / 'scenes' / scene_id / 'logs' / 'detections.json'
        if not cached.is_file():
            raise FileNotFoundError(cached)
    return SceneExportConfig(
        scene_id=scene_id,
        scene_path=scene_path,
        output_dir=scene_dir,
        mode=config.mode,
        classes=_classes(config),
        grounded_sam2_root=config.grounded_sam2_root,
        box_threshold=config.thresholds.box,
        text_threshold=config.thresholds.text,
        min_bbox_height=config.qc.min_bbox_height,
        min_bbox_width=config.qc.min_bbox_width,
        min_mask_area_ratio=config.qc.min_mask_area_ratio,
        min_largest_component_ratio=config.qc.min_largest_component_ratio,
        max_edge_touch_ratio=config.qc.max_edge_touch_ratio,
        max_asset_candidates=config.max_asset_candidates,
        sam_candidates_per_tracklet=config.sam_candidates_per_tracklet,
        dataset=config.dataset,
        camera_channels=list(config.camera_channels),
        frame_stride=config.frame_stride,
        vehicle_sam_candidates=config.vehicle_sam_candidates,
        cross_camera_dedup=config.cross_camera_dedup,
        max_mask_area_ratio=config.qc.max_mask_area_ratio,
        max_detections_per_frame=config.max_detections_per_frame,
        track_max_gap=config.qc.max_occlusion_gap * config.frame_stride,
        track_iou_threshold=config.qc.min_temporal_iou,
        real_views_max=config.real_views_max,
        real_view_quality_ratio=config.real_view_quality_ratio,
        detection_cache=cached,
    )


def _manifest_done(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        rows = read_json(path)
    except Exception:
        return False
    if not isinstance(rows, list):
        return False
    for row in rows:
        try:
            validate_manifest_item(row)
        except Exception:
            return False
        if not Path(str(row["image_path"])).is_absolute() or not Path(str(row["image_path"])).is_file():
            return False
    return True


async def _apply_scene_vlm(config: PipelineConfig, manifest_path: Path, logger: Any) -> dict[str, Any]:
    rows = read_json(manifest_path)
    if not isinstance(rows, list):
        raise ValueError(f"manifest must be a list: {manifest_path}")
    if not rows:
        write_manifest(manifest_path, [])
        return {"manifest_path": str(manifest_path), "count": 0}

    source_path = manifest_path.parents[1] / "logs" / "export_items_full.json"
    source_rows = read_json(source_path) if source_path.is_file() else []
    source_by_id = {str(row['asset_id']): row for row in source_rows}
    batches = [rigid_vlm_messages(row["image_path"], row["category"],
               source_context=source_by_id.get(str(row['obj_id']), {}).get('source')) for row in rows]
    client = AsyncOpenAIClient(config, logger)
    corrections = []
    try:
        results = await client.call_many_json(
            batches,
            concurrency=max(1, int(config.vlm.concurrency)),
            response_schema=RIGID_RESPONSE_SCHEMA,
            collect_errors=True,
        )
        # A valid JSON object may still contain an invalid angle enum. Ask the
        # model once to inspect the same evidence; never coerce side into front.
        retry_indices = [index for index, result in enumerate(results)
                         if isinstance(result, ValidationError) and result.validator == 'enum'
                         and list(result.absolute_path) == ['vehicle_view']]
        if retry_indices:
            retry_batches = [[*batches[index],
                dict(role='assistant', content=json.dumps(results[index].response)),
                dict(role='user', content=
                    'The vehicle_view value in your JSON is outside the allowed enum. Reinspect the same '
                    'target image and return the complete JSON using front_side, rear_side or unknown only. '
                    'Use front_side only for directly visible front-end cues; rear_side only for directly '
                    'visible rear-end cues. A strict side profile with neither end discernible is unknown. '
                    'Do not invent a visible end to pass validation. Preserve evidence-based quality and '
                    'completeness judgments. All original schema constraints still apply.')]
                for index in retry_indices]
            repaired = await client.call_many_json(retry_batches,
                concurrency=max(1, int(config.vlm.concurrency)),
                response_schema=RIGID_RESPONSE_SCHEMA, collect_errors=True)
            for index, result in zip(retry_indices, repaired):
                corrections.append(dict(obj_id=rows[index]['obj_id'],
                    original_response=results[index].response,
                    original_error=str(results[index]),
                    status='failed' if isinstance(result, Exception) else 'schema_valid',
                    response=getattr(result, 'response', None) if isinstance(result, Exception) else result,
                    error=f'{type(result).__name__}: {result}' if isinstance(result, Exception) else None))
                results[index] = result
    finally:
        await client.close()

    normalized: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    rigid_builder = RigidPromptBuilder()
    for row, result in zip(rows, results):
        item = dict(row)
        item["image_path"] = str(Path(item["image_path"]).resolve())
        detected_category = rigid_builder.normalize_category(item.get("category"))
        if isinstance(result, Exception):
            rejects.append({"scene_id": item.get("scene_id"), "obj_id": item.get("obj_id"),
                            "image_path": item["image_path"], "reject_stage": "vlm",
                            "reason": "vlm_request_or_schema_failed", "detected_category": detected_category,
                            "error": f"{type(result).__name__}: {result}",
                            "response": getattr(result, "response", None)})
            continue
        category = rigid_builder.normalize_category(result.get("category"))
        reason = None
        if result['asset_form'] != 'discrete_object':
            reason = result['asset_form']
        elif result['structure_extent'] != 'whole_unit':
            reason = result['structure_extent']
        elif result['source_quality'] != 'usable':
            reason = result['source_quality']
        native = source_by_id.get(str(item['obj_id']),{}).get('source',{}).get('native_foreground')
        if reason is None and category == 'pole' and native is not None and (
                max(native['width'],native['height']) < 120 or native['median_thickness_px'] < 6):
            reason = 'pole_insufficient_native_thickness'
        if reason:
            rejects.append({"scene_id": item.get("scene_id"), "obj_id": item.get("obj_id"),
                            "image_path": item["image_path"], "reject_stage": "source_quality",
                            "reason": "source_" + reason,
                            "detected_category": detected_category, "vlm": result})
            continue
        if category == "unknown":
            rejects.append({"scene_id": item.get("scene_id"), "obj_id": item.get("obj_id"),
                            "image_path": item["image_path"], "reject_stage": "vlm",
                            "reason": "unknown_rigid_category", "detected_category": detected_category,
                            "vlm": result})
            continue
        if category in rigid_builder.VEHICLE and _vehicle_end(result.get("vehicle_view")) is None:
            rejects.append({"scene_id": item.get("scene_id"), "obj_id": item.get("obj_id"),
                            "image_path": item["image_path"], "reject_stage": "vlm",
                            "reason": "unresolved_vehicle_heading", "detected_category": detected_category,
                            "vlm": result})
            continue
        item["category"] = "car" if category == "vehicle" else category
        item["prompts"] = build_diffusion_prompts(item, result)
        validate_manifest_item(item)
        normalized.append(item)
    write_manifest(manifest_path, normalized)
    if source_path.is_file():
        by_id = {str(row["obj_id"]): row for row in normalized}
        decisions = {str(row["obj_id"]): result for row, result in zip(rows, results)}
        rejected_by_id = {str(row['obj_id']): row for row in rejects}
        for source_row in source_rows if isinstance(source_rows, list) else []:
            oid = str(source_row['asset_id'])
            final_row = by_id.get(oid)
            decision = decisions.get(oid)
            if final_row is not None:
                source_row["image_path"] = final_row["image_path"]
                source_row["vlm"] = decision
                source_row['vlm_status'] = 'accepted'
            elif oid in rejected_by_id:
                source_row['vlm_status'] = 'rejected'
                source_row['vlm_rejection'] = rejected_by_id[oid]
                if isinstance(decision, dict):
                    source_row['vlm'] = decision
        write_json(source_path, source_rows)
    write_jsonl(manifest_path.parents[1] / "logs" / "vlm_rejects.jsonl", rejects)
    write_jsonl(manifest_path.parents[1] / "logs" / "vlm_schema_corrections.jsonl", corrections)
    return {
        "manifest_path": str(manifest_path),
        "count": len(normalized),
        "input_count": len(rows),
        "rejected_count": len(rejects),
        "failed_count": sum(isinstance(result, Exception) for result in results),
        "schema_correction_count": len(corrections),
        "source_context_count": sum(str(row['obj_id']) in source_by_id for row in rows),
    }


def _scene_worker(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    apply_cli_overrides(
        config,
        gpu=args.gpu,
        dataset_root=args.dataset_root,
        dataset_version=args.dataset_version,
        dataset_split=args.dataset_split,
        camera_channels=args.camera_channels,
        frame_stride=args.frame_stride,
    )
    scene_id = str(args.scene_id)
    scene_dir = Path(args.scene_output_dir).resolve()
    logger = setup_logger(f"asset_extraction.worker.{scene_id}", scene_dir / "logs" / "worker.log")
    stage = "asset-extraction"
    try:
        export = run_scene_asset_export(
            _scene_export_config(config, scene_id, Path(args.scene_path).resolve(), scene_dir)
        )
        vlm = asyncio.run(_apply_scene_vlm(config, Path(export["manifest_path"]), logger))
        status = read_json(scene_dir / "status.json")
        status["num_pre_vlm_kept"] = status.get("num_kept")
        status["num_kept"] = vlm["count"]
        status["num_vlm_rejected"] = vlm.get("rejected_count", 0)
        status["vlm"] = vlm
        write_json(scene_dir / "status.json", status)
        print(json.dumps(status, ensure_ascii=False))
        return 0
    except Exception as exc:
        failure_path = scene_dir / "status.json"
        failure = read_json(failure_path) if failure_path.exists() else {}
        failure.update(stage=stage, status='failed', scene_id=scene_id,
                       last_error=f'{type(exc).__name__}: {exc}')
        write_json(
            failure_path,
            failure,
        )
        logger.exception("scene worker failed")
        return 1


class SceneScheduler:
    def __init__(
        self,
        config: PipelineConfig,
        config_path: Path,
        scenes: list[Path],
        run_dir: Path,
        logger: Any,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.scenes = scenes
        self.run_dir = run_dir
        self.logger = logger
        self.results: list[dict[str, Any]] = []

    def _record(self, row: dict[str, Any]) -> None:
        self.results.append(row)
        write_json(self.run_dir / "reports" / "scene_progress.json", sorted(self.results, key=lambda item: item["scene_id"]))

    def _env(self, gpu: int) -> dict[str, str]:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        ensure_dir("/mnt/ojc/tmp")
        ensure_dir("/mnt/ojc/tmp/matplotlib")
        ensure_dir("/mnt/ojc/hf_cache")
        env.setdefault("TMPDIR", "/mnt/ojc/tmp")
        env.setdefault("HF_HOME", "/mnt/ojc/hf_cache")
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("HF_DATASETS_OFFLINE", "1")
        env.setdefault("ASSET_BERT_BASE_UNCASED", "/mnt/ojc/models/bert-base-uncased")
        env.setdefault("MPLCONFIGDIR", "/mnt/ojc/tmp/matplotlib")
        roots = [str(PROJECT_ROOT), str(PROJECT_ROOT.parent)]
        if env.get("PYTHONPATH"):
            roots.extend(value for value in env["PYTHONPATH"].split(":") if value)
        env["PYTHONPATH"] = ":".join(dict.fromkeys(roots))
        return env

    def _command(
        self,
        scene_id: str,
        scene_path: Path,
        scene_dir: Path,
        gpu: int,
    ) -> list[str]:
        command = [
            str(self.config.runtime.python),
            "-m",
            "asset_extraction.pipeline",
            "--scene-worker",
            "--config",
            str(self.config_path),
            "--scene-id",
            scene_id,
            "--scene-path",
            str(scene_path),
            "--scene-output-dir",
            str(scene_dir),
            "--dataset-root",
            str(self.config.dataset_root),
            "--dataset-split",
            self.config.dataset_split,
            "--frame-stride",
            str(self.config.frame_stride),
            "--gpu",
            str(gpu),
        ]
        if self.config.dataset_version:
            command.extend(["--dataset-version", self.config.dataset_version])
        if self.config.camera_channels:
            command.extend(["--camera-channels", ",".join(self.config.camera_channels)])
        return command

    def _image_completed(self, scene_id: str) -> bool:
        scene_dir = self.run_dir / "scenes" / scene_id
        status_path = scene_dir / "status.json"
        manifest_path = scene_dir / "diffusion_inputs" / "manifest.json"
        if not _manifest_done(manifest_path) or not status_path.is_file():
            return False
        try:
            status = read_json(status_path)
        except Exception:
            return False
        if status.get("status") != "completed":
            return False
        # Old single-reference exports cannot satisfy an H handoff on resume.
        try:
            sources = read_json(scene_dir / 'logs' / 'export_items_full.json')
            by_id = {r['asset_id']: r for r in sources}
            from .real_views import VEHICLES
            for row in read_json(manifest_path):
                if row['category'] in VEHICLES:
                    refs = by_id[row['obj_id']].get('real_images', [])
                    if not refs or any(not Path(r['rgba_path']).is_file() for r in refs):
                        return False
        except (OSError, ValueError, KeyError, TypeError):
            return False
        return True

    async def _run_scene(self, scene_path: Path, slots: asyncio.Queue) -> None:
        gpu, slot = await slots.get()
        scene_id = _scene_id(scene_path)
        scene_dir = self.run_dir / "scenes" / scene_id
        started = time.perf_counter()
        row = {"scene_id": scene_id, "gpu": gpu, "worker_slot": slot}
        try:
            if self.config.resume and self._image_completed(scene_id):
                row.update(status="completed", skipped=True, elapsed_sec=0.0)
                return
            log_path = scene_dir / "logs" / "worker.launch.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = self._command(scene_id, scene_path, scene_dir, gpu)
            row["log"] = str(log_path)
            with log_path.open("a", encoding="utf-8") as log:
                log.write("command: " + " ".join(command) + "\n")
                log.flush()
                process = await asyncio.create_subprocess_exec(
                    *command, cwd=str(PROJECT_ROOT), env=self._env(gpu),
                    stdout=log, stderr=asyncio.subprocess.STDOUT)
                try:
                    code = await process.wait()
                except asyncio.CancelledError:
                    if process.returncode is None:
                        process.terminate()
                        try:
                            await asyncio.wait_for(process.wait(), timeout=10)
                        except asyncio.TimeoutError:
                            process.kill()
                            await process.wait()
                    raise
            row.update(status="completed" if code == 0 else "failed", returncode=code)
        except asyncio.CancelledError:
            row["status"] = "cancelled"
            raise
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            row.setdefault("elapsed_sec", round(time.perf_counter() - started, 6))
            self._record(row)
            slots.put_nowait((gpu, slot))
            slots.task_done()

    async def _run(self) -> list[dict[str, Any]]:
        if not self.config.gpu_ids or self.config.scene_workers_per_gpu < 1:
            raise ValueError("Explicit GPU IDs and a positive worker count are required")
        slots = asyncio.Queue()
        for gpu in self.config.gpu_ids:
            for slot in range(self.config.scene_workers_per_gpu):
                slots.put_nowait((gpu, slot))
        await asyncio.gather(*(self._run_scene(scene, slots) for scene in self.scenes))
        return sorted(self.results, key=lambda row: row["scene_id"])

    def run(self) -> list[dict[str, Any]]:
        return asyncio.run(self._run())


def _merge_scene_manifests(run_dir: Path) -> dict[str, Any]:
    merged_dir = ensure_dir(run_dir / "merged")
    images_dir = ensure_dir(merged_dir / "images")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for manifest_path in sorted((run_dir / "scenes").glob("*/diffusion_inputs/manifest.json")):
        scene_rows = read_json(manifest_path)
        if not isinstance(scene_rows, list):
            raise ValueError(f"scene manifest must be a list: {manifest_path}")
        for source_row in scene_rows:
            item = dict(source_row)
            source = Path(str(item["image_path"])).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"image missing: {source}")
            if item["obj_id"] in seen:
                raise ValueError(f"duplicate obj_id: {item['obj_id']}")
            seen.add(item["obj_id"])
            destination = images_dir / source.name
            if source != destination.resolve():
                shutil.copy2(source, destination)
            item["image_path"] = str(destination.resolve())
            validate_manifest_item(item)
            rows.append(item)
    write_manifest(merged_dir / "manifest.json", rows)
    report = {
        "all_pass": True,
        "manifest": str((merged_dir / "manifest.json").resolve()),
        "total_items": len(rows),
        "unique_obj_ids": len(seen),
        "category_counts": dict(sorted(Counter(row["category"] for row in rows).items())),
        "absolute_image_paths": all(Path(row["image_path"]).is_absolute() for row in rows),
    }
    write_json(run_dir / "reports" / "manifest_check_merged.json", report)
    return report


class ImagePipeline:
    def __init__(self, config: PipelineConfig, config_path: Path):
        self.config = config
        self.config_path = config_path
        self.run_dir = run_root(config)
        self.logs_dir = ensure_dir(self.run_dir / "logs")
        self.debug_dir = ensure_dir(self.run_dir / "debug")
        ensure_dir(self.run_dir / "reports")
        self.logger = setup_logger(f"asset_extraction.{config.mode}", self.logs_dir / f"{config.mode}.log")

    def _claim_run(self) -> None:
        config = asdict(self.config)
        for key in ('workspace', 'output_root', 'run_name', 'gpu_ids', 'forbidden_gpu_ids',
                    'scene_workers_per_gpu', 'runtime', 'dry_run', 'resume'):
            config.pop(key)
        modules = ('pipeline', 'config', 'vision', 'real_views', 'vlm', 'io_utils')
        signature = dict(config=json.loads(json.dumps(config, default=str)), code={
            name: hashlib.sha256((PROJECT_ROOT / 'asset_extraction' / f'{name}.py').read_bytes()).hexdigest()
            for name in modules})
        if self.config.detection_cache_root is not None:
            files = [self.config.detection_cache_root/'input_signature.json',
                     *sorted((self.config.detection_cache_root/'scenes').glob('*/logs/detections.json'))]
            signature['detection_cache'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        path = self.run_dir / 'input_signature.json'
        if path.is_file():
            if read_json(path) != signature:
                raise ValueError('Frontend code/config changed; choose a new RUN_NAME to preserve existing results')
        elif (self.run_dir / 'scenes').exists():
            raise ValueError('Existing frontend run has no code/config signature; choose a new RUN_NAME')
        else:
            write_json(path, signature)

    def _dry_run(self) -> dict[str, Any]:
        payload = {
            "stage": "dry-run",
            "status": "completed",
            "mode": self.config.mode,
            "run_dir": str(self.run_dir),
            "workspace": str(self.config.workspace),
            "dataset": self.config.dataset,
            "dataset_root": str(self.config.dataset_root) if self.config.dataset_root else None,
            "dataset_version": self.config.dataset_version,
            "dataset_split": self.config.dataset_split,
            "gpu_ids": self.config.gpu_ids,
            "scene_ids": self.config.scene_ids,
            "limit_scenes": self.config.limit_scenes,
        }
        write_json(self.run_dir / "status.json", payload)
        return payload

    def check_ready(self) -> None:
        validate_config(self.config, check_runtime=True)
        if self.config.dataset_root is None:
            raise FileNotFoundError("dataset_root is required")
        if self.config.grounded_sam2_root is None:
            raise FileNotFoundError("grounded_sam2_root is required")
        ensure_dir(self.config.output_root)

    def run(self) -> dict[str, Any]:
        if self.config.dry_run:
            return self._dry_run()
        self.check_ready()
        self._claim_run()
        scenes = _discover_scene_dirs(self.config, self.run_dir)
        if not scenes:
            raise RuntimeError(f"No complete {self.config.dataset} scenes found under {self.config.dataset_root}")
        _write_scene_list(self.debug_dir / f"{self.config.mode}_scene_list.txt", scenes)
        started = time.perf_counter()
        results = SceneScheduler(
            self.config,
            self.config_path,
            scenes,
            self.run_dir,
            self.logger,
        ).run()
        failed = [row for row in results if row.get("status") != "completed"]
        if failed:
            status = {
                "stage": self.config.mode,
                "status": "failed",
                "failed_scenes": failed,
                "results": results,
            }
            write_json(self.run_dir / "status.asset_extraction.json", status)
            raise RuntimeError(f"{len(failed)} scenes failed; see {self.run_dir / 'reports' / 'scene_progress.json'}")
        manifest_report = _merge_scene_manifests(self.run_dir)
        status = {
            "stage": self.config.mode,
            "status": "completed",
            "scene_count": len(scenes),
            "elapsed_sec": round(time.perf_counter() - started, 6),
            "results": results,
            "manifest": manifest_report,
            "final_vlm_repeated": False,
        }
        write_json(self.run_dir / "status.asset_extraction.json", status)
        write_json(self.run_dir / "status.json", status)
        return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", default=None)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--limit-scenes", type=int)
    parser.add_argument("--scene-ids")
    parser.add_argument("--gpu")
    parser.add_argument("--scene-workers-per-gpu", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--output-root")
    parser.add_argument("--dataset-root")
    parser.add_argument("--dataset-version")
    parser.add_argument("--dataset-split")
    parser.add_argument("--camera-channels")
    parser.add_argument("--frame-stride", type=int)
    parser.add_argument("--scene-worker", action="store_true")
    parser.add_argument("--scene-id")
    parser.add_argument("--scene-path")
    parser.add_argument("--scene-output-dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.scene_worker:
        return _scene_worker(args)
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    apply_cli_overrides(
        config,
        dry_run=args.dry_run,
        resume=args.resume,
        limit_scenes=args.limit_scenes,
        scene_ids=args.scene_ids,
        gpu=args.gpu,
        scene_workers_per_gpu=args.scene_workers_per_gpu,
        run_name=args.run_name,
        output_root=args.output_root,
        dataset_root=args.dataset_root,
        dataset_version=args.dataset_version,
        dataset_split=args.dataset_split,
        camera_channels=args.camera_channels,
        frame_stride=args.frame_stride,
    )
    result = ImagePipeline(config, config_path).run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
