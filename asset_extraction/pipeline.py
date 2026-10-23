"""nuScenes rigid-asset extraction from raw six-camera scenes to a FLUX manifest."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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
                    "timestamp": int(row["timestamp"]),
                }
            )

    complete_names: list[str] = []
    incomplete: list[dict[str, Any]] = []
    for scene_name in sorted(scenes_by_name):
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
            "metadata_sample_count": len(sample_rows),
            "complete_scene_count": len(complete_names),
            "incomplete_scene_count": len(incomplete),
            "selected_scene_count": len(selected_names),
            "camera_channels": channels,
            "frame_stride": config.frame_stride,
            "incomplete_scene_examples": incomplete[:20],
        },
    )
    return scene_paths


def _discover_scene_dirs(config: PipelineConfig, run_dir: Path) -> list[Path]:
    return _prepare_nuscenes_scene_dirs(config, run_dir)


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
        max_sam_candidates=config.max_sam_candidates,
        sam_candidates_per_tracklet=config.sam_candidates_per_tracklet,
        dataset=config.dataset,
        camera_channels=list(config.camera_channels),
        frame_stride=config.frame_stride,
        vehicle_sam_candidates=config.vehicle_sam_candidates,
        real_views_max=config.real_views_max,
        real_view_quality_ratio=config.real_view_quality_ratio,
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

    batches = [rigid_vlm_messages(row["image_path"], row["category"]) for row in rows]
    client = AsyncOpenAIClient(config, logger)
    try:
        results = await client.call_many_json(
            batches,
            concurrency=max(1, int(config.vlm.concurrency)),
            response_schema=RIGID_RESPONSE_SCHEMA,
        )
    finally:
        await client.close()

    normalized: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    rigid_builder = RigidPromptBuilder()
    for row, result in zip(rows, results):
        item = dict(row)
        item["image_path"] = str(Path(item["image_path"]).resolve())
        detected_category = rigid_builder.normalize_category(item.get("category"))
        category = rigid_builder.normalize_category(result.get("category"))
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
    source_path = manifest_path.parents[1] / "logs" / "export_items_full.json"
    if source_path.is_file():
        source_rows = read_json(source_path)
        by_id = {str(row["obj_id"]): row for row in normalized}
        decisions = {str(row["obj_id"]): result for row, result in zip(rows, results)}
        synced = []
        for source_row in source_rows if isinstance(source_rows, list) else []:
            final_row = by_id.get(str(source_row.get("asset_id")))
            if final_row is None:
                continue
            source_row["image_path"] = final_row["image_path"]
            source_row["vlm"] = decisions[str(source_row["asset_id"])]
            synced.append(source_row)
        write_json(source_path, synced)
    write_jsonl(manifest_path.parents[1] / "logs" / "vlm_rejects.jsonl", rejects)
    return {
        "manifest_path": str(manifest_path),
        "count": len(normalized),
        "input_count": len(rows),
        "rejected_count": len(rejects),
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
        write_json(
            failure_path,
            {
                "stage": stage,
                "status": "failed",
                "scene_id": scene_id,
                "last_error": f"{type(exc).__name__}: {exc}",
            },
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
