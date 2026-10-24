"""Configuration for the nuScenes rigid-asset pre-FLUX pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigError(ValueError):
    pass


@dataclass
class ThresholdConfig:
    box: float = 0.15
    text: float = 0.15


@dataclass
class QCConfig:
    min_mask_area_ratio: float = 0.0005
    max_mask_area_ratio: float = 0.8
    max_edge_touch_ratio: float = 0.35
    min_largest_component_ratio: float = 0.6
    min_temporal_iou: float = 0.25
    max_occlusion_gap: int = 3
    min_bbox_height: int = 12
    min_bbox_width: int = 8


@dataclass
class VLMConfig:
    endpoint: str = "http://127.0.0.1:8080/v1"
    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    temperature: float = 0.0
    max_tokens: int = 4096
    concurrency: int = 16


@dataclass
class RuntimeConfig:
    python: Path = Path("/mnt/ojc/miniconda3/envs/vllm_vlm/bin/python")


@dataclass
class PipelineConfig:
    mode: str
    workspace: Path
    output_root: Path
    run_name: str
    dataset: str = "nuscenes"
    dataset_root: Optional[Path] = None
    dataset_version: str = "v1.0-trainval"
    dataset_split: str = "available"
    camera_channels: list[str] = field(default_factory=list)
    frame_stride: int = 1
    grounded_sam2_root: Optional[Path] = None
    detection_cache_root: Optional[Path] = None
    gpu_ids: list[int] = field(default_factory=list)
    forbidden_gpu_ids: list[int] = field(default_factory=list)
    scene_workers_per_gpu: int = 1
    classes: list[str] = field(default_factory=list)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    cross_camera_dedup: bool = True
    max_detections_per_frame: int = 0
    max_asset_candidates: int = 320
    sam_candidates_per_tracklet: int = 3
    vehicle_sam_candidates: int = 6
    real_views_max: int = 4
    real_view_quality_ratio: float = 0.70
    dry_run: bool = False
    resume: bool = False
    limit_scenes: Optional[int] = None
    scene_ids: list[str] = field(default_factory=list)
    manifest_version: str = "asset_extraction_v1"


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError("PyYAML is required") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError(f"Config must be a YAML mapping: {path}")
    return payload


def _as_path(value: Any) -> Optional[Path]:
    return None if value is None else Path(str(value)).expanduser()


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    raise ConfigError(f"Expected GPU list, got {type(value).__name__}")


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ConfigError(f"Expected string list, got {type(value).__name__}")


def _coerce(annotation: Any, value: Any) -> Any:
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation is Path:
        path = _as_path(value)
        if path is None:
            raise ConfigError("Required path cannot be null")
        return path
    if origin is list and args and args[0] is int:
        return _as_int_list(value)
    if origin is list and args and args[0] is str:
        return _as_str_list(value)
    if origin is Union and type(None) in args and Path in args:
        return _as_path(value)
    if is_dataclass(annotation) and isinstance(value, dict):
        return _dataclass_from_dict(annotation, value)
    if annotation is bool:
        return bool(value)
    if annotation is int and value is not None:
        return int(value)
    if annotation is float and value is not None:
        return float(value)
    if annotation is str and value is not None:
        return str(value)
    return value


def _dataclass_from_dict(cls: type[Any], payload: dict[str, Any]) -> Any:
    definitions = {item.name: item for item in fields(cls)}
    unknown = sorted(set(payload) - set(definitions))
    if unknown:
        raise ConfigError(f"Unknown config keys for {cls.__name__}: {unknown}")
    hints = get_type_hints(cls)
    return cls(**{
        name: _coerce(hints.get(name, definition.type), payload[name])
        for name, definition in definitions.items() if name in payload
    })


def validate_config(config: PipelineConfig, *, check_runtime: bool = False) -> None:
    if config.mode != "rigid" or config.dataset not in {"nuscenes", "frames"}:
        raise ConfigError("Expected mode=rigid and dataset=nuscenes or frames")
    if not config.workspace.exists():
        raise ConfigError(f"workspace does not exist: {config.workspace}")
    if config.dataset_root is None or not config.dataset_root.exists():
        raise ConfigError(f"dataset_root does not exist: {config.dataset_root}")
    if config.dataset == "nuscenes":
        metadata_root = config.dataset_root / config.dataset_version
        required = [metadata_root / name for name in ("scene.json", "sample.json", "sample_data.json")]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ConfigError("nuScenes metadata missing: " + ", ".join(missing))
    elif not config.dataset_root.is_dir():
        raise ConfigError("frames dataset_root must be a directory of scene JSON indexes")
    if config.max_detections_per_frame < 0:
        raise ConfigError("max_detections_per_frame must be >= 0 (0 means unlimited)")
    if config.frame_stride < 1 or config.scene_workers_per_gpu < 1:
        raise ConfigError("frame_stride and scene_workers_per_gpu must be >= 1")
    if config.max_asset_candidates < 1 or config.sam_candidates_per_tracklet < 1:
        raise ConfigError("SAM candidate limits must be >= 1")
    if config.real_views_max not in (2, 3, 4) or config.vehicle_sam_candidates < config.real_views_max:
        raise ConfigError("real_views_max must be 2..4 and vehicle_sam_candidates >= real_views_max")
    if not 0 <= config.real_view_quality_ratio <= 1:
        raise ConfigError("real_view_quality_ratio must be 0..1")
    if config.grounded_sam2_root is not None and not config.grounded_sam2_root.exists():
        raise ConfigError(f"grounded_sam2_root does not exist: {config.grounded_sam2_root}")
    forbidden = set(config.gpu_ids) & set(config.forbidden_gpu_ids)
    if forbidden:
        raise ConfigError(f"gpu_ids include forbidden IDs: {sorted(forbidden)}")
    if any(gpu < 0 for gpu in config.gpu_ids):
        raise ConfigError("gpu_ids must be non-negative")
    if check_runtime:
        missing_runtime = [str(path) for path in (config.runtime.python,) if not path.exists()]
        if missing_runtime:
            raise ConfigError("Runtime paths missing: " + ", ".join(missing_runtime))


def load_config(path: str | Path) -> PipelineConfig:
    config = _dataclass_from_dict(PipelineConfig, _load_yaml(Path(path).expanduser().resolve()))
    config.workspace = PROJECT_ROOT
    if not config.output_root.is_absolute():
        config.output_root = (PROJECT_ROOT / config.output_root).resolve()
    validate_config(config)
    return config


def apply_cli_overrides(
    config: PipelineConfig, *, dry_run: bool | None = None, resume: bool | None = None,
    limit_scenes: int | None = None, scene_ids: str | list[str] | None = None,
    gpu: str | None = None, scene_workers_per_gpu: int | None = None,
    run_name: str | None = None, output_root: str | Path | None = None,
    dataset_root: str | Path | None = None, dataset_version: str | None = None,
    dataset_split: str | None = None, camera_channels: str | list[str] | None = None,
    frame_stride: int | None = None,
) -> PipelineConfig:
    if dry_run is not None:
        config.dry_run = dry_run
    if resume is not None:
        config.resume = resume
    if limit_scenes is not None:
        config.limit_scenes = int(limit_scenes)
    if scene_ids is not None:
        config.scene_ids, config.limit_scenes = _as_str_list(scene_ids), None
    if gpu is not None:
        config.gpu_ids = _as_int_list(gpu)
    if scene_workers_per_gpu is not None:
        config.scene_workers_per_gpu = int(scene_workers_per_gpu)
    if run_name:
        config.run_name = run_name
    if output_root is not None:
        config.output_root = Path(output_root).expanduser().resolve()
    if dataset_root is not None:
        config.dataset_root = Path(dataset_root).expanduser()
    if dataset_version is not None:
        config.dataset_version = str(dataset_version)
    if dataset_split is not None:
        config.dataset_split = str(dataset_split)
    if camera_channels is not None:
        config.camera_channels = _as_str_list(camera_channels)
    if frame_stride is not None:
        config.frame_stride = int(frame_stride)
    validate_config(config)
    return config


def run_root(config: PipelineConfig) -> Path:
    return config.output_root / "runs" / config.run_name
