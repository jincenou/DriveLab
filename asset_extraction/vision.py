"""Grounded-SAM2 extraction and mask QC for nuScenes rigid assets."""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import shutil
import sys
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .io_utils import ensure_dir, write_json, write_jsonl
from .real_views import VEHICLES, diverse_order, select_real_views


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
@dataclass(frozen=True)
class FrameImage:
    path: str
    camera_name: str
    frame_id: int


@dataclass
class Candidate:
    scene_id: str
    obj_id: str
    category: str
    frame_id: str | None
    image_path: Path
    mask_path: Path | None = None
    bbox: tuple[int, int, int, int] | None = None
    score: float | None = None
    track_id: str | None = None
    camera_name: str | None = None
    source: str = "grounded_sam2"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneExportConfig:
    scene_id: str
    scene_path: Path
    output_dir: Path
    mode: str
    classes: list[str]
    grounded_sam2_root: Path
    box_threshold: float
    text_threshold: float
    min_bbox_height: int
    min_bbox_width: int
    min_mask_area_ratio: float
    min_largest_component_ratio: float
    max_edge_touch_ratio: float
    max_sam_candidates: int
    sam_candidates_per_tracklet: int
    dataset: str = "nuscenes"
    camera_channels: list[str] = field(default_factory=list)
    frame_stride: int = 1
    export_max_size: int = 768
    jpg_quality: int = 92
    vehicle_sam_candidates: int = 6
    real_views_max: int = 4
    real_view_quality_ratio: float = 0.70


@dataclass
class DetectionResult:
    bbox_xyxy: list[float]
    score: float
    label: str
    mask: Any = None
    sam_score: float | None = None


def _token(value: Any) -> str:
    return re.sub(r"_+", "_", str(value or "").strip().lower().replace("-", "_").replace(" ", "_")).strip("_")


def _scene_number(scene_id: str) -> str:
    match = re.search(r"(\d+)$", scene_id)
    return match.group(1) if match else scene_id


def normalize_rigid_class(value: Any) -> str:
    aliases = {
        "car": "vehicle",
        "truck": "vehicle",
        "bus": "vehicle",
        "van": "vehicle",
        "road_sign": "traffic_sign",
        "street_sign": "traffic_sign",
        "sign": "traffic_sign",
        "traffic_cone": "cone",
        "road_barrier": "barrier",
        "fallen_branch": "fallen_tree",
        "tree_branch_on_road": "fallen_tree",
        "large_road_obstacle": "road_debris",
        "large_obstacle": "road_debris",
        "broken_traffic_sign": "traffic_sign",
        "fallen_traffic_sign": "traffic_sign",
        "damaged_pole": "pole",
    }
    token = _token(value)
    return aliases.get(token, token or "unknown")


def _camera_name(value: Any) -> str:
    name = _token(value)
    return name.removeprefix("cam_")


def collect_nuscenes_images(
    scene_index_path: str | Path,
    *,
    cameras: list[str] | None = None,
    frame_stride: int = 1,
) -> list[FrameImage]:
    index_path = Path(scene_index_path)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    frames = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(frames, list):
        raise ValueError(f"Invalid nuScenes scene index: {index_path}")
    wanted = {_camera_name(camera) for camera in (cameras or [])}
    by_camera: dict[str, list[FrameImage]] = defaultdict(list)
    for row in frames:
        camera_name = _camera_name(row.get("camera_name"))
        if wanted and camera_name not in wanted:
            continue
        image_path = Path(str(row.get("path") or "")).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"nuScenes camera image missing: {image_path}")
        by_camera[camera_name].append(
            FrameImage(str(image_path), camera_name, int(row["frame_id"]))
        )
    stride = max(1, int(frame_stride))
    selected = [
        row
        for camera_name in sorted(by_camera)
        for row in sorted(by_camera[camera_name], key=lambda item: (item.frame_id, item.path))[::stride]
    ]
    return sorted(selected, key=lambda item: (item.camera_name, item.frame_id, item.path))


def collect_scene_images(config: SceneExportConfig) -> list[FrameImage]:
    cameras = config.camera_channels or ["front", "front_left", "front_right"]
    return collect_nuscenes_images(
        config.scene_path,
        cameras=cameras,
        frame_stride=config.frame_stride,
    )


class GroundedSAMRunner:
    """Load GroundingDINO and SAM2 once in the worker process."""

    def __init__(self, config: PipelineConfig, logger: Any):
        self.config = config
        self.logger = logger
        self._adapter: OfficialGroundedSAM2Adapter | None = None

    def check_ready(self) -> None:
        root = self.config.grounded_sam2_root
        if root is None or not root.exists():
            raise FileNotFoundError(f"Grounded-SAM2 root not found: {root}")
        required = [
            root / "checkpoints" / "sam2.1_hiera_large.pt",
            root / "gdino_checkpoints" / "groundingdino_swint_ogc.pth",
        ]
        if not self.config.dry_run:
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError("Grounded-SAM2 files missing: " + ", ".join(missing))

    def _load_adapter(self) -> "OfficialGroundedSAM2Adapter":
        if self._adapter is None:
            root = self.config.grounded_sam2_root
            if root is None:
                raise FileNotFoundError("grounded_sam2_root is required")
            self._adapter = OfficialGroundedSAM2Adapter(
                official_repo=root,
                device="cuda:0",
                box_threshold=self.config.thresholds.box,
                text_threshold=self.config.thresholds.text,
            )
        return self._adapter

    def run_on_image(self, image_path: str | Path, classes: list[str], output_dir: str | Path) -> list[Candidate]:
        adapter = self._load_adapter()
        image_source, image_tensor = adapter.load_detection_image(str(image_path))
        detections = adapter.detect_loaded(image_source, image_tensor, ". ".join(classes) + ".")
        return [
            Candidate(
                scene_id="single_image",
                obj_id=f"gsam_{index:06d}",
                category=str(row.label),
                frame_id=None,
                image_path=Path(image_path),
                bbox=_as_bbox(row.bbox_xyxy),
                score=float(row.score),
            )
            for index, row in enumerate(detections, start=1)
        ]


class OfficialGroundedSAM2Adapter:
    def __init__(
        self,
        official_repo: str | Path,
        *,
        device: str = "cuda:0",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> None:
        self.official_repo = Path(official_repo)
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.sam2_checkpoint = self.official_repo / "checkpoints" / "sam2.1_hiera_large.pt"
        self.sam2_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.original_dino_config = self.official_repo / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.bert_model_path = Path(os.environ.get("ASSET_BERT_BASE_UNCASED", "/mnt/ojc/models/bert-base-uncased"))
        self.dino_config = self._local_dino_config()
        self.dino_checkpoint = self.official_repo / "gdino_checkpoints/groundingdino_swint_ogc.pth"
        self._load_dependencies()
        self._validate_files()
        self._load_models()

    def _local_dino_config(self) -> Path:
        target = ensure_dir(Path(__file__).resolve().parents[1] / "outputs" / "tmp") / "GroundingDINO_SwinT_OGC.local_bert.py"
        source = self.original_dino_config.read_text(encoding="utf-8")
        local_bert = str(self.bert_model_path.resolve())
        patched = re.sub(
            r'text_encoder_type\s*=\s*["\']bert-base-uncased["\']',
            f'text_encoder_type = "{local_bert}"',
            source,
        )
        if patched == source:
            raise RuntimeError(f"Could not patch text_encoder_type in {self.original_dino_config}")
        target.write_text(patched, encoding="utf-8")
        return target

    def _load_dependencies(self) -> None:
        if not self.official_repo.exists():
            raise FileNotFoundError(f"Grounded-SAM2 repo not found: {self.official_repo}")
        for path in (self.official_repo, self.official_repo / "grounding_dino"):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
        extra_site_packages = os.environ.get("GSAM2_EXTRA_SITE_PACKAGES", "").strip()
        if extra_site_packages and Path(extra_site_packages).is_dir() and extra_site_packages not in sys.path:
            sys.path.append(extra_site_packages)
        import torch
        from grounding_dino.groundingdino.util.inference import load_image, load_model, predict
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from torchvision.ops import box_convert

        self.torch = torch
        self.load_image = load_image
        self.load_model = load_model
        self.predict_dino = predict
        self.build_sam2 = build_sam2
        self.SAM2ImagePredictor = SAM2ImagePredictor
        self.box_convert = box_convert

    def _validate_files(self) -> None:
        required = (
            self.sam2_checkpoint,
            self.dino_config,
            self.dino_checkpoint,
            self.bert_model_path / "config.json",
            self.bert_model_path / "tokenizer_config.json",
            self.bert_model_path / "vocab.txt",
            self.bert_model_path / "pytorch_model.bin",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Grounded-SAM2 files missing: " + ", ".join(missing))
        if self.device != "cuda:0":
            raise ValueError(f"worker sees one visible GPU; expected cuda:0, got {self.device}")
        if not self.torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this worker")

    def _load_models(self) -> None:
        cwd = Path.cwd()
        os.chdir(self.official_repo)
        try:
            self.torch.cuda.set_device(0)
            self.sam2_model = self.build_sam2(self.sam2_config, str(self.sam2_checkpoint), device=self.device)
            self.sam2_predictor = self.SAM2ImagePredictor(self.sam2_model)
            self.grounding_model = self.load_model(
                model_config_path=str(self.dino_config),
                model_checkpoint_path=str(self.dino_checkpoint),
                device=self.device,
            )
            self.sam2_model.eval()
            self.grounding_model.eval()
        finally:
            os.chdir(cwd)

    def load_detection_image(self, image_path: str):
        return self.load_image(image_path)

    def detect_loaded(self, image_source: Any, image_tensor: Any, text_prompt: str) -> list[DetectionResult]:
        prompt = text_prompt.strip().lower()
        if not prompt.endswith("."):
            prompt += "."
        with self.torch.inference_mode():
            boxes, scores, labels = self.predict_dino(
                model=self.grounding_model,
                image=image_tensor,
                caption=prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )
        if len(boxes) == 0:
            return []
        height, width, _ = image_source.shape
        boxes = boxes * self.torch.tensor([width, height, width, height], device=boxes.device)
        boxes = self.box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").detach().cpu().numpy()
        scores = scores.detach().cpu().numpy().tolist()
        return [
            DetectionResult([float(v) for v in box], float(score), str(label))
            for box, score, label in zip(boxes.tolist(), scores, labels)
        ]

    def segment_loaded_image(self, image_source: Any, boxes_xyxy: list[list[float]]) -> list[DetectionResult]:
        import numpy as np

        if not boxes_xyxy:
            return []
        self.sam2_predictor.set_image(image_source)
        input_boxes = np.asarray(boxes_xyxy, dtype=np.float32)
        with self.torch.inference_mode():
            with self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16):
                masks, scores, _ = self.sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        scores_array = scores.detach().cpu().numpy().reshape(-1) if hasattr(scores, "detach") else np.asarray(scores).reshape(-1)
        return [
            DetectionResult([float(v) for v in box], 1.0, "sam2_box", mask.astype(bool), float(scores_array[index]))
            for index, (box, mask) in enumerate(zip(input_boxes.tolist(), masks))
        ]


class ImageCache:
    def __init__(self, max_items: int = 8) -> None:
        self.max_items = max(1, int(max_items))
        self._rows: OrderedDict[str, Any] = OrderedDict()

    def put(self, path: str | Path, image: Any) -> None:
        key = str(path)
        self._rows[key] = image
        self._rows.move_to_end(key)
        while len(self._rows) > self.max_items:
            self._rows.popitem(last=False)

    def get_rgb(self, path: str | Path) -> Any:
        import numpy as np
        from PIL import Image

        key = str(path)
        if key in self._rows:
            self._rows.move_to_end(key)
            return self._rows[key]
        with Image.open(key) as image:
            value = np.asarray(image.convert("RGB")).copy()
        self.put(key, value)
        return value

    def clear(self) -> None:
        self._rows.clear()


RIGID_PROMPT_GROUPS = {
    "vehicle": "car. truck. bus. van. trailer. motorcycle. bicycle.",
    "traffic_sign": "traffic sign. road sign. street sign.",
    "traffic_light": "traffic light.",
    "cone": "traffic cone.",
    "barrier": "road barrier. guardrail. roadblock.",
    "bollard": "bollard.",
    "pole": "pole. signpost. traffic light pole.",
    "fallen_tree": "fallen tree. fallen branch. log on road.",
    "road_debris": "road debris. tire on road. wheel on road. large road obstacle.",
    "lost_cargo": "lost cargo. box on road. pallet on road.",
    "overturned_vehicle": "overturned vehicle. flipped vehicle.",
    "damaged_vehicle": "damaged vehicle. vehicle wreck.",
    "construction_equipment": "excavator. bulldozer. road construction machine.",
}

LABEL_TO_CLASS = {
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "van": "vehicle",
    "trailer": "vehicle",
    "motorcycle": "vehicle",
    "bicycle": "vehicle",
    "traffic sign": "traffic_sign",
    "road sign": "traffic_sign",
    "street sign": "traffic_sign",
    "signpost": "pole",
    "traffic light pole": "pole",
    "traffic light": "traffic_light",
    "traffic cone": "cone",
    "cone": "cone",
    "road barrier": "barrier",
    "guardrail": "barrier",
    "barrier": "barrier",
    "bollard": "bollard",
    "pole": "pole",
    "fallen tree": "fallen_tree",
    "fallen branch": "fallen_tree",
    "log on road": "fallen_tree",
    "road debris": "road_debris",
    "tire on road": "road_debris",
    "wheel on road": "road_debris",
    "large road obstacle": "road_debris",
    "lost cargo": "lost_cargo",
    "box on road": "lost_cargo",
    "pallet on road": "lost_cargo",
    "overturned vehicle": "overturned_vehicle",
    "flipped vehicle": "overturned_vehicle",
    "damaged vehicle": "damaged_vehicle",
    "vehicle wreck": "damaged_vehicle",
    "excavator": "construction_equipment",
    "bulldozer": "construction_equipment",
}


def prompts_for_classes(classes: list[str], mode: str) -> list[str]:
    normalized = [normalize_rigid_class(item) for item in classes]
    prompts = [RIGID_PROMPT_GROUPS[item] for item in dict.fromkeys(normalized) if item in RIGID_PROMPT_GROUPS]
    return prompts or [RIGID_PROMPT_GROUPS["vehicle"]]


def class_for_label(label: str, allowed: list[str], mode: str) -> str | None:
    normalized_allowed = {normalize_rigid_class(item) for item in allowed}
    text = " ".join(label.strip().lower().replace("_", " ").rstrip(".").split())
    for phrase in sorted(LABEL_TO_CLASS, key=len, reverse=True):
        class_name = LABEL_TO_CLASS[phrase]
        if class_name in normalized_allowed and re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text):
            return "car" if class_name == "vehicle" else class_name
    return None


def _as_bbox(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    x1, y1, x2, y2 = [int(round(float(item))) for item in value[:4]]
    return x1, y1, x2, y2


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _bbox_area(a) + _bbox_area(b) - inter
    return 0.0 if union <= 0 else inter / union


def _nms(rows: list[dict[str, Any]], iou_threshold: float, limit: int) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: float(item["score"]), reverse=True):
        if any(_bbox_iou(row["bbox_xyxy"], other["bbox_xyxy"]) >= iou_threshold for other in kept):
            continue
        kept.append(row)
        if limit and len(kept) >= limit:
            break
    return kept


def _pre_score(row: dict[str, Any], image_shape: tuple[int, int, int]) -> float:
    h, w = image_shape[:2]
    bbox = row["bbox_xyxy"]
    area_ratio = _bbox_area(bbox) / max(1.0, h * w)
    x1, y1, x2, y2 = bbox
    center_dist = math.hypot((x1 + x2) * 0.5 - w * 0.5, (y1 + y2) * 0.5 - h * 0.5)
    center_score = max(0.0, 1.0 - center_dist / max(1.0, math.hypot(w, h) * 0.5))
    size_score = max(0.0, min(1.0, area_ratio / 0.0008))
    return round(0.65 * float(row["score"]) + 0.20 * size_score + 0.15 * center_score, 6)


def _tracklets(rows: list[dict[str, Any]], max_gap: int = 2) -> list[list[dict[str, Any]]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["category"], row["camera_name"])].append(row)
    result: list[list[dict[str, Any]]] = []
    for _, group in sorted(buckets.items()):
        active: list[list[dict[str, Any]]] = []
        for row in sorted(group, key=lambda item: int(item["frame_id"])):
            best_index = -1
            best_score = 0.0
            for index, track in enumerate(active):
                previous = track[-1]
                if int(row["frame_id"]) - int(previous["frame_id"]) > max_gap:
                    continue
                iou = _bbox_iou(previous["bbox_xyxy"], row["bbox_xyxy"])
                if iou > best_score:
                    best_index = index
                    best_score = iou
            if best_index >= 0 and best_score >= 0.25:
                active[best_index].append(row)
            else:
                track = [row]
                active.append(track)
                result.append(track)
    return result


def _select_for_sam(tracklets: list[list[dict[str, Any]]], per_tracklet: int, limit: int,
                    vehicle_candidates: int = 1) -> list[dict[str, Any]]:
    queues: list[list[dict[str, Any]]] = []
    for index, track in enumerate(tracklets, start=1):
        count = vehicle_candidates if track[0]['category'] in VEHICLES else per_tracklet
        queue = []
        for row in diverse_order(track, score_key='pre_score')[:max(1, count)]:
            item = dict(row)
            item["tracklet_id"] = f"t{index:06d}"
            item["tracklet_len"] = len(track)
            queue.append(item)
        queues.append(queue)
    # Each identity gets its first candidate before another receives an extra.
    queues.sort(key=lambda q: float(q[0]['pre_score']), reverse=True)
    rows = [q[i] for i in range(max((len(q) for q in queues), default=0)) for q in queues if i<len(q)]
    return rows[: max(1, limit)]


def _clean_mask(mask: Any, bbox: list[float]) -> tuple[Any, int, float]:
    import numpy as np

    binary = np.asarray(mask, dtype="uint8")
    try:
        import cv2

        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if count <= 1:
            return binary.astype(bool), 0, 1.0
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + int(np.argmax(areas))
        largest = labels == keep
        ratio = float(areas.max()) / max(1.0, float(areas.sum()))
        return largest, int(count - 1), ratio
    except Exception:
        return binary.astype(bool), 1, 1.0


def _mask_quality(row: dict[str, Any], mask: Any, largest_ratio: float, shape: tuple[int, int, int], config: SceneExportConfig) -> tuple[bool, float, str]:
    h, w = shape[:2]
    x1, y1, x2, y2 = row["bbox_xyxy"]
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    box_area = box_w * box_h
    mask_area = float(mask.sum())
    area_ratio = mask_area / max(1.0, h * w)
    mask_box_ratio = mask_area / max(1.0, box_area)
    if area_ratio < config.min_mask_area_ratio:
        return False, 0.0, "mask_too_small"
    if largest_ratio < config.min_largest_component_ratio:
        return False, 0.0, "fragmented_mask"
    score = 0.50 * float(row["pre_score"]) + 0.30 * min(1.0, mask_box_ratio) + 0.20 * largest_ratio
    return True, round(max(0.0, min(1.0, score)), 6), "keep"


def _crop_box(mask: Any, bbox: list[float], width: int, height: int, pad_ratio: float = 0.12) -> tuple[int, int, int, int]:
    import numpy as np

    ys, xs = np.nonzero(mask)
    if xs.size and ys.size:
        x1, x2 = float(xs.min()), float(xs.max() + 1)
        y1, y2 = float(ys.min()), float(ys.max() + 1)
    else:
        x1, y1, x2, y2 = bbox
    pad = max(x2 - x1, y2 - y1) * pad_ratio
    return (
        max(0, int(math.floor(x1 - pad))),
        max(0, int(math.floor(y1 - pad))),
        min(width, int(math.ceil(x2 + pad))),
        min(height, int(math.ceil(y2 + pad))),
    )


def _export_white_jpg(image: Any, mask: Any, bbox: list[float], destination: Path, max_size: int, quality: int) -> None:
    from PIL import Image, ImageFilter

    height, width = image.shape[:2]
    box = _crop_box(mask, bbox, width, height)
    rgb = Image.fromarray(image).crop(box).convert("RGB")
    alpha = Image.fromarray(mask.astype("uint8") * 255, mode="L").crop(box)
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=0.6))
    isolated = Image.new("RGB", rgb.size, (255, 255, 255))
    isolated.paste(rgb, (0, 0), alpha)
    isolated.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    isolated.save(destination, format="JPEG", quality=quality, optimize=True, subsampling=0)


def _prepare_scene_dir(root: Path) -> tuple[Path, Path]:
    diffusion_dir = root / "diffusion_inputs"
    if diffusion_dir.exists():
        shutil.rmtree(diffusion_dir)
    images_dir = ensure_dir(diffusion_dir / "images")
    ensure_dir(root / "logs")
    ensure_dir(root / "profile")
    return diffusion_dir, images_dir


def _extract_scene_assets(config: SceneExportConfig) -> dict[str, Any]:
    """Run detection and SAM once and retain the rows needed by image and video exports."""

    adapter = OfficialGroundedSAM2Adapter(
        config.grounded_sam2_root,
        device="cuda:0",
        box_threshold=config.box_threshold,
        text_threshold=config.text_threshold,
    )
    cache = ImageCache(max_items=8)
    frames = collect_scene_images(config)
    if not frames:
        raise FileNotFoundError(f"No {config.dataset} RGB images found: {config.scene_path}")

    raw_rows: list[dict[str, Any]] = []
    prompt_texts = prompts_for_classes(config.classes, config.mode)
    for frame in frames:
        image_source, image_tensor = adapter.load_detection_image(frame.path)
        cache.put(frame.path, image_source)
        frame_rows: list[dict[str, Any]] = []
        for prompt in prompt_texts:
            for detection in adapter.detect_loaded(image_source, image_tensor, prompt):
                category = class_for_label(detection.label, config.classes, config.mode)
                if not category:
                    continue
                bbox = [float(value) for value in detection.bbox_xyxy]
                frame_rows.append(
                    {
                        "scene_id": config.scene_id,
                        "frame_id": int(frame.frame_id),
                        "camera_name": frame.camera_name,
                        "category": category,
                        "label": detection.label,
                        "image_path": frame.path,
                        "bbox_xyxy": bbox,
                        "score": float(detection.score),
                    }
                )
        for row in _nms(frame_rows, 0.45, 3):
            row["pre_score"] = _pre_score(row, image_source.shape)
            raw_rows.append(row)

    tracklets = _tracklets(raw_rows)
    tracklet_rows = {f"t{index:06d}": rows for index, rows in enumerate(tracklets, start=1)}
    sam_rows = _select_for_sam(tracklets, config.sam_candidates_per_tracklet, config.max_sam_candidates,
                             config.vehicle_sam_candidates)
    cache.clear()

    segmented: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sam_rows:
        by_image[row["image_path"]].append(row)
    for image_path, rows in by_image.items():
        image = cache.get_rgb(image_path)
        segments = adapter.segment_loaded_image(image, [row["bbox_xyxy"] for row in rows])
        for row, segment in zip(rows, segments):
            mask, components, largest_ratio = _clean_mask(segment.mask, row["bbox_xyxy"])
            keep, quality, reason = _mask_quality(row, mask, largest_ratio, image.shape, config)
            row["sam_score"] = segment.sam_score
            row["quality"] = quality
            row["num_components"] = components
            row["_mask"] = mask
            if keep:
                segmented.append(row)
            else:
                rejects.append(
                    {
                        "scene_id": config.scene_id,
                        "image_path": str(Path(image_path).resolve()),
                        "category": row["category"],
                        "reason": reason,
                    }
                )

    best_by_tracklet: dict[str, dict[str, Any]] = {}
    for row in segmented:
        tracklet_id = str(row["tracklet_id"])
        previous = best_by_tracklet.get(tracklet_id)
        if previous is None or float(row["quality"]) > float(previous["quality"]):
            best_by_tracklet[tracklet_id] = row

    selected = sorted(best_by_tracklet.values(), key=lambda item: (item["category"], item["tracklet_id"]))

    return {
        "frames": frames,
        "cache": cache,
        "raw_rows": raw_rows,
        "tracklets": tracklets,
        "tracklet_rows": tracklet_rows,
        "sam_rows": sam_rows,
        "segmented": segmented,
        "selected": selected,
        "rejects": rejects,
    }


def _source_record(config: SceneExportConfig, obj_id: str, row: dict[str, Any], track: list[dict[str, Any]]) -> dict[str, Any]:
    frame_records = [
        {
            "frame_index": int(item["frame_id"]),
            "source_image": str(Path(item["image_path"]).resolve()),
            "bbox_xyxy": [round(float(value), 3) for value in item["bbox_xyxy"]],
            "score": round(float(item["score"]), 6),
        }
        for item in sorted(track, key=lambda item: int(item["frame_id"]))
    ]
    return {
        "asset_id": obj_id,
        "scene_id": config.scene_id,
        "camera_name": str(row["camera_name"]),
        "track_id": str(row["tracklet_id"]),
        "frame_index": int(row["frame_id"]),
        "image_path": None,
        "quality": {
            "pre_score": float(row["pre_score"]),
            "sam_score": float(row.get("sam_score") or 0.0),
            "mask_quality": float(row["quality"]),
            "tracklet_len": int(row.get("tracklet_len") or len(track)),
        },
        "source": {
            "candidate_id": obj_id,
            "scene_id": config.scene_id,
            "camera_name": str(row["camera_name"]),
            "frame_index": int(row["frame_id"]),
            "source_image": str(Path(row["image_path"]).resolve()),
            "bbox_xyxy": [round(float(value), 3) for value in row["bbox_xyxy"]],
            "tracklet_id": str(row["tracklet_id"]),
            "frame_records": frame_records,
        },
    }


def run_scene_asset_export(config: SceneExportConfig) -> dict[str, Any]:
    started = time.perf_counter()
    diffusion_dir, images_dir = _prepare_scene_dir(config.output_dir)
    extracted = _extract_scene_assets(config)
    cache = extracted["cache"]
    selected = extracted["selected"]
    tracklet_rows = extracted["tracklet_rows"]

    manifest: list[dict[str, Any]] = []
    export_items: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        obj_id = f"{config.scene_id}_t{index:06d}"
        destination = images_dir / f"{obj_id}.jpg"
        image = cache.get_rgb(row["image_path"])
        _export_white_jpg(
            image, row["_mask"], row["bbox_xyxy"], destination,
            config.export_max_size, config.jpg_quality,
        )
        manifest.append({
            "scene_id": config.scene_id,
            "obj_id": obj_id,
            "category": str(row["category"]),
            "image_path": str(destination.resolve()),
            "prompts": [],
        })
        source_item = _source_record(config, obj_id, row, tracklet_rows[str(row["tracklet_id"])])
        source_item["image_path"] = str(destination.resolve())
        if row['category'] in VEHICLES:
            candidates = [r for r in extracted['segmented'] if r['tracklet_id']==row['tracklet_id']]
            references, selection = select_real_views(candidates, row, config.real_views_max,
                                                       config.real_view_quality_ratio)
            real_images = []
            seen_content = set()
            from PIL import Image
            # Keep RGB and alpha separate from the unchanged FLUX JPG contract.
            for reference in references:
                rgb = cache.get_rgb(reference['image_path'])
                h,w = rgb.shape[:2]
                crop = _crop_box(reference['_mask'], reference['bbox_xyxy'], w, h)
                rgba = Image.fromarray(rgb).convert('RGBA')
                rgba.putalpha(Image.fromarray(reference['_mask'].astype('uint8')*255))
                rgba = rgba.crop(crop)
                rgba.thumbnail((config.export_max_size, config.export_max_size), Image.Resampling.LANCZOS)
                digest = hashlib.sha256(rgba.tobytes()).hexdigest()
                if digest in seen_content:
                    continue
                seen_content.add(digest)
                path = images_dir / f'{obj_id}_real_{len(real_images):02}.png'
                rgba.save(path)
                real_images.append(dict(rgba_path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    source_image=str(Path(reference['image_path']).resolve()), frame_index=int(reference['frame_id']),
                    camera_name=reference['camera_name'], bbox_xyxy=reference['bbox_xyxy'], crop_box=list(crop),
                    quality=float(reference['quality']), sam_score=float(reference.get('sam_score') or 0)))
            selection['real_count'] = len(real_images)
            if len(real_images)<len(references):
                selection['reason'] = 'duplicate_pixel_content_removed'
                selection['status'] = 'preferred' if len(real_images)>=3 else ('two_view_fallback' if len(real_images)==2 else 'single_view_fallback')
            source_item.update(real_images=real_images, real_view_selection=selection)
        export_items.append(source_item)

    write_json(diffusion_dir / "manifest.json", manifest)
    write_json(config.output_dir / "logs" / "export_items_full.json", export_items)
    write_jsonl(config.output_dir / "logs" / "rejects.jsonl", extracted["rejects"])
    status = {
        "stage": "scene_image_export",
        "status": "completed",
        "scene_id": config.scene_id,
        "mode": "rigid",
        "num_images": len(extracted["frames"]),
        "num_raw_detections": len(extracted["raw_rows"]),
        "num_tracklets": len(extracted["tracklets"]),
        "num_sam_candidates": len(extracted["sam_rows"]),
        "num_kept": len(manifest),
        "num_dropped": len(extracted["sam_rows"]) - len(extracted["segmented"]),
        "elapsed_sec": round(time.perf_counter() - started, 6),
    }
    write_json(config.output_dir / "status.json", status)
    write_json(config.output_dir / "profile" / f"{config.scene_id}_profile.json", status)
    return {
        "manifest_path": str(diffusion_dir / "manifest.json"),
        "images_dir": str(images_dir),
        "items": manifest,
        "status": status,
    }
