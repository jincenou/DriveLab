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
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir, write_json, write_jsonl
from .real_views import VEHICLES, diverse_order, select_real_views


@dataclass(frozen=True)
class FrameImage:
    path: str
    camera_name: str
    frame_id: int
    timestamp_us: int | None = None
    camera_intrinsic: list | None = None
    camera_to_world: list | None = None
    motion_paths: list[str] = field(default_factory=list)


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
    max_asset_candidates: int
    sam_candidates_per_tracklet: int
    dataset: str = "nuscenes"
    camera_channels: list[str] = field(default_factory=list)
    frame_stride: int = 1
    export_max_size: int = 768
    jpg_quality: int = 92
    vehicle_sam_candidates: int = 6
    real_views_max: int = 4
    real_view_quality_ratio: float = 0.70
    max_detections_per_frame: int = 0
    track_max_gap: int = 3
    track_iou_threshold: float = 0.25
    cross_camera_dedup: bool = True
    max_mask_area_ratio: float = 0.80


@dataclass
class DetectionResult:
    bbox_xyxy: list[float]
    score: float
    label: str
    mask: Any = None
    sam_score: float | None = None


def _token(value: Any) -> str:
    return re.sub(r"_+", "_", str(value or "").strip().lower().replace("-", "_").replace(" ", "_")).strip("_")


def normalize_rigid_class(value: Any) -> str:
    aliases = {
        "garbage_bin": "trash_can",
        "rubbish_bin": "trash_can",
        "fire_hydrant": "hydrant",
        "motorbike": "motorcycle",
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


def collect_frame_images(
    scene_index_path: str | Path,
    *,
    cameras: list[str] | None = None,
    frame_stride: int = 1,
) -> list[FrameImage]:
    index_path = Path(scene_index_path)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    frames = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(frames, list):
        raise ValueError(f"Invalid scene frame index: {index_path}")
    wanted = {_camera_name(camera) for camera in (cameras or [])}
    by_camera: dict[str, list[FrameImage]] = defaultdict(list)
    for row in frames:
        camera_name = _camera_name(row.get("camera_name"))
        if wanted and camera_name not in wanted:
            continue
        image_path = Path(str(row.get("path") or "")).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Camera image missing: {image_path}")
        calibration = {key: row.get(key) for key in
                       ("timestamp_us", "camera_intrinsic", "camera_to_world")}
        if calibration["timestamp_us"] is not None:
            calibration["timestamp_us"] = int(calibration["timestamp_us"])
        by_camera[camera_name].append(
            FrameImage(str(image_path), camera_name, int(row["frame_id"]),
                       motion_paths=list(row.get("motion_paths", [])), **calibration)
        )
    stride = max(1, int(frame_stride))
    selected = []
    for camera in sorted(by_camera):
        ordered = sorted(by_camera[camera],key=lambda item:(item.frame_id,item.path))
        previous = -1
        for index in range(0,len(ordered),stride):
            paths = [path for row in ordered[previous+1:index] for path in [*row.motion_paths,row.path]]
            selected.append(replace(ordered[index],motion_paths=[*paths,*ordered[index].motion_paths]))
            previous = index
    return sorted(selected, key=lambda item: (item.camera_name, item.frame_id, item.path))


def collect_scene_images(config: SceneExportConfig) -> list[FrameImage]:
    return collect_frame_images(
        config.scene_path,
        cameras=config.camera_channels,
        frame_stride=config.frame_stride,
    )


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
    "vehicle": "car. truck. bus. van. trailer.",
    "bicycle": "bicycle.",
    "motorcycle": "motorcycle. motorbike.",
    "trash_can": "trash can. garbage bin. rubbish bin.",
    "hydrant": "fire hydrant.",
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
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bicycle": "bicycle",
    "trash can": "trash_can",
    "garbage bin": "trash_can",
    "rubbish bin": "trash_can",
    "fire hydrant": "hydrant",
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
    if not prompts:
        raise ValueError("No supported detection categories configured")
    return prompts


def class_for_label(label: str, allowed: list[str], mode: str) -> str | None:
    normalized_allowed = {normalize_rigid_class(item) for item in allowed}
    text = " ".join(label.strip().lower().replace("_", " ").rstrip(".").split())
    for phrase in sorted(LABEL_TO_CLASS, key=len, reverse=True):
        class_name = LABEL_TO_CLASS[phrase]
        if class_name in normalized_allowed and re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text):
            return "car" if class_name == "vehicle" else class_name
    return None


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
        if any(row["category"] == other["category"] and
               _bbox_iou(row["bbox_xyxy"], other["bbox_xyxy"]) >= iou_threshold for other in kept):
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


def _appearance(image: Any, bbox: list[float]) -> list[float]:
    """Small RGB histogram of the box center; no extra learned tracker."""
    import numpy as np

    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    dx, dy = (x2 - x1) * .15, (y2 - y1) * .15
    crop = image[max(0, int(y1 + dy)):min(h, math.ceil(y2 - dy)),
                 max(0, int(x1 + dx)):min(w, math.ceil(x2 - dx))]
    if crop.size == 0:
        raise ValueError("Empty detection crop")
    counts = np.concatenate([np.histogram(crop[..., c], bins=8, range=(0, 256))[0]
                             for c in range(3)]).astype(float)
    return (counts / counts.sum()).tolist()


def _appearance_similarity(a: dict, b: dict) -> float | None:
    if "appearance" not in a or "appearance" not in b:
        return None
    return sum(math.sqrt(x * y) for x, y in zip(a["appearance"], b["appearance"], strict=True))


def _motion_boxes(rows: list[dict], cache: ImageCache, max_gap: int) -> None:
    """Predict local box motion with batched sparse LK flow, without learned weights."""
    import cv2
    import numpy as np

    cameras = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cameras[row["camera_name"]][int(row["frame_id"])].append(row)
    for frames in cameras.values():
        gray_cache = OrderedDict()
        def gray(path):
            if path not in gray_cache:
                gray_cache[path] = cv2.cvtColor(cache.get_rgb(path),cv2.COLOR_RGB2GRAY)
            gray_cache.move_to_end(path)
            image = gray_cache[path]
            if len(gray_cache)>48:
                gray_cache.popitem(last=False)
            return image
        for frame, observations in sorted(frames.items()):
            source_image = gray(observations[0]['image_path'])
            points, owners = [], []
            for index, row in enumerate(observations):
                x1,y1,x2,y2 = [int(v) for v in row["bbox_xyxy"]]
                corners = cv2.goodFeaturesToTrack(source_image[y1:y2,x1:x2], 24, .01, 2)
                if corners is not None:
                    corners = corners.reshape(-1,2) + [x1,y1]
                    points.extend(corners)
                    owners.extend([index]*len(corners))
            if not points:
                continue
            points = np.asarray(points,dtype=np.float32).reshape(-1,1,2)
            owners = np.asarray(owners)
            for target in sorted(f for f in frames if 0 < f-frame <= max_gap):
                paths = [observations[0]['image_path']]
                for step in sorted(f for f in frames if frame<f<=target):
                    row = frames[step][0]
                    paths.extend([*row.get('motion_paths',[]),row['image_path']])
                moved,valid = points.copy(),np.ones(len(points),dtype=bool)
                for start,end in zip(paths[:-1],paths[1:]):
                    moved,status,_ = cv2.calcOpticalFlowPyrLK(gray(start),gray(end),moved,None,winSize=(15,15),maxLevel=4)
                    valid &= status.ravel().astype(bool)
                returned = moved.copy()
                for end,start in zip(paths[:0:-1],paths[-2::-1]):
                    returned,status,_ = cv2.calcOpticalFlowPyrLK(gray(end),gray(start),returned,None,winSize=(15,15),maxLevel=4)
                    valid &= status.ravel().astype(bool)
                valid &= np.linalg.norm(points[:,0]-returned[:,0],axis=1) <= 1.5
                for index, row in enumerate(observations):
                    chosen = (owners==index) & valid
                    if chosen.sum() < 4 or chosen.sum() < .5*(owners==index).sum():
                        continue
                    affine, inliers = cv2.estimateAffinePartial2D(points[chosen,0],moved[chosen,0],
                        method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=500)
                    if affine is None or inliers.sum() < max(4,.6*chosen.sum()):
                        continue
                    scale = np.linalg.norm(affine[:,0])
                    if not .5 <= scale <= 2.:
                        continue
                    x1,y1,x2,y2 = row["bbox_xyxy"]
                    corners = np.asarray([[x1,y1,1],[x2,y1,1],[x2,y2,1],[x1,y2,1]]) @ affine.T
                    box = [*corners.min(axis=0),*corners.max(axis=0)]
                    row.setdefault("motion_boxes",{})[str(target)] = list(map(float,box))


def _project_static_box(previous: dict, point: Any, target: dict) -> tuple[list[float], float]:
    import numpy as np

    a,b = np.asarray(previous["camera_to_world"]),np.asarray(target["camera_to_world"])
    old = a[:3,:3].T @ (point-a[:3,3])
    new = b[:3,:3].T @ (point-b[:3,3])
    if min(old[2],new[2]) <= 1.:
        return [], 0.
    pixel = np.asarray(target["camera_intrinsic"]) @ new
    x,y = pixel[:2]/pixel[2]
    x1,y1,x2,y2 = previous["bbox_xyxy"]
    scale = old[2]/new[2]
    width,height = (x2-x1)*scale,(y2-y1)*scale
    box = [x-width/2,y-height/2,x+width/2,y+height/2]
    tx1,ty1,tx2,ty2 = target["bbox_xyxy"]
    center_error = math.hypot((x-(tx1+tx2)/2)/max(1.,tx2-tx1),
                               (y-(ty1+ty2)/2)/max(1.,ty2-ty1))
    return box, _bbox_iou(box,target["bbox_xyxy"]) if center_error <= .2 else 0.


def _static_overlap(track: list[dict], row: dict, future: dict[int,list[dict]] | None = None) -> float:
    """Require a third-view reprojection before accepting a static geometric link."""
    previous = track[-1]
    point = _triangulate(track[0] if len(track)>1 else row,previous,max_ray_cos=.99999)
    if point is None:
        return 0.
    _, overlap = _project_static_box(previous,point,row)
    if overlap < .5:
        return 0.
    if len(track)>1:
        return overlap  # The candidate is already a third or later observation.
    # Two rays alone can manufacture a 3D point from two different, similar lights.
    for observations in (future or {}).values():
        matches = [item for item in observations if
            all(item.get(key) is not None for key in ("camera_intrinsic","camera_to_world")) and
            _project_static_box(previous,point,item)[1] >= .5 and
            (_appearance_similarity(previous,item) is None or _appearance_similarity(previous,item) >= .8)]
        if len(matches)==1:
            return overlap
        if len(matches)>1:
            return 0.
    return 0.


def _stitch_static_tracks(tracks: list[list[dict]], max_gap: int) -> list[list[dict]]:
    """Attach short fragments only to a unique, independently confirmed 3D track."""
    anchors = {}
    for i,track in enumerate(tracks):
        if len(track)>=3 and track[0]["category"] not in (VEHICLES | {"bicycle","motorcycle"}) and all(
                row.get(key) is not None for row in track for key in ("camera_intrinsic","camera_to_world")):
            point = _triangulate(track[0],track[-1],max_ray_cos=.99999)
            if point is not None and all(_project_static_box(track[0],point,row)[1]>=.5 for row in track):
                anchors[i] = point
    frame_sets = [{row["frame_id"] for row in track} for track in tracks]
    edges = []
    for j,fragment in enumerate(tracks):
        if len(fragment)>2:
            continue
        for i,point in anchors.items():
            anchor = tracks[i]
            if (anchor[0]["category"],anchor[0]["camera_name"]) != (fragment[0]["category"],fragment[0]["camera_name"]) or frame_sets[i]&frame_sets[j]:
                continue
            if min(abs(a-b) for a in frame_sets[i] for b in frame_sets[j])>max_gap:
                continue
            if not all(row.get(key) is not None for row in fragment for key in ("camera_intrinsic","camera_to_world")):
                continue
            appearance = [_appearance_similarity(anchor[0],row) for row in fragment]
            if any(value is not None and value<.8 for value in appearance):
                continue
            score = min(_project_static_box(anchor[0],point,row)[1] for row in fragment)
            if score>=.5:
                edges.append((score,i,j))
    merged = set()
    for score,i,j in sorted(edges,reverse=True):
        rivals = [other for other,ai,fj in edges if (ai,fj)!=(i,j) and
                  (fj==j or (ai==i and bool(frame_sets[fj]&frame_sets[j])))]
        if j in merged or frame_sets[i]&frame_sets[j] or (rivals and score-max(rivals)<.06):
            continue
        tracks[i] = sorted(tracks[i]+tracks[j],key=lambda row:row["frame_id"])
        frame_sets[i] |= frame_sets[j]
        merged.add(j)
    return [track for i,track in enumerate(tracks) if i not in merged]


def _tracklets(rows: list[dict[str, Any]], max_gap: int = 3,
               iou_threshold: float = .25) -> list[list[dict[str, Any]]]:
    """One detection per camera/frame/track; moving objects require measured flow.

    max_gap is in source frame indices. Similar color and overlapping boxes do
    not establish vehicle identity when neighboring cars pass the same pixels.
    """
    buckets: dict[tuple[str, str], dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        buckets[(row["category"], row["camera_name"])][int(row["frame_id"])].append(row)
    result: list[list[dict[str, Any]]] = []
    for _, frames in sorted(buckets.items()):
        active: list[list[dict]] = []
        for frame_id, detections in sorted(frames.items()):
            active = [track for track in active if frame_id - int(track[-1]["frame_id"]) <= max_gap]
            edges = []
            for i, track in enumerate(active):
                previous = track[-1]
                moving = previous["category"] in (VEHICLES | {"bicycle", "motorcycle"})
                box = previous["bbox_xyxy"]
                predicted = box
                motion_box = previous.get("motion_boxes", {}).get(str(frame_id))
                if motion_box is not None:
                    predicted = motion_box
                elif len(track) >= 2:
                    older = track[-2]
                    ratio = (frame_id - previous["frame_id"]) / (previous["frame_id"] - older["frame_id"])
                    predicted = [a + ratio * (a - b) for a, b in zip(box, older["bbox_xyxy"])]
                for j, row in enumerate(detections):
                    area_ratio = _bbox_area(row["bbox_xyxy"]) / max(1., _bbox_area(box))
                    overlap = _bbox_iou(predicted, row["bbox_xyxy"])
                    appearance = _appearance_similarity(previous, row)
                    if not .4 <= area_ratio <= 2.5:
                        continue
                    if moving and (motion_box is None or overlap < .5):
                        continue
                    if appearance is not None and appearance < .70:
                        continue
                    static_match = not moving and all(
                        item.get(key) is not None for item in [*track,row]
                        for key in ("camera_intrinsic","camera_to_world"))
                    if static_match:
                        if appearance is not None and appearance < .8:
                            continue
                        future = {f: items for f,items in frames.items() if 0 < f-frame_id <= max_gap}
                        geometry = _static_overlap(track,row,future)
                        overlap = max(geometry,overlap if motion_box is not None and overlap>=.7 else 0.)
                    if overlap < iou_threshold:
                        continue
                    score = .8 * overlap + .2 * _bbox_iou(box, row["bbox_xyxy"])
                    if static_match:
                        target = row["bbox_xyxy"]
                        distance = math.hypot((box[0]+box[2]-target[0]-target[2])/2,
                                              (box[1]+box[3]-target[1]-target[3])/2)
                        score -= min(.25, .03*distance/max(1.,math.sqrt(_bbox_area(box))))
                    if appearance is not None:
                        score = .8 * score + .2 * appearance
                    edges.append((score, i, j))
            used_rows = set()
            for score,i,j in sorted(edges,reverse=True):
                # Mutual best with a margin; ambiguous detections start separate
                # hypotheses instead of switching to a runner-up identity.
                alternatives = [other for other,ti,dj in edges if
                                (ti==i or dj==j) and (ti,dj)!=(i,j)]
                if alternatives and score-max(alternatives) < .06:
                    continue
                active[i].append(detections[j])
                used_rows.add(j)
            for j, row in enumerate(detections):
                if j not in used_rows:
                    track = [row]
                    active.append(track)
                    result.append(track)
    return _stitch_static_tracks(result,max_gap)


def _triangulate(a: dict, b: dict, max_ray_cos: float = .999):
    """Intersect calibrated box-center rays only when their geometry is usable."""
    import numpy as np

    origins, rays = [], []
    for row in (a, b):
        transform = np.asarray(row["camera_to_world"], dtype=float)
        intrinsic = np.asarray(row["camera_intrinsic"], dtype=float)
        if transform.shape != (4, 4) or intrinsic.shape != (3, 3):
            raise ValueError("Expected camera_to_world 4x4 and camera_intrinsic 3x3")
        x1, y1, x2, y2 = row["bbox_xyxy"]
        ray = transform[:3, :3] @ np.linalg.solve(intrinsic, [(x1+x2)/2, (y1+y2)/2, 1.])
        origins.append(transform[:3, 3])
        rays.append(ray / np.linalg.norm(ray))
    if abs(float(rays[0] @ rays[1])) > max_ray_cos:
        return None  # Near-parallel rays do not establish a reliable identity.
    depths = np.linalg.lstsq(np.stack([rays[0], -rays[1]], axis=1), origins[1]-origins[0], rcond=None)[0]
    if not np.all((depths >= 1.) & (depths <= 80.)):
        return None
    points = np.asarray(origins) + depths[:, None] * np.asarray(rays)
    if np.linalg.norm(points[0]-points[1]) > min(.35, .015 * float(depths.min())):
        return None
    return points.mean(axis=0)


def _cross_camera_tracklets(tracks: list[list[dict]]) -> tuple[list[list[dict]], int]:
    """Conservative optional merge: synchronized calibration, color, repeated geometry.

    Three matched observations and a unique reciprocal track match are required.
    This is a lightweight association heuristic, not ground-truth identity tracking.
    """
    import numpy as np

    calibrated = [all(row.get(key) is not None for row in track for key in
                      ("timestamp_us", "camera_intrinsic", "camera_to_world", "appearance"))
                  for track in tracks]
    matches: dict[int, set[int]] = defaultdict(set)
    for i, a in enumerate(tracks):
        if not calibrated[i] or len(a) < 3:
            continue
        for j in range(i + 1, len(tracks)):
            b = tracks[j]
            if not calibrated[j] or len(b) < 3 or a[0]["category"] != b[0]["category"] or a[0]["camera_name"] == b[0]["camera_name"]:
                continue
            support, used = [], set()
            for first in a:
                second = min(b, key=lambda row: abs(row["timestamp_us"] - first["timestamp_us"]))
                if second["frame_id"] in used or abs(first["timestamp_us"] - second["timestamp_us"]) > 100_000:
                    continue
                if _appearance_similarity(first, second) < .92:
                    continue
                point = _triangulate(first, second)
                if point is not None:
                    support.append((first["timestamp_us"], point))
                    used.add(second["frame_id"])
            if len(support) < 3 or len(support) < .6 * min(len(a), len(b)):
                continue
            support.sort(key=lambda pair: pair[0])
            times = (np.asarray([s[0] for s in support]) - support[0][0]) / 1e6
            if times[-1] < .5:
                continue
            points = np.asarray([s[1] for s in support])
            if a[0]["category"] in VEHICLES:
                model = np.stack([np.ones(len(times)), times], axis=1)
                coefficients = np.linalg.lstsq(model, points, rcond=None)[0]
                residual = np.linalg.norm(points - model @ coefficients, axis=1).max()
                valid = residual <= .4 and np.linalg.norm(coefficients[1]) <= 25.
            else:
                valid = np.linalg.norm(points - points.mean(axis=0), axis=1).max() <= .4
            if valid:
                matches[i].add(j)
                matches[j].add(i)
    result, merged = [], set()
    for i, track in enumerate(tracks):
        if i in merged:
            continue
        pair = next(iter(matches[i])) if len(matches[i]) == 1 else None
        if pair is not None and matches[pair] == {i} and pair not in merged:
            track = sorted(track + tracks[pair], key=lambda row: (row["frame_id"], row["camera_name"]))
            merged.add(pair)
        result.append(track)
    return result, len(tracks)-len(result)


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
    # Budget identities, then retain the full evidence allowance for each one.
    queues.sort(key=lambda q: float(q[0]['pre_score']), reverse=True)
    queues = queues[:max(1,limit)]
    return [q[i] for i in range(max((len(q) for q in queues), default=0)) for q in queues if i<len(q)]


def _clean_mask(mask: Any, bbox: list[float]) -> tuple[Any, int, float]:
    import numpy as np

    binary = np.asarray(mask, dtype="uint8")
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary.astype(bool), 0, 1.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    largest = labels == keep
    ratio = float(areas.max()) / max(1.0, float(areas.sum()))
    return largest, int(count - 1), ratio


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
    if area_ratio > config.max_mask_area_ratio:
        return False, 0.0, "mask_too_large"
    if largest_ratio < config.min_largest_component_ratio:
        return False, 0.0, "fragmented_mask"
    # A long cut along one source-frame edge must not be diluted by the other
    # three sides. A 3px band also catches SAM masks ending 1-2px inside the frame.
    import numpy as np
    ys, xs = np.nonzero(mask)
    mask_width, mask_height = int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)
    edge_fraction = max(float(mask[:, :3].any(axis=1).sum()) / mask_height,
                        float(mask[:, -3:].any(axis=1).sum()) / mask_height,
                        float(mask[:3, :].any(axis=0).sum()) / mask_width,
                        float(mask[-3:, :].any(axis=0).sum()) / mask_width)
    row['source_frame_edge'] = dict(contact_fraction=round(edge_fraction,6), band_pixels=3,
                                   limit=config.max_edge_touch_ratio)
    row['real_view_eligible'] = edge_fraction <= config.max_edge_touch_ratio
    contact = int(mask[0].sum() + mask[-1].sum() + mask[:, 0].sum() + mask[:, -1].sum())
    perimeter = 2 * (mask_width + mask_height)
    if row['category'] not in VEHICLES and contact / max(1, perimeter) > config.max_edge_touch_ratio:
        return False, 0.0, "mask_truncated_at_image_edge"
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
    coverage = dict(detector_candidates=0, bbox_rejected=0, nms_or_cap_removed=0)
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
                coverage["detector_candidates"] += 1
                bbox = [float(value) for value in detection.bbox_xyxy]
                height, width = image_source.shape[:2]
                if not all(math.isfinite(value) for value in bbox):
                    raise ValueError("Detector produced a nonfinite bounding box")
                bbox = [max(0., min(value, float(width if i % 2 == 0 else height)))
                        for i, value in enumerate(bbox)]
                if bbox[2]-bbox[0] < config.min_bbox_width or bbox[3]-bbox[1] < config.min_bbox_height:
                    coverage["bbox_rejected"] += 1
                    continue
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
                        "timestamp_us": frame.timestamp_us,
                        "camera_intrinsic": frame.camera_intrinsic,
                        "camera_to_world": frame.camera_to_world,
                        "motion_paths": frame.motion_paths,
                    }
                )
        retained = _nms(frame_rows, 0.45, config.max_detections_per_frame)
        coverage["nms_or_cap_removed"] += len(frame_rows)-len(retained)
        for row in retained:
            row["pre_score"] = _pre_score(row, image_source.shape)
            row["appearance"] = _appearance(image_source, row["bbox_xyxy"])
            raw_rows.append(row)

    _motion_boxes(raw_rows, cache, config.track_max_gap)
    write_json(config.output_dir / "logs" / "detections.json", raw_rows)
    tracklets = _tracklets(raw_rows, config.track_max_gap, config.track_iou_threshold)
    coverage["same_camera_tracklets"] = len(tracklets)
    coverage["track_length_histogram"] = dict(Counter(str(min(4,len(track))) for track in tracklets))
    coverage["track_length_histogram_last_bin"] = "4 means four or more observations"
    coverage["tracklets_by_category"] = dict(Counter(track[0]["category"] for track in tracklets))
    coverage["singleton_tracklets_by_category"] = dict(Counter(track[0]["category"] for track in tracklets if len(track)==1))
    coverage["cross_camera_merged"] = 0
    coverage["cross_camera_policy"] = "calibrated_repeated_unique_match" if config.cross_camera_dedup else "disabled"
    coverage["calibrated_observations"] = sum(all(row.get(key) is not None for key in
        ("timestamp_us", "camera_intrinsic", "camera_to_world")) for row in raw_rows)
    if config.cross_camera_dedup:
        tracklets, coverage["cross_camera_merged"] = _cross_camera_tracklets(tracklets)
    tracklet_rows = {f"t{index:06d}": rows for index, rows in enumerate(tracklets, start=1)}
    sam_rows = _select_for_sam(tracklets, config.sam_candidates_per_tracklet, config.max_asset_candidates,
                             config.vehicle_sam_candidates)
    coverage["identities_not_sampled_for_sam"] = len(tracklets)-len({row["tracklet_id"] for row in sam_rows})
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
        if previous is None or (row['real_view_eligible'],float(row['quality'])) > (
                previous['real_view_eligible'],float(previous['quality'])):
            best_by_tracklet[tracklet_id] = row

    selected = sorted(best_by_tracklet.values(), key=lambda item: (item["category"], item["tracklet_id"]))
    extras, attempted = _supplement_vehicle_views(selected, segmented, sam_rows, frames, cache, adapter, config)
    segmented.extend(extras)
    sam_rows.extend(attempted)
    for row in extras:
        tracklet_rows[row['tracklet_id']].append(row)
    coverage['intermediate_real_candidates'] = len(attempted)
    coverage['intermediate_real_accepted'] = len(extras)

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
        "coverage": coverage,
    }


def _supplement_vehicle_views(anchors, segmented, sampled, frames, cache, adapter, config):
    """Recover nearby real evidence with foreground LK and SAM, no extra detection.

    The original best image stays fixed. Each extra observation must track
    directly from its foreground and agree with a fresh SAM mask.
    """
    import cv2
    import numpy as np

    streams = defaultdict(list)
    for frame in frames:
        streams[frame.camera_name].append(frame)
    neighbors = {}
    for stream in streams.values():
        ordered = sorted(stream, key=lambda f:f.frame_id)
        for i, frame in enumerate(ordered):
            paths = []
            if i:
                before = frame.motion_paths
                paths += [(list(reversed(before[j:])),
                           ordered[i-1].frame_id+(frame.frame_id-ordered[i-1].frame_id)*(j+1)/(len(before)+1))
                          for j in range(len(before))]
            if i+1 < len(ordered):
                after = ordered[i+1].motion_paths
                paths += [(after[:j+1], frame.frame_id+(ordered[i+1].frame_id-frame.frame_id)*(j+1)/(len(after)+1))
                          for j in range(len(after))]
            neighbors[frame.path] = paths
    pending = defaultdict(list)
    for anchor in anchors:
        track = anchor['tracklet_id']
        existing = [r for r in segmented if r['tracklet_id']==track and r.get('real_view_eligible',True)]
        budget = config.vehicle_sam_candidates-sum(r['tracklet_id']==track for r in sampled)
        if (anchor['category'] not in VEHICLES or not anchor.get('real_view_eligible',True)
                or len(existing)>=config.real_views_max or budget<=0):
            continue
        start = cv2.cvtColor(cache.get_rgb(anchor['image_path']),cv2.COLOR_RGB2GRAY)
        mask = cv2.erode(anchor['_mask'].astype('uint8')*255,np.ones((3,3),'uint8'))
        points = cv2.goodFeaturesToTrack(start,80,.01,3,mask=mask)
        if points is None or len(points)<10:
            continue
        options = [dict(paths=paths,frame_id=frame,quality=anchor['quality'],camera_name=anchor['camera_name'])
                   for paths,frame in neighbors.get(anchor['image_path'],[]) if paths[-1] not in {r['image_path'] for r in existing}]
        options = diverse_order(options, dict(frame_id=anchor['frame_id'],quality=anchor['quality'],
                                camera_name=anchor['camera_name']))[1:]
        accepted_boxes = 0
        for option in options:
            if accepted_boxes >= budget:
                break
            gray = [start]+[cv2.cvtColor(cache.get_rgb(p),cv2.COLOR_RGB2GRAY) for p in option['paths']]
            moved, valid = points.copy(), np.ones(len(points),dtype=bool)
            for a,b in zip(gray[:-1],gray[1:]):
                moved,status,_ = cv2.calcOpticalFlowPyrLK(a,b,moved,None,winSize=(15,15),maxLevel=4)
                valid &= status.ravel().astype(bool)
            returned = moved.copy()
            for b,a in zip(gray[:0:-1],gray[-2::-1]):
                returned,status,_ = cv2.calcOpticalFlowPyrLK(b,a,returned,None,winSize=(15,15),maxLevel=4)
                valid &= status.ravel().astype(bool)
            valid &= np.linalg.norm(points[:,0]-returned[:,0],axis=1)<=1.
            if valid.sum()<max(10,.6*len(points)):
                continue
            affine,inliers = cv2.estimateAffinePartial2D(points[valid,0],moved[valid,0],
                method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=500)
            if affine is None or inliers.sum()<max(10,.7*valid.sum()) or not .65<=np.linalg.norm(affine[:,0])<=1.5:
                continue
            x1,y1,x2,y2=anchor['bbox_xyxy']
            support=points[valid,0][inliers.ravel().astype(bool)]
            if np.any(np.ptp(support,axis=0)/np.maximum(1,[x2-x1,y2-y1]) < [.35,.25]):
                continue
            corners=np.asarray([[x1,y1,1],[x2,y1,1],[x2,y2,1],[x1,y2,1]])@affine.T
            height,width=gray[-1].shape
            box=np.clip([*corners.min(axis=0),*corners.max(axis=0)],0,[width,height,width,height]).tolist()
            candidate=dict(anchor,image_path=option['paths'][-1],frame_id=option['frame_id'],bbox_xyxy=box,
                timestamp_us=None,camera_to_world=None,motion_paths=[],
                observation_method='foreground_lk_sam',association_source=anchor['image_path'],
                _predicted_mask=cv2.warpAffine(mask,affine,(width,height),flags=cv2.INTER_NEAREST)>127)
            candidate.pop('_mask')
            pending[candidate['image_path']].append(candidate)
            accepted_boxes += 1
    kept, attempted = [], []
    for path, rows in pending.items():
        image=cache.get_rgb(path)
        segments=adapter.segment_loaded_image(image,[r['bbox_xyxy'] for r in rows])
        for row, segment in zip(rows,segments):
            attempted.append(row)
            mask,components,ratio=_clean_mask(segment.mask,row['bbox_xyxy'])
            predicted=row.pop('_predicted_mask')
            agreement=float(np.logical_and(mask,predicted).sum())/max(1,np.logical_or(mask,predicted).sum())
            good,quality,_=_mask_quality(row,mask,ratio,image.shape,config)
            if good and row['real_view_eligible'] and agreement>=.6:
                row.update(_mask=mask,quality=quality,sam_score=segment.sam_score,num_components=components,
                           propagated_mask_iou=agreement)
                kept.append(row)
    return kept, attempted


def _source_record(config: SceneExportConfig, obj_id: str, row: dict[str, Any], track: list[dict[str, Any]]) -> dict[str, Any]:
    frame_records = [
        {
            "frame_index": item["frame_id"],
            "source_image": str(Path(item["image_path"]).resolve()),
            "bbox_xyxy": [round(float(value), 3) for value in item["bbox_xyxy"]],
            "score": round(float(item["score"]), 6),
            "camera_name": str(item["camera_name"]),
            **{key: item[key] for key in ("timestamp_us", "camera_intrinsic", "camera_to_world",
                                         "observation_method", "association_source", "propagated_mask_iou")
               if item.get(key) is not None},
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
        "detected_category": str(row["category"]),
        "detected_label": str(row.get("label", row["category"])),
        "quality": {
            "pre_score": float(row["pre_score"]),
            "sam_score": float(row.get("sam_score") or 0.0),
            "mask_quality": float(row["quality"]),
            "tracklet_len": int(row.get("tracklet_len") or len(track)),
            "real_view_eligible": row.get('real_view_eligible',True),
            "source_frame_edge": row.get('source_frame_edge'),
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
            **{key: row[key] for key in ("timestamp_us", "camera_intrinsic", "camera_to_world")
               if row.get(key) is not None},
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
        # Retain SAM evidence before Qwen finalizes the category and route.
        candidates = [r for r in extracted['segmented'] if r['tracklet_id']==row['tracklet_id']]
        from PIL import Image
        prepared = []
        anchor = None
        # Deduplicate actual pixels before limiting to four; a duplicate must
        # not occupy a slot that another valid observation could fill.
        for candidate in candidates:
            rgb = cache.get_rgb(candidate['image_path'])
            h,w = rgb.shape[:2]
            crop = _crop_box(candidate['_mask'], candidate['bbox_xyxy'], w, h)
            rgba = Image.fromarray(rgb).convert('RGBA')
            rgba.putalpha(Image.fromarray(candidate['_mask'].astype('uint8')*255))
            rgba = rgba.crop(crop)
            rgba.thumbnail((config.export_max_size, config.export_max_size), Image.Resampling.LANCZOS)
            reference = dict(candidate, _rgba=rgba, crop_box=list(crop),
                             rgba_sha256=hashlib.sha256(rgba.tobytes()).hexdigest())
            prepared.append(reference)
            if candidate is row:
                anchor = reference
        if anchor is None:
            raise ValueError("FLUX anchor is absent from the segmented identity")
        references, selection = select_real_views(prepared, anchor, config.real_views_max,
                                                   config.real_view_quality_ratio)
        real_images = []
        for reference in references:
            path = images_dir / f'{obj_id}_real_{len(real_images):02}.png'
            reference['_rgba'].save(path)
            real_images.append(dict(rgba_path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_image=str(Path(reference['image_path']).resolve()), frame_index=reference['frame_id'],
                camera_name=reference['camera_name'], bbox_xyxy=reference['bbox_xyxy'], crop_box=reference['crop_box'],
                quality=float(reference['quality']), sam_score=float(reference.get('sam_score') or 0),
                real_view_eligible=reference.get('real_view_eligible',True),
                source_frame_edge=reference.get('source_frame_edge'),
                **{key:reference[key] for key in ('observation_method','association_source','propagated_mask_iou')
                   if key in reference}))
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
        "coverage": extracted.get("coverage", {}),
        "elapsed_sec": round(time.perf_counter() - started, 6),
    }
    write_json(config.output_dir / "status.json", status)
    return {
        "manifest_path": str(diffusion_dir / "manifest.json"),
        "images_dir": str(images_dir),
        "items": manifest,
        "status": status,
    }
