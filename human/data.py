"""Known calibrated cameras and RGBA training data; held-out views stay excluded."""

import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from common import BACKEND, case, read, save

sys.path.insert(0, str(BACKEND))
from scene.cameras import Camera
from utils.graphics_utils import BasicPointCloud, focal2fov


def manifest(label):
    path = Path(case(label)["manifest"])
    data = read(path)
    assert data["heldout_views"] == [0, 8, 16]
    assert data["train_views"] == [v for v in range(24) if v not in [0, 8, 16]]
    assert data["time_denominator"] == data["frame_count"] - 1
    assert (data["width"], data["height"]) == (352, 640)
    return data, path


def camera(data, view, frame, rgb=None):
    record = next(c for c in data["cameras"] if c["view"] == view)
    K = np.asarray(record["K"], dtype=np.float64)
    w = np.linalg.inv(np.asarray(record["camtoworld"], dtype=np.float64))
    if rgb is None:
        rgb = torch.zeros(3, data["height"], data["width"], device="cuda")
    result = Camera(
        colmap_id=view,
        R=w[:3, :3].T,
        T=w[:3, 3],
        FoVx=focal2fov(K[0, 0], data["width"]),
        FoVy=focal2fov(K[1, 1], data["height"]),
        image=rgb,
        gt_alpha_mask=None,
        image_name=f"cam{view:02d}/time{frame:.6f}",
        uid=view * data["frame_count"] + frame,
        data_device="cuda",
        fid=frame / data["time_denominator"],
    )
    result.projection_matrix[2, 0] = (2 * K[0, 2] + 1) / data["width"] - 1
    result.projection_matrix[2, 1] = (2 * K[1, 2] + 1) / data["height"] - 1
    result.full_proj_transform = result.world_view_transform @ result.projection_matrix
    result.source_view, result.source_frame = view, frame
    return result


def load_train(data):
    cameras, alphas, paths = [], [], []
    for view in data["train_views"]:
        for frame in range(data["frame_count"]):
            path = (
                Path(data["source_data"])
                / "images"
                / f"cam{view:02d}"
                / f"{frame:06d}.png"
            )
            with Image.open(path) as image:
                assert image.mode == "RGBA" and image.size == (352, 640)
                rgba = (
                    torch.from_numpy(np.asarray(image).copy())
                    .to("cuda", dtype=torch.float32)
                    .permute(2, 0, 1)
                    / 255
                )
            rgb = rgba[:3] * rgba[3:] + 0.5 * (1 - rgba[3:])
            cam = camera(data, view, frame, rgb)
            cam.foreground_pixels = int((rgba[3] > 0.5).sum())
            cameras.append(cam)
            alphas.append((rgba[3] * 255).round().byte())
            paths.append(str(path))
    return cameras, alphas, paths


class KnownScene:
    def __init__(self, dataset, gaussians):
        self.model_path, self.gaussians = dataset.model_path, gaussians
        self.data, path = manifest(dataset.label)
        assert dataset.eval is True and dataset.background_value == 0.5
        with np.load(path.parent / self.data["canonical_npz"]) as c:
            cloud = BasicPointCloud(
                points=c["points"].copy(),
                colors=c["colors"].copy(),
                normals=np.zeros_like(c["points"]),
            )
        centers = np.asarray(
            [
                c["camtoworld"]
                for c in self.data["cameras"]
                if c["view"] in self.data["train_views"]
            ]
        )[:, :3, 3]
        self.cameras_extent = float(
            np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1
        )
        gaussians.create_from_pcd(cloud, self.cameras_extent)
        self.train_cameras, self.alphas, self.loaded_paths = load_train(self.data)

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return []

    def save(self, iteration, is_best=False):
        assert not is_best
        target = (
            Path(self.model_path)
            / "point_cloud"
            / f"iteration_{iteration}"
            / "point_cloud.ply"
        )
        assert not target.exists()
        self.gaussians.save_ply(str(target))
