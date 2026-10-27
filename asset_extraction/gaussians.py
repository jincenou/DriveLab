"""Original Gaussian decoding and canonical transforms, isolated from the old application."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
SH_C0 = 0.28209479177387814
UP_FIX = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
FLIP_Z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)

@dataclass
class Gaussians:
    """Frame-major gaussian bundle; rigid assets have a single frame."""

    xyz: np.ndarray  # (F, N, 3)
    rot: np.ndarray  # (F, N, 4) wxyz
    scale: np.ndarray  # (F, N, 3) linear
    opacity: np.ndarray  # (N,)
    color: np.ndarray  # (N, 3) in [0, 1]
    up: np.ndarray  # (3,) up direction in asset frame

    @property
    def frames(self) -> int:
        return self.xyz.shape[0]

def load_rigid(ply_path: Path) -> Gaussians:
    from plyfile import PlyData

    vertex = PlyData.read(str(ply_path))["vertex"]

    def stack(names: list[str]) -> np.ndarray:
        return np.stack([np.asarray(vertex[n], dtype=np.float32) for n in names], axis=-1)

    color = np.clip(0.5 + SH_C0 * stack(["f_dc_0", "f_dc_1", "f_dc_2"]), 0.0, 1.0)
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(vertex["opacity"], dtype=np.float32)))
    return Gaussians(
        xyz=stack(["x", "y", "z"])[None],
        rot=stack(["rot_0", "rot_1", "rot_2", "rot_3"])[None],
        scale=np.exp(stack(["scale_0", "scale_1", "scale_2"]))[None],
        opacity=opacity,
        color=color,
        up=np.array([0.0, -1.0, 0.0], dtype=np.float32),  # TRELLIS assets are -y up
    )

def _yaw_rotation(gs: Gaussians, mode: str, up_rotation: np.ndarray = UP_FIX) -> np.ndarray:
    """Rotation about z aligning the requested horizontal axis with x."""
    xyz = gs.xyz[0] @ up_rotation.T
    solid = xyz[gs.opacity > 0.5] if (gs.opacity > 0.5).sum() > 100 else xyz
    xy = solid[:, :2] - solid[:, :2].mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov(xy.T))  # ascending eigenvalues
    axis = vectors[:, 1] if mode == "long_axis" else vectors[:, 0]
    yaw = np.arctan2(axis[1], axis[0])
    cos, sin = np.cos(-yaw), np.sin(-yaw)
    return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float64)

def _export_ply(source: Path, target: Path, rotation: np.ndarray, translation: np.ndarray) -> None:
    from plyfile import PlyData, PlyElement
    from scipy.spatial.transform import Rotation

    ply = PlyData.read(str(source))
    vertex = ply["vertex"].data.copy()
    xyz = np.stack([vertex[n] for n in ("x", "y", "z")], axis=1).astype(np.float64)
    xyz = xyz @ rotation.T + translation
    vertex["x"], vertex["y"], vertex["z"] = xyz.astype(np.float32).T

    if "nx" in vertex.dtype.names:
        normals = np.stack([vertex[n] for n in ("nx", "ny", "nz")], axis=1).astype(np.float64)
        vertex["nx"], vertex["ny"], vertex["nz"] = (normals @ rotation.T).astype(np.float32).T

    quats_wxyz = np.stack([vertex[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)
    composed = Rotation.from_matrix(rotation) * Rotation.from_quat(quats_wxyz[:, [1, 2, 3, 0]])
    new_xyzw = composed.as_quat()
    for i, column in enumerate([3, 0, 1, 2]):  # back to wxyz
        vertex[f"rot_{i}"] = new_xyzw[:, column].astype(np.float32)

    target.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(target))

def _grounding(xyz: np.ndarray) -> tuple[np.ndarray, list[float]]:
    lo = np.quantile(xyz, 0.005, axis=0)
    hi = np.quantile(xyz, 0.995, axis=0)
    translation = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    return translation, (hi - lo).round(4).tolist()
