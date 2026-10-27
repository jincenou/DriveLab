"""DriveLab-compatible storage, complete SH and exact native-array playback.

Storage fields mirror the original pipeline; its RGB-only consumer needs an adapter.
"""

import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from PIL import Image, ImageDraw
from common import ROOT, FRONTEND, ASSET_ROOT, case, read, sha, save, gpu_identity, generation_path
from data import camera
from reconstruction import load_asset, raw_render
from utils.general_utils import quaternion_to_matrix
from utils.sh_utils import eval_sh
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

REQUIRED = (
    "format_version",
    "frame_count",
    "gaussian_count",
    "use_rgb",
    "canonical_xyz",
    "canonical_rotation",
    "canonical_scaling",
    "opacity",
    "shs",
    "frame_xyz",
    "frame_rotation",
    "frame_scaling",
)


def raw_rotation(q):
    """Match native CUDA computeCov3D: its input quaternion is NOT normalized."""
    w, x, y, z = np.moveaxis(np.asarray(q, np.float64), -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(-1, 3, 3)


def covariance(rotation, scaling):
    basis = raw_rotation(rotation) * np.asarray(scaling, np.float64)[:, None, :]
    return basis @ basis.transpose(0, 2, 1)


def transform_gaussians(xyz, rotation, scaling, transform):
    R, t = transform[:3, :3], transform[:3, 3]
    positions = np.asarray(xyz, np.float64) @ R.T + t
    source_cov = covariance(rotation, scaling)
    target_cov = R[None] @ source_cov @ R.T[None]
    values, vectors = np.linalg.eigh(target_cov)
    tolerance = np.maximum(np.linalg.norm(target_cov, axis=(1, 2)) * 1e-12, 1e-20)
    if (values[:, 0] < -tolerance).any():
        raise ValueError("Non-positive covariance beyond numerical tolerance")
    clipped = int((values < 0).sum())
    values = np.maximum(values, 1e-20)
    wrong_hand = np.linalg.det(vectors) < 0
    vectors[wrong_hand, :, 0] *= -1
    xyzw = Rotation.from_matrix(vectors).as_quat()
    quaternions = xyzw[:, [3, 0, 1, 2]].astype(np.float32)
    scales = np.sqrt(values).astype(np.float32)
    reconstructed = covariance(quaternions, scales)
    relative = np.linalg.norm(reconstructed - target_cov, axis=(1, 2)) / np.maximum(
        np.linalg.norm(target_cov, axis=(1, 2)), 1e-20
    )
    return (
        positions.astype(np.float32),
        quaternions,
        scales,
        dict(
            covariance_relative_max=float(relative.max()),
            negative_eigenvalues_clipped=clipped,
        ),
    )


def rotate_sh(coefficients, rotation, degree):
    """Solve the complete degree<=3 real-SH rotation in float64, then keep float32."""
    count = (degree + 1) ** 2
    k = torch.arange(128, device="cuda", dtype=torch.float64)
    z = 1 - 2 * (k + 0.5) / 128
    angle = k * (math.pi * (3 - math.sqrt(5)))
    directions = torch.stack(
        (torch.sqrt(1 - z * z) * angle.cos(), torch.sqrt(1 - z * z) * angle.sin(), z),
        -1,
    )
    R = torch.as_tensor(rotation, device="cuda", dtype=torch.float64)
    identity = torch.eye(count, device="cuda", dtype=torch.float64)[None].expand(
        128, -1, -1
    )
    basis_new = eval_sh(degree, identity, directions)
    basis_old = eval_sh(degree, identity, directions @ R)
    operator = torch.linalg.lstsq(basis_new, basis_old).solution
    old = torch.as_tensor(coefficients, device="cuda", dtype=torch.float64)
    new = torch.einsum("kl,nlc->nkc", operator, old[:, :count]).float()
    # Validation directions are disjoint from the fitting directions.
    v = torch.arange(97, device="cuda", dtype=torch.float64) + 0.371
    zz = 1 - 2 * v / 98
    theta = v * 2.399
    test = torch.stack(
        (
            torch.sqrt(1 - zz * zz) * theta.cos(),
            torch.sqrt(1 - zz * zz) * theta.sin(),
            zz,
        ),
        -1,
    )
    selected = torch.linspace(0, len(old) - 1, min(64, len(old)), device="cuda").long()
    a = eval_sh(degree, new[selected].double().transpose(1, 2)[None], test[:, None])
    b = eval_sh(degree, old[selected].transpose(1, 2)[None], (test @ R)[:, None])
    return new.cpu().numpy(), float((a - b).abs().max())


def load_npz(path):
    """Honor SH/RGB metadata; never silently replace view-dependent color by DC."""
    with np.load(path, allow_pickle=False) as z:
        missing = set(REQUIRED) - set(z.files)
        if missing:
            raise ValueError(f"Missing asset fields: {missing}")
        result = {key: z[key] for key in z.files}
    F, N = int(result["frame_count"][0]), int(result["gaussian_count"][0])
    assert result["frame_xyz"].shape == (F, N, 3)
    assert result["frame_rotation"].shape == (F, N, 4)
    assert result["frame_scaling"].shape == (F, N, 3)
    assert result["opacity"].shape in ((N,), (N, 1))
    use_rgb = bool(result["use_rgb"][0])
    if use_rgb:
        assert result["shs"].shape in ((N, 3), (N, 1, 3))
    else:
        degree = int(result["active_sh_degree"][0])
        assert result["shs"].shape == (N, (degree + 1) ** 2, 3)
    for value in result.values():
        if np.issubdtype(value.dtype, np.number):
            assert np.isfinite(value).all()
    assert (result["frame_scaling"] > 0).all()
    assert np.max(np.abs(np.linalg.norm(result["frame_rotation"], axis=-1) - 1)) < 1e-5
    return result


def tensor_asset(arrays):
    keys = ("frame_xyz", "frame_rotation", "frame_scaling", "opacity", "shs")
    keys += tuple(
        key
        for key in arrays
        if key.startswith("native_") and key != "native_canonical_xyz"
    )
    return {
        key: torch.as_tensor(arrays[key], device="cuda", dtype=torch.float32)
        for key in keys
    }


def render_frame(
    asset, arrays, cameras, view, frame, background=0.5, representation="exact_native"
):
    """Same rasterizer as the native model, independent of its neural runtime."""
    cam = camera(cameras, view, frame)
    degree = 0 if bool(arrays["use_rgb"][0]) else int(arrays["active_sh_degree"][0])
    settings = GaussianRasterizationSettings(
        image_height=cam.image_height,
        image_width=cam.image_width,
        tanfovx=math.tan(cam.FoVx * 0.5),
        tanfovy=math.tan(cam.FoVy * 0.5),
        bg=torch.full((3,), background, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=cam.world_view_transform,
        projmatrix=cam.full_proj_transform,
        sh_degree=degree,
        campos=cam.camera_center,
        prefiltered=False,
        debug=False,
    )
    raster = GaussianRasterizer(raster_settings=settings)
    assert representation in ("exact_native", "camera_frame")
    prefix = "native_" if representation == "exact_native" else ""
    points = asset[prefix + "frame_xyz"][frame]
    arguments = dict(
        means3D=points,
        means2D=torch.zeros_like(points),
        opacities=asset["opacity"].reshape(-1, 1),
        scales=asset[prefix + "frame_scaling"][frame],
        rotations=asset[prefix + "frame_rotation"][frame],
        cov3D_precomp=None,
    )
    use_rgb = bool(arrays["use_rgb"][0])
    rgb = raster(
        **arguments,
        shs=None if use_rgb else asset[prefix + "shs"],
        colors_precomp=asset["shs"].reshape(-1, 3) if use_rgb else None,
    )[0]
    settings = settings._replace(bg=torch.zeros(3, device="cuda"))
    white = GaussianRasterizer(raster_settings=settings)(
        **arguments, shs=None, colors_precomp=torch.ones_like(points)
    )[0]
    return rgb, white[:1]


def write_video(path, images, fps=25):
    import subprocess
    import imageio_ffmpeg

    iterator = iter(images)
    first = next(iterator)
    h, w = first.shape[:2]
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        process.stdin.write(np.ascontiguousarray(first).tobytes())
        for image in iterator:
            process.stdin.write(np.ascontiguousarray(image).tobytes())
    finally:
        process.stdin.close()
    error = process.stderr.read().decode()
    if process.wait():
        raise RuntimeError(error)


@torch.no_grad()
def export_asset(identity):
    plan = read(ROOT / "config/export.json")
    item = dict(asset=str(ROOT / "outputs" / identity / "asset"), iteration=53000)
    source = Path(item["asset"])
    target = ASSET_ROOT / identity
    assert not target.exists(), "Never overwrite a successful exported asset"
    temporary = target.with_name("." + target.name + ".exporting")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    begin = time.monotonic()
    device = gpu_identity()
    model, deform, pipe, config = load_asset(source, item["iteration"])
    data = __import__("json").loads((source / "asset_camera_data.json").read_text())
    N, F = (len(model.get_xyz), data["frame_count"])
    first = next((c for c in data["cameras"] if c["view"] == 0))
    transform = np.linalg.inv(np.asarray(first["camtoworld"], np.float64))
    assert np.linalg.det(transform[:3, :3]) > 0.99999
    cameras = dict(data)
    cameras["cameras"] = [
        dict(c, camtoworld=(transform @ np.asarray(c["camtoworld"])).tolist())
        for c in data["cameras"]
    ]
    cameras["asset_from_native"] = transform.tolist()
    cameras["coordinate_frame"] = "reference_view00_cv_camera_x_right_y_down_z_forward"
    canonical = transform_gaussians(
        model.get_xyz.cpu().numpy(),
        model.get_rotation.cpu().numpy(),
        model.get_scaling.cpu().numpy(),
        transform,
    )
    shs, sh_error = rotate_sh(
        model.get_features.cpu().numpy(), transform[:3, :3], model.active_sh_degree
    )
    xyz = []
    rot = []
    scale = []
    diagnostics = []
    native_xyz = []
    native_rotation = []
    native_scaling = []
    rotation_norm_range = [float("inf"), 0.0]
    for frame in range(F):
        t = model.get_xyz.new_tensor([[frame / (F - 1)]]).expand(N, -1)
        warp = deform.step(model.get_xyz.detach(), t)
        world = (
            quaternion_to_matrix(warp["d_xyz"][:, :4]) @ model.get_xyz[..., None]
        ).squeeze(-1) + warp["d_xyz"][:, 4:]
        q = model.get_rotation + warp["d_rotation"]
        s = model.get_scaling + warp["d_scaling"]
        native_xyz.append(world.cpu().numpy())
        native_rotation.append(q.cpu().numpy())
        native_scaling.append(s.cpu().numpy())
        norm = q.norm(dim=-1)
        rotation_norm_range = [
            min(rotation_norm_range[0], float(norm.min())),
            max(rotation_norm_range[1], float(norm.max())),
        ]
        p, rotation, scaling, diagnostic = transform_gaussians(
            world.cpu().numpy(), q.cpu().numpy(), s.cpu().numpy(), transform
        )
        xyz.append(p)
        rot.append(rotation)
        scale.append(scaling)
        diagnostics.append(diagnostic)
        if frame % 20 == 0:
            save(
                ROOT / "logs" / f"{identity}_progress.json",
                dict(
                    stage="bake", frame=frame, frames=F, pid=os.getpid(), device=device
                ),
            )
    maximum = max(
        canonical[3]["covariance_relative_max"],
        *(d["covariance_relative_max"] for d in diagnostics),
    )
    assert maximum <= plan["gates"]["covariance_relative_max"], maximum
    assert sh_error <= plan["gates"]["SH_rotation_absolute_max"], sh_error
    arrays = dict(
        format_version=np.array([1], np.int32),
        frame_count=np.array([F], np.int32),
        gaussian_count=np.array([N], np.int32),
        use_rgb=np.array([0], np.int32),
        canonical_xyz=canonical[0],
        canonical_rotation=canonical[1],
        canonical_scaling=canonical[2],
        opacity=model.get_opacity.cpu().numpy().astype(np.float32),
        shs=shs,
        frame_xyz=np.stack(xyz),
        frame_rotation=np.stack(rot),
        frame_scaling=np.stack(scale),
        active_sh_degree=np.array([model.active_sh_degree], np.int32),
        fps=np.array([25.0], np.float64),
        frame_timestamps=np.arange(F, dtype=np.float64) / 25,
        asset_from_native=transform.astype(np.float64),
        extension=np.array(["grid4d_exact_raster_v2"]),
        native_frame_xyz=np.stack(native_xyz),
        native_frame_rotation=np.stack(native_rotation),
        native_frame_scaling=np.stack(native_scaling),
        native_shs=model.get_features.cpu().numpy(),
        native_canonical_xyz=model.get_xyz.cpu().numpy(),
    )
    np.savez_compressed(temporary / "asset_4dgs.npz", **arrays)
    save(temporary / "cameras.json", cameras)
    save(temporary / "native_cameras.json", data)
    refs = {}
    for view in (0, 8, 16):
        for frame in np.rint(np.linspace(0, F - 1, 13)).astype(int):
            rgb, alpha = raw_render(model, deform, pipe, camera(data, view, int(frame)))
            refs[f"v{view}_f{frame}"] = torch.cat((rgb, alpha)).cpu().numpy()
    np.savez_compressed(temporary / "validation_reference.npz", **refs)
    from PIL import Image

    metadata_path = Path(config["provenance"]["manifest"])
    full = __import__("json").loads(metadata_path.read_text())
    (temporary / "refs").mkdir()
    refs_names = []
    for i, view in enumerate((0, 6)):
        p = Path(full["source_data"]) / "images" / f"cam{view:02d}" / "000000.png"
        with Image.open(p) as image:
            rgba = np.asarray(image, np.float32) / 255
        rgb = np.rint(
            (rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])) * 255
        ).astype(np.uint8)
        name = f"refs/{i:02d}.png"
        Image.fromarray(rgb).save(temporary / name)
        refs_names.append(name)
    layout = dict(
        static=["opacity", "shs"],
        canonical=["canonical_xyz", "canonical_rotation", "canonical_scaling"],
        per_frame=["frame_xyz", "frame_rotation", "frame_scaling"],
    )
    save(
        temporary / "motion_metadata.json",
        dict(
            frame_count=F,
            fps=25.0,
            time_unit="seconds",
            source="Grid4D learned deformation sampled at original source slots",
            smplx_retargetable=False,
            metric_world_registration=False,
            continuous_runtime_preserved_at_source=str(source),
        ),
    )
    manifest = dict(
        obj_id=identity,
        scene_id=case(identity)
        .get("selection", {})
        .get("scene", "registered_g0_" + identity),
        images=refs_names,
        motion=dict(
            clip_id=identity,
            source="generated_multiview_motion",
            path="motion_metadata.json",
        ),
        model_path=str(source),
        frame_count=F,
        asset=dict(path="asset_4dgs.npz", layout=layout),
        producer="4DAnyone Turbo + Grid4D 32999+15000+5000 updates",
        fps=25.0,
        extension=dict(
            use_rgb=False,
            sh_degree=model.active_sh_degree,
            requires_SH_aware_loader=True,
            authoritative_raster_arrays="native_frame_xyz/native_frame_rotation/native_frame_scaling/native_shs",
            authoritative_cameras="native_cameras.json",
            legacy_camera_frame_fields="Full covariance-equivalent export; subpixel rounding may change isolated raster order",
            coordinate_frame=cameras["coordinate_frame"],
            canonical_kind="learned_rest_not_SMPLX_A_pose",
            precision="float32",
            temporal_scope="baked integer slots; native source retains continuous runtime",
        ),
        quality_approved=False,
        source_scope="Grid4D 32999+15000+5000 updates; serialization does not alter its appearance",
        source_hashes={
            str(p.relative_to(source)): sha(p)
            for p in [
                source / "point_cloud/iteration_53000/point_cloud.ply",
                source / "deform/iteration_53000/deform.pth",
            ]
        },
    )
    save(temporary / "manifest.json", manifest)
    save(
        temporary / "export_validation.json",
        dict(
            state="exported_replay_pending",
            gaussians=N,
            frames=F,
            covariance_relative_max=maximum,
            SH_rotation_absolute_max=sh_error,
            native_rotation_norm_range=rotation_norm_range,
            tiny_negative_eigenvalues_clipped=sum(
                (x["negative_eigenvalues_clipped"] for x in diagnostics)
            ),
            asset_sha256=sha(temporary / "asset_4dgs.npz"),
            seconds=time.monotonic() - begin,
            device=device,
        ),
    )
    temporary.replace(target)
    print(identity, "exported", N, F, flush=True)


@torch.no_grad()
def verify_asset(identity):
    plan = read(ROOT / "config/export.json")
    item = dict(asset=str(ROOT / "outputs" / identity / "asset"))
    output = ASSET_ROOT / identity
    specification = case(identity)
    compact_media = bool(specification.get("selection"))
    input_data = read(specification["manifest"])
    denied = [
        Path(item["asset"]).resolve(),
        Path(input_data["source_data"]).resolve(),
        (ROOT / "data").resolve(),
        (FRONTEND / "models").resolve(),
    ]
    for key in ("raw_input", "conditioning", "motion"):
        if key in specification:
            denied.append(Path(specification[key]).resolve())

    def guard(event, arguments):
        if event == "open" and isinstance(arguments[0], (str, bytes)):
            path = Path(os.fsdecode(arguments[0])).resolve()
            if any((path.is_relative_to(p) for p in denied)):
                raise PermissionError(f"NPZ replay may not read original data: {path}")

    sys.addaudithook(guard)
    begin = time.monotonic()
    device = gpu_identity()
    arrays = load_npz(output / "asset_4dgs.npz")
    asset = tensor_asset(arrays)
    cameras = json.loads((output / "native_cameras.json").read_text())
    rows = []
    contacts = []
    camera_frame = json.loads((output / "cameras.json").read_text())
    camera_frame_diagnostic = []
    gates = plan["gates"]
    with np.load(output / "validation_reference.npz", allow_pickle=False) as refs:
        for key in refs.files:
            v, f = key.split("_")
            view, frame = (int(v[1:]), int(f[1:]))
            ref = torch.tensor(refs[key], device="cuda")
            rgb, alpha = render_frame(asset, arrays, cameras, view, frame, 0.0)
            mask = ref[3] > 0.01
            error = (rgb - ref[:3]).abs()
            aerr = (alpha - ref[3:]).abs()
            row = dict(
                view=view,
                frame=frame,
                foreground_MAE=float(error[:, mask].mean()),
                full_MAE=float(error.mean()),
                max_error=float(error.max()),
                alpha_MAE=float(aerr.mean()),
            )
            assert row["foreground_MAE"] <= gates["foreground_RGB_MAE_max"], row
            assert row["max_error"] <= gates["RGB_max_error"], row
            assert row["alpha_MAE"] <= gates["alpha_MAE_max"], row
            rows.append(row)
            crgb, ca = render_frame(
                asset,
                arrays,
                camera_frame,
                view,
                frame,
                0.0,
                representation="camera_frame",
            )
            camera_frame_diagnostic.append(
                dict(
                    view=view,
                    frame=frame,
                    max_error=float((crgb - ref[:3]).abs().max()),
                    foreground_MAE=float((crgb - ref[:3]).abs()[:, mask].mean()),
                )
            )
            if not compact_media and frame in (
                0,
                (int(arrays["frame_count"][0]) - 1) // 2,
                int(arrays["frame_count"][0]) - 1,
            ):
                pred = (
                    (rgb + 0.5 * (1 - alpha)).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                )
                original = (
                    (ref[:3] + 0.5 * (1 - ref[3:]))
                    .clamp(0, 1)
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )
                pair = np.concatenate([original, pred], 1)
                canvas = Image.new("RGB", (704, 668), (80, 80, 80))
                canvas.paste(
                    Image.fromarray(np.rint(pair * 255).astype(np.uint8)), (0, 28)
                )
                ImageDraw.Draw(canvas).text(
                    (5, 7),
                    f"Native | DriveLab NPZ; {identity} v{view} f{frame}",
                    fill="white",
                )
                contacts.append(canvas)
    assert len(rows) == 39
    if not compact_media:
        sheet = Image.new("RGB", (704 * 3, 668 * 3), (128, 128, 128))
        for i, image in enumerate(contacts):
            sheet.paste(image, (i % 3 * 704, i // 3 * 668))
        sheet.save(output / "comparison.jpg", quality=94)

    def images():
        F = int(arrays["frame_count"][0])
        for frame in range(F):
            tiles = []
            for view in (0, 8, 16):
                rgb, alpha = render_frame(asset, arrays, cameras, view, frame, 0.5)
                assert torch.isfinite(rgb).all() and torch.isfinite(alpha).all()
                assert (alpha > 0.01).any(), (identity, view, frame)
                tiles.append(
                    np.rint(
                        rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                    ).astype(np.uint8)
                )
            yield np.concatenate(tiles, 1)

    if compact_media:
        for frame in images():
            assert frame.shape == (640, 1056, 3) and np.isfinite(frame).all()
    else:
        write_video(output / "preview.mp4", images(), 25)
    save(
        output / "replay_validation.json",
        dict(
            state="NPZ_integer_frame_replay_passed",
            device=device,
            probes=rows,
            reference_count=39,
            source_asset_reads=0,
            source_image_reads=0,
            camera_frame_reparameterization_diagnostic=camera_frame_diagnostic,
            authoritative_representation="exact_native; no quaternion renormalization, no precision reduction or coordinate conversion during replay",
            denied_source_roots=list(map(str, denied)),
            seconds=time.monotonic() - begin,
            foreground_MAE_max=max((x["foreground_MAE"] for x in rows)),
            RGB_max_error=max((x["max_error"] for x in rows)),
            alpha_MAE_max=max((x["alpha_MAE"] for x in rows)),
            quality_scope="Serialization equivalence only; inherited reconstruction defects remain",
            asset_sha256=sha(output / "asset_4dgs.npz"),
            full_time_render_frames=int(arrays["frame_count"][0]),
            preview_sha256=None if compact_media else sha(output / "preview.mp4"),
        ),
    )
    (output / "validation_reference.npz").unlink()
    print(identity, "NPZ replay passed", flush=True)


@torch.no_grad()
def five_column_video(identity):
    """One presentation video; source rows use the saved nearest native timestamp."""
    import cv2

    output = ASSET_ROOT / identity
    assert read(output / "replay_validation.json")["state"] == "NPZ_integer_frame_replay_passed"
    assert read(output / "runtime_replay.json")["state"] == "published_runtime_replay_passed"
    gpu_identity()
    arrays = load_npz(output / "asset_4dgs.npz")
    asset = tensor_asset(arrays)
    cameras = read(output / "native_cameras.json")
    count = int(arrays["frame_count"][0])
    mapping = read(output / "refs/frame_mapping.json")["frames"]
    assert len(mapping) == count and float(arrays["fps"][0]) == 25
    sources = {
        "observed": output / "refs/observed_native.mp4",
        "skeleton_pair": output / "refs/skeleton.mp4",
        "generated_view00": generation_path(identity) / "videos/00.mp4",
    }
    observed = []
    cap = cv2.VideoCapture(str(sources["observed"]))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        observed.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    indices = [int(row["left_native_index"] if row["interpolation_fraction"] < 0.5
                   else row["right_native_index"]) for row in mapping]
    assert observed and min(indices) >= 0 and max(indices) < len(observed)
    caps = {key: cv2.VideoCapture(str(sources[key])) for key in ("skeleton_pair", "generated_view00")}
    assert all(c.isOpened() for c in caps.values())
    titles = ["Original observation", "Skeleton overlay", "4DAnyone view 00 / 24", "Grid4D view 00", "Grid4D view 08"]

    def fit(image):
        tile = Image.new("RGB", (352, 640), (128, 128, 128))
        image = image.copy()
        image.thumbnail((352, 640), Image.Resampling.LANCZOS)
        tile.paste(image, ((352-image.width)//2, (640-image.height)//2))
        return tile

    def images():
        for frame in range(count):
            tiles = [fit(observed[indices[frame]])]
            for key, capture in caps.items():
                ok, bgr = capture.read()
                assert ok, (identity, key, frame)
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                if key == "skeleton_pair":
                    # Saved G0 diagnostic is original input | skeleton overlay.
                    assert image.width == 2 * observed[0].width and image.height == observed[0].height
                    image = image.crop((image.width//2, 0, image.width, image.height))
                tiles.append(fit(image))
            for view in (0, 8):
                rgb, alpha = render_frame(asset, arrays, cameras, view, frame, 0.5)
                assert torch.isfinite(rgb).all() and (alpha > 0.01).any()
                tiles.append(Image.fromarray(np.rint(rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)))
            canvas = Image.new("RGB", (1760, 672), (40, 40, 40))
            draw = ImageDraw.Draw(canvas)
            for column, (title, tile) in enumerate(zip(titles, tiles, strict=True)):
                canvas.paste(tile, (column*352, 32))
                draw.text((column*352+6, 8), title, fill="white")
            yield np.asarray(canvas)

    temporary = output / ".demo.writing.mp4"
    try:
        write_video(temporary, images(), 25)
        assert all(not capture.read()[0] for capture in caps.values()), "Video has extra frames"
    finally:
        for capture in caps.values():
            capture.release()
    capture = cv2.VideoCapture(str(temporary))
    assert abs(capture.get(cv2.CAP_PROP_FPS)-25) < 1e-6
    decoded = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        assert image.shape == (672, 1760, 3)
        decoded += 1
    capture.release()
    assert decoded == count
    temporary.replace(output / "demo.mp4")
    save(output / "presentation.json", dict(
        state="five_column_video_verified", file="demo.mp4", sha256=sha(output / "demo.mp4"),
        columns=titles, frame_count=count, fps=25, size=[1760, 672],
        observed_native_indices=indices, reconstructed_views=[0, 8], generated_view=0,
        source_sha256={key: sha(path) for key, path in sources.items()},
        timing="Nearest observed native sample from frozen frame mapping; skeleton/generated/reconstruction use the same interpolated output slot",
        quality_scope="Readable synchronized presentation, not visual quality acceptance",
    ))
    manifest = read(output / "manifest.json")
    manifest["preview_paths"] = {"five_column": "demo.mp4"}
    save(output / "manifest.json", manifest)
    complete = read(output / "package_complete.json")
    complete.update(
        state="complete_pending_visual_review", runtime_replay="runtime_replay.json",
        presentation="presentation.json", demo_sha256=sha(output / "demo.mp4"),
    )
    save(output / "package_complete.json", complete)
    publish_catalogue()
    print(identity, "five-column video verified", flush=True)


def publish_catalogue():
    """Publish only verified packages, keeping DriveLab's relative path layout."""
    import fcntl

    catalogue_root = ASSET_ROOT if "HUMAN_ASSET_ROOT" in os.environ else ROOT
    folder = catalogue_root / "assets/nonrigid/human"
    folder.mkdir(parents=True, exist_ok=True)
    from common import asset_uid, merge_catalogue
    with (catalogue_root / ".catalogue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        items = []
        for identity in read(ROOT / "config/cases.json"):
            package = ASSET_ROOT / identity
            proof = package / "replay_validation.json"
            quality = ROOT / "reports" / f"{identity}_quality.json"
            if not proof.exists() or not quality.exists():
                continue
            if read(proof)["state"] != "NPZ_integer_frame_replay_passed" or read(
                quality
            )["state"] not in ("passed", "measured_visual_review_pending"):
                continue
            specification = case(identity)
            if specification.get("selection"):
                completion = package / "package_complete.json"
                if not completion.exists() or read(completion).get("state") != "complete_pending_visual_review":
                    continue
                if not (package / "runtime_replay.json").exists() or read(package / "runtime_replay.json").get("state") != "published_runtime_replay_passed":
                    continue
                if not (package / "presentation.json").exists() or read(package / "presentation.json").get("state") != "five_column_video_verified":
                    continue
            review = read(package / "visual_review.json") if (package / "visual_review.json").exists() else {}
            severe = review.get("severe_failure", False)
            approved = review.get("final_asset_quality_certified", False) and not severe
            quality_state = (
                "severe_failure" if severe else "accepted_with_recorded_limitations" if approved
                else "flagged_for_review" if review.get("quality_flags") else "visual_review_pending"
            )
            with np.load(package / "asset_4dgs.npz", allow_pickle=False) as a:
                from placement import standard_transform
                with np.load(package / "insertion_transforms.npz", allow_pickle=False) as maps:
                    T, model_scale, up_error = standard_transform(maps["asset_camera_to_root_local"])
                # Legacy canonical fields remain in asset units. The complete
                # metric-estimate similarity is supplied separately, exactly once.
                R = T[:3, :3] / model_scale
                translation = (T[:3, 3] / model_scale).tolist()
                points = a["canonical_xyz"] @ R.T + translation
                lo, hi = points.min(0), points.max(0)
                frames = int(a["frame_count"][0])
            placement = read(package / "placement.json") if (package / "placement.json").exists() else None
            insertion_record=read(package/'insertion.json')
            practical=insertion_record.get('practical_playback')
            skip_insertion=bool(insertion_record.get('skip_insertion', False))
            from placement import motion_review_route
            link = folder / f"{identity}.npz"
            if link.is_symlink():
                assert link.resolve() == (package / "asset_4dgs.npz").resolve()
            elif link.exists():
                raise FileExistsError(link)
            else:
                link.symlink_to(
                    os.path.relpath(package / "asset_4dgs.npz", link.parent)
                )
            items.append(
                dict(
                    global_uid=asset_uid(identity, specification.get("source_dataset", "waymo")),
                    dataset=specification.get("source_dataset", "waymo"),
                    **{"class": "pedestrian"},
                    category="human",
                    asset_path=str(link.relative_to(catalogue_root)),
                    size_xyz=(hi - lo).tolist(),
                    scale_m=None,
                    scale_axis="z",
                    canonical=dict(rotation=R.tolist(), translation=translation,
                                   units="asset_units", convention="x_forward_y_left_z_up",
                                   ground_origin="generator_fitted_ground_not_verified_gaussian_sole"),
                    asset_unit_to_model_meter=model_scale,
                    asset_to_standard_model_meters=T.tolist(),
                    placement_metadata=str((package / "placement.json").relative_to(catalogue_root)) if placement else None,
                    placement_ready=False,
                    approximate_placement_available=bool(practical and practical['approximate_placement_available']),
                    approximate_playback=str((package/'insertion.json').relative_to(catalogue_root))+'#/practical_playback' if practical else None,
                    skip_insertion=skip_insertion,
                    insertion_exclusion_reason=insertion_record.get('insertion_exclusion_reason') if skip_insertion else None,
                    motion_review_group='excluded_from_insertion' if skip_insertion else motion_review_route(placement),
                    person=dict(
                        frame_count=frames,
                        fps=25.0,
                        loopable=False if placement else None,
                        action=placement["action"]["label"] if placement else "unknown",
                        yaw_deg_start=None,
                    ),
                    producer="Grid4D 32999+15000+5000 updates",
                    metric_scale_verified=False,
                    original_motion_binding_verified=False,
                    requires_SH_aware_loader=True,
                    quality_approved=approved,
                    quality_state=quality_state,
                    quality_flags=review.get("quality_flags", []),
                    package_path=str(package.relative_to(catalogue_root)),
                    presentation_path=str((package / "demo.mp4").relative_to(catalogue_root)) if specification.get("selection") else None,
                    validation_scope="Independent serialization replay; per-asset appearance review is separate",
                )
            )
        run_items = items
        catalogue_path = catalogue_root / "drivelab_asset.json"
        existing = read(catalogue_path).get("assets", []) if catalogue_path.exists() else []
        owned = {asset_uid(i, case(i).get("source_dataset", "waymo"))
                 for i in read(ROOT / "config/cases.json")}
        items = merge_catalogue(existing, run_items, owned)
        save(
            catalogue_root / "drivelab_asset.json",
            dict(
                schema_version=1,
                manifest_type="drivelab_asset",
                path_policy="repository_relative_single_asset_path",
                total=len(items),
                quality_qualified_total=sum(item["quality_approved"] for item in items),
                severe_failure_total=sum(item["quality_state"] == "severe_failure" for item in items),
                insertion_excluded_total=sum(item['skip_insertion'] for item in items),
                motion_groups={name:[item['global_uid'] for item in items if item['motion_review_group']==name]
                               for name in sorted({item['motion_review_group'] for item in items})},
                motion_group_policy='Validation candidates and separate review; no group grants trajectory walking or looping permission',
                assets=items,
                scale_policy="asset_unit_to_model_meter restores fitted body-model meters, not measured personal height. Canonical fields stay in asset units; use the full similarity once. null scale_m is not a measured height",
                producer_extension="Grid4D full SH; SH-aware consumer required",
            ),
        )
        if (ROOT/'PRODUCTION_INPUTS.json').exists():
            items = run_items
            from collections import Counter
            production=read(ROOT/'PRODUCTION_INPUTS.json')
            jobs={p.stem:read(p) for p in (ROOT/'jobs').glob('H*.json')}
            workers={p.stem:read(p) for p in (ROOT/'jobs').glob('worker_gpu*.json')}
            snapshot=dict(updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                selected_inputs=len(production['ids']),target_count=production['target_count'],
                technical_complete=len(items),completed_ids=[Path(i['package_path']).name for i in items],
                approximate_playback_complete=sum(i['approximate_placement_available'] for i in items),
                insertion_excluded_total=sum(i['skip_insertion'] for i in items),
                quality_qualified_total=sum(i['quality_approved'] for i in items),
                job_states=dict(Counter(j.get('state','unknown') for j in jobs.values())),workers=workers,
                quality_scope='Technical completion and approximate metadata availability do not certify appearance, no-slip or looping; inspect per-asset caveats.',
                source='Current published catalogue plus job/worker snapshots; worker liveness requires process checks')
            save(ROOT/'STATUS.json',snapshot)
            (ROOT/'PROGRESS.txt').write_text(
                f"更新时间：{snapshot['updated_at']}\n已选输入：{snapshot['selected_inputs']}；技术完成：{snapshot['technical_complete']}；近似插入/播放信息已补齐：{snapshot['approximate_playback_complete']}。\n"
                "以上不是全部质量合格数。每个资产读取 insertion.json 的 practical_playback 和 placement_playback.npz；当前默认单次播放，接地与步速按估计精度使用。\n"
                "后台运行状态见jobs/worker_gpu*.json；tmux服务名称以本轮PLAN.txt为准。文件状态需结合进程存活核实。\n")


@torch.no_grad()
def verify_runtime_package(identity):
    """Read only the published package and replay its continuous network runtime."""
    output = ASSET_ROOT / identity
    specification = case(identity)
    denied = [
        (ROOT / "outputs").resolve(),
        (ROOT / "data").resolve(),
        (FRONTEND / "models").resolve(),
    ]
    for key in ("raw_input", "conditioning", "motion"):
        denied.append(Path(specification[key]).resolve())

    def guard(event, arguments):
        if event == "open" and isinstance(arguments[0], (str, bytes)):
            path = Path(os.fsdecode(arguments[0])).resolve()
            if any(path.is_relative_to(p) for p in denied):
                raise PermissionError(
                    f"Published runtime may not read original work files: {path}"
                )

    sys.addaudithook(guard)
    device = gpu_identity()
    begin = time.monotonic()
    model, deform, pipe, _ = load_asset(output / "runtime", 53000)
    data = read(output / "native_cameras.json")
    arrays = load_npz(output / "asset_4dgs.npz")
    asset = tensor_asset(arrays)
    gates = read(ROOT / "config/export.json")["gates"]
    records = []
    for view in (0, 8, 16):
        for frame in (0, data["frame_count"] // 2, data["frame_count"] - 1):
            rgb, alpha = raw_render(model, deform, pipe, camera(data, view, frame))
            baked_rgb, baked_alpha = render_frame(asset, arrays, data, view, frame, 0.0)
            difference = (rgb - baked_rgb).abs()
            foreground = alpha[0] > 0.01
            assert foreground.any() and torch.isfinite(difference).all()
            row = dict(
                view=view,
                frame=frame,
                foreground_MAE=float(difference[:, foreground].mean()),
                RGB_max_error=float(difference.max()),
                alpha_MAE=float((alpha - baked_alpha).abs().mean()),
            )
            assert row["foreground_MAE"] <= gates["foreground_RGB_MAE_max"]
            assert row["RGB_max_error"] <= gates["RGB_max_error"]
            assert row["alpha_MAE"] <= gates["alpha_MAE_max"]
            records.append(row)
    fractional = []
    for frame in (12.25, data["frame_count"] / 2 + 0.25):
        rgb, alpha = raw_render(model, deform, pipe, camera(data, 0, frame))
        assert (
            torch.isfinite(rgb).all()
            and torch.isfinite(alpha).all()
            and (alpha > 0.01).any()
        )
        fractional.append(
            dict(frame=frame, finite=True, foreground_pixels=int((alpha > 0.01).sum()))
        )
    save(
        output / "runtime_replay.json",
        dict(
            state="published_runtime_replay_passed",
            original_work_reads_forbidden=True,
            integer_comparisons=records,
            fractional_time_smoke=fractional,
            seconds=time.monotonic() - begin,
            device=device,
            scope="Packaged continuous runtime agrees with baked integer asset; fractional samples finite, not arbitrary new motion validation",
        ),
    )


@torch.no_grad()
def render_placement_trial(identity):
    """Read-only, continuous-runtime ground/path diagnostic on held-out points."""
    from placement import transform_points, camera_in_native_frame, vertical_sigma
    from common import CODE_ROOT
    begin=time.monotonic();package=ASSET_ROOT/identity
    destination=ROOT/'reports/placement_trials'/identity
    trial=read(destination/'candidate.json');policy=read(CODE_ROOT/'config/placement_trial.json')
    assert sha(package/'asset_4dgs.npz')==trial['asset_sha256']
    assert sha(package/'placement.json')==trial['placement_sha256']
    assert sha(destination/'candidate.npz')==trial['candidate_sha256']
    with np.load(destination/'candidate.npz',allow_pickle=False) as z:track={k:z[k] for k in z.files}
    with np.load(package/'placement_motion.npz',allow_pickle=False) as z:motion={k:z[k] for k in z.files}
    with np.load(package/'asset_4dgs.npz',allow_pickle=False) as z:
        native_to_asset=z['asset_from_native'];opacity=z['opacity'].reshape(-1)
        standard=transform_points(z['frame_xyz'],motion['asset_to_standard_model_meters'])
    excluded=np.concatenate([motion[s+'-foot_gaussian_indices'] for s in ('left','right')])
    heldout={};selection={}
    for side in ('left','right'):
        proxy=motion[side+'-foot_proxy_position']
        distance=np.median(np.linalg.norm(standard-proxy[:,None],axis=-1),axis=0)
        ids=np.flatnonzero((opacity>=.5)&(distance<=.15)&~np.isin(np.arange(len(opacity)),excluded))
        ids=ids[np.argsort(distance[ids])[:policy['heldout_max_points']]]
        heldout[side]=ids
        selection[side]=dict(count=len(ids),indices=ids.tolist(),sufficient=len(ids)>=policy['heldout_min_points'])
    del standard
    model,deform,pipe,_=load_asset(package/'runtime',53000)
    times=track['times'];F=len(times);T=motion['asset_to_standard_model_meters']
    width,height=640,384;K=np.array([[550.,0,319.5],[0,550.,191.5],[0,0,1.]])
    center=trial['native_forward_speed_model_mps']*(times[-1]-times[0])*.5
    target=np.array([center,0,.85]);world_cameras=[];floors=[]
    yy,xx=np.mgrid[:height,:width];rays=np.stack([(xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1],np.ones_like(xx)],axis=-1)
    for eye in (np.array([center,-6,2.3]),np.array([center+3.5,-5.5,2.6])):
        forward=target-eye;forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
        c2w=np.eye(4);c2w[:3,:3]=np.stack([right,np.cross(forward,right),forward],axis=1);c2w[:3,3]=eye
        world_cameras.append(c2w);directions=rays@c2w[:3,:3].T
        hit=-eye[2]/directions[:,:,2];xy=eye[:2]+directions[:,:,:2]*hit[:,:,None]
        checker=(np.floor(xy[:,:,0]/.25)+np.floor(xy[:,:,1]/.25))%2
        floor=np.repeat(np.where(checker[:,:,None]>0,.66,.44),3,axis=2)
        floor[hit<=0]=.82;floors.append(floor)
    diagnostics={branch:{side:dict(points=[],height=[]) for side in heldout} for branch in ('baseline','candidate')}
    cameras=dict(width=width,height=height,frame_count=F,time_denominator=F-1)
    def frames():
        for i in range(F):
            rows=[]
            for branch in ('baseline','candidate'):
                frame=float(track[branch+'_frames'][i]);W=track[branch+'_world_from_standard'][i];M=W@T@native_to_asset
                warp=deform.step(model.get_xyz.detach(),model.get_xyz.new_tensor([[frame/(F-1)]]).expand(len(model.get_xyz),-1))
                xyz=((quaternion_to_matrix(warp['d_xyz'][:,:4])@model.get_xyz[...,None]).squeeze(-1)+warp['d_xyz'][:,4:]).cpu().numpy()
                q=(model.get_rotation+warp['d_rotation']).cpu().numpy();s=(model.get_scaling+warp['d_scaling']).cpu().numpy()
                placed=transform_points(xyz,M)
                for side,ids in heldout.items():
                    if len(ids):
                        diagnostics[branch][side]['points'].append(placed[ids])
                        sigma=vertical_sigma(q[None,ids],s[None,ids],M[2,:3])[0]
                        diagnostics[branch][side]['height'].append(np.quantile(placed[ids,2]-2*sigma,.1))
                panels=[]
                for view,c2w in enumerate(world_cameras):
                    native_cam,_=camera_in_native_frame(c2w,W,T,native_to_asset)
                    cameras['cameras']=[dict(view=view,K=K.tolist(),camtoworld=native_cam.tolist())]
                    rgb,alpha=raw_render(model,deform,pipe,camera(cameras,view,frame))
                    rgb=rgb.permute(1,2,0).cpu().numpy();alpha=alpha.permute(1,2,0).cpu().numpy()
                    assert np.isfinite(rgb).all() and np.isfinite(alpha).all()
                    panels.append(np.clip((rgb+floors[view]*(1-alpha))*255,0,255).astype(np.uint8))
                row=np.concatenate(panels,axis=1);image=Image.fromarray(row)
                ImageDraw.Draw(image).text((10,10),f'{identity} {branch} | frame {frame:.2f} | experimental, unapproved',fill='white',stroke_width=1,stroke_fill='black')
                rows.append(np.asarray(image))
            result=np.concatenate(rows,axis=0)
            if i in (0,F//2,F-1):Image.fromarray(result).save(destination/f'frame_{i:03d}.jpg',quality=90)
            yield result
    write_video(destination/'comparison.mp4',frames(),fps=1/float(np.median(np.diff(times))))
    metrics={};evidence={}
    for branch,feet in diagnostics.items():
        metrics[branch]={};sampled=track[branch+'_frames']
        nearest=np.clip(np.rint(sampled).astype(int),0,F-1)
        for side,values in feet.items():
            if not selection[side]['sufficient']:
                metrics[branch][side]=dict(state='unknown_insufficient_heldout_points');continue
            points=np.asarray(values['points']);h=np.asarray(values['height'])
            support=motion[side+'_foot_contact_candidate'][nearest]
            core=support.copy();core[1:]&=support[:-1];core[:-1]&=support[1:]
            speed=np.linalg.norm(np.gradient(points,times,axis=0)[:,:,:2],axis=-1)
            evidence[branch+'_'+side+'_points']=points;evidence[branch+'_'+side+'_height']=h
            evidence[branch+'_'+side+'_contact']=support
            metrics[branch][side]=dict(state='diagnostic_not_contact_certification',contact_frames=int(support.sum()),core_frames=int(core.sum()),
                slip_p95_model_mps=float(np.quantile(speed[support],.95)) if support.any() else None,
                core_slip_p95_model_mps=float(np.quantile(speed[core],.95)) if core.any() else None,
                support_height_min=float(h.min()),support_height_max=float(h.max()),
                contact_height_p05=float(np.quantile(h[support],.05)) if support.any() else None,
                contact_height_p95=float(np.quantile(h[support],.95)) if support.any() else None)
    np.savez_compressed(destination/'heldout_diagnostics.npz',**evidence)
    assert sha(package/'asset_4dgs.npz')==trial['asset_sha256']
    save(destination/'render_validation.json',dict(identity=identity,state='experimental_render_complete_not_production_accepted',
        asset_unchanged=True,frames=F,views=2,device=gpu_identity(),seconds=time.monotonic()-begin,
        heldout_selection=selection,metrics=metrics,candidate_bounded=trial['bounded'],
        geometry='Actual packaged runtime at fractional time; no baked-array interpolation',
        limitations='Held-out IDs are disjoint from fitted proxies, but share reconstruction and contact hypotheses; semantic foot identity is unverified. Checker floor is composited behind the subject, not a depth occluder. No loop verification.'))


@torch.no_grad()
def practical_playback_preview(identity):
    """One temporary comparison row, combined into a single requested preview."""
    from playback import sample
    from placement import camera_in_native_frame
    package=ASSET_ROOT/identity;r=read(package/'insertion.json')['practical_playback']
    destination=ROOT/'reports/practical_playback';destination.mkdir(parents=True,exist_ok=True)
    with np.load(package/'placement_playback.npz',allow_pickle=False) as z:times=z['frame_timestamps'];T=z['asset_to_standard']
    with np.load(package/'asset_4dgs.npz',allow_pickle=False) as z:native_to_asset=z['asset_from_native']
    speed=r['native_speed_model_mps'] if r['suggested_motion']=='trajectory_with_estimated_retiming' else 0.
    positions=np.c_[(times-times[0])*speed,np.zeros((len(times),2))]
    after,frames=sample(package,times,positions,np.zeros(len(times)))
    before=np.broadcast_to(T,(len(times),4,4)).copy();before[:,:3,3]+=positions
    model,deform,pipe,_=load_asset(package/'runtime',53000)
    width,height=640,384;K=np.array([[550.,0,319.5],[0,550.,191.5],[0,0,1.]])
    center=positions[-1,0]/2;target=np.array([center,0,.85]);eye=np.array([center+2,-6,2.4])
    forward=target-eye;forward/=np.linalg.norm(forward);right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    world_camera=np.eye(4);world_camera[:3,:3]=np.stack([right,np.cross(forward,right),forward],axis=1);world_camera[:3,3]=eye
    yy,xx=np.mgrid[:height,:width];rays=np.stack([(xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1],np.ones_like(xx)],axis=-1)@world_camera[:3,:3].T
    hit=-eye[2]/rays[:,:,2];xy=eye[:2]+rays[:,:,:2]*hit[:,:,None]
    checker=(np.floor(xy[:,:,0]/.25)+np.floor(xy[:,:,1]/.25))%2
    floor=np.repeat(np.where(checker[:,:,None]>0,.66,.44),3,axis=2);floor[hit<=0]=.82
    def images():
        for i,frame in enumerate(frames):
            panels=[]
            for name,matrices in [('Before placement adapter',before),('Approximate placement adapter',after)]:
                native_cam,_=camera_in_native_frame(world_camera,matrices[i]@np.linalg.inv(T),T,native_to_asset)
                config=dict(width=width,height=height,frame_count=len(times),time_denominator=len(times)-1,
                            cameras=[dict(view=0,K=K.tolist(),camtoworld=native_cam.tolist())])
                rgb,alpha=raw_render(model,deform,pipe,camera(config,0,float(frame)))
                pixels=rgb.permute(1,2,0).cpu().numpy()+floor*(1-alpha.permute(1,2,0).cpu().numpy())
                assert np.isfinite(pixels).all()
                image=Image.fromarray(np.clip(pixels*255,0,255).astype(np.uint8));draw=ImageDraw.Draw(image)
                draw.rectangle((0,0,width,42),fill=(35,35,35));draw.text((8,5),f'{identity} | {name}',fill='white')
                draw.text((8,23),f't={times[i]-times[0]:.2f}s | speed estimate={speed:.2f} m/s | original appearance',fill='white')
                panels.append(np.asarray(image))
            yield np.concatenate(panels,axis=1)
    write_video(destination/f'.{identity}.preview.mp4',images(),fps=25)


@torch.no_grad()
def verify_practical_playback(identity):
    from playback import sample,validate_estimate
    from placement import camera_in_native_frame
    package=ASSET_ROOT/identity;readback=validate_estimate(package)
    model,deform,pipe,_=load_asset(package/'runtime',53000)
    cameras=read(package/'native_cameras.json')
    with np.load(package/'placement_playback.npz',allow_pickle=False) as z:
        times=z['frame_timestamps'][[0,len(z['frame_timestamps'])//2,-1]];T=z['asset_to_standard']
    with np.load(package/'asset_4dgs.npz',allow_pickle=False) as z:native_to_asset=z['asset_from_native']
    positions=np.c_[times*.6,np.ones(3),np.zeros(3)];yaw=np.linspace(-.3,.5,3)
    matrices,frames=sample(package,times,positions,yaw)
    measurements=[]
    for i,frame in enumerate(frames):
        record=next(c for c in cameras['cameras'] if c['view']==0)
        world_from_native=matrices[i]@native_to_asset
        scale=np.linalg.svd(world_from_native[:3,:3],compute_uv=False).mean()
        world_camera=world_from_native@np.asarray(record['camtoworld']);world_camera[:3,:3]/=scale
        recovered,_=camera_in_native_frame(world_camera,matrices[i]@np.linalg.inv(T),T,native_to_asset)
        np.testing.assert_allclose(recovered,record['camtoworld'],atol=1e-7)
        reference,ra=raw_render(model,deform,pipe,camera(cameras,0,float(frame)))
        changed=dict(cameras);changed['cameras']=[dict(record,camtoworld=recovered.tolist())]
        actual,aa=raw_render(model,deform,pipe,camera(changed,0,float(frame)))
        delta=(actual-reference).abs();alpha=(aa-ra).abs()
        assert torch.isfinite(actual).all() and float(delta.mean())<=1e-5 and float(alpha.mean())<=1e-5
        measurements.append(dict(frame=float(frame),rgb_mae=float(delta.mean()),rgb_max=float(delta.max()),alpha_mae=float(alpha.mean())))
    save(ROOT/'reports/practical_playback'/f'{identity}.json',dict(state='approximate_transform_camera_replay_passed',
         readback=readback,measurements=measurements,device=gpu_identity(),
         scope='Actual packaged runtime roundtrip after root/yaw removal and planned placement; not a perceptual contact or no-slip certificate'))


@torch.no_grad()
def verify_placement(identity):
    """Replay an actual similarity transform with full SH/covariances on CUDA."""
    output = ASSET_ROOT / identity
    device = gpu_identity()
    metadata = read(output / "placement.json")
    assert metadata['asset_sha256'] == sha(output / 'asset_4dgs.npz')
    with np.load(output/'asset_4dgs.npz', allow_pickle=False) as z:
        arrays = {k:z[k] for k in ['native_frame_xyz','native_frame_rotation','native_frame_scaling',
                  'native_shs','opacity','use_rgb','active_sh_degree','asset_from_native','frame_count']}
    transform = np.asarray(metadata['coordinates']['asset_to_standard_model_meters']) @ arrays['asset_from_native']
    scale = np.linalg.svd(transform[:3,:3],compute_uv=False).mean()
    rotation = transform[:3,:3]/scale
    cameras = read(output/'native_cameras.json')
    standard_cameras = dict(cameras)
    standard_cameras['cameras'] = []
    for c in cameras['cameras']:
        c2w = np.asarray(c['camtoworld'])
        new = np.eye(4)
        new[:3,:3] = rotation @ c2w[:3,:3]
        new[:3,3] = transform[:3,:3] @ c2w[:3,3] + transform[:3,3]
        standard_cameras['cameras'].append(dict(c,camtoworld=new.tolist()))
    sh, sh_error = rotate_sh(arrays['native_shs'],rotation,int(arrays['active_sh_degree'][0]))
    gates = read(ROOT/'config/export.json')['gates']
    assert sh_error <= gates['SH_rotation_absolute_max']
    rows = []
    native_camera_rows = []
    frames = int(arrays['frame_count'][0])
    for frame in sorted({0,frames//2,frames-1}):
        points, quats, scales, cov = transform_gaussians(arrays['native_frame_xyz'][frame],
              arrays['native_frame_rotation'][frame],arrays['native_frame_scaling'][frame],transform)
        assert cov['covariance_relative_max'] <= gates['covariance_relative_max']
        original = {k:torch.as_tensor(v[frame:frame+1] if k.startswith('native_frame_') else v,
                    device='cuda',dtype=torch.float32) for k,v in arrays.items()
                    if k in ['native_frame_xyz','native_frame_rotation','native_frame_scaling','native_shs','opacity']}
        transformed = dict(opacity=original['opacity'],native_shs=torch.as_tensor(sh,device='cuda'),
                    native_frame_xyz=torch.as_tensor(points[None],device='cuda'),
                    native_frame_rotation=torch.as_tensor(quats[None],device='cuda'),
                    native_frame_scaling=torch.as_tensor(scales[None],device='cuda'))
        for view in [0,8,16]:
            rgb, alpha = render_frame(original,arrays,cameras,view,0,0.)
            target, a = render_frame(transformed,arrays,standard_cameras,view,0,0.)
            error = (rgb-target).abs()
            mask = alpha[0] > .01
            assert mask.any()
            row = dict(frame=frame,view=view,foreground_MAE=float(error[:,mask].mean()),
                    max_error=float(error.max()),alpha_MAE=float((a-alpha).abs().mean()),**cov)
            rows.append(row)
            from placement import camera_in_native_frame
            world_camera=next(c for c in standard_cameras['cameras'] if c['view']==view)
            recovered,_=camera_in_native_frame(np.asarray(world_camera['camtoworld']),np.eye(4),
                 np.asarray(metadata['coordinates']['asset_to_standard_model_meters']),arrays['asset_from_native'])
            recovered_cameras=dict(cameras,cameras=[dict(c,camtoworld=recovered.tolist()) if c['view']==view else c for c in cameras['cameras']])
            rrgb,ralpha=render_frame(original,arrays,recovered_cameras,view,0,0.)
            native_camera_rows.append(dict(frame=frame,view=view,foreground_MAE=float((rgb-rrgb).abs()[:,mask].mean()),
                   max_error=float((rgb-rrgb).abs().max()),alpha_MAE=float((alpha-ralpha).abs().mean())))
    materialized_passed = all(r['foreground_MAE']<=gates['foreground_RGB_MAE_max'] and
                 r['max_error']<=gates['RGB_max_error'] and r['alpha_MAE']<=gates['alpha_MAE_max'] for r in rows)
    passed = all(r['foreground_MAE']<=gates['foreground_RGB_MAE_max'] and
                 r['max_error']<=gates['RGB_max_error'] and r['alpha_MAE']<=gates['alpha_MAE_max'] for r in native_camera_rows)
    save(output/'placement_replay.json',dict(state='native_camera_placement_render_passed' if passed else 'failed',
         authoritative_representation='Untouched native Gaussians, inverse-transformed placed-world camera',
         rows=native_camera_rows,materialized_similarity_diagnostic=dict(passed=materialized_passed,rows=rows),SH_error=sh_error,gates=gates,device=device,asset_sha256=metadata['asset_sha256'],
         placement_sha256=sha(output/'placement.json'),
         scope='Nine CUDA RGB/alpha renders with placed-world camera expressed in native frame; full direct point transformation separately diagnosed. No original data/model loaded. Does not validate gait, feet or loops.'))
    assert passed, native_camera_rows
    publish_catalogue()
    print(identity, 'native_camera_placement_render_passed', 'materialized_passed',materialized_passed, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["export", "replay", "runtime-package", "showcase", "placement-check", "placement-trial", "playback-check", "playback-preview", "catalogue"])
    ap.add_argument("identity")
    args = ap.parse_args()
    {
        "export": export_asset,
        "replay": verify_asset,
        "runtime-package": verify_runtime_package,
        "showcase": five_column_video,
        "placement-check": verify_placement,
        "placement-trial": render_placement_trial,
        "playback-check": verify_practical_playback,
        "playback-preview": practical_playback_preview,
        "catalogue": lambda identity: publish_catalogue(),
    }[args.stage](args.identity)
    if args.stage == "replay" and not case(args.identity).get("selection"):
        publish_catalogue()
