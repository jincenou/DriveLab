"""Generated video -> fixed soft RGBA -> training-only hull -> Grid4D inputs.

The original camera parser/normalization is vendored to preserve its numerical
conventions. Masks run in four independent view groups, not one reused process.
"""
import argparse, math, json, sys, shutil
from pathlib import Path
import numpy as np
from PIL import Image
from common import (
    ROOT,
    CODE_ROOT,
    FRONTEND,
    RESOURCES,
    case,
    read,
    save,
    sha,
    gpu_identity,
    generation_path,
)


def locations(identity):
    spec = case(identity)
    return (
        generation_path(identity),
        ROOT / "data" / identity / "rgba",
        spec["prepared_frames"],
    )


def masks(identity, group):
    import av, torch

    from frontend import apply_temporal_policy
    apply_temporal_policy(identity)
    source, dest, count = locations(identity)
    device = gpu_identity()
    assert read(source / "metadata.json")["output"]["frames_per_video"] == count
    sys.path[:0] = [str(CODE_ROOT / "vendor/segmentation"), str(FRONTEND)]
    from sam2.build_sam import build_sam2_video_predictor
    from fdanyone.foreground import predict_foreground_masks
    from fdanyone.assets import resolve_foreground_model
    from vendor.sam_transfer_guard import install_transfer_guard

    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        RESOURCES["sam2_weights"],
        device="cuda",
        apply_postprocessing=False,
    )
    install_transfer_guard(predictor)
    records = []
    for view in range(group, 24, 4):
        folder = dest / "images" / f"cam{view:02d}"
        folder.mkdir(parents=True, exist_ok=False)
        tmp = dest / "sam2_input" / f"cam{view:02d}"
        tmp.mkdir(parents=True, exist_ok=False)
        frames = []
        with av.open(str(source / "videos" / f"{view:02d}.mp4")) as con:
            con.streams.video[0].thread_count = 1
            for i, f in enumerate(con.decode(video=0)):
                assert abs(float(f.pts * f.time_base) - i / 25) < 1e-7
                im = f.to_image().resize((352, 640), Image.Resampling.LANCZOS)
                frames.append(np.array(im))
                im.save(tmp / f"{i:05d}.jpg", quality=95, subsampling=0)
        assert len(frames) == count
        keys = list(range(0, count, 15))
        ref = predict_foreground_masks(
            tuple(frames[i] for i in keys),
            resolve_foreground_model(FRONTEND / "models"),
            "cuda:0",
        )
        alpha = np.empty((count, 640, 352), np.uint8)
        seen = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(
                video_path=str(tmp),
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
            )
            predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=ref[0] > 127)
            for i, ids, logits in predictor.propagate_in_video(state):
                assert ids == [1] and torch.isfinite(logits).all()
                alpha[i] = (
                    logits[0, 0]
                    .sigmoid()
                    .float()
                    .cpu()
                    .numpy()
                    .__mul__(255)
                    .round()
                    .astype(np.uint8)
                )
                seen.append(i)
        assert seen == list(range(count))
        checks = []
        for i, r in zip(keys, ref, strict=True):
            a, b = alpha[i] > 127, r > 127
            checks.append(float((a & b).sum() / max((a | b).sum(), 1)))
        for i, (rgb, a) in enumerate(zip(frames, alpha, strict=True)):
            assert (
                int((a > 127).sum()) >= 100 and a.max() > 200 and np.unique(a).size > 2
            )
            im = Image.fromarray(rgb)
            im.putalpha(Image.fromarray(a))
            im.save(folder / f"{i:06d}.png")
        records.append(
            dict(
                view=view,
                frames=count,
                keyframe_sam2_birefnet_iou=checks,
                quality_warning=bool(min(checks) < 0.65 or np.median(checks) < 0.8),
                device=device,
            )
        )
        save(ROOT / "reports" / f"{identity}_masks_{group}.json", records)
        shutil.rmtree(tmp)
        del state, frames, alpha, ref
        torch.cuda.empty_cache()
        print(identity, view, "RGBA complete", flush=True)


def project(points, camera):
    w = np.linalg.inv(camera["camera_to_world"])
    p = points @ w[:3, :3].T + w[:3, 3]
    K = np.array(camera["K"])
    K[:2] *= 0.5
    q = p @ K.T
    uv = np.rint(q[:, :2] / q[:, 2:]).astype(int)
    valid = (
        (p[:, 2] > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < 352)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < 640)
    )
    return uv, valid


def hull(identity):
    from scipy.ndimage import binary_erosion
    from scipy.spatial.transform import Rotation

    source, dest, count = locations(identity)
    assert not (dest / "init.npz").exists()
    camera_info = read(source / "cameras.json")
    cams = camera_info["cameras"]
    assert len(cams) == 24
    assert camera_info["world_frame"]["name"] == "root_local_human_world"
    train = [i for i in range(24) if i not in (0, 8, 16)]
    shape = (64, 96, 64)
    axes = np.meshgrid(
        np.linspace(-1.2, 1.2, 64),
        np.linspace(-0.1, 2.1, 96),
        np.linspace(-1.2, 1.2, 64),
        indexing="ij",
    )
    grid = np.stack(axes, -1).reshape(-1, 3)
    projections = [project(grid, c) for c in cams]
    positions = []
    colors = []
    times = []
    checks = []
    for frame in range(0, count, 15):
        images = [
            np.array(Image.open(dest / "images" / f"cam{i:02d}" / f"{frame:06d}.png"))
            for i in range(24)
        ]
        assert all(im.shape == (640, 352, 4) for im in images)
        votes = np.zeros(len(grid), np.int16)
        for i in train:
            uv, valid = projections[i]
            hit = np.zeros(len(grid), bool)
            hit[valid] = images[i][uv[valid, 1], uv[valid, 0], 3] >= 128
            votes += hit
        occupied = (votes >= math.ceil(0.95 * len(train))).reshape(shape)
        if any(
            face.any()
            for face in (
                occupied[0],
                occupied[-1],
                occupied[:, 0],
                occupied[:, -1],
                occupied[:, :, 0],
                occupied[:, :, -1],
            )
        ):
            raise RuntimeError(
                "Hull touches bounds; inspect input, do not clip silently"
            )
        idx = np.flatnonzero((occupied & ~binary_erosion(occupied)).ravel())
        assert len(idx) >= 50
        pts = grid[idx]
        rgb = np.zeros((len(pts), 3))
        n = np.zeros(len(pts))
        for i in train:
            uv, valid = project(pts, cams[i])
            valid[valid] &= images[i][uv[valid, 1], uv[valid, 0], 3] >= 128
            rgb[valid] += images[i][uv[valid, 1], uv[valid, 0], :3]
            n[valid] += 1
        assert n.min() > 0
        rgb /= n[:, None]
        positions.append(pts.astype(np.float32))
        colors.append(rgb.astype(np.float32))
        times.append(np.full(len(pts), frame / (count - 1), np.float32))
        checks.append(dict(frame=frame, points=len(pts)))
    np.savez_compressed(
        dest / "init.npz",
        positions=np.concatenate(positions),
        colors=np.concatenate(colors),
        times=np.concatenate(times),
        time_denominator=count - 1,
    )
    sparse = dest / "sparse/0"
    sparse.mkdir(parents=True, exist_ok=False)
    cameras = []
    images = []
    for idx, c in enumerate(cams, 1):
        K = np.array(c["K"])
        K[:2] *= 0.5
        cameras.append(
            f"{idx} PINHOLE 352 640 {K[0,0]:.12g} {K[1,1]:.12g} {K[0,2]:.12g} {K[1,2]:.12g}"
        )
        w = np.linalg.inv(c["camera_to_world"])
        q = Rotation.from_matrix(w[:3, :3]).as_quat()
        q = np.r_[q[3], q[:3]]
        values = " ".join(f"{v:.15g}" for v in np.r_[q, w[:3, 3]])
        images.append(f"{idx} {values} {idx} cam{idx-1:02d}/000000.png\n")
    (sparse / "cameras.txt").write_text("\n".join(cameras) + "\n")
    (sparse / "images.txt").write_text("\n".join(images) + "\n")
    lines = [
        f"{i+1} {p[0]:.8g} {p[1]:.8g} {p[2]:.8g} {int(c[0])} {int(c[1])} {int(c[2])} 0"
        for i, (p, c) in enumerate(zip(positions[0], colors[0], strict=True))
    ]
    (sparse / "points3D.txt").write_text("\n".join(lines) + "\n")
    save(
        ROOT / "reports" / f"{identity}_hull.json",
        dict(keyframes=checks, heldout_excluded=[0, 8, 16], source=str(source)),
    )


def configure(identity):
    from vendor.ftgs_io.FreeTime_dataset import FreeTimeParser
    from vendor.ftgs_io.normalize import transform_points

    source, dest, count = locations(identity)
    out = ROOT / "data" / identity
    assert not (out / "fdanyone.json").exists()
    parsed = FreeTimeParser(str(dest), normalize=True, start_frame=0, end_frame=count)
    with np.load(dest / "init.npz") as z:
        points = transform_points(parsed.transform, z["positions"])
        at_zero = np.isclose(z["times"], 0, atol=1e-7)
        assert at_zero.sum() > 100
        np.savez(
            out / "canonical.npz",
            points=points[at_zero],
            colors=np.clip(z["colors"][at_zero] / 255.0, 0, 1),
        )
    cameras = []
    hashes = []
    for index, cid in enumerate(parsed.camera_ids):
        view = int(cid) - 1
        assert parsed.camera_names[index] == f"cam{view:02d}"
        K = parsed.Ks_dict[cid]
        assert np.allclose(K[:2, 2], [176, 320])
        cameras.append(
            dict(view=view, K=K.tolist(), camtoworld=parsed.camtoworlds[index].tolist())
        )
        assert sorted(
            int(p.stem) for p in (dest / "images" / f"cam{view:02d}").glob("*.png")
        ) == list(range(count))
        for frame in range(count):
            p = dest / "images" / f"cam{view:02d}" / f"{frame:06d}.png"
            with Image.open(p) as im:
                assert im.mode == "RGBA" and im.size == (352, 640)
            hashes.append(dict(view=view, frame=frame, sha256=sha(p)))
    low, high = points.min(0), points.max(0)
    margin = np.maximum((high - low) * 0.1, 0.02)
    centers = parsed.camtoworlds[:, :3, 3]
    radius = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
    manifest = dict(
        version=1,
        label=identity,
        source_data=str(dest),
        frame_count=count,
        time_denominator=count - 1,
        fps=25,
        width=352,
        height=640,
        sample_span_seconds=(count - 1) / 25,
        train_views=[v for v in range(24) if v not in (0, 8, 16)],
        heldout_views=[0, 8, 16],
        cameras=cameras,
        canonical_npz="canonical.npz",
        initial_points=int(at_zero.sum()),
        scene_radius=radius,
        source_to_normalized=parsed.transform.tolist(),
        motion_aabb_min=(low - margin).tolist(),
        motion_aabb_max=(high + margin).tolist(),
        source_init_sha256=sha(dest / "init.npz"),
        input_hashes=hashes,
        initialization="time0 training-only visual hull; all-time training hull bounds",
    )
    save(out / "fdanyone.json", manifest)
    config = read(CODE_ROOT / "config/N04.json")
    bound = (
        math.ceil(
            float(
                np.abs(
                    np.array([manifest["motion_aabb_min"], manifest["motion_aabb_max"]])
                ).max()
            )
            * 1.1
            * 20
        )
        / 20
    )
    config.update(
        label=identity,
        created_at=__import__("datetime").datetime.now().isoformat(),
        saving_iterations=[33000],
    )
    config["dataset"].update(label=identity, source_path=str(out), model_path="")
    config["dataset"]["grid_args"]["bound"] = bound
    config["provenance"].update(
        manifest=str(out / "fdanyone.json"),
        manifest_sha256=sha(out / "fdanyone.json"),
        initialization=str(out / "canonical.npz"),
        initialization_sha256=sha(out / "canonical.npz"),
        motion_aabb=[manifest["motion_aabb_min"], manifest["motion_aabb_max"]],
        bound=bound,
        official_recipe=str(
            CODE_ROOT / "vendor/grid4d/arguments/neu3d/cook_spinach.py"
        ),
    )
    config["invariants"].update(frame_count=count)
    save(ROOT / "config" / f"{identity}.json", config)
    save(
        ROOT / "reports" / f"{identity}_dataset_ready.json",
        dict(
            state="ready",
            frame_count=count,
            initial_points=int(at_zero.sum()),
            bound=bound,
            manifest_sha256=sha(out / "fdanyone.json"),
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["masks", "hull", "configure"])
    ap.add_argument("identity")
    ap.add_argument("--group", type=int, choices=range(4))
    a = ap.parse_args()
    if a.stage == "masks":
        assert a.group is not None
        masks(a.identity, a.group)
    else:
        {"hull": hull, "configure": configure}[a.stage](a.identity)
