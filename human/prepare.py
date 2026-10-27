"""Restore audited inputs, repair carried-object alpha, and prepare generated RGBA.

All algorithms are the fixed baseline; no changes to motion/identity or hidden pixels.
"""
import argparse, io, json, sys
from pathlib import Path
from fractions import Fraction
import numpy as np
from PIL import Image, ImageDraw
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
)


def encode(path, images, *, fps=25, timestamps=None):
    import av

    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264rgb", rate=Fraction(fps))
        stream.width, stream.height = images[0].size
        stream.pix_fmt = "rgb24"
        stream.thread_count = 1
        stream.options = {"crf": "0", "preset": "fast", "threads": "1"}
        for i, im in enumerate(images):
            frame = av.VideoFrame.from_image(im)
            frame.pts = i if timestamps is None else round((timestamps[i] - timestamps[0]) * 1000000)
            frame.time_base = 1 / Fraction(fps) if timestamps is None else Fraction(1, 1000000)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def inspect_input(identity):
    """Small, time-aligned source/input/pose contact sheet; never auto-accept quality."""
    import av

    spec = case(identity)
    source = Path(spec["selection"]["package"]) / "clip0"
    mapping = read(source / "inputs/frame_mapping.json")
    count = spec["prepared_frames"]
    with np.load(source / "segmentation/target_masks.npz") as archive:
        masks = archive["masks"]
    foreground = (masks > 127).sum((1, 2))
    opaque = (masks > 230).sum((1, 2)) / np.maximum(foreground, 1)
    alpha_metrics = dict(
        mask_source_sha256=sha(source / "segmentation/target_masks.npz"),
        first_area_relative_to_median=float(
            foreground[0] / max(float(np.median(foreground)), 1)
        ),
        first_opaque_fraction=float(opaque[0]),
        median_opaque_fraction=float(np.median(opaque)),
        p10_opaque_fraction=float(np.quantile(opaque, 0.1)),
        note="Cached mask diagnostics, not anatomical ground truth; brief later occlusion is not an automatic rejection",
    )
    output = ROOT / "review"
    old = output / f"{identity}.json"
    if old.exists():
        previous = read(old)
        if (
            all(sha(x["path"]) == x["sha256"] for x in previous["sources"])
            and sha(output / f"{identity}.jpg") == previous["image_sha256"]
        ):
            previous["alpha_metrics"] = alpha_metrics
            save(old, previous)
            return
    slots = [0, count // 2, count - 1]
    native = [
        mapping["frames"][i]["left_native_index"]
        if mapping["frames"][i]["interpolation_fraction"] < 0.5
        else mapping["frames"][i]["right_native_index"]
        for i in slots
    ]
    columns = [
        ("observed", Path(spec["selection"]["source"]), native),
        ("model input", Path(spec["raw_input"]), slots),
        ("pose comparison", Path(spec["selection"]["pose"]), slots),
    ]
    page = Image.new("RGB", (1080, 900), (55, 55, 55))
    records = []
    for col, (label, path, indices) in enumerate(columns):
        decoded = {}
        with av.open(str(path)) as container:
            container.streams.video[0].thread_count = 1
            for i, frame in enumerate(container.decode(video=0)):
                if i in indices:
                    decoded[i] = frame.to_image()
                if i >= max(indices):
                    break
        assert all(i in decoded for i in indices), (
            identity,
            label,
            indices,
            sorted(decoded),
        )
        for row, index in enumerate(indices):
            im = decoded[index].copy()
            im.thumbnail((354, 270))
            page.paste(im, (col * 360 + (360 - im.width) // 2, row * 300 + 28))
            ImageDraw.Draw(page).text(
                (col * 360 + 5, row * 300 + 5),
                f"{identity} {label} slot {slots[row]} / frame {index}",
                fill="white",
            )
        records.append(
            dict(kind=label, path=str(path), sha256=sha(path), selected_indices=indices)
        )
    output = ROOT / "review"
    output.mkdir(exist_ok=True)
    page.save(output / f"{identity}.jpg", quality=92)
    save(
        output / f"{identity}.json",
        dict(
            state="contact_ready_not_quality_approved",
            frames=slots,
            sources=records,
            image_sha256=sha(output / f"{identity}.jpg"),
            alpha_metrics=alpha_metrics,
        ),
    )


def restore_native(identity):
    import cv2

    cv2.setNumThreads(2)
    source = Path(case(identity)["selection"]["package"]) / "clip0"
    meta = read(source / "native.json")
    audit = Path(meta["source_camera_audit"])
    assert sha(audit) == meta["source_camera_audit_sha256"]
    with np.load(audit.parent / "native_cameras.npz") as z:
        K = z["K"]
        dist = z["distortion"]
    out = ROOT / "data" / identity / "repair"
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for row in meta["frames"]:
        assert sha(row["raw_source_path"]) == row["raw_source_sha256"]
        rgb = np.asarray(Image.open(row["raw_source_path"]).convert("RGB"))
        A = np.asarray(row["undistorted_to_input"])
        maps = cv2.initUndistortRectifyMap(
            K, dist, np.eye(3), A @ K, tuple(meta["size"]), cv2.CV_32FC1
        )
        im = Image.fromarray(
            cv2.remap(
                rgb,
                *maps,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
        )
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        import hashlib

        assert (
            hashlib.sha256(buf.getvalue()).hexdigest() == row["sha256"]
        ), "Native crop regeneration differs"
        frames.append(np.asarray(im))
    Image.fromarray(frames[0]).save(out / "source_first.png")
    save(
        out / "native_restore.json",
        dict(
            frames=len(frames),
            all_original_native_sha256_matched=True,
            source=str(source),
        ),
    )
    return source, frames, out


def restore(identity):
    import torch
    from torch.nn import functional as F

    device = gpu_identity()
    source, rgb, out = restore_native(identity)
    sys.path.insert(0, str(CODE_ROOT / "vendor/rife"))
    from train_log.IFNet_HDv3 import IFNet

    model = IFNet().eval().cuda()
    weights = torch.load(
        RESOURCES["rife_weights"], map_location="cpu", weights_only=True
    )
    weights = {k.removeprefix("module."): v for k, v in weights.items()}
    model.load_state_dict(
        {
            k: v
            for k, v in weights.items()
            if not k.startswith(("teacher.", "caltime."))
        },
        strict=True,
    )
    h, w = rgb[0].shape[:2]
    mapping = read(source / "inputs/frame_mapping.json")
    rows = mapping["frames"]
    tensors = [
        F.pad(
            torch.from_numpy(x[:, :, ::-1].copy())
            .permute(2, 0, 1)
            .float()
            .div(255)
            .unsqueeze(0),
            (0, (-w) % 64, 0, (-h) % 64),
        )
        for x in rgb
    ]
    images = [None] * len(rows)
    pending = []
    for row in rows:
        l, r = row["left_native_index"], row["right_native_index"]
        if l == r:
            images[row["slot"]] = rgb[l]
        else:
            pending.append(row)
    with torch.inference_mode():
        for start in range(0, len(pending), 4):
            batch = pending[start : start + 4]
            a = torch.cat([tensors[x["left_native_index"]] for x in batch]).cuda()
            b = torch.cat([tensors[x["right_native_index"]] for x in batch]).cuda()
            t = torch.tensor(
                [x["interpolation_fraction"] for x in batch], device="cuda"
            ).reshape(-1, 1, 1, 1)
            pred = model(torch.cat((a, b), 1), t, [16, 8, 4, 2, 1])[2][-1]
            im = (
                pred[:, :, :h, :w]
                .clamp(0, 1)
                .mul(255)
                .round()
                .byte()
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()[:, :, :, ::-1]
            )
            for x, arr in zip(batch, im, strict=True):
                images[x["slot"]] = arr.copy()
    assert all(x is not None for x in images)
    # The original compositing used lossless RIFE RGB. Reapply old masks and
    # compare pixels before allowing the restored source to repair anything.
    import av

    with np.load(source / "segmentation/target_masks.npz") as z:
        masks = z["masks"]
    with av.open(str(source / "inputs/main.mp4")) as c:
        old = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    errors = []
    for x, m, y in zip(images, masks, old, strict=True):
        alpha = m.astype(np.float32)[..., None] / 255
        z = np.rint(x * alpha + 255 * (1 - alpha)).clip(0, 255).astype(np.uint8)
        errors.append(int(np.abs(z.astype(int) - y.astype(int)).max()))
    assert max(errors) <= 1, ("Restored source not equivalent", max(errors))
    frames = out / "frames"
    frames.mkdir(exist_ok=False)
    for i, arr in enumerate(images):
        Image.fromarray(arr).save(frames / f"{i:05d}.jpg", quality=95, subsampling=0)
    encode(out / "unsegmented.mp4", [Image.fromarray(x) for x in images])
    save(
        out / "restore.json",
        dict(
            state="restored",
            frames=len(images),
            original_composite_max_RGB_error=max(errors),
            device=device,
        ),
    )


def accessory(identity):
    import av, torch

    gpu_identity()
    spec = case(identity)
    out = ROOT / "data" / identity / "repair"
    plan = read(out / "accessory_plan.json")
    assert read(out / "restore.json")["state"] == "restored"
    sys.path.insert(0, str(CODE_ROOT / "vendor/segmentation"))
    from sam2.build_sam import build_sam2_video_predictor
    from vendor.sam_transfer_guard import install_transfer_guard

    source = Path(spec["selection"]["package"]) / "clip0"
    with np.load(source / "segmentation/target_masks.npz") as z:
        body = z["masks"]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            RESOURCES["sam2_weights"],
            device="cuda",
            apply_postprocessing=False,
        )
        install_transfer_guard(predictor)
        state = predictor.init_state(
            video_path=str(out / "frames"),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
        predictor.add_new_points_or_box(
            state,
            frame_idx=0,
            obj_id=2,
            points=np.array(plan["points"], np.float32),
            labels=np.array(plan["labels"], np.int32),
            box=np.array(plan["box"], np.float32),
        )
        masks = []
        for i, ids, logits in predictor.propagate_in_video(state):
            assert i == len(masks) and ids == [2] and torch.isfinite(logits).all()
            masks.append(
                logits[0, 0]
                .sigmoid()
                .float()
                .cpu()
                .numpy()
                .__mul__(255)
                .round()
                .astype(np.uint8)
            )
    bag = np.array(masks)
    assert bag.shape == body.shape
    combined = np.maximum(body, bag)
    np.savez_compressed(out / "accessory_masks.npz", accessory=bag, combined=combined)
    with av.open(str(out / "unsegmented.mp4")) as c:
        rgb = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    images = [
        Image.fromarray(
            np.rint(x * (m[..., None] / 255) + 255 * (1 - m[..., None] / 255))
            .clip(0, 255)
            .astype(np.uint8)
        )
        for x, m in zip(rgb, combined, strict=True)
    ]
    encode(out / "corrected.mp4", images)
    probes = np.rint(np.linspace(0, len(images) - 1, 9)).astype(int)
    page = Image.new("RGB", (960, 9 * 260), (70, 70, 70))
    for row, i in enumerate(probes):
        before = (
            np.rint(
                rgb[i] * (body[i][..., None] / 255)
                + 255 * (1 - body[i][..., None] / 255)
            )
            .clip(0, 255)
            .astype(np.uint8)
        )
        for col, (label, im) in enumerate(
            [
                ("source", Image.fromarray(rgb[i])),
                ("old", Image.fromarray(before)),
                ("repaired", images[i].copy()),
            ]
        ):
            im.thumbnail((320, 232))
            page.paste(im, (col * 320, row * 260 + 25))
            ImageDraw.Draw(page).text(
                (col * 320 + 4, row * 260 + 4), f"{identity} {label} f{i}", fill="white"
            )
    page.save(out / "accessory_review.jpg", quality=93)
    save(
        out / "accessory_result.json",
        dict(
            state="needs_visual_review",
            frames=len(images),
            body_alpha_unchanged_or_increased=bool((combined >= body).all()),
            source_sha256=sha(source / "inputs/main.mp4"),
            corrected_sha256=sha(out / "corrected.mp4"),
            min_area=int((bag > 127).sum((1, 2)).min()),
            max_area=int((bag > 127).sum((1, 2)).max()),
        ),
    )


def bind_repair(identity):
    from dataclasses import replace

    sys.path.insert(0, str(FRONTEND))
    from fdanyone.video import (
        decode_canonical_clip,
        write_lossless_video,
        verify_lossless_video,
    )
    from fdanyone.motion.result import MotionResult

    spec = case(identity)
    out = ROOT / "data" / identity / "repair"
    assert read(out / "visual_review.json")["allow_generation"]
    old = decode_canonical_clip(
        spec["raw_input"], num_frames=spec["prepared_frames"], fps=Fraction(25)
    )
    new = decode_canonical_clip(
        out / "corrected.mp4", num_frames=spec["prepared_frames"], fps=Fraction(25)
    )
    assert (old.width, old.height) == (new.width, new.height)
    assert [
        (x.source_index, x.source_pts, x.canonical_timestamp) for x in old.frames
    ] == [(x.source_index, x.source_pts, x.canonical_timestamp) for x in new.frames]
    # Use the same explicit clip length as generation, including 61-frame inputs.
    MotionResult.validate.__defaults__ = (len(old.frames),)
    motion = MotionResult.load(spec["motion"])
    motion.validate_against_clip(old)
    revised = replace(
        motion,
        source_size_bytes=new.source_size_bytes,
        source_mtime_ns=new.source_mtime_ns,
    )
    revised.validate_against_clip(new)
    revised.save(out / "motion")
    assert sha(out / "motion/motion.safetensors") == sha(
        Path(spec["motion"]) / "motion.safetensors"
    )
    original = Path(spec["conditioning"])
    meta = read(original / "metadata.json")
    cond = out / "conditioning"
    cond.mkdir(exist_ok=False)
    for p in original.iterdir():
        if p.name not in ("metadata.json", meta["source_video"]):
            (cond / p.name).symlink_to(p.resolve(), target_is_directory=p.is_dir())
    write_lossless_video(new, cond / meta["source_video"])
    verify_lossless_video(new, cond / meta["source_video"])
    meta["appearance_repair"] = {
        "motion_tensors_unchanged": True,
        "input_pixels": "Union of observed bag and original body masks; no recoloring; unobserved surface unknown",
    }
    save(cond / "metadata.json", meta)
    cases = read(ROOT / "config/cases.json")
    cases[identity].update(
        raw_input=str(out / "corrected.mp4"),
        motion=str(out / "motion"),
        conditioning=str(cond),
        input_review=read(out / "visual_review.json"),
    )
    save(ROOT / "config/cases.json", cases)
    save(
        out / "binding.json",
        dict(
            motion_sha256=sha(out / "motion/motion.safetensors"),
            cameras_sha256=sha(cond / "cameras.json"),
            source_input=spec["raw_input"],
            old_conditioning=str(original),
            corrected_input_sha256=sha(out / "corrected.mp4"),
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "stage", choices=["inspect", "native", "restore", "accessory", "bind"]
    )
    ap.add_argument("identity")
    a = ap.parse_args()
    {
        "inspect": inspect_input,
        "native": restore_native,
        "restore": restore,
        "accessory": accessory,
        "bind": bind_repair,
    }[a.stage](a.identity)
