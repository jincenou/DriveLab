"""Fixed reference metrics and native-runtime regression; no best-checkpoint search."""

import argparse
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
import lpips
from common import ROOT, case, read, save, sha, gpu_identity
from data import manifest, camera
from reconstruction import load_asset, raw_render
from vendor.metrics_loss import ssim


def scalar(pred, target, alpha, mask):
    error = (pred - target) ** 2
    s = alpha > 0.5
    union = (s | mask).sum()
    return {
        "full_psnr": -10 * math.log10(max(float(error.mean()), 1e-12)),
        "foreground_psnr": -10 * math.log10(max(float(error[mask].mean()), 1e-12)),
        "silhouette_iou": float((s & mask).sum() / max(union, 1)),
        "outside_alpha_mean": float(alpha[~mask].mean()),
    }


def crop(masks):
    y, x = np.where(np.logical_or.reduce(masks))
    assert len(x) > 100
    pad = max(4, round(0.05 * max(x.max() - x.min() + 1, y.max() - y.min() + 1)))
    return [
        max(0, int(x.min()) - pad),
        max(0, int(y.min()) - pad),
        min(352, int(x.max()) + pad + 1),
        min(640, int(y.max()) + pad + 1),
    ]


@torch.no_grad()
def verify_native(identity):
    """Pinned existing asset loaded through the new local vendor runtime."""
    spec = case(identity)
    run = Path(spec["baseline_asset"])
    model, deform, pipe, config = load_asset(run, 53000)
    data = read(run / "asset_camera_data.json")
    errors = []
    with np.load(run / "snapshots/53000_reference.npz") as refs:
        for key in refs.files:
            v, f = key.split("_")
            rgb, a = raw_render(
                model, deform, pipe, camera(data, int(v[1:]), float(f[1:]))
            )
            errors.append(
                float(np.abs(torch.cat((rgb, a)).cpu().numpy() - refs[key]).max())
            )
    assert len(errors) == 12 and max(errors) < 1e-5, errors
    save(
        ROOT / "reports" / f"{identity}_runtime_equivalence.json",
        dict(
            state="passed",
            probes=12,
            max_error=max(errors),
            vendor_code_only=True,
            baseline_asset=str(run),
        ),
    )


@torch.no_grad()
def evaluate(identity):
    run = ROOT / "outputs" / identity / "asset"
    spec = case(identity)
    model, deform, pipe, config = load_asset(run, 53000)
    data, _ = manifest(identity)
    saved = read(run / "asset_camera_data.json")
    old = read(spec["baseline_metrics"]) if "baseline_metrics" in spec else None
    network = lpips.LPIPS(net="alex").cuda().eval()
    frames = np.rint(np.linspace(0, data["frame_count"] - 1, 13)).astype(int).tolist()
    rows = []
    for view in (0, 8, 16):

        def path(frame):
            return (
                Path(data["source_data"])
                / "images"
                / f"cam{view:02d}"
                / f"{frame:06d}.png"
            )

        masks = [np.asarray(Image.open(path(frame)))[..., 3] > 127 for frame in frames]
        box = crop(masks)
        x0, y0, x1, y1 = box
        for frame in frames:
            rgba = np.asarray(Image.open(path(frame)), np.float32) / 255
            target = rgba[..., :3] * rgba[..., 3:] + 0.5 * (1 - rgba[..., 3:])
            mask = rgba[..., 3] > 0.5
            rgb, a = raw_render(model, deform, pipe, camera(saved, view, frame))
            rgb = rgb.permute(1, 2, 0).cpu().numpy()
            a = a[0].cpu().numpy()
            pred = np.clip(rgb + 0.5 * (1 - a[..., None]), 0, 1)
            x = torch.tensor(pred.transpose(2, 0, 1).copy(), device="cuda")[None]
            y = torch.tensor(target.transpose(2, 0, 1).copy(), device="cuda")[None]
            row = dict(
                view=view,
                frame=frame,
                **scalar(pred, target, a, mask),
                ssim=float(ssim(x, y)),
                crop_lpips=float(
                    network(
                        x[:, :, y0:y1, x0:x1] * 2 - 1, y[:, :, y0:y1, x0:x1] * 2 - 1
                    )
                ),
                crop_xyxy=box,
                input_sha256=sha(path(frame)),
            )
            rows.append(row)
    if old is not None:
        assert sorted(
            (r["view"], r["frame"], r["crop_xyxy"], r["input_sha256"]) for r in rows
        ) == sorted(
            (r["view"], r["frame"], r["crop_xyxy"], r["input_sha256"])
            for r in old["samples"]
        )
    keys = (
        "full_psnr",
        "foreground_psnr",
        "ssim",
        "crop_lpips",
        "silhouette_iou",
        "outside_alpha_mean",
    )
    means = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    if old is None:
        save(
            ROOT / "reports" / f"{identity}_quality.json",
            dict(
                state="measured_visual_review_pending",
                mean=means,
                samples=rows,
                quality_approved=False,
                baseline=None,
                protocol=f"39 generated development references; not real unseen {spec.get('source_dataset', 'waymo')} ground truth",
            ),
        )
        return
    delta = {k: means[k] - old["mean"][k] for k in keys}
    gate = read(ROOT / "config/recipe.json")["guardrails"]
    accepted = (
        delta["foreground_psnr"] >= -gate["psnr_drop_db"]
        and delta["ssim"] >= -gate["ssim_drop"]
        and delta["crop_lpips"] <= gate["lpips_increase"]
    )
    save(
        ROOT / "reports" / f"{identity}_quality.json",
        dict(
            state="passed" if accepted else "quality_regression",
            mean=means,
            baseline=old["mean"],
            delta=delta,
            samples=rows,
            protocol="Same fixed generated H development references,39/person; not real unseen-view truth",
            thresholds=gate,
        ),
    )
    # Full-time old/new/reference video is streamed instead of writing redundant render arrays.
    del model, deform
    torch.cuda.empty_cache()
    new_model, new_deform, new_pipe, _ = load_asset(run, 53000)
    old_model, old_deform, old_pipe, _ = load_asset(Path(spec["baseline_asset"]), 53000)
    from assets import write_video

    def images():
        for frame in range(data["frame_count"]):
            tiles = []
            p = Path(data["source_data"]) / "images/cam00" / f"{frame:06d}.png"
            rgba = np.asarray(Image.open(p), np.float32) / 255
            tiles.append(rgba[..., :3] * rgba[..., 3:] + 0.5 * (1 - rgba[..., 3:]))
            for m, d, pipe in (
                (old_model, old_deform, old_pipe),
                (new_model, new_deform, new_pipe),
            ):
                rgb, a = raw_render(m, d, pipe, camera(saved, 0, frame))
                tiles.append(
                    (rgb + 0.5 * (1 - a)).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                )
            strip = np.rint(np.concatenate(tiles, 1) * 255).astype(np.uint8)
            canvas = Image.new("RGB", (1056, 668), (80, 80, 80))
            canvas.paste(Image.fromarray(strip), (0, 28))
            ImageDraw.Draw(canvas).text(
                (5, 7),
                f"{identity} v00 f{frame}: Generated reference | Historical A | Refactored A",
                fill="white",
            )
            if frame in (0, (data["frame_count"] - 1) // 2, data["frame_count"] - 1):
                canvas.save(
                    ROOT / "reports" / f"{identity}_f{frame:03d}.jpg", quality=94
                )
            yield np.asarray(canvas)

    write_video(ROOT / "reports" / f"{identity}_comparison.mp4", images(), 25)
    if not accepted:
        raise RuntimeError(
            f"{identity}: quality changed; keep historical implementation"
        )


@torch.no_grad()
def compare_prefix(identity):
    """Compare old/full and new/truncated assets against identical retained targets."""
    from common import ASSET_ROOT
    from assets import load_npz, tensor_asset, render_frame

    specification = case(identity)
    assert "temporal_subset" in specification
    data, _ = manifest(identity)
    frames = np.rint(np.linspace(0, data["frame_count"]-1, 13)).astype(int).tolist()
    source_start = specification["temporal_subset"].get("range_start", 0)
    network = lpips.LPIPS(net="alex").cuda().eval()
    records, hashes = {}, {}
    for method, folder in (("original_full", Path(specification["comparison_asset"])),
                           ("trimmed_retrained", ASSET_ROOT / identity)):
        assert read(folder / "replay_validation.json")["state"] == "NPZ_integer_frame_replay_passed"
        arrays = load_npz(folder / "asset_4dgs.npz")
        offset = source_start if method == "original_full" else 0
        assert int(arrays["frame_count"][0]) > max(frames)+offset and float(arrays["fps"][0]) == 25
        asset, cameras = tensor_asset(arrays), read(folder / "native_cameras.json")
        rows = []
        for view in (0, 8, 16):
            paths = [Path(data["source_data"])/"images"/f"cam{view:02d}"/f"{frame:06d}.png" for frame in frames]
            masks = [np.asarray(Image.open(p))[..., 3] > 127 for p in paths]
            box = crop(masks)
            x0, y0, x1, y1 = box
            for frame, path, mask in zip(frames, paths, masks, strict=True):
                rgba = np.asarray(Image.open(path), np.float32)/255
                target = rgba[..., :3]*rgba[..., 3:] + 0.5*(1-rgba[..., 3:])
                rgb, alpha = render_frame(asset, arrays, cameras, view, frame+offset, 0.0)
                a = alpha[0].cpu().numpy()
                pred = np.clip(rgb.permute(1, 2, 0).cpu().numpy()+0.5*(1-a[..., None]), 0, 1)
                x = torch.tensor(pred.transpose(2, 0, 1).copy(), device="cuda")[None]
                y = torch.tensor(target.transpose(2, 0, 1).copy(), device="cuda")[None]
                rows.append(dict(view=view, frame=frame, rendered_frame=frame+offset,
                                 original_source_frame=frame+source_start, input_sha256=sha(path), crop_xyxy=box,
                                 **scalar(pred, target, a, mask), ssim=float(ssim(x,y)),
                                 crop_lpips=float(network(x[:,:,y0:y1,x0:x1]*2-1, y[:,:,y0:y1,x0:x1]*2-1))))
        records[method] = rows
        hashes[method] = sha(folder / "asset_4dgs.npz")
        del asset, arrays
        torch.cuda.empty_cache()
    assert [(r['view'],r['frame'],r['input_sha256'],r['crop_xyxy']) for r in records['original_full']] == [(r['view'],r['frame'],r['input_sha256'],r['crop_xyxy']) for r in records['trimmed_retrained']]
    keys = ('full_psnr','foreground_psnr','ssim','crop_lpips','silhouette_iou','outside_alpha_mean')
    means = {method:{k:float(np.mean([r[k] for r in rows])) for k in keys} for method,rows in records.items()}
    result = dict(state='paired_same_prefix_measured', means=means,
                  delta={k:means['trimmed_retrained'][k]-means['original_full'][k] for k in keys},
                  samples=records, asset_sha256=hashes, quality_approved=False,
                  scope='Both assets compared on identical retained generated-reference frames/masks/crops. Not real Waymo novel-view truth or automatic visual acceptance.')
    save(ROOT / 'reports' / f'{identity}_paired_prefix_metrics.json', result)
    save(ASSET_ROOT / identity / 'paired_prefix_metrics.json', result)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["runtime", "quality", "compare-prefix"])
    ap.add_argument("identity")
    args = ap.parse_args()
    print(gpu_identity(), flush=True)
    {"runtime": verify_native, "quality": evaluate, "compare-prefix": compare_prefix}[args.stage](args.identity)
