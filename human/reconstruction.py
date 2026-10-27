"""Native Grid4D math and the unchanged three-stage complete-A optimizer."""

import argparse, gc, json, math, os, random, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from random import randint, choice
from common import ROOT, BACKEND, case, read, save, sha, gpu_identity, disk_check

sys.path.insert(0, str(BACKEND))
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from data import KnownScene, camera, manifest, load_train

PARAMETERS = [
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_scaling",
    "_rotation",
    "_opacity",
]
Scene = KnownScene


def new_deform(config, regularize=False):
    d, opt = config["dataset"], config["optimization"]
    return DeformModel(
        grid_args=d["grid_args"],
        net_args=d["network_args"],
        scale_xyz=d["scale_xyz"],
        spatial_downsample_ratio=opt["spatial_downsample_ratio"],
        spatial_perturb_range=opt["spatial_perturb_range"],
        temporal_downsample_ratio=opt["temporal_downsample_ratio"],
        temporal_perturb_range=opt["temporal_perturb_range"],
        reg_spatial_able=regularize and opt["lambda_spatial_tv"] > 0,
        reg_temporal_able=regularize and opt["lambda_temporal_tv"] > 0,
    )


def raw_render(model, deform, pipe, cam, background=0.0):
    time = cam.fid.unsqueeze(0).expand(len(model.get_xyz), -1)
    warp = deform.step(model.get_xyz.detach(), time)
    image = render(
        cam,
        model,
        pipe,
        torch.full((3,), background, device="cuda"),
        warp["d_xyz"],
        warp["d_rotation"],
        warp["d_scaling"],
    )["render"]
    white = render(
        cam,
        model,
        pipe,
        torch.zeros(3, device="cuda"),
        warp["d_xyz"],
        warp["d_rotation"],
        warp["d_scaling"],
        override_color=torch.ones_like(model.get_xyz),
    )["render"]
    assert float((white.max(0).values - white.min(0).values).max()) < 1e-5
    return image, white[:1]


def load_asset(run, iteration):
    config = json.loads((run / "config.json").read_text())
    model = GaussianModel(config["dataset"]["sh_degree"])
    model.load_ply(
        str(run / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply")
    )
    snapshot = json.loads((run / "snapshots" / f"{iteration}.json").read_text())
    model.active_sh_degree = snapshot["active_sh_degree"]
    deform = new_deform(config)
    # Native loader restores BOTH hash encoders and deformation network, strict by default.
    deform.load_weights(str(run), iteration)
    deform.grid4d.eval()
    deform.deform.eval()
    return model, deform, SimpleNamespace(**config["pipeline"]), config


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all(),
    )


def set_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a.detach().cpu(), b.detach().cpu())
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def native_training(dataset, opt, pipe, saving_iterations, monitor):
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(
        grid_args=dataset.grid_args,
        net_args=dataset.network_args,
        spatial_downsample_ratio=opt.spatial_downsample_ratio,
        spatial_perturb_range=opt.spatial_perturb_range,
        temporal_downsample_ratio=opt.temporal_downsample_ratio,
        temporal_perturb_range=opt.temporal_perturb_range,
        scale_xyz=dataset.scale_xyz,
        reg_spatial_able=opt.lambda_spatial_tv > 0.0,
        reg_temporal_able=opt.lambda_temporal_tv > 0.0,
    )
    deform.train_setting(opt)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    bg_color = [dataset.background_value] * 3
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = None
    first_iteration = 1
    first_iteration, viewpoint_stack = monitor.initialize(scene, gaussians, deform)
    for iteration in range(first_iteration, opt.iterations + 1):
        monitor.before(iteration, viewpoint_stack, gaussians, deform)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        if opt.data_sample == "random":
            viewpoint_cam = choice(scene.getTrainCameras())
        elif opt.data_sample == "order":
            viewpoint_cam = viewpoint_stack.pop(0)
        elif opt.data_sample == "stack":
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid
        reg = 0.0
        if iteration < opt.warm_up:
            d_rotation, d_scaling = (0.0, 0.0)
            d_xyz = 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)
            xyz = gaussians.get_xyz.detach()
            deform_pkgs = deform.step(xyz, time_input)
            d_xyz, d_rotation, d_scaling = (
                deform_pkgs["d_xyz"],
                deform_pkgs["d_rotation"],
                deform_pkgs["d_scaling"],
            )
            if opt.lambda_spatial_tv > 0.0 and (
                not opt.reg_after_densify or iteration > opt.densify_until_iter
            ):
                reg += torch.mean(deform_pkgs["reg_spatial"]) * opt.lambda_spatial_tv
            if opt.lambda_temporal_tv > 0.0 and (
                not opt.reg_after_densify or iteration > opt.densify_until_iter
            ):
                reg += torch.mean(deform_pkgs["reg_temporal"]) * opt.lambda_temporal_tv
        render_pkg_re = render(
            viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling
        )
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg_re["render"],
            render_pkg_re["viewspace_points"],
            render_pkg_re["visibility_filter"],
            render_pkg_re["radii"],
        )
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (
            (1.0 - opt.lambda_dssim) * Ll1
            + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            + reg
        )
        loss.backward()
        monitor.backward(
            iteration, viewpoint_cam, gaussians, deform, render_pkg_re, loss
        )
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device("cpu")
        with torch.no_grad():
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            if iteration in saving_iterations:
                scene.save(iteration)
                deform.save_weights(scene.model_path, iteration)
                monitor.saved(iteration, scene, gaussians, deform)
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                        opt.disable_ws_prune,
                    )
                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)
            monitor.after(
                iteration,
                viewpoint_cam,
                gaussians,
                deform,
                loss,
                iteration < opt.iterations,
                viewpoint_stack,
            )


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_suffix(".writing")
    torch.save(value, temporary)
    temporary.replace(path)


def restore(source, regularize=True):
    source = Path(source)
    config = read(source / "config.json")
    state = torch.load(source / "recovery.pt", map_location="cpu", weights_only=False)
    model = GaussianModel(config["dataset"]["sh_degree"])
    for name in PARAMETERS:
        setattr(model, name, torch.nn.Parameter(state["gaussians"][name].cuda()))
    opt = SimpleNamespace(**config["optimization"])
    model.training_setup(opt)
    model.optimizer.load_state_dict(state["gaussian_optimizer"])
    for name in ("max_radii2D", "xyz_gradient_accum", "denom"):
        setattr(model, name, state["statistics"][name].cuda())
    model.active_sh_degree = state["active_sh_degree"]
    deform = new_deform(config, regularize=regularize)
    deform.train_setting(opt)
    deform.grid4d.load_state_dict(state["hash"], strict=True)
    deform.deform.load_state_dict(state["deform"], strict=True)
    deform.optimizer.load_state_dict(state["deform_optimizer"])
    for name in PARAMETERS:
        equal(getattr(model, name), state["gaussians"][name])
    equal(model.optimizer.state_dict(), state["gaussian_optimizer"])
    equal(deform.optimizer.state_dict(), state["deform_optimizer"])
    set_rng(state["rng"])
    return (
        model,
        deform,
        SimpleNamespace(**config["pipeline"]),
        config,
        read(source / "asset_camera_data.json"),
        state,
    )


def capture(run, nominal, actual, model, deform, pipe, data, stack, counts):
    """One self-contained state, preserving RNG across reference rendering."""
    saved = rng_state()
    flags = deform.reg_spatial_able, deform.reg_temporal_able
    deform.reg_spatial_able = deform.reg_temporal_able = False
    refs = {}
    try:
        with torch.no_grad():
            for view in (1, 9, 17):
                for frame in (
                    0,
                    (data["frame_count"] - 1) / 2,
                    data["frame_count"] - 1,
                    12.25,
                ):
                    rgb, alpha = raw_render(
                        model, deform, pipe, camera(data, view, frame)
                    )
                    refs[f"v{view}_f{frame:g}"] = torch.cat((rgb, alpha)).cpu().numpy()
    finally:
        deform.reg_spatial_able, deform.reg_temporal_able = flags
        set_rng(saved)
    target = run / "snapshots"
    target.mkdir(exist_ok=True)
    np.savez_compressed(target / f"{nominal}_reference.npz", **refs)
    files = [
        run / "point_cloud" / f"iteration_{nominal}" / "point_cloud.ply",
        run / "deform" / f"iteration_{nominal}" / "deform.pth",
    ]
    save(
        target / f"{nominal}.json",
        dict(
            iteration=nominal,
            actual_optimizer_updates=actual,
            active_sh_degree=model.active_sh_degree,
            points=len(model.get_xyz),
            files={str(p.relative_to(run)): sha(p) for p in files},
        ),
    )
    checkpoint(run, nominal, actual, model, deform, stack, counts)


def checkpoint(run, iteration, actual, model, deform, stack, counts):
    atomic_torch(
        run / "recovery.pt",
        dict(
            iteration=iteration,
            global_actual_updates=actual,
            gaussians={name: getattr(model, name).detach() for name in PARAMETERS},
            gaussian_optimizer=model.optimizer.state_dict(),
            statistics={
                name: getattr(model, name)
                for name in ("max_radii2D", "xyz_gradient_accum", "denom")
            },
            active_sh_degree=model.active_sh_degree,
            hash=deform.grid4d.state_dict(),
            deform=deform.deform.state_dict(),
            deform_optimizer=deform.optimizer.state_dict(),
            rng=rng_state(),
            stack=stack,
            counts=counts,
        ),
    )


class Monitor:
    """Small recovery/finite-value monitor; no diagnostic render dumping."""

    def __init__(self, run, config, resume=False):
        self.run, self.config, self.resume = run, config, resume
        self.counts = dict(
            optimizer_calls=0, dynamic_gradient_updates=0, training_renders=0
        )
        self.started = time.monotonic()
        self.trace = (run / "trace.jsonl").open("a" if resume else "x", buffering=1)

    def initialize(self, scene, model, deform):
        self.scene = scene
        if not self.resume:
            return 1, None
        state = torch.load(
            self.run / "recovery.pt", map_location="cpu", weights_only=False
        )
        for name in PARAMETERS:
            setattr(model, name, torch.nn.Parameter(state["gaussians"][name].cuda()))
        model.training_setup(SimpleNamespace(**self.config["optimization"]))
        model.optimizer.load_state_dict(state["gaussian_optimizer"])
        for name in ("max_radii2D", "xyz_gradient_accum", "denom"):
            setattr(model, name, state["statistics"][name].cuda())
        model.active_sh_degree = state["active_sh_degree"]
        deform.grid4d.load_state_dict(state["hash"], strict=True)
        deform.deform.load_state_dict(state["deform"], strict=True)
        deform.optimizer.load_state_dict(state["deform_optimizer"])
        self.counts = state["counts"]
        set_rng(state["rng"])
        by_uid = {int(c.uid): c for c in scene.getTrainCameras()}
        return state["iteration"] + 1, [by_uid[uid] for uid in state["stack"]]

    def before(self, iteration, stack, model, deform):
        if iteration % 1000 == 0:
            disk_check()

    def backward(self, iteration, cam, model, deform, package, loss):
        assert (
            bool(torch.isfinite(loss))
            and cam.source_view in self.config["invariants"]["train_views"]
        )
        self.counts["training_renders"] += 1

    def saved(self, iteration, scene, model, deform):
        capture(
            self.run,
            iteration,
            iteration - 1,
            model,
            deform,
            SimpleNamespace(**self.config["pipeline"]),
            scene.data,
            [],
            self.counts,
        )

    def after(self, iteration, cam, model, deform, loss, updated, stack):
        self.counts["optimizer_calls"] += int(updated)
        self.counts["dynamic_gradient_updates"] += int(
            updated and iteration >= self.config["optimization"]["warm_up"]
        )
        row = dict(
            iteration=iteration,
            uid=int(cam.uid),
            view=cam.source_view,
            frame=cam.source_frame,
            loss=float(loss),
            points=len(model.get_xyz),
        )
        self.trace.write(json.dumps(row) + "\n")
        if iteration % 100 == 0 or iteration == 1:
            save(
                self.run / "progress.json",
                dict(
                    state="native_training",
                    **row,
                    counts=self.counts,
                    seconds=time.monotonic() - self.started,
                ),
            )
        if iteration % 1000 == 0:
            checkpoint(
                self.run,
                iteration,
                self.counts["optimizer_calls"],
                model,
                deform,
                [int(c.uid) for c in stack],
                self.counts,
            )


def train_native(label, resume=False):
    run = ROOT / "outputs" / label / "native"
    config = read(case(label)["config"])
    if resume:
        assert (run / "recovery.pt").is_file() and not (run / "complete.json").exists()
    else:
        run.mkdir(parents=True, exist_ok=False)
        save(run / "config.json", config)
    data, _ = manifest(label)
    save(run / "asset_camera_data.json", data)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    dataset = SimpleNamespace(**dict(config["dataset"], model_path=str(run)))
    monitor = Monitor(run, config, resume)
    try:
        native_training(
            dataset,
            SimpleNamespace(**config["optimization"]),
            SimpleNamespace(**config["pipeline"]),
            [33000],
            monitor,
        )
    finally:
        monitor.trace.close()
    assert monitor.counts["optimizer_calls"] == 32999
    save(
        run / "complete.json",
        dict(state="native_complete", actual_updates=32999, counts=monitor.counts),
    )


def alpha_loss(prediction, target):
    foreground = target > 0.5
    assert bool(foreground.any()) and bool((~foreground).any())
    error = (prediction - target).abs()
    return 0.5 * (error[foreground].mean() + error[~foreground].mean())


def continue_training(label, stage):
    """Exact historical alpha15k and quarter-LR A5k objectives."""
    source = ROOT / "outputs" / label / ("native" if stage == "alpha" else "alpha")
    run = ROOT / "outputs" / label / ("alpha" if stage == "alpha" else "asset")
    assert (source / "complete.json").exists()
    run.mkdir(exist_ok=False)
    model, deform, pipe, config, data, state = restore(source)
    base, budget = (32999, 15000) if stage == "alpha" else (47999, 5000)
    assert state["global_actual_updates"] == base
    save(run / "config.json", config)
    save(run / "asset_camera_data.json", data)
    cams, alphas, paths = load_train(data)
    by_uid = {int(c.uid): i for i, c in enumerate(cams)}
    all_uids = list(by_uid)
    stack = list(state["stack"])
    sampler = random.Random()
    sampler.setstate(state["rng"]["python"])
    probe_stack = list(stack)
    sequence = []
    for step in range(budget):
        if not probe_stack:
            probe_stack = list(all_uids)
        sequence.append(probe_stack.pop(sampler.randrange(len(probe_stack))))
    save(run / "sampling.json", sequence)
    base_lrs = {
        g["name"]: g["lr"]
        for o in (model.optimizer, deform.optimizer)
        for g in o.param_groups
    }
    lrs = []
    for j in range(budget):
        mapped = j if stage == "alpha" else int(14999 * j / 4999)
        virtual = 15000 + mapped * 17999 / 14999
        model.update_learning_rate(virtual)
        deform.update_learning_rate(virtual)
        row = {
            g["name"]: g["lr"]
            if g["name"] in ("xyz", "deform", "grid")
            else base_lrs[g["name"]]
            for o in (model.optimizer, deform.optimizer)
            for g in o.param_groups
        }
        lrs.append({k: v * (0.25 if stage == "A" else 1) for k, v in row.items()})
    save(run / "learning_rates.json", lrs)
    set_rng(state["rng"])
    del state
    opt = SimpleNamespace(**config["optimization"])
    n = len(model.get_xyz)
    background = torch.full((3,), 0.5, device="cuda")
    started = time.monotonic()
    with (run / "trace.jsonl").open("x", buffering=1) as trace:
        for step in range(1, budget + 1):
            if not stack:
                stack = list(all_uids)
            uid = stack.pop(random.randrange(len(stack)))
            assert uid == sequence[step - 1]
            i = by_uid[uid]
            cam = cams[i]
            for optimizer in (model.optimizer, deform.optimizer):
                for group in optimizer.param_groups:
                    group["lr"] = lrs[step - 1][group["name"]]
            warp = deform.step(
                model.get_xyz.detach(), cam.fid.unsqueeze(0).expand(n, -1)
            )
            rgb = render(
                cam,
                model,
                pipe,
                background,
                warp["d_xyz"],
                warp["d_rotation"],
                warp["d_scaling"],
            )["render"]
            alpha = render(
                cam,
                model,
                pipe,
                torch.zeros_like(background),
                warp["d_xyz"],
                warp["d_rotation"],
                warp["d_scaling"],
                override_color=torch.ones_like(model.get_xyz),
            )["render"][0]
            delta = (rgb - cam.original_image).abs()
            l1 = delta.mean(0).mean() if stage == "alpha" else delta.mean()
            rgb_loss = 0.8 * l1 + 0.2 * (1 - ssim(rgb, cam.original_image))
            a_loss = alpha_loss(alpha, alphas[i].float() / 255)
            reg = (
                warp["reg_spatial"].mean() * opt.lambda_spatial_tv
                + warp["reg_temporal"].mean() * opt.lambda_temporal_tv
            )
            loss = rgb_loss + 0.1 * a_loss + reg
            assert bool(torch.isfinite(loss))
            loss.backward()
            model.optimizer.step()
            deform.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
            deform.optimizer.zero_grad(set_to_none=True)
            assert len(model.get_xyz) == n
            trace.write(
                json.dumps(
                    dict(
                        step=step,
                        uid=uid,
                        view=cam.source_view,
                        frame=cam.source_frame,
                        loss=float(loss),
                        points=n,
                    )
                )
                + "\n"
            )
            if step % 100 == 0:
                save(
                    run / "progress.json",
                    dict(
                        state=stage,
                        step=step,
                        budget=budget,
                        actual_updates=base + step,
                        seconds=time.monotonic() - started,
                    ),
                )
            if step % 1000 == 0:
                disk_check()
                checkpoint(
                    run,
                    base + step + 1,
                    base + step,
                    model,
                    deform,
                    stack,
                    dict(optimizer_calls=base + step),
                )
    assert random.getstate() == sampler.getstate() and stack == probe_stack
    nominal = base + budget + 1
    model.save_ply(
        str(run / "point_cloud" / f"iteration_{nominal}" / "point_cloud.ply")
    )
    deform.save_weights(str(run), nominal)
    capture(
        run,
        nominal,
        base + budget,
        model,
        deform,
        pipe,
        data,
        stack,
        dict(optimizer_calls=base + budget),
    )
    save(
        run / "complete.json",
        dict(
            state="complete_A" if stage == "A" else "alpha_complete",
            actual_updates=base + budget,
            points=n,
            seconds=time.monotonic() - started,
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["native", "alpha", "A"])
    ap.add_argument("identity")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    device = gpu_identity()
    print(device, flush=True)

    def protect_heldout(event, arguments):
        if event == "open" and isinstance(arguments[0], (str, bytes)):
            path = os.fsdecode(arguments[0])
            if any(f"/cam{view:02d}/" in path for view in (0, 8, 16)) and Path(
                path
            ).suffix in (
                ".png",
                ".jpg",
                ".npz",
            ):
                raise PermissionError(
                    "Heldout image access forbidden during optimization: " + path
                )

    sys.addaudithook(protect_heldout)
    if args.stage == "native":
        train_native(args.identity, args.resume)
    else:
        continue_training(args.identity, args.stage)
