"""Bounded async FLUX → H/B0 jobs; intermediates live only in the work directory."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from .io_utils import read_json, write_json, validate_manifest_item
from .real_views import VEHICLES

PREFIX = "ASSET_RESULT "
ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = ("flux_python", "h_python", "b0_python", "legacy_root", "recon_root", "flux_model",
             "h_model", "vggt_model", "b0_model", "torch_home", "segmentation_model")


async def gpu_free_mib():
    process = await asyncio.create_subprocess_exec(
        'nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits',
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        output, error = await asyncio.wait_for(process.communicate(), 15)
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    if process.returncode:
        raise RuntimeError('Cannot inspect GPU memory: ' + error.decode(errors='replace'))
    return {int(index): int(free) for index, free in
            (line.split(',') for line in output.decode().splitlines() if line.strip())}


class GpuPool:
    """Admit one worker per local slot only when measured free memory is sufficient."""
    def __init__(self, gpus, *, poll_seconds=5, timeout_seconds=3600):
        if not gpus or len(set(gpus)) != len(gpus):
            raise ValueError('Provide distinct GPU indices')
        self.available = set(gpus)
        self.lock = asyncio.Lock()
        self.poll_seconds, self.timeout_seconds = poll_seconds, timeout_seconds

    async def acquire(self, stage):
        required = {'flux': 24, 'H': 12, 'B0': 8, 'render': 8}[stage] * 1024
        started = time.monotonic()
        logged = -30.0
        while True:
            async with self.lock:
                free = await gpu_free_mib()
                ready = [gpu for gpu in self.available if free.get(gpu, 0) >= required]
                if ready:
                    gpu = min(ready, key=lambda value: (-free[value], value))
                    self.available.remove(gpu)
                    return gpu
            elapsed = time.monotonic() - started
            if elapsed >= self.timeout_seconds:
                raise TimeoutError(f'GPU memory wait exceeded {self.timeout_seconds}s for {stage}')
            if elapsed - logged >= 30:
                print(f'Waiting for {stage}: need {required // 1024} GiB free; free MiB={free}', flush=True)
                logged = elapsed
            await asyncio.sleep(self.poll_seconds)

    def release(self, gpu):
        self.available.add(gpu)


def is_memory_failure(result):
    error = result.get('error', '').lower()
    return result.get('status') == 'failed' and ('out of memory' in error or 'cuda oom' in error)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare_jobs(handoff, routes, config):
    rows, routing = read_json(handoff), read_json(routes)
    if routing["flux_input_sha256"] != sha256(handoff):
        raise ValueError("Routing sidecar does not match the FLUX manifest")
    if not rows:
        raise ValueError("No generation queries")
    if set(config) != set(PATH_KEYS) | {"seed"}:
        raise ValueError("Generation config requires exactly: " + ", ".join((*PATH_KEYS, "seed")))
    if any(not Path(config[key]).is_absolute() or not Path(config[key]).exists() for key in PATH_KEYS):
        raise ValueError("All runtime/model paths must exist and be absolute")
    if config["segmentation_model"].split("/")[-1] != "birefnet-general-lite.onnx":
        raise ValueError("Expected the original birefnet-general-lite.onnx")
    index = {item["obj_id"]: item for item in routing["assets"]}
    if len(index) != len(routing["assets"]) or len({r["obj_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate object identity")
    if set(index) != {r["obj_id"] for r in rows}:
        raise ValueError("Routes must cover exactly the supplied query set")
    jobs = []
    for item_index, row in enumerate(rows):
        validate_manifest_item(row)
        route = index[row["obj_id"]]
        expected = "H" if row["category"].lower() in VEHICLES else "B0"
        if (route["route"], route["category"], route["scene_id"]) != (expected, row["category"], row["scene_id"]):
            raise ValueError("Route/category/scene mismatch: " + row["obj_id"])
        real = route["real_images"]
        if (expected == "H" and not 0 <= len(real) <= 4) or (expected == "B0" and real):
            raise ValueError("H accepts 0..4 real images; B0 takes generated views only")
        if routing.get('schema_version') == 2 and any(not r.get('quality_review', {}).get('accepted') for r in real):
            raise ValueError('Unapproved real evidence in reviewed routes')
        hashes = [sha256(image["rgba_path"]) for image in real]
        if hashes != [image["sha256"] for image in real] or len(hashes) != len(set(hashes)):
            raise ValueError("Changed or duplicated real evidence: " + row["obj_id"])
        jobs.append(dict(item_index=item_index, **row, route=expected, real_images=real,
                         source=route["source"], real_view_selection=route["real_view_selection"],
                         input_mode='flux_plus_real' if real else 'flux_only',
                         image_sha256=sha256(row["image_path"])))
    code = [Path(__file__), Path(config["legacy_root"]) / "flux.py",
            Path(config["legacy_root"]) / "gen_gs.py",
            Path(config["recon_root"]) / "trellis/pipelines/trellis_image_to_3d.py"]
    signature = digest(dict(config=config, jobs=jobs, code={str(p): sha256(p) for p in code}))
    return dict(schema_version=1, signature=signature, config=config, assets=jobs)


def checked_outputs(result):
    paths = result["images"] if "images" in result else [dict(path=result["raw_path"], sha256=result["raw_sha256"])]
    if not paths or any(not Path(r["path"]).is_file() or sha256(r["path"]) != r["sha256"] for r in paths):
        raise ValueError("A completed generation output is missing or has changed")


async def _launch(stage, indices, gpu, jobs_path, accept):
    bundle = read_json(jobs_path)
    config = bundle["config"]
    python = config[{"flux": "flux_python", "H": "h_python", "B0": "b0_python"}[stage]]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TORCH_HOME=config["torch_home"], U2NET_HOME=str(Path(config["segmentation_model"]).parent),
               ATTN_BACKEND="flash-attn", SPCONV_ALGO="native", OMP_NUM_THREADS="4",
               NUMBA_CACHE_DIR=str(jobs_path.parent / "cache/numba"))
    command = [python, "-m", "asset_extraction.generation", "--worker", stage, "--jobs", str(jobs_path),
               "--indices", *map(str, indices)]
    process = await asyncio.create_subprocess_exec(*command, cwd=ROOT, env=env,
                                                 stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    received = set()
    memory_error = False
    with (jobs_path.parent / f"{stage}_gpu{gpu}.log").open("a") as log:
        try:
            async for raw in process.stdout:
                line = raw.decode(errors="replace")
                log.write(line); log.flush()
                memory_error |= 'out of memory' in line.lower() or 'cuda oom' in line.lower()
                if line.startswith(PREFIX):
                    result = json.loads(line[len(PREFIX):])
                    if result["item_index"] not in indices or result["item_index"] in received:
                        raise ValueError("Unexpected worker result identity")
                    accept(result); received.add(result["item_index"])
            returncode = await process.wait()
        finally:
            if process.returncode is None:
                process.terminate()
                await process.wait()
    for index in set(indices) - received:
        accept(dict(item_index=index, stage=stage, status="failed",
                    error=("CUDA OOM worker stopped; " if returncode == 86 or memory_error else "") +
                          f"Worker exited {returncode} without a result; see {stage}_gpu{gpu}.log"))


async def generate(bundle, work, gpus):
    work = Path(work).resolve(); work.mkdir(parents=True, exist_ok=True)
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("Provide a nonempty list of distinct GPU indices")
    jobs_path, state_path = work / "jobs.json", work / "generation.json"
    if state_path.exists():
        state = read_json(state_path)
        if state["signature"] != bundle["signature"]:
            raise ValueError("Inputs/config/code changed: use a new work directory")
        for row in state["assets"]:
            for result in row["stages"].values():
                if result["status"] == "succeeded":
                    checked_outputs(result)
    else:
        state = dict(schema_version=1, signature=bundle["signature"], config=bundle["config"],
                     assets=[dict(row, stages={}, status="pending") for row in bundle["assets"]])
    write_json(jobs_path, bundle)
    available = GpuPool(gpus)

    def accept(result):
        row = state["assets"][result["item_index"]]
        if result["status"] == "succeeded":
            checked_outputs(result)
        previous = row["stages"].get(result["stage"])
        if previous is not None:
            row.setdefault('stage_attempts', {}).setdefault(result['stage'], []).append(previous)
        row["stages"][result["stage"]] = result
        row["status"] = "generated" if result["status"] == "succeeded" and result["stage"] != "flux" else result["status"]
        if row["status"] == "generated":
            row.update({k: result[k] for k in ("raw_path", "raw_sha256", "raw_up_axis", "gaussian_count")})
        write_json(state_path, state)

    async def batch(stage, indices):
        remaining = indices
        for attempt in range(3):
            gpu = None
            try:
                gpu = await available.acquire(stage)
                await _launch(stage, remaining, gpu, jobs_path, accept)
            except Exception as exc:
                for index in remaining:
                    if state["assets"][index]["stages"].get(stage, {}).get("status") != "succeeded":
                        accept(dict(item_index=index, stage=stage, status="failed", error=f"{type(exc).__name__}: {exc}"))
            finally:
                if gpu is not None:
                    available.release(gpu)
            remaining = [index for index in remaining if is_memory_failure(state['assets'][index]['stages'][stage])]
            if not remaining or attempt == 2:
                break
            await asyncio.sleep(5)

    def pending(stage):
        return [r["item_index"] for r in state["assets"] if
                r["stages"].get(stage, {}).get("status") != "succeeded" and
                (stage == "flux" or r["route"] == stage and r["stages"].get("flux", {}).get("status") == "succeeded")]

    async def dispatch(stage):
        ids = pending(stage)
        await asyncio.gather(*(batch(stage, ids[i::len(gpus)]) for i in range(min(len(gpus), len(ids)))))

    write_json(state_path, state)
    await dispatch("flux")
    await asyncio.gather(dispatch("H"), dispatch("B0"))
    state["counts"] = dict(queries=len(state["assets"]), generated=sum(r["status"] == "generated" for r in state["assets"]))
    state["status"] = "generated_pending_curation" if state["counts"]["queries"] == state["counts"]["generated"] else "incomplete"
    write_json(state_path, state)
    return state


def load_reconstruction(config, stage):
    import torch
    from safetensors.torch import load_file
    from torchvision import transforms
    root = Path(config["recon_root"] if stage == "H" else config["legacy_root"])
    sys.path[:0] = [str(root), str(root / "wheels/vggt")]
    from trellis import models
    from trellis.pipelines import samplers
    from trellis.pipelines.base import Pipeline
    from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline
    if stage == "H":
        from trellis.pipelines import TrellisVGGTTo3DPipeline
        from wheels.vggt.vggt.models.vggt import VGGT
    model_root = Path(config["h_model"] if stage == "H" else config["b0_model"])
    spec = read_json(model_root / "pipeline.json")["args"]
    modules, audit = {}, {}
    for key, value in spec["models"].items():
        if key in {"slat_decoder_rf", "slat_decoder_mesh"}:
            continue
        path = model_root / value
        definition = read_json(path.with_suffix(".json"))
        model = getattr(models, definition["name"])(**definition["args"])
        weights = load_file(str(path.with_suffix(".safetensors")))
        extra = set(weights) - set(model.state_dict())
        allowed = {f"fuse_blocks.{i}.0.{part}" for i in range(4) for part in ("weight", "bias")} if stage == "H" and key == "slat_vggt_cond" else set()
        if extra - allowed:
            raise ValueError(f"Unrecognized checkpoint tensors for {key}: {extra - allowed}")
        for name in extra:
            del weights[name]
        model.load_state_dict(weights, strict=True)
        modules[key] = model.eval(); audit[key] = sorted(extra)
    pipe = TrellisVGGTTo3DPipeline() if stage == "H" else TrellisImageTo3DPipeline()
    Pipeline.__init__(pipe, modules)
    pipe.model_load_audit = audit
    for name in ("sparse_structure", "slat"):
        sampler = spec[name + "_sampler"]
        setattr(pipe, name + "_sampler", getattr(samplers, sampler["name"])(**sampler["args"]))
        setattr(pipe, name + "_sampler_params", sampler["params"])
    pipe.slat_normalization = spec["slat_normalization"]
    dino = Path(config["torch_home"]) / "hub/facebookresearch_dinov2_main"
    pipe.models["image_cond_model"] = torch.hub.load(str(dino), spec["image_cond_model"], source="local", pretrained=True).eval()
    pipe.image_cond_model_transform = transforms.Compose([transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
    if stage == "H":
        vggt_root = Path(config["vggt_model"])
        vggt = VGGT(**read_json(vggt_root / "config.json"))
        vggt.load_state_dict(load_file(str(vggt_root / "model.safetensors")), strict=True)
        del vggt.depth_head; del vggt.track_head
        pipe.VGGT_model = vggt.eval(); pipe.VGGT_dtype = torch.bfloat16
        pipe.low_vram = True; pipe._device = torch.device("cuda")
    else:
        pipe.rembg_session = None
        pipe.cuda()
    return pipe


def save_gaussian(asset, target):
    """Preserve native coordinates; serialize opacity logits without sigmoid roundtrip."""
    import numpy as np
    from plyfile import PlyData
    if asset.get_xyz.shape[0] == 0:
        raise ValueError("Empty Gaussian asset")
    asset.save_ply(str(target))
    payload = PlyData.read(target, mmap=False)
    payload["vertex"]["opacity"] = (asset._opacity + asset.opacity_bias).detach().cpu().numpy().ravel()
    if any(not np.isfinite(payload["vertex"][name]).all() for name in payload["vertex"].data.dtype.names):
        raise ValueError("Nonfinite Gaussian attributes")
    payload.write(target)


def worker(stage, jobs_path, indices):
    import numpy as np
    import torch
    from PIL import Image
    bundle = read_json(jobs_path); config = bundle["config"]; work = jobs_path.parent
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    started = time.monotonic()
    if stage == "flux":
        sys.path.insert(0, config["legacy_root"])
        import flux
        from diffusers import Flux2KleinPipeline
        flux.PIPE = Flux2KleinPipeline.from_pretrained(config["flux_model"], torch_dtype=torch.bfloat16, local_files_only=True)
        flux.PIPE.enable_model_cpu_offload(); flux.PIPE.set_progress_bar_config(disable=True)
    else:
        import rembg
        session = rembg.new_session("birefnet-general-lite", providers=["CPUExecutionProvider"])
        pipe = load_reconstruction(config, stage)
    load_seconds = time.monotonic() - started
    for index in indices:
        row = bundle["assets"][index]; output = work / f"asset_{index:05d}"; output.mkdir(exist_ok=True)
        torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
        result = dict(item_index=index, stage=stage, load_seconds=load_seconds)
        try:
            if sha256(row["image_path"]) != row["image_sha256"] or any(sha256(r["rgba_path"]) != r["sha256"] for r in row["real_images"]):
                raise ValueError("Input changed after job registration")
            if stage == "flux":
                seed_index = row.get('flux_seed_index', index)
                paths = [str(output / f"flux_{i}.png") for i in range(len(row["prompts"]))]
                flux._generate(dict(item_index=seed_index, image_path=row["image_path"], prompts=row["prompts"], output_paths=paths))
                result.update(images=[dict(path=p, sha256=sha256(p)) for p in paths], seed_per_view=[seed_index * 1000 + i for i in range(len(paths))])
            else:
                flux_result = read_json(work / "generation.json")["assets"][index]["stages"]["flux"]
                checked_outputs(flux_result)
                paths = [r["path"] for r in flux_result["images"]] + [r["rgba_path"] for r in row["real_images"]]
                images = [rembg.remove(Image.open(r["path"]).convert("RGB"), session=session) for r in flux_result["images"]]
                images += [Image.open(r["rgba_path"]).convert("RGBA") for r in row["real_images"]]
                if any((np.asarray(im.getchannel("A")) > 204).sum() < 16 or np.all(np.asarray(im.getchannel("A")) == 255) for im in images):
                    raise ValueError("Reconstruction requires valid segmented foregrounds")
                prepared = [pipe.preprocess_image(im) for im in images]
                with torch.no_grad():
                    if stage == "H":
                        outputs, _, _ = pipe.run(prepared, seed=config["seed"], formats=["gaussian"], preprocess_image=False)
                    elif len(prepared) == 1:
                        outputs = pipe.run(prepared[0], seed=config["seed"], formats=["gaussian"], preprocess_image=False)
                    else:
                        outputs = pipe.run_multi_image(prepared, seed=config["seed"], formats=["gaussian"], mode="multidiffusion", preprocess_image=False)
                asset = outputs["gaussian"][0]; target = output / "raw.ply"
                save_gaussian(asset, target)
                result.update(raw_path=str(target), raw_sha256=sha256(target), raw_up_axis="z" if stage == "H" else "-y", gaussian_count=int(asset.get_xyz.shape[0]),
                              real_count=len(row["real_images"]), seed=config["seed"], input_images=paths,
                              sparse_structure_sampler=pipe.sparse_structure_sampler_params, slat_sampler=pipe.slat_sampler_params,
                              model_load_audit=pipe.model_load_audit,
                              serialization="opacity logits directly from _opacity + opacity_bias; finite PLY fields checked")
                del outputs, asset
            torch.cuda.synchronize()
            result.update(status="succeeded", seconds=time.monotonic() - started,
                          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        except Exception as exc:
            result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(PREFIX + json.dumps(result), flush=True)
        if is_memory_failure(result):
            # A fresh process releases offload hooks and CUDA state before another attempt.
            raise SystemExit(86)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path); parser.add_argument("--routes", type=Path)
    parser.add_argument("--config", type=Path); parser.add_argument("--work", type=Path)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--worker", choices=["flux", "H", "B0"])
    parser.add_argument("--jobs", type=Path); parser.add_argument("--indices", type=int, nargs="+")
    args = parser.parse_args()
    if args.worker:
        if args.jobs is None or not args.indices:
            parser.error("Workers require --jobs and --indices")
        worker(args.worker, args.jobs, args.indices)
    else:
        if any(getattr(args, name) is None for name in ("handoff", "routes", "config", "work")):
            parser.error("Require --handoff, --routes, --config and --work")
        state = asyncio.run(generate(prepare_jobs(args.handoff, args.routes, read_json(args.config)), args.work, args.gpus))
        print(json.dumps(dict(status=state["status"], **state["counts"])))
        if state["status"] == "incomplete":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
