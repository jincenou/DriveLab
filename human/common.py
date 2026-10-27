"""Project paths, atomic records, frozen configuration and isolated subprocesses."""

import asyncio
import ctypes
import uuid
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import shutil
import subprocess
import re
from functools import lru_cache
from copy import deepcopy

CODE_ROOT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("HUMAN_RUN_ROOT", str(CODE_ROOT))).resolve()
ASSET_ROOT = Path(
    os.environ.get("HUMAN_ASSET_ROOT", str(ROOT / "output/person_gs"))
).resolve()
FRONTEND = CODE_ROOT / "vendor/4danyone"
BACKEND = CODE_ROOT / "vendor/grid4d"
RUNTIME_FILE = os.environ.get("HUMAN_RUNTIME_CONFIG")
RUNTIME = json.loads(Path(RUNTIME_FILE).expanduser().read_text()) if RUNTIME_FILE else {}
RESOURCES = {**json.loads((CODE_ROOT / "config/resources.json").read_text()),
             **RUNTIME.get("resources", {})}
ENV_ROOT = Path(os.environ.get("HUMAN_ENV_ROOT", str(Path.home()/"miniconda3/envs")))
PYTHONS = {name: Path(RUNTIME.get("pythons", {}).get(name, str(ENV_ROOT/name/"bin/python")))
    for name in ("4danyone", "grid4d", "4danyone-sam2", "4danyone-ftgs", "vllm_vlm")}
GPU_LEASE_ROOT = Path(os.environ.get("DRIVELAB_GPU_LOCK_DIR", "/tmp/drivelab-gpu-locks"))


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def asset_uid(identity, dataset):
    """Preserve existing Waymo IDs while giving new datasets their own namespace."""
    if not dataset or not dataset.replace('_', '').isalnum():
        raise ValueError('Invalid dataset namespace')
    return f'{dataset}_grid4d_{identity}'


def merge_catalogue(previous, current, owned_uids):
    """A run may update its own entries, never erase another run's assets."""
    owned_uids = set(owned_uids)
    if len({r['global_uid'] for r in current}) != len(current):
        raise ValueError('Duplicate published UID')
    if not all(r['global_uid'] in owned_uids for r in current):
        raise ValueError('Cannot publish an unowned asset')
    retained = [r for r in previous if r['global_uid'] not in owned_uids]
    combined = retained + current
    if len({r['global_uid'] for r in combined}) != len(combined):
        raise ValueError('Duplicate existing catalogue UID')
    paths = [r['package_path'] for r in combined]
    if len(set(paths)) != len(paths):
        raise ValueError('Different identities collide at one package path')
    return combined


def production_paused(identity=None):
    """Drain admitted assets while retaining the pending queue for explicit resume."""
    path = ROOT / "PRODUCTION_PAUSE.json"
    if not path.exists():
        return False
    policy = read(path)
    return bool(policy.get("paused")) and identity not in policy.get("drain_ids", [])


def wait_for_production_permission():
    # Also protects children launched by controllers that predate this pause
    # implementation. Runs before model/CUDA initialization, without restarting
    # the controller or interrupting any admitted asset's later stages.
    script = Path(sys.argv[0]).resolve()
    if script == CODE_ROOT / "assets.py" and sys.argv[1:2] == ["placement-check"]:
        return  # Explicit read-only verification, never generation/training.
    if script == CODE_ROOT / "placement.py" and "--track-view" in sys.argv:
        return  # Explicit image-tracking diagnostic; no asset mutation.
    identities = [arg for arg in sys.argv[1:] if re.fullmatch(r"HN?\d{4}", arg)]
    if (not os.environ.get("HUMAN_GPU_UUID") or script.parent != CODE_ROOT
            or len(identities) != 1):
        return
    identity = identities[0]
    receipt = ROOT / "jobs" / f"pause_wait_{os.getpid()}.json"
    waiting = False
    while production_paused(identity):
        if not waiting:
            save(receipt, dict(state="paused_before_model_load", identity=identity,
                               pid=os.getpid(), script=script.name, time=time.time()))
            print(f"Production paused: {identity} waiting before model load", flush=True)
            waiting = True
        time.sleep(5)
    if waiting:
        save(receipt, dict(state="resumed", identity=identity, pid=os.getpid(), time=time.time()))


wait_for_production_permission()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _case_registry(path, inode, modified_ns, size):
    return read(path)


def case(name):
    path = ROOT / "config/cases.json"
    stat = path.stat()
    config = _case_registry(str(path), stat.st_ino, stat.st_mtime_ns, stat.st_size)
    if name not in config:
        raise ValueError(f"Unknown case {name}; register its input explicitly")
    spec = deepcopy(config[name])
    override = ROOT / "config/case_overrides" / f"{name}.json"
    if override.exists():
        record = read(override)
        digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        assert digest == record["original_spec_sha256"], "Temporal source registration changed"
        spec = record["specification"]
    return spec


def generation_path(identity):
    return Path(case(identity).get("generation", ROOT / "outputs" / identity / "generation"))


def disk_check(reserve_gib=1):
    if shutil.disk_usage(ROOT).free < reserve_gib * 2**30:
        raise RuntimeError("Insufficient disk for the next atomic checkpoint")


def gpu_identity():
    import torch

    expected = os.environ["CUDA_VISIBLE_DEVICES"]
    assert expected.startswith("GPU-") and "," not in expected
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.empty(1, device="cuda")
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    assert driver.cuInit(0) == 0
    device = ctypes.c_int()
    driver.cuCtxGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuCtxGetDevice.restype = ctypes.c_int
    assert driver.cuCtxGetDevice(ctypes.byref(device)) == 0
    value = (ctypes.c_ubyte * 16)()
    driver.cuDeviceGetUuid.argtypes = [ctypes.c_void_p, ctypes.c_int]
    driver.cuDeviceGetUuid.restype = ctypes.c_int
    assert driver.cuDeviceGetUuid(ctypes.byref(value), device.value) == 0
    actual = "GPU-" + str(uuid.UUID(bytes=bytes(value)))
    assert actual == expected, (actual, expected)
    return dict(
        uuid=actual,
        name=torch.cuda.get_device_name(0),
        pid=os.getpid(),
        torch=torch.__version__,
    )


def environment(gpu, library):
    uuid = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if not uuid.startswith("GPU-"):
        raise RuntimeError("Cannot resolve assigned physical GPU")
    env = os.environ.copy()
    # Do not inherit another project's Python source search path.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    toolkit = Path(RUNTIME.get("cuda_home", os.environ.get("CUDA_HOME", "/usr/local/cuda")))
    env.update(
        CUDA_VISIBLE_DEVICES=uuid,
        HUMAN_GPU_UUID=uuid,
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="2",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        TORCH_CUDA_ARCH_LIST=RUNTIME.get("cuda_arch", os.environ.get("TORCH_CUDA_ARCH_LIST", "8.9")),
        MAX_JOBS="4",
        CC=RUNTIME.get("cc", os.environ.get("CC", "gcc")),
        CXX=RUNTIME.get("cxx", os.environ.get("CXX", "g++")),
        CUDA_HOME=str(toolkit),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PATH=str(PYTHONS[library].parent)
        + ":"
        + str(toolkit / "bin")
        + ":"
        + env["PATH"],
    )
    for key, folder in dict(
        TMPDIR="tmp",
        GRID4D_HASH_BUILD="hash",
        TORCH_EXTENSIONS_DIR="extensions",
        TORCH_HOME="torch",
        MPLCONFIGDIR="matplotlib",
        CUDA_CACHE_PATH="cuda",
    ).items():
        if key == "TORCH_HOME":
            path = CODE_ROOT / "vendor/metric_weights"
            if not path.is_dir():
                raise FileNotFoundError(
                    "Metric weights reference is unavailable: " + str(path)
                )
        else:
            path = ROOT / "cache" / folder
            path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    return env


def doctor():
    """Audit local code and external resources without starting GPU work."""
    import ast

    applications = sorted(CODE_ROOT.glob("*.py"))
    for path in applications:
        ast.parse(path.read_text(), filename=str(path))
    local_links = []
    for path in (CODE_ROOT / "vendor").rglob("*"):
        if path.is_symlink():
            local_links.append(
                dict(
                    path=str(path.relative_to(CODE_ROOT)),
                    target=str(path.resolve()),
                    exists=path.exists(),
                )
            )
    resources = {
        key: dict(path=value, exists=Path(value).exists())
        for key, value in RESOURCES.items()
    }
    check = "import importlib.util,json; names=__import__('sys').argv[1:]; print(json.dumps({n:({'origin':s.origin,'locations':list(s.submodule_search_locations or [])} if (s:=importlib.util.find_spec(n)) else None) for n in names}))"
    packages = {}
    for library, modules in {
        "4danyone": ["torch", "transformers", "av"],
        "grid4d": ["torch", "diff_gaussian_rasterization", "simple_knn"],
        "4danyone-sam2": ["torch", "hydra"],
        "4danyone-ftgs": ["torch", "pycolmap", "scipy"],
    }.items():
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env["PYTHONNOUSERSITE"] = "1"
        packages[library] = json.loads(
            subprocess.check_output(
                [str(PYTHONS[library]), "-c", check, *modules],
                env=env,
                cwd=CODE_ROOT,
                text=True,
            )
        )
        for item in packages[library].values():
            if item is not None:
                for origin in [item["origin"], *item["locations"]]:
                    if origin and origin.startswith("/mnt/"):
                        assert any(Path(origin).is_relative_to(p.parent.parent.parent) for p in PYTHONS.values()) or Path(
                            origin
                        ).is_relative_to(CODE_ROOT), origin
    result = dict(
        state="ready"
        if all(v["exists"] for v in resources.values())
        and all(v["exists"] for v in local_links)
        and all(v is not None for group in packages.values() for v in group.values())
        else "missing_dependency",
        code_root=str(CODE_ROOT),
        work_root=str(ROOT),
        asset_root=str(ASSET_ROOT),
        application_sha256={p.name: sha(p) for p in applications},
        external_resources=resources,
        weight_links=local_links,
        packages=packages,
        scope="Static application syntax and core dependency locations; full generation/reconstruction/replay are separate acceptance tests",
    )
    save(ROOT / "reports/code_audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["state"] != "ready":
        raise RuntimeError("Missing dependency; see reports/code_audit.json")


async def worker(module, arguments, gpu, library="grid4d", log_name=None):
    """Bounded concurrency is owned by the caller; failures always propagate."""
    env = environment(gpu, library)
    log = ROOT / "logs" / (log_name or f"{module}_{arguments[0]}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab", buffering=0) as stream:
        process = await asyncio.create_subprocess_exec(
            str(PYTHONS[library]),
            "-u",
            str(CODE_ROOT / f"{module}.py"),
            *map(str, arguments),
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=asyncio.subprocess.STDOUT,
        )
        code = await process.wait()
    if code:
        raise RuntimeError(f"{module} exited {code}; see {log}")
    return dict(module=module, arguments=arguments, gpu=gpu, log=str(log))
