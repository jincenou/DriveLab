"""One CLI and bounded async workers for prepare/generate/reconstruct/export."""

import argparse
import asyncio
import fcntl
import os
import shutil
from pathlib import Path
import time
from common import ROOT, GPU_LEASE_ROOT, read, save, worker


async def one(identity, gpu, mode):
    # Existing jobs in this project cooperate through the historical lease file.
    GPU_LEASE_ROOT.mkdir(parents=True, exist_ok=True)
    shared = GPU_LEASE_ROOT / f"gpu{gpu}.lock"
    with shared.open("a+") as lock:
        await asyncio.to_thread(fcntl.flock, lock.fileno(), fcntl.LOCK_EX)
        if mode == "runtime":
            return await worker(
                "evaluation",
                ["runtime", identity],
                gpu,
                log_name=f"{identity}_runtime.log",
            )
        if mode == "frontend":
            return await worker(
                "frontend",
                ["generate", identity],
                gpu,
                library="4danyone",
                log_name=f"{identity}_generate.log",
            )
        stages = [("evaluation", ["runtime", identity])]
        stages += [
            ("reconstruction", [stage, identity]) for stage in ("native", "alpha", "A")
        ]
        stages += [
            ("evaluation", ["quality", identity]),
            ("assets", ["export", identity]),
            ("assets", ["replay", identity]),
        ]
        for module, arguments in stages:
            await worker(
                module,
                arguments,
                gpu,
                log_name=f"{identity}_{module}_{arguments[0]}.log",
            )
            if module == "reconstruction" and arguments[0] in ("alpha", "A"):
                obsolete = (
                    ROOT
                    / "outputs"
                    / identity
                    / ("native" if arguments[0] == "alpha" else "alpha")
                )
                successor = (
                    ROOT
                    / "outputs"
                    / identity
                    / ("alpha" if arguments[0] == "alpha" else "asset")
                )
                assert (successor / "complete.json").is_file() and (
                    successor / "recovery.pt"
                ).is_file()
                # Final state is fully self-contained. Keep history/config, not redundant stage tensors.
                removed = []
                for name in ("point_cloud", "deform", "recovery.pt"):
                    path = obsolete / name
                    if path.is_file():
                        path.unlink()
                        removed.append(name)
                    elif path.is_dir():
                        shutil.rmtree(path)
                        removed.append(name)
                for path in (obsolete / "snapshots").glob("*_reference.npz"):
                    path.unlink()
                    removed.append(str(path.relative_to(obsolete)))
                save(
                    obsolete / "PAYLOAD_RETIRED.json",
                    dict(
                        successor=str(successor),
                        removed=removed,
                        reason="Following full recovery verified; avoid duplicate intermediate storage",
                    ),
                )
        return dict(identity=identity, state="quality_and_export_passed")


def status_update(stage, value):
    save(ROOT / f"STATUS_{stage}.json", value)
    with (ROOT / "cache/status.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        jobs = {
            p.stem.removeprefix("STATUS_"): read(p) for p in ROOT.glob("STATUS_*.json")
        }
        state = (
            "running"
            if any(j["state"] == "running" for j in jobs.values())
            else (
                "completed"
                if all(j["state"] == "completed" for j in jobs.values())
                else "needs_review"
            )
        )
        save(ROOT / "STATUS.json", dict(state=state, jobs=jobs, updated_at=time.time()))


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "stage", choices=["runtime", "regression", "frontend", "prepare", "status"]
    )
    ap.add_argument("--cases", nargs="+", default=["N04", "N06", "N11"])
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    if args.stage != "status":
        (ROOT / "cache").mkdir(parents=True, exist_ok=True)
        (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    if args.stage == "status":
        print(
            __import__("json").dumps(
                read(ROOT / "STATUS.json")
                if (ROOT / "STATUS.json").exists()
                else {"state": "no_run_state_retained"},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    assert all(g in range(8) for g in args.gpus)
    if args.stage == "prepare":
        from frontend import prepare_registered

        for name in args.cases:
            prepare_registered(name)
        return
    assert len(args.gpus) >= len(args.cases), "Assign one GPU per concurrent case"
    assert len(set(args.gpus)) == len(args.gpus)
    status_update(
        args.stage,
        dict(
            state="running",
            stage=args.stage,
            cases=args.cases,
            gpus=args.gpus,
            pid=os.getpid(),
            started_at=time.time(),
        ),
    )
    results = await asyncio.gather(
        *(one(name, args.gpus[i], args.stage) for i, name in enumerate(args.cases)),
        return_exceptions=True,
    )
    errors = [str(value) for value in results if isinstance(value, BaseException)]
    status_update(
        args.stage,
        dict(
            state="failed" if errors else "completed",
            stage=args.stage,
            cases=args.cases,
            errors=errors,
            finished_at=time.time(),
            results=[r for r in results if not isinstance(r, BaseException)],
        ),
    )
    if errors:
        raise RuntimeError(errors)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "doctor":
        from common import doctor

        doctor()
    elif len(sys.argv) > 1 and sys.argv[1] == "recover-generation":
        ap = argparse.ArgumentParser(
            description="Validate already published videos and recover only completion metadata"
        )
        ap.add_argument("identity")
        ap.add_argument("--gpu", type=int, choices=range(8), default=0)
        args = ap.parse_args(sys.argv[2:])
        from batch import active_child

        assert not active_child("frontend", args.identity), "Generation is still active"
        asyncio.run(
            worker(
                "frontend",
                ["recover", args.identity],
                args.gpu,
                library="4danyone",
                log_name=f"{args.identity}_generation_recovery.log",
            )
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "asset":
        ap = argparse.ArgumentParser(
            description="Export or verify a registered asset using the isolated backend environment"
        )
        ap.add_argument("command", choices=["export", "replay", "runtime-package", "showcase"])
        ap.add_argument("identity")
        ap.add_argument("--gpu", type=int, choices=range(8), default=0)
        args = ap.parse_args(sys.argv[2:])
        asyncio.run(
            worker(
                "assets",
                [args.command, args.identity],
                args.gpu,
                log_name=f"{args.identity}_asset_{args.command}.log",
            )
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "input":
        ap = argparse.ArgumentParser(
            description="Restore source pixels and repair an observed accessory; no motion changes"
        )
        ap.add_argument("command", choices=["native", "restore", "accessory", "bind"])
        ap.add_argument("identity")
        ap.add_argument("--gpu", type=int, choices=range(8), default=0)
        args = ap.parse_args(sys.argv[2:])
        library = "4danyone-sam2" if args.command == "accessory" else "4danyone"
        asyncio.run(
            worker(
                "prepare",
                [args.command, args.identity],
                args.gpu,
                library=library,
                log_name=f"{args.identity}_input_{args.command}.log",
            )
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "intake":
        sys.argv.pop(1)
        from intake import main as intake_main
        intake_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "batch":
        sys.argv.pop(1)
        from batch import main as batch_main

        asyncio.run(batch_main())
    else:
        asyncio.run(main())
