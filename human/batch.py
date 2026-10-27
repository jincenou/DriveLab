"""Resumable async asset queue; pilot quality gates precede large-batch work."""
import argparse, asyncio, fcntl, os, json, time, shutil, subprocess
from pathlib import Path
from common import ROOT, CODE_ROOT, ASSET_ROOT, case, read, save, worker, sha, production_paused


def replenish_inputs(plan_path):
    """Swap explicitly reviewed inputs for terminal failures; retain all evidence."""
    from contextlib import ExitStack
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    pairs = plan['pairs']
    assert pairs and plan['approved'], 'Explicit reviewed replacement plan required'
    old_ids, new_ids = ([p[k] for p in pairs] for k in ('old', 'new'))
    assert len(set(old_ids)) == len(pairs) == len(set(new_ids))
    assert not set(old_ids) & set(new_ids)
    target = ROOT/'PRODUCTION_INPUTS.json'
    with ExitStack() as locks:
        guard = locks.enter_context((ROOT/'jobs/replenish.lock').open('a+'))
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        production = read(target)
        before = sha(target)
        assert len(production['ids']) == production['target_count'] == 300
        source_keys = {case(i)['selection']['identity_key'] for i in production['ids']}
        receipts = []
        for pair in pairs:
            old, new = pair['old'], pair['new']
            assert old in production['ids'] and new not in production['ids']
            lock = locks.enter_context((ROOT/'jobs'/f'{old}.lock').open('a+'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            job = ROOT/'jobs'/f'{old}.json'
            assert read(job)['state'] == 'failed_or_needs_review', 'Never replace live/pending jobs'
            assert not any(active_child(m, old) for m in ('frontend', 'dataset', 'reconstruction', 'assets'))
            assert not (ASSET_ROOT/old/'package_complete.json').exists(), 'Published assets require separate review'
            assert not (ROOT/'jobs'/f'{new}.json').exists(), 'Do not rerun an attempted identity'
            assert not (ROOT/'outputs'/new).exists() and not (ASSET_ROOT/new).exists()
            spec = case(new)
            review = spec['input_review']
            assert review['allow_generation'] and review.get('reviewed_slots'), 'Multi-time review required'
            assert sha(spec['raw_input']) == review['input_sha256'] == pair['input_sha256']
            contact = ROOT/'review'/f'{new}.json'
            evidence = read(contact)
            assert sha(ROOT/'review'/f'{new}.jpg') == evidence['image_sha256']
            assert all(sha(x['path']) == x['sha256'] for x in evidence['sources'])
            assert review['reviewed_slots'] == evidence['frames']
            key = spec['selection']['identity_key']
            assert key not in source_keys, 'Source track already scheduled'
            source_keys.add(key)
            assert pair['temporal_decision'] == 'keep_full'
            assert pair['temporal_source_sha256'], 'Frozen temporal evidence required'
            assert plan['temporal_policy_sha256'] == sha(ROOT/'TEMPORAL_POLICY.json')
            assert all(sha(p) == h for p, h in pair['temporal_source_sha256'].items())
            production['ids'][production['ids'].index(old)] = new
            receipts.append(dict(old=old,new=new,reason=pair['reason'],old_job_sha256=sha(job)))
        assert len(set(production['ids'])) == 300
        event = dict(time=time.time(), plan=str(plan_path), plan_sha256=sha(plan_path),
                     previous_production_sha256=before, pairs=receipts)
        production.setdefault('replacement_history', []).append(event)
        production['replacement_reserve'] = [i for i in production.get('replacement_reserve', []) if i not in new_ids]
        assert sha(target) == before, 'Concurrent selection change; retry from current state'
        save(target, production)
    return dict(replaced=len(pairs), production_sha256=sha(target), event=event,
                scope='Queued inputs, not successful or quality-approved assets; old jobs and published files unchanged')


def progress_snapshot(process_root=Path('/proc')):
    """Read current delivery evidence without changing jobs or published assets."""
    from collections import Counter
    from playback import validate_estimate
    production = read(ROOT / 'PRODUCTION_INPUTS.json')
    identities = production['ids']
    processes = []
    for folder in process_root.iterdir():
        if not folder.name.isdigit():
            continue
        try:
            args = (folder / 'cmdline').read_bytes().split(b'\0')
            if not any(a.startswith(str(CODE_ROOT).encode()+b'/') and a.endswith(b'.py') for a in args):
                continue
            if str(CODE_ROOT/'run.py').encode() in args and b'batch' in args and b'progress' in args:
                continue
            env = (folder / 'environ').read_bytes().split(b'\0')
            if ('HUMAN_RUN_ROOT='+str(ROOT)).encode() not in env:
                continue
            processes.append(dict(pid=int(folder.name), args=[os.fsdecode(a) for a in args if a]))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    rows = []
    for identity in identities:
        job_path = ROOT / 'jobs' / f'{identity}.json'
        job = read(job_path) if job_path.exists() else {}
        item = dict(identity=identity, job_state=job.get('state', 'not_started'),
                    stage=job.get('stage'), gpu=job.get('gpu'),
                    live_pids=[p['pid'] for p in processes if identity in p['args']])
        package = ASSET_ROOT / identity
        marker = package / 'package_complete.json'
        if marker.exists() and read(marker).get('state') == 'complete_pending_visual_review':
            try:
                assert read(package/'runtime_replay.json')['state'] == 'published_runtime_replay_passed'
                assert read(package/'presentation.json')['state'] == 'five_column_video_verified'
                validate_estimate(package)
                item.update(state='technical_complete_visual_review_pending',
                            skip_insertion=read(package/'insertion.json').get('skip_insertion',False))
            except Exception as error:
                item.update(state='published_evidence_needs_review', error=f'{type(error).__name__}: {error}')
        elif item['live_pids']:
            item['state'] = 'active_process_confirmed'
        elif item['job_state'] == 'failed_or_needs_review':
            item.update(state='failed_or_needs_review',error=job.get('error'))
        elif item['job_state'] == 'not_started':
            item['state'] = 'not_started'
        else:
            item['state'] = 'job_state_unverified_process_not_visible'
        rows.append(item)
    counts = dict(Counter(r['state'] for r in rows))
    controllers = [p for p in processes if str(CODE_ROOT/'run.py') in p['args']
                   and 'batch' in p['args'] and 'run' in p['args']]
    snapshot = dict(updated_at=time.time(), registered=len(identities), target_count=production['target_count'],
        counts=counts, insertion_excluded=sum(r.get('skip_insertion',False) for r in rows),
        controllers=controllers, visible_processes=processes, assets=rows,
        scope='Current completion/replay/presentation records and metadata hashes, not fresh full-payload hashing or visual/ground/loop acceptance. Process visibility may be restricted; missing PID alone does not prove a job stopped.')
    save(ROOT/'PROGRESS_LATEST.json',snapshot)
    return snapshot


async def watch_progress(interval):
    """Independent lightweight watcher; never starts, stops, or edits asset jobs."""
    if interval < 30:
        raise ValueError('Progress interval must be at least 30 seconds')
    absent = 0
    while True:
        result = progress_snapshot()
        print(json.dumps(dict(updated_at=result['updated_at'],counts=result['counts'],
                              controller_pids=[p['pid'] for p in result['controllers']])),flush=True)
        absent = absent+1 if not result['visible_processes'] else 0
        # Two observations bridge the deferred retry controller's 30-second handoff.
        # Exit means monitor idle, never completion of the asset goal.
        if absent >= 2:
            save(ROOT/'reports/progress_watch_exit.json',dict(state='no_visible_pipeline_processes',
                updated_at=time.time(),counts=result['counts'],goal_complete=False))
            return
        await asyncio.sleep(interval)


def handoff_ready(plan):
    """Drain complete assets before changing devices; never terminate a CUDA child."""
    policy = read(ROOT/'PRODUCTION_PAUSE.json')
    if (not policy.get('paused') or policy.get('next_gpus') != plan['gpus']
            or set(policy.get('drain_ids', [])) != set(plan['drain_ids'])):
        raise RuntimeError('Handoff pause policy changed; leave the queue untouched')
    terminal = {'replayed_visual_review_pending', 'failed_or_needs_review'}
    jobs = [ROOT/'jobs'/f'{i}.json' for i in plan['drain_ids']]
    workers = [ROOT/'jobs'/f'worker_gpu{g}.json' for g in plan['previous_gpus']]
    return (all(p.exists() and read(p).get('state') in terminal for p in jobs)
            and all(p.exists() and read(p).get('state') ==
                    'paused_pending_asset_metadata_validation' for p in workers))


def handoff_cases(plan):
    """An OOM can be retried only after its failed stage was explicitly recovered."""
    production = read(ROOT/'PRODUCTION_INPUTS.json')
    assert set(plan['cases']).issubset(production['ids']), 'Admission changed'
    ready, withheld = [], []
    for identity in plan['cases']:
        if verified_package(identity):
            continue
        path = ROOT/'jobs'/f'{identity}.json'
        job = read(path) if path.exists() else {}
        if job.get('state') == 'failed_or_needs_review':
            expected = plan['prepared_failures'].get(identity)
            if expected is None or sha(path) != expected:
                withheld.append(dict(identity=identity, reason='Unprepared or changed failure', job=job))
                continue
        elif job.get('state', 'not_started') != 'not_started':
            raise RuntimeError(f'{identity}: unexpected job state at drained handoff')
        ready.append(identity)
    return ready, withheld


async def handoff_queue(plan_path):
    """Explicit, PID-bound one-time device handoff; training recipes are unchanged."""
    import signal
    from common import PYTHONS
    from audit import watched_process
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    assert plan['state'] == 'prepared' and plan['approved']
    assert plan['gpus'] and len(set(plan['gpus'])) == len(plan['gpus'])
    assert all(g in range(8) for g in plan['gpus'])
    pid, start = plan['controller_pid'], plan['controller_start_ticks']
    assert watched_process(pid, start), 'Original controller identity is no longer live'
    with (ROOT/'jobs/device_handoff.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan.update(state='waiting_current_assets', pid=os.getpid(), started_at=time.time())
        save(plan_path, plan)
        while not handoff_ready(plan):
            if not watched_process(pid, start):
                raise RuntimeError('Controller exited before verified drain; inspect before recovery')
            await asyncio.sleep(10)
        assert watched_process(pid, start)
        # No active child, even in a brief gap between stage status updates.
        children = Path(f'/proc/{pid}/task/{pid}/children').read_text().strip()
        assert not children, 'Drained controller still has children; do not stop it'
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        env = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
        assert b'batch' in args and b'run' in args
        assert any(Path(os.fsdecode(a)).name == 'run.py' for a in args if a)
        assert ('HUMAN_RUN_ROOT='+str(ROOT)).encode() in env
        plan['drained_jobs'] = {i:read(ROOT/'jobs'/f'{i}.json') for i in plan['drain_ids']}
        cases, withheld = handoff_cases(plan)
        plan.update(state='stopping_drained_controller', next_cases=cases,
                    withheld_failures=withheld, drain_verified_at=time.time())
        save(plan_path, plan)
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic()+30
        while watched_process(pid, start):
            if time.monotonic() >= deadline:
                raise RuntimeError('Controller did not stop; pause retained, no second queue started')
            await asyncio.sleep(1)
        # The original controller has released its queue lock before unpausing.
        assert handoff_ready(plan)
        policy = read(ROOT/'PRODUCTION_PAUSE.json')
        policy.update(paused=False, handoff_completed_at=time.time(), active_gpus=plan['gpus'])
        save(ROOT/'PRODUCTION_PAUSE.json', policy)
        if not cases:
            plan.update(state='drained_no_pending_cases', finished_at=time.time())
            save(plan_path, plan)
            return
        command = [str(PYTHONS['4danyone']), '-u', str(CODE_ROOT/'run.py'),
                   'batch', 'run', '--cases', *cases, '--gpus', *map(str,plan['gpus']),
                   '--min-free-mib', str(plan['min_free_mib'])]
        child = await asyncio.create_subprocess_exec(*command, cwd=CODE_ROOT)
        plan.update(state='running_on_requested_gpus', child_pid=child.pid,
                    command=command, resumed_at=time.time())
        save(plan_path, plan)
        code = await child.wait()
        plan.update(state='completed_pending_quality_review' if code == 0 else 'queue_finished_with_failures',
                    return_code=code, finished_at=time.time())
        save(plan_path, plan)


def initialize(pool, limit, extend=False):
    """Rank already screened tracks, without rescanning Waymo or accepting quality."""
    import csv

    if not extend:
        assert not (
            ROOT / "selection.json"
        ).exists(), "Do not overwrite a registered batch"
    pool = Path(pool)
    rows = list(csv.DictReader(pool.open()))
    for x in rows:
        candidate = read(Path(x["package"]) / "candidate.json")
        w, h = float(x["source_width"]), float(x["source_height"])
        boxes = [o["bbox_xyxy"] for o in candidate["observations"]]
        edge = sum(
            b[0] <= 2 or b[1] <= 2 or b[2] >= w - 2 or b[3] >= h - 2 for b in boxes
        ) / len(boxes)
        height = float(x["height_median_px"])
        p95 = float(x["pose_reprojection_p95"])
        median = float(x["pose_reprojection_median"])
        x["bbox_edge_fraction"] = edge
        x["rank_score"] = (
            0.45 * min(height, 600) / 600
            + 0.35 * max(0, 1 - p95 / 0.15)
            + 0.15 * max(0, 1 - median / 0.08)
            + (0.05 if x["identity_scope"] == "official_2d3d_association" else 0)
            - 0.5 * edge
        )
    rows.sort(key=lambda x: (-x["rank_score"], x["identity_key"]))
    selected = []
    seen = set()
    for x in rows:
        if x["identity_key"] in seen:
            continue
        seen.add(x["identity_key"])
        selected.append(dict(x, id=f"H{len(selected)+1:04d}"))
        if len(selected) == limit:
            break
    assert len(selected) == limit
    registry = read(ROOT / "config/cases.json") if extend else {}
    if extend:
        selection = read(ROOT / "selection.json")
        assert selection["source_sha256"] == sha(pool), "Candidate source changed"
        known = {x["selection"]["identity_key"] for x in registry.values()}
        selected = [x for x in selected if x["identity_key"] not in known]
        next_id = max(int(x[1:]) for x in registry) + 1
        for index, item in enumerate(selected):
            item["id"] = f"H{next_id+index:04d}"
        prior = sha(ROOT / "selection.json")
        selection["items"].extend(selected)
        selection.setdefault("extensions", []).append(
            dict(
                previous_sha256=prior,
                added_ids=[x["id"] for x in selected],
                requested_rank_pool=limit,
                purpose="Replace visibly weak inputs from cached screened pool; production target remains about 300",
            )
        )
        selection["production_target"] = 300
    else:
        selection = dict(
            source=str(pool),
            source_sha256=sha(pool),
            items=selected,
            quality_approved=False,
            scope="Ranked existing track keys; not certified distinct natural persons",
            demo_ids=[x["id"] for x in selected[:2]],
            score_rule="height .45 + detector/pose p95 .35 + median .15 + official association .05 - border fraction .5",
        )
    save(ROOT / "selection.json", selection)
    for x in selected:
        p = Path(x["package"])
        identity = x["id"]
        registry[identity] = dict(
            prepared_frames=int(x["frames"]),
            raw_input=x["input"],
            conditioning=str(p / "conditions/preflight/conditioning"),
            motion=str(p / "conditions/preflight/motion"),
            root_motion=str(p / "conditions/preflight/root_motion.npz"),
            source_dataset="waymo",
            selection=x,
            manifest=str(ROOT / "data" / identity / "fdanyone.json"),
            config=str(ROOT / "config" / f"{identity}.json"),
            input_review=dict(allow_generation=False, accessory="pending"),
        )
    save(ROOT / "config/cases.json", registry)
    if not extend:
        for name in ("export.json", "recipe.json"):
            shutil.copy2(CODE_ROOT / "config" / name, ROOT / "config" / name)
        shutil.copy2(CODE_ROOT / "config/temporal_policy.json", ROOT / "TEMPORAL_POLICY.json")
    print("Registered", len(registry), "candidates; input review precedes generation")


def active_child(module, identity):
    found = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().split(b"\0")
            env = (p / "environ").read_bytes().split(b"\0")
            if (
                str(CODE_ROOT / f"{module}.py").encode() in cmd
                and identity.encode() in cmd
                and ("HUMAN_RUN_ROOT=" + str(ROOT)).encode() in env
            ):
                found.append(int(p.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


async def wait_existing_generation(identity):
    receipt = ROOT / "outputs" / identity / "generation/human_generation_receipt.json"
    while not receipt.exists():
        pids = active_child("frontend", identity)
        if not pids:
            raise RuntimeError(
                f"{identity}: existing generation exited without completion; inspect log"
            )
        save(
            ROOT / "jobs" / f"{identity}.json",
            dict(
                state="waiting_existing_generation", pids=pids, updated_at=time.time()
            ),
        )
        await asyncio.sleep(15)


def memory_free(gpu):
    return int(
        subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    )


async def execute(identity, gpu):
    spec = case(identity)
    path = ROOT / "jobs" / f"{identity}.json"
    assert spec["input_review"]["allow_generation"], "Input/accessory review pending"
    if verified_package(identity):
        save(
            path,
            dict(
                state="replayed_visual_review_pending",
                gpu=gpu,
                updated_at=time.time(),
                asset=str(ASSET_ROOT / identity),
                resumed_existing_complete_package=True,
            ),
        )
        return
    stages = []

    async def stage(module, args, library="grid4d", tag=None):
        tag = tag or f"{module}_{args[0]}"
        save(
            path,
            dict(
                state="running",
                stage=tag,
                gpu=gpu,
                updated_at=time.time(),
                finished_stages=stages,
            ),
        )
        result = await worker(
            module, args, gpu, library=library, log_name=f"{identity}_{tag}.log"
        )
        stages.append(tag)
        return result

    gen = ROOT / "outputs" / identity / "generation"
    if not (gen / "human_generation_receipt.json").exists():
        if active_child("frontend", identity):
            await wait_existing_generation(identity)
        else:
            assert (
                not gen.exists()
            ), "Partial generation exists; explicit recovery required"
            await stage("frontend", ["generate", identity], "4danyone")
    if not (ROOT / "reports" / f"{identity}_frontend_validation.json").exists():
        await stage("frontend", ["validate", identity], "4danyone")
    for group in range(4):
        proof = ROOT / "reports" / f"{identity}_masks_{group}.json"
        if proof.exists() and len(read(proof)) == 6:
            continue
        await stage(
            "dataset",
            ["masks", identity, "--group", group],
            "4danyone-sam2",
            f"masks_{group}",
        )
    if not (ROOT / "data" / identity / "rgba/init.npz").exists():
        await stage("dataset", ["hull", identity], "4danyone-ftgs")
    if not (ROOT / "reports" / f"{identity}_dataset_ready.json").exists():
        await stage("dataset", ["configure", identity], "4danyone-ftgs")
    assert read(ROOT / "CODE_VALIDATION.json")[
        "ready_for_demo_reconstruction"
    ], "Preparation migration validation required before reconstruction"
    for label, folder in (("native", "native"), ("alpha", "alpha"), ("A", "asset")):
        dest = ROOT / "outputs" / identity / folder
        if (dest / "complete.json").exists():
            continue
        args = [label, identity]
        if dest.exists():
            if label == "native" and (dest / "recovery.pt").is_file():
                args.append("--resume")
            else:
                raise RuntimeError(
                    f"{identity}/{folder}: partial stage requires explicit recovery; no overwrite"
                )
        await stage("reconstruction", args)
        # Only retire earlier tensors once the successor's full recovery exists.
        if label in ("alpha", "A"):
            assert (dest / "recovery.pt").is_file()
            prior = (
                ROOT
                / "outputs"
                / identity
                / ("native" if label == "alpha" else "alpha")
            )
            removed = []
            for name in ("point_cloud", "deform", "recovery.pt"):
                p = prior / name
                if p.is_dir():
                    shutil.rmtree(p)
                    removed.append(name)
                elif p.is_file():
                    p.unlink()
                    removed.append(name)
            for p in (prior / "snapshots").glob("*_reference.npz"):
                p.unlink()
                removed.append(str(p.relative_to(prior)))
            save(
                prior / "PAYLOAD_RETIRED.json",
                dict(successor=str(dest), removed=removed),
            )
    if not (ROOT / "reports" / f"{identity}_quality.json").exists():
        await stage("evaluation", ["quality", identity])
    if not (ASSET_ROOT / identity / "manifest.json").exists():
        await stage("assets", ["export", identity])
    if not (ASSET_ROOT / identity / "replay_validation.json").exists():
        await stage("assets", ["replay", identity])
    await stage("batch", ["package", identity], "4danyone")
    if (ROOT / "BATCH_APPROVAL.json").exists() and read(ROOT / "BATCH_APPROVAL.json")[
        "approved"
    ]:
        retire_verified_work(identity)
    save(
        path,
        dict(
            state="replayed_visual_review_pending",
            gpu=gpu,
            updated_at=time.time(),
            finished_stages=stages,
            asset=str(ASSET_ROOT / identity),
        ),
    )


def verified_package(identity):
    out = ASSET_ROOT / identity
    marker = out / "package_complete.json"
    if not marker.exists():
        return False
    complete = read(marker)
    if complete["state"] != "complete_pending_visual_review":
        return False
    assert (
        read(out / "runtime_replay.json")["state"] == "published_runtime_replay_passed"
    )
    assert (
        read(out / "replay_validation.json")["state"]
        == "NPZ_integer_frame_replay_passed"
    )
    assert complete["asset_sha256"] == sha(out / "asset_4dgs.npz")
    from placement import validate_output_contract
    validate_output_contract(out)
    assert complete["refs_sha256"]["input.mp4"] == sha(case(identity)["raw_input"])
    for name, digest in read(out / "manifest.json")["source_hashes"].items():
        assert sha(out / "runtime" / name) == digest
    return True


def reconcile_completed_jobs():
    """Recover stale failure states only from a fully verified, unlocked package."""
    results = []
    for path in sorted((ROOT / 'jobs').glob('H*.json')):
        if read(path).get('state') != 'failed_or_needs_review':
            continue
        identity = path.stem
        with path.with_suffix('.lock').open('a+') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                results.append(dict(identity=identity, state='active_job_untouched'))
                continue
            previous = read(path)
            if previous.get('state') != 'failed_or_needs_review':
                continue
            try:
                complete = verified_package(identity)
            except Exception as error:
                results.append(dict(identity=identity, state='verification_failed_untouched',
                                    error=f'{type(error).__name__}: {error}'))
                continue
            if not complete:
                results.append(dict(identity=identity, state='incomplete_untouched'))
                continue
            save(path, dict(state='replayed_visual_review_pending',
                 updated_at=time.time(), asset=str(ASSET_ROOT / identity),
                 gpu=previous.get('gpu'), previous_failure=previous,
                 reconciled_existing_complete_package=True,
                 reconciliation_scope='Current payload hashes, runtime/source hashes, placement contract and existing replay evidence; no new training or quality approval'))
            results.append(dict(identity=identity, state='complete_package_reconciled'))
    report=dict(updated_at=time.time(), results=results,
                reconciled=sum(r['state']=='complete_package_reconciled' for r in results))
    save(ROOT / 'reports/job_reconciliation.json', report)
    return report


def exclude_from_insertion(identity, reason):
    """Retain a completed asset while marking it unusable for scene insertion."""
    if not reason or not reason.strip():
        raise ValueError('An observed reason is required')
    out = ASSET_ROOT / identity
    with (ROOT / 'jobs' / f'{identity}.lock').open('a+') as job_lock, \
         (ROOT / 'jobs' / f'{identity}_placement.lock').open('a+') as placement_lock:
        fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(placement_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not verified_package(identity):
            raise ValueError('Only a verified completed asset can be reviewed here')
        flags=dict(skip_insertion=True, insertion_exclusion_reason=reason.strip(),
                   insertion_review_basis='explicit_review', insertion_reviewed_at=time.time())
        for name in ('insertion.json', 'manifest.json', 'motion_metadata.json'):
            data=read(out/name);data.update(flags);save(out/name,data)
        readme=out/'README.txt'
        lines=readme.read_text().splitlines()
        if lines and lines[0].startswith('【插入时跳过】'):
            lines=lines[1:]
        readme.write_text('【插入时跳过】'+reason.strip()+'\n'+'\n'.join(lines)+'\n')
    catalogue_root=ASSET_ROOT if 'HUMAN_ASSET_ROOT' in os.environ else ROOT
    catalogue=catalogue_root/'drivelab_asset.json'
    with (catalogue_root/'.catalogue.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if catalogue.exists():
            data=read(catalogue)
            for item in data['assets']:
                if item['global_uid']==__import__('common').asset_uid(identity,case(identity).get('source_dataset','waymo')):
                    item.update(skip_insertion=True, insertion_exclusion_reason=reason.strip(),
                                motion_review_group='excluded_from_insertion')
            data['insertion_excluded_total']=sum(bool(i.get('skip_insertion')) for i in data['assets'])
            data['motion_groups']={group:[i['global_uid'] for i in data['assets'] if i['motion_review_group']==group]
                 for group in sorted({i['motion_review_group'] for i in data['assets']})}
            save(catalogue,data)
    return dict(identity=identity, **flags, payload_changed=False)


def retire_verified_work(identity):
    """Remove only reproducible/duplicated payloads after packaged replay succeeds."""
    assert verified_package(identity)
    original = ROOT / "outputs" / identity / "asset"
    published = ASSET_ROOT / identity / "runtime"
    for name in ("point_cloud", "deform"):
        folder = original / name
        if folder.exists():
            assert not folder.is_symlink()
            for path in folder.rglob("*"):
                if path.is_file():
                    assert sha(path) == sha(published / path.relative_to(original))
    targets = [
        original / "point_cloud",
        original / "deform",
        original / "recovery.pt",
        ROOT / "data" / identity / "rgba/images",
        ROOT / "data" / identity / "rgba/sam2_input",
    ]
    targets += list((original / "snapshots").glob("*_reference.npz"))
    removed = []
    for path in targets:
        if not path.exists():
            continue
        assert not path.is_symlink() and path.resolve().is_relative_to(ROOT)
        size = (
            sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
            if path.is_dir()
            else path.stat().st_size
        )
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(dict(path=str(path.relative_to(ROOT)), bytes=size))
    save(
        ROOT / "reports" / f"{identity}_working_payload_retired.json",
        dict(
            removed=removed,
            kept="Published NPZ, runtime, input/camera/root refs, generated videos, metrics and logs",
            resume_scope="Training optimizer retired; exact native training resume is no longer available, full final runtime retained",
        ),
    )


def retire_redundant_media(identity):
    """Retire named legacy presentation files only after the five-column replacement."""
    assert verified_package(identity)
    out = ASSET_ROOT / identity
    presentation = read(out / "presentation.json")
    assert presentation["state"] == "five_column_video_verified"
    assert presentation["sha256"] == sha(out / "demo.mp4")
    assert read(out / "manifest.json")["preview_paths"] == {"five_column": "demo.mp4"}
    targets = [out / name for name in (
        "preview.mp4", "comparison.mp4", "comparison.jpg",
        "multiview_preview.mp4", "multiview_preview.jpg",
    )]
    generated = ROOT / "outputs" / identity / "generation"
    targets += [generated / "preview.mp4", generated / "preview.jpg"]
    targets += [ROOT / "reports" / f"{identity}_comparison.mp4"]
    targets += list((ROOT / "reports").glob(f"{identity}_f[0-9][0-9][0-9].jpg"))
    record_path = ROOT / "reports" / f"{identity}_media_retired.json"
    record = read(record_path) if record_path.exists() else {
        "removed": [], "replacement": str(out / "demo.mp4"),
        "scope": "Legacy presentation only; historical validation hashes remain as evidence",
    }
    pending = []
    for path in targets:
        if not path.exists():
            continue
        assert path.is_file() and not path.is_symlink()
        assert path.resolve().is_relative_to(ROOT) or path.resolve().is_relative_to(out)
        pending.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
    record.update(state="cleanup_in_progress", pending=pending)
    save(record_path, record)
    for item in pending:
        Path(item["path"]).unlink()
        record["removed"].append(item)
    record.update(state="complete", pending=[], demo_sha256=presentation["sha256"])
    save(record_path, record)
    return record


def required_memory(identity):
    marker = ASSET_ROOT / identity / "package_complete.json"
    if marker.exists() and read(marker).get("state") == "complete_pending_visual_review":
        return 0
    if (
        ROOT / "outputs" / identity / "generation/human_generation_receipt.json"
    ).exists():
        return 18000
    # 61-frame Turbo was measured at 17.85 GB peak reserved on this pipeline;
    # retain over 6 GB admission headroom for shared-device inference.
    return 38000 if case(identity)["prepared_frames"] > 61 else 24000


def package(identity):
    """Preserve insertion provenance alongside the old-compatible NPZ layout."""
    s = case(identity)
    out = ASSET_ROOT / identity
    assert (
        read(out / "replay_validation.json")["state"]
        == "NPZ_integer_frame_replay_passed"
    )
    source = Path(s["selection"]["package"]) / "clip0"
    refs = out / "refs"
    refs.mkdir(exist_ok=True)
    inputs = {
        "input.mp4": Path(s["raw_input"]),
        "observed_native.mp4": Path(s["selection"]["source"]),
        "skeleton.mp4": Path(s["selection"]["pose"]),
        "frame_mapping.json": source / "inputs/frame_mapping.json",
        "native.json": source / "native.json",
        "source_cameras.npz": source / "inputs/cameras.npz",
        "root_motion.npz": Path(s["root_motion"]),
        "motion.json": Path(s["motion"]) / "motion.json",
        "motion.safetensors": Path(s["motion"]) / "motion.safetensors",
    }
    from placement import write_source_motion_hints
    hint_info,hint_arrays=write_source_motion_hints(identity,refs)
    inputs['source_motion_hints.json']=hint_info
    inputs['source_motion_hints.npz']=hint_arrays
    for name, p in inputs.items():
        assert p.is_file(), str(p)
        if not (refs / name).exists():
            shutil.copy2(p, refs / name)
        assert sha(refs / name) == sha(p)
    native = ROOT / "outputs" / identity / "asset"
    runtime = out / "runtime"
    runtime.mkdir(exist_ok=True)
    for name in (
        "point_cloud",
        "deform",
        "config.json",
        "asset_camera_data.json",
        "snapshots/53000.json",
        "complete.json",
    ):
        p = native / name
        q = runtime / name
        if p.is_dir():
            if not q.exists():
                shutil.copytree(p, q)
        else:
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)
    for name in ("quality.json",):
        p = ROOT / "reports" / f"{identity}_{name}"
        if p.is_file():
            shutil.copy2(p, out / name)
    manifest = read(out / "manifest.json")
    manifest["model_path"] = "runtime"
    motion_metadata = read(out / "motion_metadata.json")
    motion_metadata.pop("continuous_runtime_preserved_at_source", None)
    motion_metadata["continuous_runtime"] = "runtime"
    save(out / "motion_metadata.json", motion_metadata)
    manifest["provenance"] = {
        "selection": s["selection"],
        "source_refs": list(inputs),
        "coordinate_and_time_limitations": "Original 25fps; root-local estimated motion, no measured metric world registration; no SMPL-X retargeting guarantee",
    }
    if "temporal_subset" in s:
        manifest["temporal_subset"] = s["temporal_subset"]
    temporal = ROOT / "temporal" / f"{identity}.json"
    if temporal.exists():
        shutil.copy2(temporal, out / "temporal_screen.json")
        manifest["temporal_screen"] = "temporal_screen.json"
    manifest["preview_paths"] = {"five_column": "demo.mp4"}
    save(out / "manifest.json", manifest)
    import numpy as np

    native_to_asset = np.asarray(read(out / "cameras.json")["asset_from_native"])
    local_to_native = np.asarray(
        read(native / "asset_camera_data.json")["source_to_normalized"]
    )
    asset_to_local = np.linalg.inv(local_to_native) @ np.linalg.inv(native_to_asset)
    with np.load(refs / "root_motion.npz") as root:
        assert "local_to_smpl_incam" in root and "frame_times" in root
        source_cameras = root["local_to_smpl_incam"].copy()
        times = root["frame_times"].copy()
    asset_to_source = source_cameras @ asset_to_local
    assert np.isfinite(asset_to_source).all() and len(asset_to_source) == spec_frames(s)
    roundtrip_error = float(
        np.abs(
            asset_to_source @ native_to_asset @ local_to_native - source_cameras
        ).max()
    )
    assert roundtrip_error < 1e-8
    np.savez_compressed(
        out / "insertion_transforms.npz",
        asset_camera_to_root_local=asset_to_local,
        asset_camera_to_estimated_source_camera=asset_to_source,
        frame_timestamps=times,
    )
    save(
        out / "insertion.json",
        dict(
            native_to_reference_camera=native_to_asset.tolist(),
            source_to_normalized=local_to_native.tolist(),
            transforms="insertion_transforms.npz",
            transform_scope="Estimated same-body source camera; not verified dataset-world registration",
            root_motion="refs/root_motion.npz",
            source_motion_hints="refs/source_motion_hints.json",
            source_hints_are_not_final_gaussian_motion=True,
            source_camera="refs/source_cameras.npz",
            source_timestamps="refs/frame_mapping.json",
            asset_fps=25,
            full_SH=True,
            requires_SH_aware_consumer=True,
            world_metric_verified=False,
            root_horizontal_translation_already_removed=True,
            do_not_double_subtract_root=True,
            arbitrary_motion_retargeting=False,
            coordinate_composition_max_abs_error=roundtrip_error,
            coordinate_composition_scope="Algebraic consistency only; source motion and metric world registration remain estimated",
            **{key: manifest[key] for key in (
                'skip_insertion', 'insertion_exclusion_reason', 'insertion_review_basis',
                'insertion_reviewed_at'
            ) if key in manifest},
        ),
    )
    (out / "README.txt").write_text(
        "asset_4dgs.npz 保留旧接口键与完整SH；manifest.json、insertion.json、refs/保存来源/相机/时间/root信息；runtime/保留连续形变推理权重。独立回放见replay_validation.json。原旧读取器若只支持RGB或写死30fps，必须使用SH与25fps适配，不能直接当作完全兼容。root水平位移已分离，不能重复去除；原人体估计不等于真实米制轨迹。preview为已有估计动作，不是任意新动作重定向。\n"
    )
    from placement import enrich
    enrich(identity)
    from playback import build_estimates
    practical_playback = build_estimates(out)
    save(
        out / "package_complete.json",
        dict(
            state="payload_ready_runtime_check_pending",
            asset_sha256=sha(out / "asset_4dgs.npz"),
            refs_sha256={n: sha(refs / n) for n in inputs},
            practical_playback=practical_playback,
        ),
    )


def spec_frames(spec):
    return int(spec["prepared_frames"])


def gallery(identities):
    """One useful page with existing inputs, intermediate videos and final assets."""
    from html import escape
    from urllib.parse import quote

    def link(path):
        return escape(quote(os.path.relpath(path, ROOT), safe="/"))

    def video(path, title):
        if not path.is_file():
            return f"<section><h3>{escape(title)}</h3><p>尚未完成</p></section>"
        return f'<section><h3>{escape(title)}</h3><video controls preload="none"><source src="{link(path)}" type="video/mp4"></video><p><a href="{link(path)}">直接打开视频</a></p></section>'

    cards = []
    for identity in identities:
        spec = case(identity)
        job = ROOT / "jobs" / f"{identity}.json"
        stage = (
            read(job).get("stage", read(job).get("state"))
            if job.exists()
            else "input_review_pending"
        )
        final = ASSET_ROOT / identity
        review = read(final / "visual_review.json") if (final / "visual_review.json").exists() else {}
        if review.get("severe_failure", False):
            quality = "已发现严重片段问题，未计入质量合格"
        elif review.get("final_asset_quality_certified", False):
            quality = "已通过记录中的视觉验收"
        else:
            quality = "待完整视觉验收，未计入质量合格"
        flags = "; ".join(review.get("quality_flags", []))
        insertion = read(final/'insertion.json') if (final/'insertion.json').exists() else {}
        if insertion.get('skip_insertion'):
            quality = '插入时跳过'
            flags = insertion.get('insertion_exclusion_reason', '已标记待处理')
        cards.append(
            f'<article><h2>{escape(identity)} · {escape(spec.get("source_dataset", "waymo"))} {escape(spec["selection"]["scene"])}</h2><p>阶段：{escape(str(stage))}。{quality} {escape(flags)}</p>'
            + video(final / "demo.mp4", "原始观测 | 骨骼叠加 | 生成视角00 | 重建视角00 | 重建视角08")
            + "</article>"
        )
    html = (
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>人体资产生成</title><style>body{font-family:system-ui;margin:24px;background:#f4f5f7;color:#20232a}article{background:white;padding:18px;margin:20px 0;border-radius:10px}.videos{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}video{width:100%;max-height:430px;background:#222}a{color:#2458a4}</style><h1>人体资产生成：输入 → 多视角视频 → 动态高斯</h1><p>每人仅一个五列展示视频。第三列显示24个生成视角中的00视角，后两列显示重建00/08视角。原始观测按源时间映射取最近真实帧。程序完成不等于外观质量通过；未完成项目留空。</p>'
        + "".join(cards)
        + "</html>"
    )
    temporary = ROOT / f"index.html.{os.getpid()}.writing"
    temporary.write_text(html)
    temporary.replace(ROOT / "index.html")


def input_overviews(identities, prefix="overview"):
    """Compact first-frame triage plus links to three-time detailed evidence."""
    from PIL import Image, ImageDraw
    from html import escape

    folder = ROOT / "review"
    pages = []
    for start in range(0, len(identities), 20):
        ids = identities[start : start + 20]
        canvas = Image.new("RGB", (1440, 1650), "#333333")
        for pos, identity in enumerate(ids):
            path = folder / f"{identity}.jpg"
            if not path.is_file():
                continue
            with Image.open(path) as im:
                pair = Image.new("RGB", (360, 300), "#333333")
                # Remove only the blank column margins; never stretch a person.
                pair.paste(im.crop((90, 25, 270, 300)), (0, 25))
                pair.paste(im.crop((450, 25, 630, 300)), (180, 25))
            x, y = pos % 4 * 360, pos // 4 * 330
            canvas.paste(pair, (x, y + 25))
            ImageDraw.Draw(canvas).text(
                (x + 5, y + 5),
                identity + " observed / input, first slot only",
                fill="white",
            )
        name = f"{prefix}_{start//20+1:02d}.jpg"
        canvas.save(folder / name, quality=93)
        pages.append(
            dict(
                path=name,
                identities=ids,
                scope="First slot only; see individual contact for middle/end and pose",
            )
        )
    save(folder / f"{prefix}s.json", pages)
    body = "".join(
        f'<p><a href="{x["path"]}">{escape(", ".join(x["identities"]))}</a></p>'
        for x in pages
    )
    body += "".join(
        f'<a href="{i}.jpg" style="display:inline-block;margin:8px">{escape(i)} 原观测/输入/骨架</a>'
        for i in identities
        if (folder / f"{i}.jpg").is_file()
    )
    (folder / ("index.html" if prefix == "overview" else f"{prefix}.html")).write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>输入复核</title><h1>已筛选候选：输入复核</h1><p>总览只显示第一时刻，个人详情显示首/中/末三个时刻。预览制作不代表输入或资产质量通过。</p>'
        + body
        + "</html>"
    )


def triage_inputs():
    """Prioritize existing inputs; measurements never constitute visual acceptance."""
    policy = read(CODE_ROOT / "config/input_priority.json")
    registry = read(ROOT / "config/cases.json")
    records = []
    for identity, spec in registry.items():
        report = ROOT / "review" / f"{identity}.json"
        if not report.exists():
            continue
        item = read(report)
        if "alpha_metrics" not in item:
            continue
        metrics = item["alpha_metrics"]
        checks = {
            key: metrics[key] >= value
            for key, value in policy["prefer_clear_reference"].items()
        }
        records.append(
            dict(
                identity=identity,
                rank_score=spec["selection"]["rank_score"],
                preferred=all(checks.values()),
                checks=checks,
                contact_sha256=item["image_sha256"],
                input_review=spec["input_review"],
                metrics=metrics,
            )
        )
    records.sort(key=lambda x: (not x["preferred"], -x["rank_score"], x["identity"]))
    ids = [x["identity"] for x in records if x["preferred"]]
    save(
        ROOT / "reports/input_priority.json",
        dict(
            policy=policy,
            measured=len(records),
            preferred=len(ids),
            records=records,
            scope="Prioritization only; other tracks are retained, not declared unusable; accessory and actual appearance review still required",
        ),
    )
    input_overviews(ids, prefix="preferred")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "mode",
        choices=[
            "init",
            "extend",
            "pilot",
            "run",
            "package",
            "status",
            "progress",
            "reconcile",
            "exclude",
            "masks_check",
            "inspect",
            "gallery",
            "triage",
            "replenish",
            "repair",
            "handoff",
        ],
    )
    ap.add_argument("identity", nargs="?")
    ap.add_argument("--reason")
    ap.add_argument("--replacement-plan")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--no-overviews", action="store_true",
                    help="Inspect inputs without regenerating aggregate sheets or gallery")
    ap.add_argument("--interval", type=int, default=60)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--cases", nargs="+")
    group.add_argument("--all", action="store_true")
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 2])
    ap.add_argument("--auto-gpu", action="store_true", help="Pilot only: wait for a suitable device from --gpus")
    ap.add_argument("--min-free-mib", type=int, default=0,
                    help="Additional GPU memory admission floor for an isolated queue")
    ap.add_argument("--queue-name", default="production",
                    help="Separate repair controller lock and receipt; case locks still prevent duplicate work")
    ap.add_argument(
        "--pool",
        default="inputs/priority_candidates.csv",
    )
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()
    if args.mode == 'handoff':
        if not args.replacement_plan:
            ap.error('handoff requires --replacement-plan')
        await handoff_queue(args.replacement_plan)
        return
    if args.auto_gpu and args.mode != "pilot":
        ap.error("--auto-gpu applies only to pilot")
    if args.queue_name != "production":
        assert args.mode == "repair" and args.queue_name.replace("_", "").isalnum()
    if args.mode == 'replenish':
        if not args.replacement_plan:
            ap.error('replenish requires --replacement-plan')
        print(json.dumps(replenish_inputs(args.replacement_plan), ensure_ascii=False))
        return
    if args.mode == 'progress':
        if args.watch:
            await watch_progress(args.interval)
        else:
            report=progress_snapshot()
            print(json.dumps(dict(counts=report['counts'],registered=report['registered'],
                report=str(ROOT/'PROGRESS_LATEST.json')),ensure_ascii=False))
        return
    if args.mode == "repair" and (args.all or not args.cases):
        ap.error("repair requires --cases with explicit identities; --all is forbidden")
    args.cases = (
        sorted(read(ROOT / "config/cases.json"))
        if args.all
        else (args.cases or ["H0001", "H0002"])
    )
    if args.mode == "repair":
        assert not args.all and args.cases, "Repair requires explicit identities"
        assert (ROOT / "REPAIR_PLAN.json").is_file(), "Freeze a bounded repair plan first"
        assert (ROOT / "CODE_VALIDATION.json").is_file(), "Repair root needs referenced pipeline validation before starting work"
        assert read(ROOT / "CODE_VALIDATION.json")["ready_for_demo_reconstruction"], "Pipeline validation must pass before repair"
        assert ASSET_ROOT.is_relative_to(ROOT), "Repair candidates must stay inside their isolated work root"
        assert args.min_free_mib >= 26000, "Repair shares GPUs with production; require memory headroom"
        assert all(case(i)["input_review"]["allow_generation"] for i in args.cases)
    if args.mode in ("init", "extend"):
        initialize(args.pool, args.limit, extend=args.mode == "extend")
        return
    if args.mode == "exclude":
        if not args.identity or not args.reason:
            ap.error('exclude requires an identity and --reason')
        print(json.dumps(exclude_from_insertion(args.identity, args.reason), ensure_ascii=False))
        return
    if args.mode == "package":
        package(args.identity)
        await worker(
            "assets",
            ["runtime-package", args.identity],
            os.environ["HUMAN_GPU_UUID"],
            log_name=f"{args.identity}_runtime_package.log",
        )
        validation = read(ASSET_ROOT / args.identity / "runtime_replay.json")
        assert validation["state"] == "published_runtime_replay_passed"
        await worker("assets", ["placement-check", args.identity],
                     os.environ["HUMAN_GPU_UUID"], log_name=f"{args.identity}_placement_check.log")
        await worker(
            "assets", ["showcase", args.identity], os.environ["HUMAN_GPU_UUID"],
            log_name=f"{args.identity}_showcase.log",
        )
        presentation = read(ASSET_ROOT / args.identity / "presentation.json")
        assert presentation["state"] == "five_column_video_verified"
        from placement import validate_output_contract
        placement_contract = validate_output_contract(ASSET_ROOT / args.identity)
        complete = read(ASSET_ROOT / args.identity / "package_complete.json")
        complete.update(
            state="complete_pending_visual_review", runtime_replay="runtime_replay.json",
            presentation="presentation.json", demo_sha256=presentation["sha256"],
            placement_contract=placement_contract,
        )
        save(ASSET_ROOT / args.identity / "package_complete.json", complete)
        await worker("assets", ["catalogue", args.identity], os.environ["HUMAN_GPU_UUID"],
                     log_name=f"{args.identity}_catalogue.log")
        retire_redundant_media(args.identity)
        if "comparison_asset" in case(args.identity) and "temporal_subset" in case(args.identity):
            await worker("evaluation", ["compare-prefix", args.identity],
                         os.environ["HUMAN_GPU_UUID"], log_name=f"{args.identity}_paired_prefix.log")
        gallery(read(ROOT / "PRODUCTION_INPUTS.json")["ids"] if (ROOT / "PRODUCTION_INPUTS.json").exists() else ["H0001", "H0002"])
        return
    if args.mode == "gallery":
        if args.all and (ROOT / "PRODUCTION_INPUTS.json").exists():
            args.cases = read(ROOT / "PRODUCTION_INPUTS.json")["ids"]
        gallery(args.cases)
        return
    if args.mode == "triage":
        triage_inputs()
        return
    if args.mode == "inspect":
        semaphore = asyncio.Semaphore(8)

        async def inspect_one(identity):
            async with semaphore:
                prior = ROOT / "review" / f"{identity}.json"
                if not prior.exists() or "alpha_metrics" not in read(prior):
                    await worker(
                        "prepare",
                        ["inspect", identity],
                        0,
                        library="4danyone",
                        log_name=f"input_review/{identity}.log",
                    )

        results = await asyncio.gather(
            *(inspect_one(identity) for identity in args.cases), return_exceptions=True
        )
        errors = [
            dict(identity=i, error=str(r))
            for i, r in zip(args.cases, results, strict=True)
            if isinstance(r, BaseException)
        ]
        save(
            ROOT / "reports/input_contacts.json",
            dict(
                cases=args.cases,
                completed=len(args.cases) - len(errors),
                errors=errors,
                quality_approved=False,
            ),
        )
        if not args.no_overviews:
            input_overviews(args.cases)
            gallery(args.cases[:2])
        if errors:
            raise RuntimeError(errors)
        return
    if args.mode == "status":
        print(
            json.dumps(
                {p.stem: read(p) for p in (ROOT / "jobs").glob("*.json")}, indent=2
            )
        )
        return
    if args.mode == "reconcile":
        report = reconcile_completed_jobs()
        print(json.dumps(dict(reconciled=report['reconciled'],
              inspected=len(report['results']), report=str(ROOT / 'reports/job_reconciliation.json'))))
        return
    if args.mode == "masks_check":
        assert args.identity and len(args.gpus) == 4
        results = await asyncio.gather(
            *(
                worker(
                    "dataset",
                    ["masks", args.identity, "--group", g],
                    gpu,
                    library="4danyone-sam2",
                    log_name=f"group{g}.log",
                )
                for g, gpu in enumerate(args.gpus)
            ),
            return_exceptions=True,
        )
        errors = [str(r) for r in results if isinstance(r, BaseException)]
        save(
            ROOT / "mask_check_exit.json",
            dict(state="failed" if errors else "completed", errors=errors),
        )
        if errors:
            raise RuntimeError(errors)
        return
    if args.mode == "run":
        gate = read(ROOT / "BATCH_APPROVAL.json")
        assert gate["approved"] and set(gate["demo_ids"]) == set(read(ROOT / "selection.json")["demo_ids"])
        assert len(gate["demo_ids"]) == 2
        production = read(ROOT / "PRODUCTION_INPUTS.json")
        assert production["target_count"] > 0
        assert len(production["ids"]) <= production["target_count"] and len(set(production["ids"])) == len(
            production["ids"]
        )
        if args.all:
            args.cases = production["ids"]
        assert set(args.cases).issubset(production["ids"])
        assert all(case(i)["input_review"]["allow_generation"] for i in args.cases)
        validation_mode = gate.get("validation_mode", "two_complete_demos")
        assert validation_mode in ("two_complete_demos", "explicit_user_expansion")
        required_demos = gate["demo_ids"]
        if validation_mode == "explicit_user_expansion":
            assert gate.get("user_authorization"), "Record the explicit scope change"
            required_demos = gate["required_complete_demo_ids"]
            assert required_demos and set(required_demos).issubset(gate["demo_ids"])
        assert all(verified_package(i) for i in required_demos)
        for i in required_demos:
            assert read(ASSET_ROOT / i / "visual_review.json")["allow_batch_baseline"]
    (ROOT / "jobs").mkdir(exist_ok=True)
    (ROOT / "cache").mkdir(exist_ok=True)
    assert len(set(args.gpus)) == len(args.gpus) and all(
        g in range(8) for g in args.gpus
    )
    if args.mode == "pilot":
        assert args.auto_gpu or len(args.cases) == len(args.gpus)
        reserved = set()

        async def pilot_one(identity, gpu):
            try:
                with (ROOT / "jobs" / f"{identity}.lock").open("a+") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if args.auto_gpu:
                        while True:
                            available = [(memory_free(g), g) for g in args.gpus if g not in reserved]
                            if available and max(available)[0] >= max(required_memory(identity), args.min_free_mib):
                                gpu = max(available)[1]
                                reserved.add(gpu)
                                break
                            save(ROOT / "jobs" / f"{identity}.json", dict(
                                state="waiting_resources", stage="before_generation", gpu=None,
                                free_mib={str(g):m for m,g in available}, updated_at=time.time()))
                            await asyncio.sleep(30)
                    await execute(identity, gpu)
            except Exception as e:
                save(
                    ROOT / "jobs" / f"{identity}.json",
                    dict(
                        state="failed_or_needs_review",
                        error=str(e),
                        gpu=gpu,
                        updated_at=time.time(),
                    ),
                )
                return dict(identity=identity, error=str(e))
            finally:
                reserved.discard(gpu)

        results = await asyncio.gather(
            *(
                pilot_one(identity, gpu)
                for identity, gpu in zip(args.cases, [None] * len(args.cases) if args.auto_gpu else args.gpus, strict=True)
            )
        )
        errors = [r for r in results if r is not None]
        save(
            ROOT / "queue_exit.json",
            dict(
                state="needs_review" if errors else "pilot_complete_pending_review",
                errors=errors,
                finished_at=time.time(),
                cases=args.cases,
            ),
        )
        if errors:
            raise RuntimeError(errors)
        return
    queue_lock = (ROOT / "jobs" / f"{args.queue_name}_queue.lock").open("a+")
    fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pending = list(args.cases)
    registered = set(pending)
    errors = []
    # Adopt existing generation children on their original GPU after a controller
    # restart. Their allocated memory must not be mistaken for another task.
    adopted = {}
    for identity in pending:
        path = ROOT / "jobs" / f"{identity}.json"
        if path.exists():
            job = read(path)
            if job.get("stage") == "frontend_generate" and active_child("frontend", identity):
                gpu = job["gpu"]
                assert gpu in args.gpus and gpu not in adopted
                adopted[gpu] = identity

    def refresh_inputs():
        if not args.all:
            return True
        current = read(ROOT / "PRODUCTION_INPUTS.json")
        assert current["target_count"] > 0 and len(current["ids"]) <= current["target_count"]
        assert len(set(current["ids"])) == len(current["ids"])
        for identity in current["ids"]:
            if identity not in registered:
                assert case(identity)["input_review"]["allow_generation"]
                pending.append(identity)
                registered.add(identity)
        return current.get("input_selection_complete", True)

    async def consume(gpu):
        while True:
            if production_paused():
                save(ROOT / "jobs" / f"worker_gpu{gpu}.json",
                     dict(state="paused_pending_asset_metadata_validation", updated_at=time.time()))
                await asyncio.sleep(5)
                continue
            input_complete = refresh_inputs()
            reservation = ROOT / "INPUT_GPU_RESERVATION.json"
            if not input_complete and reservation.exists() and gpu in read(reservation).get("gpus", []):
                save(ROOT / "jobs" / f"worker_gpu{gpu}.json",
                     dict(state="reserved_for_input_preparation", updated_at=time.time()))
                await asyncio.sleep(30)
                continue
            if not pending:
                if input_complete:
                    break
                save(
                    ROOT / "jobs" / f"worker_gpu{gpu}.json",
                    dict(
                        state="waiting_reviewed_inputs",
                        registered=len(registered),
                        updated_at=time.time(),
                    ),
                )
                await asyncio.sleep(30)
                continue
            free = memory_free(gpu)
            if gpu in adopted:
                eligible = pending.index(adopted[gpu])
            else:
                reserved = set(adopted.values())
                eligible = next(
                    (i for i, name in enumerate(pending)
                     if name not in reserved and max(required_memory(name), args.min_free_mib) <= free),
                    None,
                )
            if eligible is None or shutil.disk_usage(ROOT).free < 12 * 2**30:
                save(
                    ROOT / "jobs" / f"worker_gpu{gpu}.json",
                    dict(
                        state="waiting_resources",
                        memory_free_MiB=free,
                        pending=len(pending),
                        updated_at=time.time(),
                    ),
                )
                await asyncio.sleep(30)
                continue
            identity = pending.pop(eligible)
            adopted.pop(gpu, None)
            save(
                ROOT / "jobs" / f"worker_gpu{gpu}.json",
                dict(state="running", identity=identity, updated_at=time.time()),
            )
            try:
                with (ROOT / "jobs" / f"{identity}.lock").open("a+") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pending.append(identity)
                        save(ROOT / "jobs" / f"worker_gpu{gpu}.json",
                             dict(state="asset_owned_by_other_controller", identity=identity,
                                  updated_at=time.time()))
                        await asyncio.sleep(5)
                        continue
                    await execute(identity, gpu)
            except Exception as e:
                errors.append(dict(identity=identity, error=str(e)))
                save(
                    ROOT / "jobs" / f"{identity}.json",
                    dict(
                        state="failed_or_needs_review",
                        error=str(e),
                        gpu=gpu,
                        updated_at=time.time(),
                    ),
                )
        save(
            ROOT / "jobs" / f"worker_gpu{gpu}.json",
            dict(state="queue_drained", updated_at=time.time()),
        )

    await asyncio.gather(*(consume(g) for g in args.gpus))
    save(
        ROOT / ("queue_exit.json" if args.queue_name == "production" else f"queue_exit_{args.queue_name}.json"),
        dict(
            state="needs_review" if errors else "batch_complete_pending_visual_review",
            errors=errors,
            finished_at=time.time(),
            cases=sorted(registered),
        ),
    )
    queue_lock.close()
    if errors:
        raise RuntimeError(errors)


if __name__ == "__main__":
    asyncio.run(main())
