"""Audited G0 input packets and the pinned official 4DAnyone Turbo inference.

The input boundary is an existing calibrated/cropped G0 packet, not an untracked
Waymo raw frame. Detection/tracking and human quality acceptance remain upstream.
"""

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
import numpy as np
from common import ROOT, CODE_ROOT, FRONTEND, PYTHONS, case, read, save, sha, gpu_identity


def classify_temporal_signals(confidence, angular_step, projection_error, policy):
    """Visibility loss must coincide with a fitting anomaly, not occlusion alone.

    Inputs are per-frame/per-body-group scalars. Angular differences are SO(3)
    geodesics. Generated/heldout RGB never selects a favorable metric window.
    """
    confidence, angular_step, projection_error = map(
        lambda x: np.asarray(x, dtype=np.float64), (confidence, angular_step, projection_error))
    assert confidence.shape == angular_step.shape == projection_error.shape
    assert confidence.ndim == 2 and all(np.isfinite(x).all() for x in (confidence, angular_step, projection_error))
    count = len(confidence)
    baseline_start = max(1, int(count * policy["baseline_start_fraction"]))
    baseline_end = int(count * policy["baseline_end_fraction"])
    reference = np.median(confidence[baseline_start:baseline_end], axis=0)
    weak = (confidence < policy["confidence_low"]) | (reference - confidence > policy["confidence_drop"])
    reliable = [angular_step[baseline_start:baseline_end, i][~weak[baseline_start:baseline_end, i]]
                for i in range(confidence.shape[1])]
    supported_groups = np.array([len(values) >= 5 for values in reliable])
    if not supported_groups.any():
        return dict(decision="skip", start_frame=None, end_frame_exclusive=None, frame_count=count,
                    reason="Insufficient trustworthy interior to estimate motion baseline", quality_approved=False)
    angular_limit = np.maximum(policy["angular_step_radians"],
        policy["angular_baseline_multiplier"] * np.array([np.quantile(values, .95) if len(values) >= 5 else 0
                                                          for values in reliable]))
    abnormal = (angular_step > angular_limit) | (projection_error > policy["projection_error_height"])
    abnormal &= supported_groups
    evidence = np.flatnonzero((weak & abnormal).any(axis=1))
    runs = np.split(evidence, np.flatnonzero(np.diff(evidence) > policy["maximum_evidence_gap"] + 1) + 1)
    supported = [r for r in runs if len(r) >= policy["minimum_bad_frames"]]
    decision, start, end, reason = "keep_full", 0, count, "No sustained corroborated anomaly detected"
    if supported:
        middle_anomaly = False
        degraded_groups = ((reference - confidence > policy["onset_confidence_drop"]) &
                           (confidence < policy["onset_confidence_low"]))
        for run in supported:
            first, last = int(run[0]), int(run[-1])
            head = first <= policy["maximum_evidence_gap"]
            tail = last >= count - 1 - policy["maximum_evidence_gap"]
            if head:
                responsible = weak[last] & abnormal[last]
                degraded = degraded_groups[:, responsible].any(axis=1)
                boundary = last + 1
                while boundary < count and degraded[boundary]:
                    boundary += 1
                start = max(start, boundary + policy["boundary_margin_frames"])
            elif tail:
                responsible = weak[first] & abnormal[first]
                degraded = degraded_groups[:, responsible].any(axis=1)
                boundary = first
                while boundary > 0 and degraded[boundary-1]:
                    boundary -= 1
                end = min(end, max(0, boundary - policy["boundary_margin_frames"]))
            else:
                middle_anomaly = True
        removed = start + count - end
        short_edges = (not middle_anomaly and removed <= policy["maximum_trim_frames"] and
                      removed / count <= policy["maximum_trim_fraction"] and
                      end-start >= policy["minimum_retained_frames"])
        decision = "trim_edges" if short_edges else "skip"
        reason = ("Corroborated short boundary anomalies; keep one continuous interior" if short_edges
                  else "Interior anomaly or total trimming exceeds the fixed small-cut limit")
        if decision == "skip":
            start = end = None
    return dict(decision=decision, start_frame=start, end_frame_exclusive=end, reason=reason,
        frame_count=count, reference_confidence=reference.tolist(), angular_limits=angular_limit.tolist(),
        unmeasured_groups=np.flatnonzero(~supported_groups).tolist(),
        supported_bad_runs=[r.tolist() for r in supported], bad_frames=evidence.tolist(),
        confidence=confidence.tolist(), angular_step=angular_step.tolist(), projection_error=projection_error.tolist(),
        quality_approved=False)


def temporal_diagnosis(identity, policy=None):
    from safetensors.numpy import load
    from scipy.spatial.transform import Rotation

    policy = policy or read(CODE_ROOT / "config/temporal_policy.json")
    spec = case(identity)
    motion = Path(spec["motion"]) / "motion.safetensors"
    projection = Path(spec["selection"]["package"]) / "clip0/motion_pair/calibrated_moving/projection.npz"
    if not projection.is_file():
        return dict(identity=identity, decision="skip", reason="Required calibrated pose diagnostic missing",
                    start_frame=None, end_frame_exclusive=None, quality_approved=False, policy=policy)
    # These packets are small. Own the bytes and avoid concurrent mmap/native
    # tensor-reader initialization in the bounded CPU diagnostic workers.
    tensors = load(motion.read_bytes())
    count = spec["prepared_frames"]
    pose, observed = tensors["smpl_params_incam.body_pose"], tensors["observed_keypoints_2d"]
    assert pose.shape == (count, 63) and observed.shape == (count, 17, 3)
    rotation = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(count, 21, 3, 3)
    relative = rotation[1:] @ rotation[:-1].swapaxes(-1, -2)
    steps = Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude().reshape(count-1, 21)
    steps = np.concatenate((np.zeros((1, 21)), steps))
    with np.load(projection) as p:
        assert p["observed_coco17"].shape == observed.shape
        assert np.allclose(p["observed_coco17"], observed, atol=1e-5, rtol=0)
        height = p["source_boxes"][:, 3] - p["source_boxes"][:, 1]
        assert (height > 0).all()
        errors = p["detector_distance_px"] / height[:, None]
    # COCO joints and corresponding SMPL body-pose indices, per limb.
    groups = [([11, 13, 15], [0, 3, 6, 9]), ([12, 14, 16], [1, 4, 7, 10]),
              ([5, 7, 9], [12, 15, 17, 19]), ([6, 8, 10], [13, 16, 18, 20])]
    confidence = np.stack([observed[:, ids, 2].mean(1) for ids, _ in groups], axis=1)
    angular = np.stack([steps[:, ids].max(1) for _, ids in groups], axis=1)
    error = np.stack([np.quantile(errors[:, ids], .75, axis=1) for ids, _ in groups], axis=1)
    result = classify_temporal_signals(confidence, angular, error, policy)
    result.update(identity=identity, policy=policy, source_sha256={str(p): sha(p) for p in (motion, projection)},
        groups=["left_leg", "right_leg", "left_arm", "right_arm"],
        limitations="Detector confidence and estimated pose consistency are not truth; gradual or confidently wrong poses and RGB-only artifacts may escape detection")
    return result


def apply_temporal_policy(identity):
    """Automatic admission before RGBA creation, without touching original files.

    Prefixes live in an isolated sub-run; publish only a verified per-case
    pointer. Child processes of an existing queue can use this without restart.
    """
    import fcntl, hashlib, subprocess

    policy_file = ROOT / "TEMPORAL_POLICY.json"
    if not policy_file.exists():
        return
    policy = read(policy_file)
    assert policy["mode"] in ("audit", "enforce")
    folder = ROOT / "temporal"
    folder.mkdir(exist_ok=True)
    with (folder / f"{identity}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = folder / f"{identity}.json"
        if marker.exists():
            record = read(marker)
            assert record["policy_sha256"] == sha(policy_file), "Policy changed during batch"
            if record["state"] == "skipped":
                raise RuntimeError(f"Temporal screening skipped {identity}: {record['diagnosis']['reason']}")
            if record["state"] in ("admitted_full", "admitted_prefix", "audit_only", "existing_preparation_unchanged", "admitted_reviewed_subset"):
                for path, digest in record.get("source_sha256", {}).items():
                    assert sha(path) == digest, "Screened source changed"
                if record["state"] == "admitted_prefix":
                    assert sha(ROOT / "config/case_overrides" / f"{identity}.json") == record["override_sha256"]
                return
        if (ROOT / "data" / identity / "rgba").exists():
            save(marker, dict(state="existing_preparation_unchanged", policy_sha256=sha(policy_file),
                              reason="Preparation predates policy; not automatically quality accepted"))
            return
        spec = case(identity)
        plan_path = ROOT / "REPAIR_PLAN.json"
        if "temporal_subset" in spec and plan_path.is_file():
            # A reviewed generated interval has no new pose-fit diagnostic.
            # Require the frozen per-asset interval and the lossless slice proof;
            # do not misclassify this explicit repair as an unreviewed input.
            planned = next(x for x in read(plan_path)["repairs"] if x["identity"] == identity)
            subset = spec["temporal_subset"]
            assert planned["reviewed_boundary_all_views"] is True
            assert (planned["start"], planned["end_exclusive"]) == (subset["range_start"], subset["range_end_exclusive"])
            evidence = Path(planned["evidence"])
            assert sha(evidence) == planned["evidence_sha256"]
            proof_path = Path(spec["selection"]["package"]) / "video_slice_validation.json"
            proof = read(proof_path)
            assert proof["all_retained_rgb_exact"] and len(proof["videos"]) == 50
            assert all(v["frames"] == spec["prepared_frames"] and
                       v["source_start"] == planned["start"] and
                       v["source_end_exclusive"] == planned["end_exclusive"] and
                       sha(v["target"]) == v["sha256"] for v in proof["videos"])
            save(marker, dict(state="admitted_reviewed_subset", policy_sha256=sha(policy_file),
                source_sha256={str(plan_path): sha(plan_path), str(evidence): sha(evidence),
                               str(proof_path): sha(proof_path)},
                reason="Frozen manually reviewed interval; not automatic pose or final quality acceptance"))
            return
        diagnosis = temporal_diagnosis(identity, policy)
        record = dict(diagnosis=diagnosis, policy_sha256=sha(policy_file),
                      source_sha256=diagnosis.get("source_sha256", {}))
        if policy["mode"] == "audit":
            record["state"] = "audit_only"
        elif diagnosis["decision"] == "skip":
            record["state"] = "skipped"
        elif diagnosis["decision"] == "keep_full":
            record["state"] = "admitted_full"
        else:
            child = ROOT / "temporal_inputs" / identity
            env = os.environ.copy()
            env["HUMAN_RUN_ROOT"] = str(child)
            env["HUMAN_ASSET_ROOT"] = str(child / "unused_assets")
            if not (child / "outputs" / identity / "generation/human_generation_receipt.json").exists():
                assert not child.exists(), "Partial temporal input needs explicit recovery"
                subprocess.run([str(PYTHONS["4danyone"]), str(CODE_ROOT / "frontend.py"), "subset", identity,
                    "--source-root", str(ROOT), "--source-identity", identity,
                    "--start-frame", str(diagnosis["start_frame"]),
                    "--end-frame", str(diagnosis["end_frame_exclusive"])], env=env, check=True)
            derived = read(child / "config/cases.json")[identity]
            assert derived["prepared_frames"] == diagnosis["end_frame_exclusive"] - diagnosis["start_frame"]
            assert read(child / "data" / identity / "packet/video_slice_validation.json")["all_retained_rgb_exact"]
            original = read(ROOT / "config/cases.json")[identity]
            derived.update(manifest=original["manifest"], config=original["config"],
                           generation=str(child / "outputs" / identity / "generation"))
            override = ROOT / "config/case_overrides" / f"{identity}.json"
            save(override, dict(original_spec_sha256=hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest(),
                                specification=derived, automatic_temporal_diagnosis=diagnosis))
            record.update(state="admitted_prefix", override_sha256=sha(override),
                          retained_frames=derived["prepared_frames"], derived_root=str(child))
        save(marker, record)
        if record["state"] == "skipped":
            raise RuntimeError(f"Temporal screening skipped {identity}: {diagnosis['reason']}")


def prepare_registered(identity):
    specification = case(identity)
    folder = ROOT / "data" / identity
    folder.mkdir(parents=True, exist_ok=True)
    if "prepared_frames" in specification:
        assert specification["input_review"][
            "allow_generation"
        ], "Review input and accessories first"
        frame_count = specification["prepared_frames"]
        if "input_sha256" in specification["input_review"]:
            assert (
                sha(specification["raw_input"])
                == specification["input_review"]["input_sha256"]
            ), "Reviewed appearance input changed"
    else:
        data = read(specification["manifest"])
        assert sha(specification["manifest"]) == specification["manifest_sha256"]
        frame_count = data["frame_count"]
    conditioning = Path(specification["conditioning"])
    motion = Path(specification["motion"])
    assert (conditioning / "cameras.json").is_file() and (
        motion / "motion.safetensors"
    ).is_file()
    records = {}
    for key, target in dict(
        input_video=specification["raw_input"],
        conditioning=str(conditioning),
        motion=str(motion),
    ).items():
        path = Path(target)
        assert path.exists(), path
        link = folder / key
        if link.is_symlink():
            assert link.resolve() == path.resolve()
        elif not link.exists():
            link.symlink_to(path.resolve(), target_is_directory=path.is_dir())
        else:
            raise FileExistsError(link)
    files = [
        Path(specification["raw_input"]),
        conditioning / "cameras.json",
        conditioning / "metadata.json",
        motion / "motion.json",
        motion / "motion.safetensors",
    ]
    for file in files:
        records[str(file)] = sha(file)
    hints=dict(state='unavailable_missing_registered_root_motion')
    if 'root_motion' in specification:
        from placement import write_source_motion_hints
        hint_info,hint_arrays=write_source_motion_hints(identity,folder/'motion_hints')
        hints=dict(state='source_hints_saved_not_final_motion_validation',
                   metadata=str(hint_info),metadata_sha256=sha(hint_info),arrays_sha256=sha(hint_arrays))
    save(
        folder / "READY.json",
        dict(
            state="registered_G0_packet",
            identity=identity,
            input_boundary="Previously prepared frozen G0 packet; raw Waymo tracker/segmentation not rerun",
            frames=frame_count,
            fps=25,
            source_files_sha256=records,
            source_motion_hints=hints,
            new_conditioning_generated=False,
            new_RGB_generated=False,
        ),
    )
    return folder


def generate(identity):
    specification = case(identity)
    # Each generation child reads this at launch; an already running batch
    # controller cannot bypass a newly enabled input policy.
    attachment_policy=ROOT/'ATTACHMENT_POLICY.json'
    if (attachment_policy.exists() and read(attachment_policy).get('mode')=='enforce'
            or (ROOT/'review'/f'{identity}_generation_exclusion.json').exists()):
        # Use the existing SDK environment; do not add API dependencies to the
        # frozen diffusion environment or import source from another project.
        api_env=os.environ.copy();api_env.pop('PYTHONPATH',None)
        __import__('subprocess').run([str(PYTHONS['vllm_vlm']),str(CODE_ROOT/'intake.py'),
            'attachment',identity],env=api_env,check=True)
    assert "temporal_subset" not in specification, "A generated-video subset must not rerun diffusion"
    policy_file = ROOT / "TEMPORAL_POLICY.json"
    if policy_file.exists() and read(policy_file)["mode"] == "enforce":
        diagnosis = temporal_diagnosis(identity, read(policy_file))
        save(ROOT / "reports" / f"{identity}_temporal_precheck.json", diagnosis)
        if diagnosis["decision"] == "skip":
            save(ROOT / "temporal" / f"{identity}.json", dict(state="skipped", diagnosis=diagnosis,
                policy_sha256=sha(policy_file), source_sha256=diagnosis.get("source_sha256", {}),
                stage="before_diffusion", quality_approved=False))
            raise RuntimeError(f"Temporal screening skipped {identity}: {diagnosis['reason']}")
    folder = prepare_registered(identity)
    ready = read(folder / "READY.json")
    for path, digest in ready["source_files_sha256"].items():
        assert sha(path) == digest
    gpu_identity()
    sys.path.insert(0, str(FRONTEND))
    # Match the historical 61-slot config and its motion validator. Pose-feature
    # shape is the same pinned dynamic implementation retained in vendor/.
    import fdanyone.config as configuration

    configuration.INFERENCE = dataclasses.replace(
        configuration.INFERENCE, num_frames=ready["frames"]
    )
    from fdanyone import pipeline
    from fdanyone.motion.result import MotionResult

    MotionResult.validate.__defaults__ = (ready["frames"],)
    from fdanyone.skeleton.pipeline import Conditioning
    from inference import inference

    pipeline._run_motion = lambda **kwargs: MotionResult.load(folder / "motion")
    pipeline._build_conditioning = lambda **kwargs: Conditioning.load(
        folder / "conditioning"
    )
    target = ROOT / "outputs" / identity / "generation"
    if target.exists():
        raise FileExistsError(target)
    import torch

    torch.cuda.reset_peak_memory_stats()
    result = inference(
        video_path=str(folder / "input_video"),
        output_dir=str(target),
        views_per_layer=24,
        layer_pitches=[15],
        start_yaw=0,
        yaw_span=360,
        enable_rcp=True,
        enable_tcr=True,
        enable_turbo=True,
        target_fps=25,
        seed=42,
        gpu_ids=[0],
        attention_backend="sdpa",
        model_dir=str(FRONTEND / "models"),
        gvhmr_root=str(FRONTEND / "third_party/GVHMR"),
    )
    finalize_generated(identity, result)
    save(
        target / "memory.json",
        dict(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            frames=ready["frames"],
            scope="This inference process only; other GPU tasks excluded",
        ),
    )


def validate_cameras(published, conditioning):
    """Publication renames angle/path fields; physical camera values must match exactly."""
    assert len(published["cameras"]) == len(conditioning["cameras"]) == 24
    for index, (output, source) in enumerate(
        zip(published["cameras"], conditioning["cameras"], strict=True)
    ):
        assert output["camera_id"] == source["camera_id"] == index
        for field in (
            "K",
            "camera_to_world",
            "image_height",
            "image_width",
            "layer_index",
        ):
            assert output[field] == source[field], (index, field)
        assert output["yaw"] == source["yaw_degrees"]
        assert output["pitch"] == source["pitch_degrees"]
        assert output["video"] == f"videos/{index:02d}.mp4"
        assert output["skeleton_video"] == f"skeletons/{index:02d}.mp4"
    assert published["world_frame"] == conditioning["world_frame"]


def finalize_generated(identity, result=None, recovered=False):
    specification = case(identity)
    folder = ROOT / "data" / identity
    target = ROOT / "outputs" / identity / "generation"
    ready = read(folder / "READY.json")
    for path, digest in ready["source_files_sha256"].items():
        assert sha(path) == digest
    metadata = read(target / "metadata.json")
    assert metadata["output"] == dict(
        fps="25/1",
        frames_per_video=ready["frames"],
        height=1280,
        target_views=24,
        width=704,
    )
    assert sorted(p.name for p in (target / "videos").glob("*.mp4")) == [
        f"{i:02d}.mp4" for i in range(24)
    ]
    if "root_motion" in specification:
        import shutil

        root_motion = Path(specification["root_motion"])
        shutil.copy2(root_motion, target / "root_motion.npz")
        cameras = read(target / "cameras.json")
        original_cameras = read(folder / "conditioning/cameras.json")
        validate_cameras(cameras, original_cameras)
        cameras["world_frame"] = original_cameras["world_frame"]
        cameras["root_motion"] = dict(
            file="root_motion.npz",
            sha256=sha(root_motion),
            local_to_canonical_key="local_to_canonical",
            time_key="frame_times",
            note="Estimated root motion; not verified Waymo metric trajectory",
        )
        save(target / "cameras.json", cameras)
    save(
        target / "human_generation_receipt.json",
        dict(
            input_packet=str(folder),
            frames=ready["frames"],
            views=24,
            seed=42,
            steps=4,
            backend="sdpa",
            result=result if result is not None else metadata,
            recovered_published_outputs=recovered,
            note="Frontend weights unchanged; not retrained. Generated quality is not automatically accepted.",
        ),
    )


def validate_generated(identity):
    """Decode all outputs; compare cameras and fixed frames with the frozen frontend."""
    import av
    from PIL import Image, ImageDraw

    target = ROOT / "outputs" / identity / "generation"
    metadata = read(target / "metadata.json")
    frames = read(ROOT / "data" / identity / "READY.json")["frames"]
    assert metadata["output"] == dict(
        fps="25/1", frames_per_video=frames, height=1280, target_views=24, width=704
    )
    if "baseline_metrics" not in case(identity):
        cameras = read(target / "cameras.json")
        original = read(Path(case(identity)["conditioning"]) / "cameras.json")
        validate_cameras(cameras, original)
        rows = []
        for view in range(24):
            file = target / "videos" / f"{view:02d}.mp4"
            with av.open(str(file)) as video:
                video.streams.video[0].thread_count = 1
                count = 0
                for i, f in enumerate(video.decode(video=0)):
                    assert (f.width, f.height) == (704, 1280) and abs(
                        float(f.pts * f.time_base) - i / 25
                    ) < 1e-7
                    count += 1
            assert count == frames
            rows.append(dict(view=view, frames=count, sha256=sha(file)))
        save(
            ROOT / "reports" / f"{identity}_frontend_validation.json",
            dict(
                state="all_videos_decoded_cameras_matched",
                videos=rows,
                quality_approved=False,
                scope="Technical validation only; generated appearance and motion need inspection",
            ),
        )
        return
    old = Path(case(identity)["generation_root"]) / "outputs/turbo"
    cameras = read(target / "cameras.json")
    reference = read(old / "cameras.json")
    assert len(cameras["cameras"]) == 24
    for a, b in zip(cameras["cameras"], reference["cameras"], strict=True):
        for key in (
            "camera_id",
            "K",
            "camera_to_world",
            "image_height",
            "image_width",
            "yaw",
            "pitch",
        ):
            assert np.allclose(a[key], b[key], atol=1e-9), key
    selected = {0, (frames - 1) // 2, frames - 1}
    rows = []
    pictures = {}
    for view in range(24):
        with av.open(str(target / "videos" / f"{view:02d}.mp4")) as video:
            stream = video.streams.video[0]
            stream.thread_count = 2
            count = 0
            for i, frame in enumerate(video.decode(video=0)):
                assert (frame.width, frame.height) == (704, 1280)
                assert abs(float(frame.pts * frame.time_base) - i / 25) < 1e-7
                if view in (0, 8, 16) and i in selected:
                    pictures[view, i] = frame.to_image().resize(
                        (176, 320), Image.Resampling.LANCZOS
                    )
                count += 1
            assert count == frames
        rows.append(dict(view=view, frames=count, width=704, height=1280, fps=25))
    comparisons = []
    sheet = Image.new("RGB", (1056, 1008), (80, 80, 80))
    for column, view in enumerate((0, 8, 16)):
        with av.open(str(old / "videos" / f"{view:02d}.mp4")) as video:
            video.streams.video[0].thread_count = 2
            for i, frame in enumerate(video.decode(video=0)):
                if i not in selected:
                    continue
                image = frame.to_image().resize((176, 320), Image.Resampling.LANCZOS)
                new = pictures[view, i]
                difference = np.asarray(image, dtype=np.float32) - np.asarray(
                    new, dtype=np.float32
                )
                comparisons.append(
                    dict(
                        view=view,
                        frame=i,
                        thumbnail_RGB_MAE=float(np.abs(difference).mean() / 255),
                    )
                )
                row = sorted(selected).index(i)
                x = column * 352
                y = row * 336
                ImageDraw.Draw(sheet).text(
                    (x + 3, y + 3), f"v{view:02d} f{i}: old | new", fill="white"
                )
                sheet.paste(image, (x, y + 16))
                sheet.paste(new, (x + 176, y + 16))
    sheet.save(ROOT / "reports" / f"{identity}_frontend_comparison.jpg", quality=94)
    save(
        ROOT / "reports" / f"{identity}_frontend_validation.json",
        dict(
            state="all_videos_decoded_cameras_matched",
            videos=rows,
            reference_generation=str(old),
            fixed_frame_diagnostic=comparisons,
            scope="Frozen packet, weights and cameras; pixel similarity is diagnostic, not true Waymo novel-view accuracy",
        ),
    )


def recover_published(identity):
    """Explicitly validate a fully published run before recovering its receipt."""
    target = ROOT / "outputs" / identity / "generation"
    assert not (target / "human_generation_receipt.json").exists()
    validate_generated(identity)
    finalize_generated(identity, recovered=True)


def trim_video_interval(src, dst, start_frame, end_frame):
    import av, hashlib
    from fractions import Fraction
    src, dst = Path(src), Path(dst)
    count = end_frame - start_frame
    dst.parent.mkdir(parents=True, exist_ok=True)
    assert not dst.exists()
    hashes = []
    with av.open(str(src)) as inp, av.open(str(dst), "w") as out:
        video = inp.streams.video[0]
        video.thread_count = 1
        assert float(video.average_rate) == 25
        stream = out.add_stream("libx264rgb", rate=25)
        stream.width, stream.height = video.width, video.height
        stream.pix_fmt = "rgb24"
        stream.options = {"crf": "0", "preset": "ultrafast", "threads": "1"}
        for index, frame in enumerate(inp.decode(video=0)):
            if index == end_frame:
                break
            assert abs(float(frame.pts * frame.time_base) - index/25) < 1e-7
            if index < start_frame:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
            converted = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            converted.pts, converted.time_base = index-start_frame, Fraction(1, 25)
            for encoded in stream.encode(converted):
                out.mux(encoded)
        for encoded in stream.encode():
            out.mux(encoded)
    assert len(hashes) == count
    checked = []
    with av.open(str(dst)) as inp:
        inp.streams.video[0].thread_count = 1
        for index, frame in enumerate(inp.decode(video=0)):
            assert abs(float(frame.pts * frame.time_base) - index/25) < 1e-7
            checked.append(hashlib.sha256(frame.to_ndarray(format="rgb24").tobytes()).hexdigest())
    assert checked == hashes, (src, "retained RGB changed")
    return dict(source=str(src), target=str(dst), frames=count,
                source_start=start_frame, source_end_exclusive=end_frame,
                source_sha256=sha(src), sha256=sha(dst), retained_rgb_sha256=hashes)


async def prepare_temporal_subset(identity, source_root, source_identity, end_frame, start_frame=0):
    """Losslessly retain one generated interval and matching G0 provenance.

    This is backend input selection, not a new diffusion run or pose repair.
    Source files and completed assets are never changed. End frame is exclusive.
    """
    import asyncio, copy, hashlib, shutil
    import av
    from fractions import Fraction
    from safetensors.numpy import load_file, save_file

    source_root = Path(source_root).resolve()
    assert source_root != ROOT
    original = read(source_root / "config/cases.json")[source_identity]
    total = int(original["prepared_frames"])
    count = end_frame - start_frame
    assert 0 <= start_frame < end_frame <= total and 16 <= count < total
    registry_file = ROOT / "config/cases.json"
    registry = read(registry_file) if registry_file.exists() else {}
    assert identity not in registry
    source = source_root / "outputs" / source_identity / "generation"
    receipt = read(source / "human_generation_receipt.json")
    assert receipt["frames"] == total and receipt["views"] == 24
    assert read(source_root / "reports" / f"{source_identity}_frontend_validation.json")["state"] == "all_videos_decoded_cameras_matched"
    output = ROOT / "outputs" / identity / "generation"
    packet = ROOT / "data" / identity / "packet"
    assert not output.exists() and not packet.exists()
    output.mkdir(parents=True)
    packet.mkdir(parents=True)
    clip = packet / "clip0"
    (clip / "inputs").mkdir(parents=True)
    motion = packet / "motion"
    motion.mkdir()
    provenance = dict(source_root=str(source_root), source_identity=source_identity,
                      source_frame_count=total, retained_source_frames=list(range(start_frame, end_frame)),
                      range_start=start_frame, range_end_exclusive=end_frame, fps=25,
                      source_time_offset_sec=start_frame/25, operation="generated_video_interval_selection",
                      diffusion_rerun=False, pose_refit=False, display_only_edit=False,
                      source_receipt_sha256=sha(source / "human_generation_receipt.json"))
    save(packet / "temporal_subset.json", provenance)


    pairs = [(source / folder / f"{v:02d}.mp4", output / folder / f"{v:02d}.mp4")
             for folder in ("videos", "skeletons") for v in range(24)]
    pairs += [(Path(original["raw_input"]), clip / "inputs/main.mp4"),
              (Path(original["selection"]["pose"]), clip / "skeleton.mp4")]
    sem = asyncio.Semaphore(2)
    async def bounded(pair):
        async with sem:
            # Keep native codec contexts in separate processes: concurrent Python
            # threads can stall inside PyAV teardown even with codec threads=1.
            proc = await asyncio.create_subprocess_exec(
                sys.executable, __file__, "slice-video", "clip",
                "--source-video", str(pair[0]), "--target-video", str(pair[1]),
                "--start-frame", str(start_frame), "--end-frame", str(end_frame),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await proc.communicate()
            if proc.returncode:
                raise RuntimeError(stderr.decode()[-3000:])
            return json.loads(stdout)
    records = await asyncio.gather(*(bounded(pair) for pair in pairs))
    save(packet / "video_slice_validation.json", dict(videos=records, all_retained_rgb_exact=True))

    source_clip = Path(original["selection"]["package"]) / "clip0"
    mapping = read(source_clip / "inputs/frame_mapping.json")
    assert len(mapping["frames"]) == total
    mapping["frames"] = mapping["frames"][start_frame:end_frame]
    for slot, row in enumerate(mapping["frames"]):
        row["original_slot"] = row["slot"]
        row["original_video_seconds"] = row["video_seconds"]
        row.update(slot=slot, video_seconds=slot/25)
    mapping["temporal_subset"] = provenance
    # The original native observations remain available as source context.
    shutil.copy2(source_clip / "native.json", clip / "native.json")
    for src, dst in ((Path(original["root_motion"]), packet / "root_motion.npz"),
                     (source_clip / "inputs/cameras.npz", clip / "inputs/cameras.npz")):
        with np.load(src) as data:
            assert all(data[k].shape[0] == total for k in data.files)
            arrays = {k: data[k][start_frame:end_frame].copy() for k in data.files}
        for key in ("frame_times", "video_seconds"):
            if key in arrays:
                arrays[key] = np.arange(count, dtype=arrays[key].dtype) / 25
        np.savez_compressed(dst, **arrays)
        with np.load(dst) as check:
            assert all(np.array_equal(check[k], arrays[k]) for k in arrays)
    tensors = load_file(str(Path(original["motion"]) / "motion.safetensors"))
    assert all(t.shape[0] == total for t in tensors.values())
    save_file({k: t[start_frame:end_frame].copy() for k, t in tensors.items()}, str(motion / "motion.safetensors"))
    meta = read(Path(original["motion"]) / "motion.json")
    for key in ("frame_timestamps_sec", "source_frame_indices", "source_pts"):
        assert len(meta[key]) == total
        meta[key] = meta[key][start_frame:end_frame]
    meta["frame_timestamps_sec"] = [i/25 for i in range(count)]
    meta.update(num_frames=count, temporal_subset=provenance)
    save(motion / "motion.json", meta)
    mapping.update(output_frames=count, sample_span_s=(count-1)/25,
        nominal_frame_span_s=(count-1)/25, sampled_source_span_s=(count-1)/25,
        video_sha256=sha(clip / "inputs/main.mp4"), camera_npz_sha256=sha(clip / "inputs/cameras.npz"),
        direct_native_output_slots=sum(not row["interpolated"] for row in mapping["frames"]),
        interpolated_slots=sum(row["interpolated"] for row in mapping["frames"]))
    save(clip / "inputs/frame_mapping.json", mapping)
    metadata = read(source / "metadata.json")
    metadata["output"]["frames_per_video"] = count
    metadata["temporal_subset"] = provenance
    save(output / "metadata.json", metadata)
    cameras = read(source / "cameras.json")
    shutil.copy2(packet / "root_motion.npz", output / "root_motion.npz")
    cameras["root_motion"]["sha256"] = sha(output / "root_motion.npz")
    save(output / "cameras.json", cameras)
    spec = copy.deepcopy(original)
    spec.update(prepared_frames=count, raw_input=str(clip / "inputs/main.mp4"),
                motion=str(motion), root_motion=str(packet / "root_motion.npz"),
                manifest=str(ROOT / "data" / identity / "fdanyone.json"),
                config=str(ROOT / "config" / f"{identity}.json"), temporal_subset=provenance)
    spec["selection"].update(id=identity, package=str(packet), pose=str(clip / "skeleton.mp4"),
                             input=spec["raw_input"], frames=str(count))
    spec["input_review"].update(input_sha256=sha(spec["raw_input"]),
                                scope="Recorded continuous interval selection; no new diffusion or pose repair")
    # Register only after all video and temporal-array checks pass.
    registry[identity] = spec
    save(registry_file, registry)
    prepare_registered(identity)
    validate_generated(identity)
    save(output / "human_generation_receipt.json", dict(
        input_packet=str(ROOT / "data" / identity), frames=count, views=24,
        temporal_subset=provenance, inherited_generation_receipt=receipt,
        note="Exact decoded RGB interval of existing generation; no new diffusion inference"))
    print(identity, "temporal subset verified", start_frame, end_frame, count, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "generate", "validate", "recover", "subset", "temporal-check", "slice-video"])
    ap.add_argument("identity")
    ap.add_argument("--source-root")
    ap.add_argument("--source-identity")
    ap.add_argument("--source-video")
    ap.add_argument("--target-video")
    ap.add_argument("--end-frame", type=int)
    ap.add_argument("--start-frame", type=int, default=0)
    args = ap.parse_args()
    if args.stage == "slice-video":
        print(json.dumps(trim_video_interval(args.source_video, args.target_video, args.start_frame, args.end_frame)))
        raise SystemExit(0)
    if args.stage == "temporal-check":
        result = temporal_diagnosis(args.identity)
        save(ROOT / "reports" / f"{args.identity}_temporal_diagnosis.json", result)
        print(args.identity, result["decision"], result["end_frame_exclusive"], flush=True)
        raise SystemExit(0)
    if args.stage == "subset":
        import asyncio
        asyncio.run(prepare_temporal_subset(args.identity, args.source_root, args.source_identity, args.end_frame, args.start_frame))
        raise SystemExit(0)
    {
        "prepare": prepare_registered,
        "generate": generate,
        "validate": validate_generated,
        "recover": recover_published,
    }[args.stage](args.identity)
