# Rigid pipeline

## Stages

1. **Input adapter.** nuScenes camera metadata, rectified Waymo TFRecord images, local video frames, or a custom frame index enter one shared scene contract. Video timestamps come from decoder presentation timestamps. No camera pose is fabricated for video.
2. **Detection and association.** GroundingDINO detects configured classes. Temporal overlap, available camera geometry and optical flow associate observations. Candidates retain their source frame and mask provenance. Near-identical foregrounds and repeated tracklets are screened before generation.
3. **Foreground and completeness checks.** SAM2 masks, pixel extent, connected foreground, clipping, blur and semantic completeness help reject tiny fragments, isolated pole pieces and unusable observations. `max_asset_candidates: 0` means no per-scene identity quota; explicit positive limits remain available for resource-constrained experiments.
4. **Attributes and prompts.** Qwen verifies category and instance attributes. The finalized prompt strings are written once. JSON responses use the OpenAI SDK with bounded asynchronous concurrency and schema validation.
5. **Real-reference selection.** Vehicles prefer 3–4 high-quality, diverse references. Only genuinely usable references count; 2, 1, and 0 are valid. There is no duplicate padding. Decisions remain in the routing sidecar.
6. **FLUX.2 Klein.** Generate standardized views at 1024×1024, 8 steps, guidance 1.0. Vehicles use the observed end plus a side view. Category-dependent single-view cases remain supported.
7. **3D reconstruction.** Vehicles use H / TRELLIS-VGGT with the generated views plus 0–4 real references. Other rigid objects use B0 / original TRELLIS with generated evidence. Both export Gaussian PLY payloads.
8. **Render, curate and normalize.** Validate Gaussian fields, render 16 azimuths, assess category/shape/completeness and estimate canonical front and dimensions. Normalize the payload to centered XY, ground at Z=0 and verified +X heading where applicable. Keep/review/reject are recorded separately. Only otherwise-usable vehicle identity variations are exempt from identity-only rejection.
9. **Publish.** Store a categorized package with source evidence, metadata and hashes. Merge the catalogue under a lock. Record rejected decisions and hashes before retiring that run's rejected generated outputs.

## Configuration

`configs/pipeline.example.yaml` selects input type/path, scene IDs, run directory, central library, GPU IDs, frontend/model configs and chunk size. Omit scene IDs or use an empty list to process all complete selected inputs. There is no 500-per-dataset target in the generic runner. Separate runs can publish to a common library. Use separate output libraries for deliberate alternative recipes of the same identities.

Scenes are processed in order. Within a scene, VLM calls and model jobs use bounded async execution and GPU leases. The production-scale historical scheduler is not a required dependency. Shared leases coordinate DriveLab processes, while memory checks coexist with non-cooperating external workloads; neither mechanism guarantees exclusive GPU performance.

Repeat the same command to resume completed chunks. Changed run configuration is rejected. Completed stage evidence is checked by the generation/curation components. A completed scene may still contain failed/review/rejected objects; read the counts, not only the scene completion flag.

## Stable FLUX contract

```json
[
  {
    "scene_id": "scene-0001",
    "obj_id": "scene-0001_t000007",
    "category": "truck",
    "image_path": "/data/run/crops/scene-0001_t000007.png",
    "prompts": ["... finalized front-view prompt ...", "... finalized side-view prompt ..."]
  }
]
```

These five fields remain the handoff boundary. `asset_routes.json` records H/B0 routing, real references, checksums and provenance separately. Prompts are passed directly to FLUX.

## Custom frame index

A directory can contain one JSON file per continuous scene. Its filename stem must match `scene_id`:

```json
{
  "scene_id": "clip_001",
  "frames": [
    {"path": "/data/frames/000000.jpg", "camera_name": "front", "frame_id": 0, "timestamp_us": 0},
    {"path": "/data/frames/000015.jpg", "camera_name": "front", "frame_id": 1, "timestamp_us": 500000}
  ]
}
```

Optional `motion_paths` contains intermediate images since the previous detector frame. When measured camera metadata is available, provide both `camera_intrinsic` (3×3) and `camera_to_world` (4×4, OpenCV camera convention). Relative image paths resolve against the index directory. Do not invent calibration or native track identities for uncalibrated videos.

RGB association is conservative but not infallible. Similar nearby instances can remain ambiguous, and unrelated clips are not globally deduplicated by appearance. The earlier 1,000-vehicle production batch also used dataset-native identity metadata for deduplication; that experiment-specific logic is not silently presented as RGB-only tracking in this generic workflow.
