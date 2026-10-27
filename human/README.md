# Dynamic human asset reconstruction

This directory contains the maintained **screened input → 4DAnyone Turbo → Grid4D → dynamic Gaussian export → independent replay** workflow. It is the current human route, replacing the earlier LHM++ implementation for these assets.

## Input boundary

- **nuScenes:** `intake.py` reads camera metadata and annotated person tracks, crops/interpolates/segments the visible person and estimates motion with GVHMR. Intermediate boxes are interpolated geometric estimates, not extra ground truth.
- **Waymo:** `batch init` consumes an existing screened candidate CSV and G0 packets. The historical raw-Waymo detection/mining scripts are not part of this supported entry point.
- **Generic video:** the model source can process video, but the maintained batch pipeline requires a screened single-person packet with motion, skeletons and cameras. Arbitrary raw web video is not claimed to be an automatic approved input here.

A G0 packet supplies the actual RGB input, source observations, motion tensors, 24 skeleton conditions, cameras, source-frame mapping and root motion. `batch.py:init` and `frontend.py` validate the consumed fields. Multiple track keys are not a guarantee of distinct natural people.

## Setup

Follow [installation](../docs/INSTALL.md). `config/runtime.example.json` selects stage interpreters, CUDA/compiler paths and external checkpoints. Configure the ignored `runtime.local.json` and bind its three model-directory links with `../scripts/configure_human.py`.

```bash
# From the repository root; use absolute work/output directories.
export HUMAN_RUNTIME_CONFIG="$PWD/human/config/runtime.local.json"
export HUMAN_RUN_ROOT=/data/runs/human_scene
export HUMAN_ASSET_ROOT=/data/asset_library/human
export NUSCENES_ROOT=/data/nuScenes
cd human
python run.py doctor
```

`doctor` audits source imports, interpreters and checkpoint references without starting GPU generation. It is a deployment check, not a visual quality approval.

## Preparation and pilot

For prepared Waymo candidates:

```bash
python run.py batch init --limit 10 --pool /data/inputs/priority_candidates.csv
python run.py batch inspect --all --no-overviews
```

For native nuScenes intake:

```bash
python run.py intake rank --limit 10
python run.py intake prepare --cases HN0001 HN0002 --gpus 0 1
python run.py intake checks
python run.py intake review
```

A new native-intake work directory must also contain copies of `config/recipe.json`, `config/export.json`, and `TEMPORAL_POLICY.json` (from `config/temporal_policy.json`). Run `intake review` with an OpenAI-SDK-capable controller interpreter. Model subprocesses use the configured stage interpreters.

Inspect the actual prepared input, including visible bags or other carried objects. `input_review.allow_generation` remains false until evidence supports admission. Missing visible accessories can be repaired with the existing `input restore`, `input accessory`, and `input bind` stages before generation; this restores observed pixels, not unseen anatomy.

The reconstruction driver requires `HUMAN_RUN_ROOT/CODE_VALIDATION.json` with `ready_for_demo_reconstruction: true` and the deployment/preparation validation evidence. This is an explicit deployment record: do not copy a successful record from another input or manufacture a passing result. Configuration examples and CPU unit tests alone are not the evidence for a new full run.

Once the input and deployment checks are recorded:

```bash
python run.py batch pilot --cases HN0001 HN0002 --gpus 0 1
```

This executes 24-view generation, four independent six-view segmentation groups, visual-hull initialization, all reconstruction stages, export and independent replay. The fixed reconstruction schedule is 32,999 base updates, 15,000 silhouette-supervised updates and 5,000 low-learning-rate refinement updates. Holdout views remain isolated from training.

## Batch execution

The queue consumes an explicit `PRODUCTION_INPUTS.json` with unique admitted `ids` and a positive configurable `target_count`. The recorded `BATCH_APPROVAL.json` must match the two pilot IDs in `selection.json`, and each required pilot must have a verified package and a `visual_review.json` allowing that baseline. This preserves the existing pilot review policy; repository integration does not authorize unseen inputs.

```bash
python run.py batch run --all --gpus 0 1 2 3 4 5 6 7
python run.py batch status
```

Jobs are isolated by identity, with bounded asynchronous workers, memory admission and locks. Existing stage records are reused where supported. Partial non-base optimization stages require explicit recovery; arbitrary failures are not silently treated as completed training. Do not run a second controller for the same IDs without the existing job-lock protocol.

`PRODUCTION_PAUSE.json` supports finishing already admitted identities while pausing new work. Read-only incremental auditing is available through `audit.py watch`. It records technical/visual findings and does not automatically delete assets or grant quality approval.

## Outputs and playback

See [asset contracts](../docs/ASSETS.md). Every completed identity contains dynamic Gaussian NPZ, continuous runtime, source/input/camera/motion evidence, placement transforms, validation records and a single five-column `demo.mp4`. Production counts distinguish admitted inputs, technically complete packages and visually accepted assets.

```python
from pathlib import Path
import numpy as np
from playback import sample

package = Path('/data/asset_library/human/HN0001')
time_s = np.arange(61) / 25
positions = np.column_stack((0.7 * time_s, np.zeros(len(time_s)), np.zeros(len(time_s))))
yaw = np.zeros(len(time_s))
world_from_asset, source_frame = sample(package, time_s, positions, yaw,
                                        retime_by_distance=False)
```

Only sample inside the actual package's usable duration. This estimates trajectory-driven playback; it does not generate new articulation. Honor `skip_insertion`, use the documented native-to-asset transform once, and retain full SH/covariance. Neither a renderable package nor a successful coordinate round-trip certifies ground contact, appearance, measured metric scale, or seamless looping.

## Tests and provenance

```bash
python -m unittest discover -s tests -v
```

The integrated source comes from the maintained human checkout at commit `ebbd2c2`; runtime-path configuration was made portable while preserving the reconstruction schedule. Detailed historical evidence and limitations remain in [operations history](../docs/HUMAN_OPERATIONS_HISTORY.md), [temporal validation](TEMPORAL_VALIDATION.md), and [third-party notices](THIRD_PARTY_NOTICES.md).
