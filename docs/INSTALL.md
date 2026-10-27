# Installation and external resources

DriveLab is a multi-environment research pipeline. Linux, an NVIDIA CUDA GPU, FFmpeg, and compatible C++/CUDA build tools are required for the model stages. CPU contract tests can run without the model checkpoints.

## Environments

Keep the following runtimes separate. `environments/*.constraints.txt` records the package versions observed on the validation host; these are environment inventories, not a promise that every compiled CUDA package is available from PyPI on another machine.

| Runtime | Used for | Setup reference |
|---|---|---|
| Controller / frontend | GroundingDINO, SAM2, async OpenAI client, source preparation | `requirements.txt`, Grounded-SAM-2 installation |
| FLUX | FLUX.2 Klein through Diffusers | `environments/flux.constraints.txt` |
| TRELLIS | H and B0 reconstruction | `vendor/reconviagen/requirements.txt`, `environments/trellis.constraints.txt` |
| Renderer | gsplat CUDA rasterization | `environments/renderer.constraints.txt` |
| 4DAnyone | Motion conditioning and 24-view video generation | `human/vendor/4danyone`, `environments/4danyone.constraints.txt` |
| SAM2 / preparation | Grouped masks and visual-hull preparation | `environments/4danyone-sam2.constraints.txt`, `environments/4danyone-ftgs.constraints.txt` |
| Grid4D | Dynamic reconstruction and replay | `human/vendor/grid4d`, `environments/grid4d.constraints.txt` |

Install the rasterizer, sparse convolution, FlashAttention and hash-encoding extensions against the chosen interpreter's PyTorch/CUDA ABI. The model source snapshots are included; compiled extensions are not. For Grid4D, obtain the upstream `depth-diff-gaussian-rasterization` and `simple-knn` submodules and build them in the Grid4D environment. Human runtime code checks the selected GPU and isolates Python import paths.

The controller's basic dependencies can be installed with:

```bash
python -m pip install -r requirements.txt
```

This does not install all model-specific compiled runtimes. The Waymo source adapter additionally requires `waymo_open_dataset.dataset_pb2` in its selected interpreter; it reads TFRecord framing directly and does not require TensorFlow to decode records. The adapter uses `google-crc32c` to validate record checksums. Video URL ingestion uses `yt-dlp` and FFmpeg; access-restricted sites require their normal authorization.

## Rigid paths

Export these variables before using the portable example configs:

```bash
export NUSCENES_ROOT=/data/nuScenes
export GROUNDED_SAM2_ROOT=/opt/Grounded-SAM-2
export RIGID_PYTHON=/opt/conda/envs/controller/bin/python
export FLUX_PYTHON=/opt/conda/envs/flux/bin/python
export TRELLIS_PYTHON=/opt/conda/envs/trellis/bin/python
export RENDER_PYTHON=/opt/conda/envs/renderer/bin/python
export FLUX_MODEL=/data/models/FLUX.2-klein-9B
export H_MODEL=/data/models/trellis-vggt-v0-2
export VGGT_MODEL=/data/models/vggt-object-v0-1
export B0_MODEL=/data/models/TRELLIS-image-large-gaussian
export TRELLIS_TORCH_HOME=/data/models/trellis_torch
export BIREFNET_MODEL=/data/models/birefnet-general-lite.onnx
```

`GROUNDED_SAM2_ROOT` must contain the expected source tree plus `checkpoints/sam2.1_hiera_large.pt`, `gdino_checkpoints/groundingdino_swint_ogc.pth`, and the GroundingDINO/SAM2 configs. The TRELLIS Torch cache must include the DINOv2 hub source and weights required by the pipeline. Models run from local files; missing resources cause an explicit error.

The new runner expands environment variables and resolves relative paths against the repository root. `configs/generation.example.json` and `scripts/run_nuscenes.sh` retain the original host-specific entry point for compatibility; use `configs/pipeline.example.yaml` and `configs/generation.portable.json` for a new deployment.

## Visual language service

An existing compatible OpenAI endpoint can be reused. To start a local service explicitly:

```bash
PYTHON=/opt/conda/envs/vllm/bin/python \
MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct \
VLM_CUDA_HOME=/usr/local/cuda VLM_GPU=6 VLM_PORT=8082 \
bash scripts/start_vlm.sh
```

The script requests `--gpu-memory-utilization 0.90`. Select a GPU with sufficient capacity and adjust this option for co-resident jobs. The pipeline does not start or stop a shared service implicitly. The new rigid runner forces `json_object` responses and validates their schema locally.

## Human paths

Copy `human/config/runtime.example.json` to `human/config/runtime.local.json`. Supply absolute interpreter, CUDA/compiler, RIFE and SAM2 paths, and the three external checkpoint directories. Then run:

```bash
export HUMAN_RUNTIME_CONFIG="$PWD/human/config/runtime.local.json"
python scripts/configure_human.py "$HUMAN_RUNTIME_CONFIG"        # inspect
python scripts/configure_human.py "$HUMAN_RUNTIME_CONFIG" --apply
```

This creates only three ignored symlinks, and refuses to overwrite a different existing path. It does not download models or alter the original checkpoints. `4danyone/models` requires the checkpoint layout named in `human/vendor/4danyone/fdanyone/assets.py`, including 4DAnyone, Turbo LoRA, VAE, motion/pose models, BiRefNet and the licensed body-model resources. GVHMR checkpoint and metric-weight locations are separate links.

Set `HUMAN_RUN_ROOT` and `HUMAN_ASSET_ROOT` to separate writable work/delivery directories. Use `NUSCENES_ROOT` for native human intake. Runtime config overrides affect this checkout only, not another running checkout. Optional `DRIVELAB_GPU_LOCK_DIR` lets cooperating launchers share GPU leases.

## Tests

For CPU checks, install `numpy scipy Pillow plyfile PyYAML ijson jsonschema openai opencv-python-headless pytest av`, then run the commands in the main README. The optional old trajectory-loader integration tests are skipped when that external consumer is absent. Full GPU validation also requires the environments and checkpoints above.
