# Third-party code and model resources

This integration does not relicense upstream implementations, checkpoints or datasets.

| Component | Source / snapshot |
|---|---|
| TRELLIS B0 runtime | Microsoft TRELLIS; the exact local production source snapshot is recorded in `vendor/source_hashes.json`. MIT notice: `vendor/trellis_b0/LICENSE`. |
| ReconViaGen H runtime | Local production ReconViaGen snapshot, including TRELLIS and VGGT adapters; per-file hashes in `vendor/source_hashes.json`. Existing source notices and `vendor/reconviagen/LICENSE` remain applicable. |
| FLUX reference preprocessing | The used padding/resizing/generation functions from the original project pipeline, consolidated into `asset_extraction/flux.py`. FLUX checkpoints and Diffusers are external dependencies. |
| GroundingDINO and SAM2 frontend | External Grounded-SAM-2 checkout and checkpoints, explicitly configured; not redistributed as model weights. |
| Human runtime sources | See `human/THIRD_PARTY_NOTICES.md` for 4DAnyone, GVHMR, Grid4D, RIFE, SAM2 and supporting implementations. |

Official projects: [TRELLIS](https://github.com/microsoft/TRELLIS), [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2), [4DAnyone](https://github.com/ant-research/4DAnyone), [Grid4D](https://github.com/JiaweiXu8/Grid4D), [GVHMR](https://github.com/zju3dv/GVHMR). Obtain models, body-model files and dataset access under their respective terms. No model weights or raw/derived dataset imagery are included in this Git repository.
