# Vehicle asset diagnostics

Measured on 2026-10-27, using NVIDIA GeForce RTX 4090 (not H20), 512×512, batch-one gsplat rendering. Other jobs were allowed to share the GPU.

| Cohort | Assets timed | Matched quality samples | PSNR ↑ (dB) | LPIPS ↓ | FPS ↑ | Mean peak allocation (MiB) | Maximum peak allocation (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| nuscenes | 500 | 313 | 14.15 | 0.306 | 693.0 | 74.5 | 199.8 |
| waymo | 500 | 0 | N/A | N/A | 747.6 | 68.4 | 152.0 |
| all | 1000 | 313 | 14.15 | 0.306 | 719.3 | 71.5 | 199.8 |

PSNR/LPIPS use the original segmented source and a native-pose matched render, independently centered and isotropically normalized with 5% padding on white. The quality columns cover 313 nuScenes identities only; Waymo lacks the verified pose linkage needed for this protocol. No best-view search is used.

FPS uses 32 warmup frames followed by 128 measured frames per asset, cycling over 16 orbit views. Aggregate FPS is total frames / total CUDA event time. Loading, generation, image saving and perceptual networks are excluded. Memory reports PyTorch peak allocation for the resident single asset plus rasterization buffers, not total device usage or process context; maximum reserved memory was 232 MiB.

These are source-conditioned appearance diagnostics and standalone rasterization measurements, not heldout scene-reconstruction quality or full-scene insertion throughput. They cannot be ranked against external 3DGS/LightGaussian/DecoupleGS numbers without rerunning all methods under the same protocol.

Script: [`benchmarks/vehicles.py`](../benchmarks/vehicles.py). Full per-asset records are generated in each run, rather than shipping dataset imagery in Git.
