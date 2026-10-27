# Integration validation

This document distinguishes integration checks from new reconstruction experiments.

- Rigid application source starts from `pre_flux_pipeline` commit `e13dcda`, including the vehicle identity-variation admission rule.
- Human source starts from `human_cvpr` commit `ebbd2c2`; its current 4DAnyone/Grid4D training schedule is retained.
- The existing rigid CPU suite passed 73 tests before the new source/lease checks. Five new tests exercise real local-video decoding/timestamps, unlimited candidate scheduling and cross-controller GPU exclusion, plus idempotent/tamper-evident publication and empty-scene completion.
- The human CPU suite passed 62 tests, covering source timing, admission, catalogue merging, motion/placement/playback, temporal screening and pause behavior.
- A six-native-frame Waymo TFRecord fixture passed CRC validation, RGB rectification and frame-index export. Local video timestamps and a controlled local HTTP download passed. Native nuScenes scene indexing was also exercised. These adapter checks do not certify arbitrary video-site availability.
- Human runtime resource/import inspection passed using explicitly configured local resources. This is not a new 52,999-step human training run.
- A real three-frame nuScenes source passed the new scene entry point and produced 16 filtered FLUX handoff rows. One H truck (two accepted real references) and one B0 traffic-light candidate were selected to test both reconstruction branches; both FLUX and Gaussian generation completed using the repository-local model code.
- Portable canonical transforms, 16-view rendering, quality curation and categorized package publication completed: the H truck was kept and published; the B0 traffic light was generated but rejected by the quality audit. GPU evidence and quality decisions from integration samples live in the ignored validation run directory; generated assets and original images are not committed.

The original dataset jobs continue from their original checkouts. This publication does not restart them. A new machine still requires its own model resources, compiled CUDA dependencies, deployment checks and per-input visual acceptance.

The current standalone vehicle evaluation script is `benchmarks/vehicles.py`. It measures matched-source appearance and batch-one Gaussian rasterization at 512×512. These diagnostics do not reproduce unknown settings from external 3DGS/compression papers. Detailed results/protocols are written with each run.
