# Asset delivery and insertion contract

## Rigid assets

Each kept/review package contains:

| File | Meaning |
|---|---|
| `asset.ply` | Canonical Gaussian payload: position, DC color, opacity logits, log scales, quaternion |
| `asset.json` | Category, H/B0 route, quality decision, coordinate/scale information and provenance |
| `placement.json` | Coordinate convention, estimated dimensions and downstream placement requirements |
| `source_camera.json` | Source observation/camera evidence; not a planned scene camera |
| `refs/` | Source crop and selected real images |
| `previews/` | Matched source/evidence sheet, orbit evidence and heading evidence |
| `package.json` | Package identity, publication signature and per-file SHA256 |

The central `manifests/asset_library_rigid.json` uses library-relative canonical PLY paths. `keep` means admitted under the recorded policy; `review` remains in a separate directory. The canonical transform is already baked into the PLY. Do not apply the original TRELLIS up-axis correction twice. Estimated metric size is separate from normalized coordinates; apply one isotropic scale and a target-scene placement transform downstream.

The package does not invent a trajectory or target background checkpoint. Those depend on the reconstructed scene into which the asset will be inserted. Historical source paths inside provenance may point to the original dataset, while operational payload/reference paths inside `asset.json` are package-relative.

## Human assets

| File / directory | Meaning |
|---|---|
| `asset_4dgs.npz` | Framewise dynamic Gaussians, full SH, float32 native parameters |
| `runtime/` | Canonical Gaussians and deformation/hash-network state for continuous replay |
| `manifest.json`, `motion_metadata.json` | Identity, timing, representation and source information |
| `cameras.json`, `native_cameras.json` | Render and native camera conventions |
| `insertion.json`, `insertion_transforms.npz` | Root-local/source-camera transforms and timestamps |
| `placement.json`, `placement_motion.npz`, `placement_playback.npz` | Estimated metric scale, standard coordinates, root/heading handling and playback support |
| `refs/` | Actual input clip, skeletons, motion, source cameras and original frame mapping |
| `quality.json`, replay/placement validation JSON | Technical and appearance evidence, kept as separate claims |
| `demo.mp4` | Five columns: observation, skeleton, generated view 00, reconstructed view 00, reconstructed view 08 |

The human path runs at 25 fps; do not force the old reader's 30 fps convention. Root horizontal motion has already been separated; do not subtract it again. Human scale and world motion are estimates from images. Ground contact, seamless looping and arbitrary SMPL-X retargeting are not guaranteed. Consumers must filter `skip_insertion` and obey the valid playback duration. See `human/playback.py` for trajectory sampling.
