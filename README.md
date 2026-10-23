# nuScenes pre-FLUX rigid asset pipeline

当前交付范围：pre-FLUX 前端及 H/B0 路由交接。2026-10-23 的自适应真实图扩展已通过 CPU 选图、导出和接口回归测试；修改后的 nuScenes 全链路模型运行、FLUX 后自动重建及场景插入验证尚未完成。服务器路径需按部署环境调整。

这是可下载、可独立检查的 **FLUX 之前**流程。它从原始 nuScenes 六相机图像开始，最终停在 `flux_input.json`；包内不会启动 FLUX，也不包含 TRELLIS、PLY、人体、Waymo、pair 构造或 dataset 发布代码。

## 数据流

```text
nuScenes v1.0-trainval 元数据 + 六相机图片
  -> 场景/相机索引（任一相机均可产生资产候选）
  -> GroundingDINO 检测
  -> 跨帧 tracklet 与候选排序
  -> SAM2 mask
  -> mask 连通域、碎裂度、面积等 QC
  -> 白底参考图 + 完整 source/frame/bbox 溯源
  -> Qwen3-VL 类别、颜色、车辆前/后朝向判定
  -> 绑定原资产身份与朝向的两个提示词
  -> 宽松尺寸/清晰度过滤与场景内近重复去重
  -> FLUX handoff JSON
```

车辆朝向无法判断、类别为 unknown、参考图不存在、提示词缺少身份约束的条目不会进入 handoff。该门槛只处理明确错误和明显不可用的小图/模糊图，不用于苛刻视觉打分。

## 文件职责

- `asset_extraction/config.py`：nuScenes-only 配置与校验。
- `asset_extraction/pipeline.py`：场景索引、GPU 调度、VLM、manifest 合并。
- `asset_extraction/vision.py`：GroundingDINO、SAM2、mask QC、白底参考图和溯源。
- `asset_extraction/vlm.py`：刚性类别、车辆前后朝向、身份绑定提示词。
- `asset_extraction/io_utils.py`：原子 JSON/JSONL 与 manifest 校验。
- `nuscenes/quality_filter.py`：生产流程使用的宽松刚性图像过滤与去重。
- `nuscenes/prepare_handoff.py`：生成 FLUX 输入，严格检查 ID、图片、提示词和车辆朝向。
- `nuscenes/validate_handoff.py`：只读复验 handoff。
- `configs/nuscenes.yaml`：六相机默认配置。
- `scripts/run_nuscenes.sh`：完整的一键 pre-FLUX 入口。

## 运行

先修改 `configs/nuscenes.yaml` 中的数据集、Grounded-SAM2 和 Python 路径，并保证一个或多个 OpenAI-compatible Qwen3-VL 服务在线。

```bash
cd pre_flux_pipeline
GPUS=4,5,6,7 \
RUN_NAME=rigid_nusc_train \
bash scripts/run_nuscenes.sh --scene-ids scene-0001,scene-0002
```

只检查配置，不运行模型：

```bash
python -m asset_extraction.pipeline --config configs/nuscenes.yaml --dry-run \
  --run-name config_smoke --output-root /tmp/pre_flux_smoke
```

只处理一个已经生成的 merged manifest：

```bash
python -m nuscenes.quality_filter /path/to/manifest.json \
  --output /path/to/manifest.filtered.json \
  --reject-log /path/to/rejects.jsonl
python -m nuscenes.prepare_handoff /path/to/manifest.filtered.json \
  --output /path/to/flux_input.json
python -m nuscenes.validate_handoff /path/to/flux_input.json
```

最终交给外部 FLUX 程序的唯一必需文件是：

```text
outputs/runs/<RUN_NAME>/handoff/flux_input.json
```

每行合同为 `scene_id / obj_id / category / image_path / prompts`。外部生产 `flux.py` 可直接读取；FLUX 状态、输出图片及后续 TRELLIS 不属于本包。

## H 真实多图适配（2026-10-23）

FLUX 仍接收一张最佳白底 JPG 和原字段清单；本次没有改变 `flux_input.json`。
车辆额外保存同一 tracklet 的 1–4 张真实 RGBA 前景，目标是 3–4 张；两张可用时保留两张，仅一张时标记 `single_view_fallback`，不重复图片凑数。
这里的 H 路线指真实图与 FLUX 图共同送往 ReconViaGen；非车辆路由 B0，继续使用原 TRELLIS。

车辆先最多选择 6 个候选进行 SAM 分割，再通过已有 mask QC 和相对于最佳观测 0.70 的质量门槛选择最多 4 张。该相对阈值是可配置的工程初值，不是由十辆车指标拟合出的最优阈值。
没有标定视角时用帧间距作为观测差异的轻量代理，不声称它是真实视角变化。候选预算按轮次分配：先让不同对象各获得一张，再分配追加观测，避免多图车辆挤占其他对象的首张机会。
默认场景 SAM 预算从 80 增到 320；若对象数量太多或 mask 未过检，仍可能只有 1–2 张，实际计数与原因写入清单。

新增交接文件 `outputs/runs/<RUN_NAME>/handoff/asset_routes.json`：

- `assets[].obj_id` 对应 FLUX 原清单的对象，`route` 为 H 或 B0，按 Qwen 最终 `category` 确定。
- `real_images` 提供 RGBA 路径、SHA256、来源图片、帧号、相机、bbox 和裁剪范围。
- `real_view_selection` 记录有效图数、降级原因和选图规则。
- `flux_output_keys` 为外部 FLUX 的预期输出文件名，如 `<obj_id>_0.png`；不是已经生成的图片。

该文件只包含通过最终过滤并进入 FLUX 清单的对象。外部消费者按 `obj_id` 联结两份文件，等待 FLUX 成功后，将车辆的真实 RGBA 加上 FLUX 前景送入 H；非车辆送入 B0。该目录本身仍不启动三维重建。
旧的单图缓存没有真实多图记录时，不会被当作已适配的成功任务复用。

CPU 回归检查：`python -m unittest discover -s tests -v`。检查包含 1–6 个候选、低质量与重复观测、场景预算公平分配、真实图导出、哈希校验、H/B0 分流以及 FLUX JSON 字段与字节不被路由阶段修改。

## 边界与可移植性

- 不打包模型权重或 nuScenes 数据。
- 不打包任何输出资产、中间缓存或测试目录。
- 场景选择可通过 `--scene-ids` 精确指定；六个相机由 YAML 中 `camera_channels` 控制。
- 代码保留已验证生产逻辑而删除无关分支，因此 Python 文件数量比早期估计的 25–30 个更少；减少的是重复封装，不是功能。
