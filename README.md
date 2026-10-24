# DriveLab rigid asset pipeline

保留“检测 → 关联去重 → 分割 → 属性分析 → FLUX”的主流程，车辆在 FLUX 后进入 H，其他刚体进入 B0，最后统一标准化、检查和导出。**小规模生成、质量检查和旧资产库加载已经实测；整场景多图覆盖及当前版本插入效果仍在验收。**

## 当前证据与运行环境

- 最近一次完整回归为 **55 项通过**，覆盖轻量关联、中间帧补图、参考图、接口、异步 JSON、资产导出及原资产库加载/筛选/绑定。
- 固定的四个较清晰对象使用自动简洁提示词，完成 FLUX→H/B0→标准化→16 视角渲染：**4/4 可导出且各视角非空；两辆车和一个锥桶通过自动质量准入，交通灯因颜色等错误被拒绝**。原加载器接纳 3 件。此为小样本结果，不是整场景成功率。更早的两个四对象实验均失败；保留原结果，没有重选或覆盖失败样本。
- nuScenes scene-0103 的 240 张六相机关键帧及 1,144 张中间图已完成前端测试。旧版 257 条交接候选发现车辆串轨，不能用于验收。修复关联后的 300 条属性有效候选，经尺寸过滤、同帧及跨帧去重剩 **265 条：H 152、B0 113**。跨帧去重新增合并 8 个任务，10 组已知串轨反例均未误合并；本次去重约 12 秒。候选数量不是独立对象覆盖率。
- 上述 H 的 1/2/3/4 图分布为 **130/8/4/10**，尚未达到大多数 3–4 图。新增中间帧补图在预先固定的 12 个完整单图候选上，获得 1/2/3/4 图分布 **7/2/0/3**；新增 15 次 SAM，未重新检测，含模型加载约 14 秒。目视复核补图身份一致，但它们是邻近时刻观测，不能等同于独立相机或大视差。
- 三件新准入资产已完成旧加载器绑定和原 SplatAD 插入：每件 6 帧 × 6 相机，三份视频均成功解码，FRONT 6/6 可见且没有边缘截断，渲染与绑定尺度的相对差异均低于 0.004%。两辆车沿原刹车轨迹，锥桶固定放置；验证位于 `auto_concise_v1/admitted_insertion_v2/verification.json`。这些是 0.5 秒的工程检查，不能代替整段轨迹验证。目视仍有前景偏锐、缺少接触阴影的问题，未宣称照片级融合。旧拒绝资产的 `engineering_only` 诊断与本轮准入验证分别保存。
- 工作证据：`outputs/runs/nusc0103_complete_sift/dedup_audit.json`、`outputs/validation/intermediate_real_v1/report.json`、`outputs/validation/source_validation_v2/auto_concise_v1/curation/curation.json`。当前完整前端复验使用新运行目录 `outputs/runs/nusc0103_current_v1`；不要将阶段验证拼接成当前版本已全部验收的结论。工作证据不进入正式资产包。
- 本机实际使用 **Qwen3-VL-8B-Instruct、8 张 NVIDIA 4090（单卡约 48 GB）**；论文中的 32B/H20 配置不是这里的实测配置。
- 本仓库不打包数据、权重和第三方模型代码。`configs/generation.example.json` 提供本机已配置的模型与运行环境路径；异地部署需修改这些路径并安装对应模型依赖。

## 流程与接口

```text
nuScenes 六相机 / 通用帧索引
  → GroundingDINO → 轻量跨帧关联与保守去重
  → SAM2 → 白底最佳图 + 车辆自适应真实 RGBA
  → Qwen 属性、类别、提示词 → 原格式 FLUX 清单
  → FLUX.2 Klein → 前景分割
      ├─ 车辆 H：FLUX 前景 + 1–4 张真实前景 → ReconViaGen
      └─ 其他 B0：FLUX 前景 → 原 TRELLIS
  → 坐标统一、接地、16 视角渲染、Qwen 朝向/质量/尺度估计
  → 内容去重的 Gaussian PLY + 两份集中清单
```

FLUX 清单每条仍只有五个字段，`prompts` 原文直接传给 FLUX，不追加语义重写：

```json
{
  "scene_id": "scene-0103",
  "obj_id": "vehicle_scene-0103_t000001",
  "category": "car",
  "image_path": "/absolute/path/to/best_white.jpg",
  "prompts": ["final visible-end prompt", "final side-view prompt"]
}
```

`image_path` 是一张最佳白底图。`handoff/asset_routes.json` 单独记录 `obj_id`、H/B0 路线、真实图、SHA256、来源和选图原因；按 Qwen 最终 `category` 确定路线，与提示词模板一致，不靠自由文本猜测路线。路由文件校验对应 FLUX 清单的哈希。

车辆真实图优先 3–4 张，只有两张时保留两张，单张标记 `single_view_limited`。仍走 H，不复制图片凑数。每辆车最多选 6 个候选进行分割，再以质量、观测差异选择最多 4 张；默认相对最佳观测质量门槛 0.70 是工程初值。边缘截断的观测不作为补充图；只有截断图时保留受限单图并记录原因。无标定视角时使用帧间距代理，不能解释为真实视角差。

关键帧有效观测不足时，在剩余 SAM 预算内，从固定最佳图的 SAM 前景跟踪附近中间帧。需要往返光流、分布足够广的特征、稳健仿射拟合及新 SAM 掩码与预测前景的交并比 ≥0.6；不额外运行检测，不更换 FLUX 最佳图。补图记录原图路径、小数帧位置、关联来源和掩码一致性；不会沿用关键帧的外参假装获得了中间帧标定。

检测默认不设每帧三个对象的截断（`max_detections_per_frame: 0`）。`max_asset_candidates: 320` 限制进入分割的候选身份数，每个入选车辆另外获得最多 6 个观测的分割预算；不会因为其他候选多而让所有车辆只剩一张真实图。预算内未入选的轨迹和实际 SAM 请求数分别统计；候选轨迹数量不等于真实独立对象数量。同相机车辆连接必须有实测光流支持；有标定时可使用保守跨相机几何证据。

属性确定后的去重先检查同帧前景支持，再检查同类别、同相机、短时间内的掩码 SIFT 特征：双向匹配、RANSAC 及分布范围均须通过；同一帧中分处不同位置的对象不能合并。只允许唯一匹配到固定代表图，不进行传递合并。别名来源保留；只有逐个验证过的不同帧最佳图才可补充车辆参考图。同帧重复掩码不计为新增真实图。低纹理、小物体仍可能重复，当前没有全场景身份真值覆盖率。主处理不要求对象真值 ID。

## 一条命令运行

先核对 `configs/nuscenes.yaml` 与 `configs/generation.example.json` 中的本机路径。另一个终端启动 Qwen：

```bash
VLM_GPU=2 bash scripts/start_vlm.sh
```

该入口使用 `--gpu-memory-utilization 0.90`、JSON 约束后端、8192 上下文和每次一张图片。0.90 是 vLLM 显存预算，不是 GPU 计算利用率。前端默认使用 GPU 4–7；GPU 分配由显式配置决定，不会中止其他用户进程。

从场景提取到正式资产库的完整入口如下。`--all-assets` **明确授权生成该场景通过过滤的全部对象**，没有隐式数量截断：

```bash
RUN_NAME=scene0103_all \
GENERATION_CONFIG=configs/generation.example.json \
GPUS=4,5,6,7 GENERATION_GPUS=4,5,6,7 RENDER_GPUS=7 \
LIBRARY_DIR=/mnt/ojc/new_rigid_library_scene0103 \
bash scripts/run_nuscenes.sh --all-assets --scene-ids scene-0103
```

也可以先完成提取并查看完整清单，再选择少量任务验证：

```bash
RUN_NAME=scene0103 bash scripts/run_nuscenes.sh --handoff-only --scene-ids scene-0103

# 使用上一步清单中真实存在的 obj_id；以下名称仅展示参数形式。
RUN_NAME=scene0103 WORK_DIR=/mnt/ojc/rigid_work/check01 \
LIBRARY_DIR=/mnt/ojc/rigid_library/check01 \
bash scripts/run_nuscenes.sh --reuse-handoff --asset-ids vehicle_scene-0103_t000001,cone_scene-0103_t000002
```

选择方式必须是 `--handoff-only`、`--all-assets`、`--asset-ids`、`--task-manifest` 四者之一。不带选择参数仅显示用法并退出，不会自动生成整场景。当前整场景关联质量尚在修复，推荐先检查交接中的真实参考图，再启动选定对象的验证。

`--task-manifest /absolute/path/selected.json` 接受原完整清单中若干条记录组成的 JSON 数组；必须保留原条目全部五字段及内容。程序校验身份、重复项及内容，并在工作目录记录完整/选中数量。重复、未知、空选择明确报错。阶段级错误会终止脚本；前端单对象 Qwen 请求或 Schema 错误记录为该对象失败，其余并发对象继续。不会改写非法枚举以强行通过校验。

`PYTHON`、`CONFIG`、`GPUS`、`RUN_NAME`、`OUTPUT_ROOT`、`WORKERS_PER_GPU`、`GENERATION_CONFIG`、`GENERATION_GPUS`、`RENDER_GPUS`、`RENDER_PYTHON`、`WORK_DIR`、`LIBRARY_DIR` 均可通过环境变量设置。相对路径以仓库根目录为基准；正式目录必须尚不存在。运行 `bash scripts/run_nuscenes.sh --help` 查看默认值。

只检查前端配置而不运行模型：

```bash
python -m asset_extraction.pipeline --config configs/nuscenes.yaml --dry-run \
  --run-name config_check --output-root /tmp/pre_flux_config_check
```

## 工作目录和正式资产库

默认 `OUTPUT_ROOT/runs/RUN_NAME` 中保存前端结果；默认 `WORK_DIR=RUN_DIR/work` 保存选定任务、FLUX 图、原始 Gaussian、过程日志和质量证据。这些内容服务调试和溯源，不复制到正式资产库。

`LIBRARY_DIR` 只包含：

```text
payloads/candidate/ply/<category>/<sha-prefix>/<sha256>.ply
manifests/global_asset_manifest.jsonl
manifests/asset_library_rigid.json
```

相同 PLY 内容只保存一份。逻辑清单保留 `uid/category/status/scale/label/canonical/provenance`，包含坐标变换、尺寸、来源、真实图选择、质量判断及渲染记录；工作证据通过路径与哈希引用，不重复打包图片或 raw PLY。正式包自带可加载的几何及插入元数据，追查原图和过程证据仍需访问原数据/工作目录。

H 原生 z-up、B0 原生 -y-up，统一到 z-up、水平居中、稳健下边界接地。车辆、路牌需要朝向判定；交通灯先识别信号面角度，再独立验证旋转后 +X 朝向。横向路障等按几何长轴对齐 X，避免旧绑定器按薄边缩放。尺度是带置信度的视觉估计，不是传感器测量。`keep` 要求 16 视角非空、类别和身份一致、质量达标，以及所需朝向和尺度置信度足够；不确定时为 `review`，明显失败为 `reject`。对称性是描述属性，不作为准入门槛。直接导出接口也拒绝缺失或不可信的准入尺寸。

原加载器只接纳 `status=keep` 的条目：

```python
library = AssetLibrary(
    library_json="/path/to/library/manifests/asset_library_rigid.json",
    pipeline_root="/path/to/library",
)
```

导出不会更改旧 `/mnt/ojc/generated_asset_library`。原加载器会对可信尺度施加自身的尺寸扰动；精确轨迹/场景插入还需在原渲染流程验证。前端运行目录保存语义配置及代码签名，代码/配置变化须选择新运行名；generation 可复用输入、代码签名和文件哈希匹配的成功阶段。策展签名包括模型调用客户端代码；未提交中间几何不能直接继续，需新工作目录。不要把这些机制描述为任意阶段均能无损恢复。

## 通用帧输入

入口同样接受 `CONFIG=/absolute/path/frames.yaml`，其中 `dataset: frames`、`dataset_root` 指向场景索引目录。其余检测、分割、模型、Qwen 配置保持一致；单相机视频可设置 `camera_channels: [front]`。每个 `<scene_id>.json`：

```json
{
  "scene_id": "clip01",
  "frames": [
    {"path": "/absolute/path/frame000.jpg", "camera_name": "front", "frame_id": 0, "timestamp_us": 0},
    {"path": "/absolute/path/frame001.jpg", "camera_name": "front", "frame_id": 1, "timestamp_us": 100000}
  ]
}
```

`path/camera_name/frame_id` 必需，`timestamp_us`、3×3 `camera_intrinsic`、4×4 `camera_to_world` 可选；同场景相机/帧号组合必须唯一。相对图像路径相对于索引文件。可选 `motion_paths` 是前一关键帧之后、当前关键帧之前的中间图路径数组，须按时间排序；用于轻量光流及预算内真实补图，不额外执行检测。nuScenes 读取器从同相机 sweeps 构造这一字段。当前提供帧索引接口，**没有内建视频下载或解码器**；网络视频和其他数据集需先转成这一格式，不能据此声称已经完成所有数据源适配。

## 模块和验证

- `pipeline/config/vision/real_views/vlm/io_utils.py`：输入、检测关联、选图、异步属性分析和严格接口。
- `nuscenes/quality_filter.py`、`prepare_handoff.py`、`prepare_routes.py`：来源质量检查与兼容交接。
- `generation.py`：异步 GPU 槽位调度，原 FLUX、H、B0 模型工作进程。
- `export.py`：坐标统一、16 视角渲染、异步质量策展和精简导出。

模型请求统一为 OpenAI 库的 `await client.chat.completions.create(...)`。已知输出字段时，使用 `response_format={"type":"json_schema","json_schema":{"name":"asset_response","strict":true,"schema":...}}` 在服务端强制 JSON 结构和枚举，再执行本地 JSON Schema 校验；不带 Schema 的通用调用使用 `json_object`。仅限制 JSON 语法不能约束枚举：实测原 16 条枚举错误源图在严格 Schema 下 16/16 通过格式检查，仍需独立判断语义与质量。FLUX 的五字段交接接口不变。

本机约束解码器对数值 `exclusiveMinimum: 0` 会把亚米小数错误截为 0；最小请求已复现。尺寸字段改用 `minimum: 0.001`，仍要求严格正值且本地校验，未经改写地保存完整模型响应。对照证据为 `outputs/validation/strict_decimal_probe.json`。

前端将 SAM 前景和带目标框的原图局部上下文在内存中组成一张图，用于排除把底座、轮子等部件当成完整对象的错误；FLUX 仍只接收清单中的原白底前景。后端将源图与 16 视角组成一张有标签的工作拼图，满足单图请求上限。`asyncio.gather`、信号量和 GPU 槽位队列管理并发；数据遍历与时间关联可以使用普通循环，不是逐条等待网络请求。

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/ojc/miniconda3/envs/vllm_vlm/bin/python -m unittest discover -s tests -v
bash -n scripts/run_nuscenes.sh
bash scripts/run_nuscenes.sh --help
```

已有十辆车真实图数量实验位于本机 `/mnt/ojc/workplace1/reconviagen_demo/h_count_ablation/REPORT.html`。H+1/2/3/4 real 的 FID 为 91.968 / 88.796 / 87.405 / 87.248；三图和四图整体接近，支持优先 3–4 张的已选策略。该实验是固定对象的小样本离线比较，既不是新 nuScenes 前端覆盖率结果，也不能代替场景插入验证。
