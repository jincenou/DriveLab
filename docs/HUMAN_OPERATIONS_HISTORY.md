# human：人体多视角生成与动态高斯资产

本目录是后续人体资产生成的唯一业务代码入口。固定流程为：

**已筛选的人体输入、骨架与相机 → 必要的携带物掩码修复 → 冻结的 4DAnyone Turbo → 多视角软分割与初始化 → Grid4D 动态重建 → 动态资产导出 → 独立回放。**

Grid4D 保持 **32,999 次基础优化 + 15,000 次轮廓监督 + 5,000 次低学习率细化**，共 52,999 次实际参数更新。不加入 Query，不因代码精简减少训练预算、分辨率、帧数或 SH 阶数。

## 入口与分工

统一执行 `run.py`，不调用旧实验目录中的 Python 脚本。11 个应用文件按职责维护；上游模型放在 `vendor/`，保留原结构和许可，不为了凑文件数合并网络实现。

|文件|职责|
|---|---|
|run.py|命令入口，输入修复、批次、历史回归入口|
|batch.py|已有候选排名、async 队列、完成状态、资产包汇总|
|prepare.py|校验恢复原始裁剪和插帧，携带物分割、像素合成、条件重新绑定|
|frontend.py|登记输入契约，冻结 Turbo 24 视角生成，完整解码与相机验证|
|dataset.py|四个独立视角组软分割，训练视角 visual hull，相机和点云归一化|
|data.py|Grid4D 相机、RGBA 读取，隔离评价视角|
|reconstruction.py|固定三阶段优化和恢复点；内部历史阶段名仅用于旧文件兼容|
|evaluation.py|固定生成参考指标与历史回归|
|assets.py|NPZ、完整 SH、相机导出、独立回放与统一五列展示视频|
|placement.py|正式坐标/估计尺度、高斯绑定根运动、动作与接触诊断、插入验证和历史资产补齐|
|common.py|资源、环境、设备、哈希、JSON、子进程与依赖审计|

`vendor/ftgs_io` 只保留历史数据解析与归一化，后端重建使用 Grid4D。RIFE、SAM2、GVHMR、4DAnyone、Grid4D 所需源码均在本目录；PyTorch、编译后的光栅器等依赖来自已有隔离环境。权重、原始数据、已筛选输入包可在外部只读引用。

## 当前输入边界

本轮从**已有筛选完成的 G0 输入包**开始，不重新筛选整个 Waymo。输入包提供视频、GVHMR 动作、24 路骨架、相机、原始帧映射与 root motion。`batch init` 读取其 CSV 登记候选，分数只用于排序，不等于质量合格或跨场景自然人去重。

从任意原始 Waymo 场景重新检测跟踪、拟合骨架的一键入口尚未迁入。已有包的恢复、携带物修复、生成和全部后端衔接在本目录完成。携带物修复只恢复真实观测像素，保留原运动张量与相机；没有自动承诺遮挡补全或任意动作重定向。

## 运行目录与环境

```bash
cd /mnt/ojc/workplace1/human
export HUMAN_RUN_ROOT=/mnt/ojc/workplace1/asset_work/human300_20261024
export HUMAN_ASSET_ROOT=/mnt/ojc/workplace1/asset_library/human
/mnt/ojc/miniconda3/envs/4danyone/bin/python run.py doctor
/mnt/ojc/miniconda3/envs/4danyone/bin/python run.py batch status
```

代码在 `human`，工作文件在 `HUMAN_RUN_ROOT`，成品在 `HUMAN_ASSET_ROOT`。不设置环境变量时保留旧回归目录约定。资源位置见 `config/resources.json`；其他模型引用位于 `vendor/4danyone/models`、`vendor/4danyone/third_party/GVHMR/inputs/checkpoints`、`vendor/metric_weights`。克隆后须恢复这些本地权重引用；Git 不提供权重或数据。

子进程按阶段使用已有 `4danyone`、`4danyone-sam2`、`4danyone-ftgs`、`grid4d` 环境，不安装或升级原环境。后两者部分依赖继承已有环境；`doctor` 列出实际包位置。CUDA 编译使用已有 toolkit 11.8/gcc9，hash 源码从本目录编译到批次缓存。子进程清除继承的 PYTHONPATH/PYTHONHOME，禁用用户 site-packages，避免意外导入其他实验代码。

## nuScenes 输入适配（2026-10-26）

nuScenes 批次使用 `HUMAN_RUN_ROOT=/mnt/ojc/workplace1/asset_work/human_nuscenes100_20261026`、资产前缀 `HN`。Waymo 原有 `H` 编号、候选登记和训练配方保留。共享资产目录的索引按数据集 UID 合并，并使用目录级锁，不能由新批次覆盖旧 Waymo 条目。

`intake.py` 负责原始数据读取、候选排序、裁剪/插帧/分割、输入检查与补选；`intake_motion.py` 负责 GVHMR 与原 G0 的 root-local 骨架条件。代码都在本仓库，权重保留原位置。之后两种数据集共用 `frontend.py → data.py → reconstruction.py → assets.py / batch.py`，不为 nuScenes 修改重建损失或训练次数。

原始 `v1.0-trainval` 的 camera sample_data、ego_pose、calibrated_sensor 提供真实曝光时间、K 与相机姿态。人体标注约 2 Hz；中间时刻的身份框由 3D 框插值得到，不冒充额外真值。真实相机图像约 12 Hz，RIFE 按实际时间插值为 61 帧/25 fps，覆盖 2.4 秒，RGB、裁剪 K 与旋转使用相同时间对应。GVHMR 接收 K 和相机角速度，但不消费相机平移，其人体世界轨迹仍是估计。不能将此写成真实世界米制运动恢复。

入口（必须使用正确工作目录环境变量）：

```bash
python run.py intake rank --limit 100          # 仅用于全新批次，冻结候选
python run.py intake rank-reserve --limit 150  # 排除已选身份，冻结备用池
python run.py intake prepare --cases HN0004 HN0019 --gpus 0 1
python run.py intake checks                    # 实际白底输入的多人/alpha检查
# review/fill 使用已有含 OpenAI SDK 的 vllm_vlm Python；模型子进程仍进入各自环境
python run.py intake review                    # 本地 VLM、JSON 强制结构，非质量认证
python run.py batch pilot --cases HN0004 HN0019 --gpus 0 1
python run.py intake fill --gpus 0 2 3 4 7     # 从备用池分波补选，最多目标名额
python run.py batch run --all --gpus 0 1 2 3 4 5 6 7
```

新批次仍需冻结 `config/recipe.json`、`config/export.json`、`TEMPORAL_POLICY.json`，以及有证据的 `CODE_VALIDATION.json`；不能把复制文件当作新数据质量通过。上述命令是阶段入口，不应在已有任务运行时重复执行。批量启动还要求两例完整资产独立回放及实际视觉复核的 `BATCH_APPROVAL.json`。`fill` 只登记通过输入筛查的候选，不签署这份完整样例验收。

输入自动检查联合首/中/末携带物对照、七个时刻多人检测、alpha 透明度及既有时序骨架规则。阈值是工程筛查，不是真值或最终质量保证；轻微遮挡不单独排除。已验证单靠 VLM 会漏报旁人混入，故它不能独立决定准入。失败保留原因，备用池补位，不为了凑数放宽规则。当前候选数、允许生成数、完整导出数、视觉通过数分别统计。

本轮 HN0004、HN0019 已实际完成全部训练、导出、独立回放和六个时刻的五列视频抽查，背包/肩包保留；脚趾、手脸、包体背面形状仍有模糊或推断，不声明完整质量认证。用户曾在首例完成后明确要求立即扩展，该次放行记录使用 `explicit_user_expansion`，保留第二例未完成状态；第二例完成后已补齐两例证据。默认仍要求两例完成，不能把用户扩展授权写成尚未完成的样例已经通过。

队列支持多个控制器共享对象锁，已有对象被另一控制器处理时等待，不覆盖其运行状态。61帧生成实测峰值 reserved 为17.85GB，显存准入为24000MiB；121帧保持原38000MiB门槛。`INPUT_GPU_RESERVATION.json` 可在补选期间留卡给输入准备，名单结束后自动释放。`intake maintain --wait-pids <当前fill控制器PID> --gpus 4 7` 在当前补选结束后接管，必要时只再扩展一批150个不同候选；仍不足则明确记录差额，不把不通过的输入补进名额，也不让生产队列永久等待。一次视觉服务错误只保留该对象待查，其他证据完整者继续。

用户授权扩大到200–300条时，使用 SDK 环境执行 `python run.py intake expand --target 300 --limit 800 --gpus 4 7 --wait-pids <现有补选PID>`。先按相同规则排序尚未尝试的有限候选池，排除全部已有/备用身份，沿用像素尺寸、可见性、时长、分割、骨架和附着物检查；新扩展还将每场景5人的上限累计到先前候选池。候选数不等于允许生成数。`EXPANSION.json` 固定实际规则函数、提示和配置哈希；现有补选结束后才合并备用池、更新生产上限，正在训练的对象继续完成。候选耗尽时按实际通过数停止筛选，记录差额，不降门槛、不重复挑选失败身份。300是上限，不保证能凑满。

按用户授权，扩展队列会自动清理未排入生产且明确筛选失败的输入缓存，保留来源、失败原因和删除清单。使用对象锁并保护所有已发布资产、已开始生成的输出和待复核对象；接口错误/不确定结论不当作质量失败清理。已清理对象写入禁止重新生成记录，后续检查不再访问被删除的媒体。手动执行同样的清理入口为 `python run.py intake cleanup`；应在当前旧补选控制器退出后使用。原始数据集、有效输入和已生成成品不会被这项清理删除。

`audit.py watch` 为新增成品做后台只读增量复核：全视频解码、交付哈希/坐标侧车检查，再用五个时刻的原观测/骨架/生成/重建对照进行 VLM 问题标记。它只扫描本轮登记的人体，`--reuse` 复用同一资产、同一提示和抽样规则的已完成证据；`--wait-pids` 绑定生产、筛选和补跑进程的实际启动时间，等待它们退出后检查最后一批成品。每批最多16条，CPU检查并发3、视觉服务并发2。API错误记为待复核，不重复无限请求，不自动通过或删除资产。完整人体资产目录不增加展示图片，审计证据留在工作目录。复核控制器的停止不代表全部生产完成，状态文件会列出尚未发布的准入ID。首次增量复核已包含对子集、旧证据复用、错误保留、不修改资产库和未完成ID的回归检查。

每个完整输出仍只有一份五列 `demo.mp4` 和必要资产/来源/插入信息。首批早期输入的 `observed_native.mp4` 采用帧索引容器；五列视频始终按真实时间映射选帧，不按该容器帧率重采样。新输入的原观测容器也保存真实曝光时间戳。所有模型输入均为真实时长的 25 fps 映射。

## 新批次与两例测试

仅在新建的工作目录初始化，已登记批次禁止覆盖：

```bash
/mnt/ojc/miniconda3/envs/4danyone/bin/python run.py batch init --limit 300 --pool /path/to/priority_candidates.csv
/mnt/ojc/miniconda3/envs/4danyone/bin/python run.py batch pilot --cases H0001 H0002 --gpus 0 2
```

候选的 `input_review.allow_generation` 默认为 false，检查实际输入及携带物后再登记依据。大批量 `batch run --all --gpus 0 1 2 3 4 5 6 7` 只消费 `PRODUCTION_INPUTS.json` 的显式名单，最多300条，不会自动运行整个候选池；还需要 `BATCH_APPROVAL.json` 记录两例视觉检查和继续依据。当前用户已明确允许保留局部质量缺陷继续批量；严重失败不计入质量合格。现有 tmux 服务为 `human300`，重连可执行 `tmux -L human300 list-sessions`。关闭对话不会终止已启动的 tmux 任务；不要主动杀死或重启运行中的会话。

补选可运行 `batch inspect --all --no-overviews`，只准备必要的逐人复核证据，不重复生成总览或覆盖生产浏览入口。复核后使用 `batch replenish --replacement-plan /path/to/reviewed_plan.json`：计划明确列出 old/new、原因、输入哈希和冻结姿态证据；新输入还必须有首中末审核记录。命令只替换已结束且没有发布包的失败任务，拒绝活动子进程、重复轨迹或变更过的输入，保留旧任务和替换历史。现有 `--all` 队列会自动领取新增项。候选库大小、累计尝试数、当前300个队列名额和最终质量合格资产数必须分别统计；不同轨迹键不保证属于不同自然人。

修复遗漏携带物时：

```bash
/mnt/ojc/miniconda3/envs/4danyone/bin/python run.py input restore H0001 --gpu 0
# 在 data/H0001/repair/accessory_plan.json 登记首帧框、正负点和依据
/mnt/ojc/miniconda3/envs/4danyone/bin/python run.py input accessory H0001 --gpu 0
# 检查 accessory_review.jpg，记录 visual_review.json 后绑定
/mnt/ojc/miniconda3/envs/4danyone/bin/python run.py input bind H0001 --gpu 0
```

`restore` 逐帧核对原始裁剪 PNG 哈希，并核对旧掩码合成后的像素；不一致明确退出。该修复必须在生成开始前完成。不能修改正在生成的对象输入。

携带物也属于输入完整性检查范围：背包、挎包、手提物不能只因人体掩码面积或骨架检查通过就被忽略。新补选至少对照首、中、末的原始观测与白底输入，检查物体漏分、变白、裁剪及人物混入；记录实际查看范围，不把抽查写成全片质量认证。明显缺失的对象先保持未批准状态，使用上述 SAM2 可见物体跟踪与人体掩码并集修复，检查后再绑定。修复不会恢复原裁剪之外或被遮挡的像素。


仅对于已经完整发布但完成记录写入失败的任务，可显式执行 `run.py recover-generation H0002 --gpu 2`：先逐帧解码全部视频，校验内外参、视角编号、尺寸、角度和输入哈希，再补完成记录。活动生成进程存在时拒绝执行，不会重跑 DiT 或覆盖视频。

队列按对象隔离失败，保留日志；已完成阶段可跳过。原生重建有完整恢复点时支持继续，部分 alpha/细化阶段仍需明确处理，不能把当前队列称为任意故障自动恢复。批量按显存和空间准入，不停止其他项目。

### 完成当前对象后暂停

工作目录的 `PRODUCTION_PAUSE.json` 设置 `paused: true` 并在 `drain_ids` 登记当时已启动的对象，允许这些对象完成所有后续阶段和导出。其他对象在模型加载前等待，不取消现有训练。新控制器直接暂停领取；早于此功能启动的控制器可能仍显示下一对象为 running，但其子进程会写入 `jobs/pause_wait_<pid>.json` 并在加载模型前等待，不代表开始了训练。补跑队列同样受暂停控制。不要为启用暂停重启 tmux 或杀死控制器。

恢复必须在约定的资产补齐与验证完成后，原子更新该文件为 `paused: false`；等待进程自动继续，不重复启动新队列。该标记只控制生产调度，不代表元数据优化、Git 发布或验收已经完成。

## 输出与插入信息

每个成品 `Hxxxx/` 保存：

- `asset_4dgs.npz`：旧接口键、完整 SH、float32 和原生光栅参数。
- `manifest.json`、`motion_metadata.json`、`cameras.json`、`native_cameras.json`：坐标、帧率、帧数及来源。
- `insertion.json`、`insertion_transforms.npz`：参考相机到 root-local/估计源相机的变换与时间戳。
- `placement.json`、`placement_motion.npz`：资产到 X前/Y左/Z上的完整相似变换、人体模型估计米制比例、高斯绑定根/朝向、动作估计、接触与循环诊断；`placement_validation.json` 做独立数组读回，`placement_replay.json` 做实际 CUDA 放置相机回放。未验收的脚部接触、步速、动作标签有明确状态，不能默认用于无滑步自动行走。
- 新包的 `motion_review_group` 区分行走验证候选、其他动作候选、待接触证据和单独运动复核；目录索引同步该字段。分组不代表外观或防滑步验收，困难对象不阻塞其他资产生产，也不因代理关联失败直接删除。`feet.pair_assignment_diagnostic` 自动检查疑似左右脚交换，只做诊断、不修改动画。旧流程借鉴结论和有限实验范围见 [FOOT_MOTION_RESEARCH.md](FOOT_MOTION_RESEARCH.md)。
- `refs/source_motion_hints.json/.npz`：从冻结的G0骨骼提取脚跟/脚尖轨迹、候选支撑帧、踝部观测可信度、身体朝向、已移除根位移与原帧映射。新任务在生成前保存，打包时按最终保留帧重新对齐；旧包可运行 `placement.py --all-complete --backfill-source-hints --concurrency 2` 补齐。它们是辅助证据，不是最终高斯的接触或循环认证，也不改变模型输入或训练损失。缺少注册根运动的旧测试入口明确记录信息不可用，不填造数值。
- `refs/`：实际输入、原观测、骨架、相机、原帧映射、root motion 和动作参数。
- `runtime/`：高斯 PLY、形变/hash 网络、固定配置和相机；不包含优化器。
- `demo.mp4`：唯一展示视频，五列为原始观测、骨骼叠加、4DAnyone生成视角00、重建视角00、重建视角08。1760×672、25fps，原始观测按已保存时间映射选最近真实帧；第三列只展示24路中的一个固定视角。
- `presentation.json`、`quality.json`、导出与独立回放检查：记录时间对应、来源哈希及验证结果。新批次不再额外生成预览拼图、关键帧图片或重复对比视频。`refs`中的输入资料和旧接口要求的参考图是资产来源/插入数据，不作为额外展示产物。

统一输出规则：每人只发布一个五列 `demo.mp4`。打包与独立回放验证完成后，自动删除同一对象遗留的 preview/comparison 展示视频、拼图和评测截图，删除清单保存在工作目录 `reports/<id>_media_retired.json`；技术回放所需的资产、输入参考及插入信息保留。训练缓存和重复权重仅在已发布 runtime 哈希核对通过后清理，不改变训练预算或质量判据。

首尾异常由 `config/temporal_policy.json` 控制；新 `batch init` 将它冻结为工作目录的 `TEMPORAL_POLICY.json`。生成前先检查已有姿态，生成后、创建RGBA之前自动决定保留全段、同步截取短首尾段、或跳过对象并继续队列。不依赖在线Codex逐个批准。只出现遮挡或正常快速运动不单独触发截取；需关键点可信度下降与骨骼SO(3)旋转突变/投影残差共同支持。长期不可测肢体记unknown，不把它当作整个人异常，也不当作质量通过。

首尾合计最多删原帧数的15%，且最多12帧；61帧最多删9帧、121帧最多删12帧。正常片段不固定删首尾帧。超限或中间异常自动跳过，不剪接中间动作；少于3个异常证据帧不自动截取。参数是本项目工程规则，检测不到所有渐变错误、可信度虚高的错误或纯RGB生成问题。`keep_full`只表示未触发本检测器，不代表质量合格。跳过记录在`temporal/<id>.json`，旧控制器也会保留对应阶段失败日志并继续下一个对象。

自动截取在独立的 `temporal_inputs/<id>/` 中完成，通过逐帧RGB无损核验后发布 `config/case_overrides/<id>.json`。24路生成和骨架、实际输入、动作、root、相机数组及时间映射同步选择同一个连续区间；新播放时间从零开始，保留原始帧索引/时间偏移和空间变换。已开始准备或训练的对象不原地改时间轴，既有资产不覆盖。每份新资产带`temporal_screen.json`；仍只输出一份五列展示视频。

独立对照命令：`frontend.py subset <id> --source-root <原工作目录> --source-identity <原id> --start-frame <含起始帧> --end-frame <不含结束帧>`。必须设置独立的`HUMAN_RUN_ROOT`和`HUMAN_ASSET_ROOT`，再用`run.py batch pilot`执行完整后端。省略`--start-frame`表示从0开始。此操作是时间区间选择，不是动作补全；不能只裁展示视频就宣称重建已修复。实际测试与限制见[首尾处理验证记录](TEMPORAL_VALIDATION.md)。

目录中 `assets/nonrigid/human/<id>.npz` 和 `drivelab_asset.json` 提供旧式索引。NPZ 独立回放一致性与外观质量分开记录，新对象没有旧终态对照时不虚构质量回归通过。

原 `/mnt/wmj/workspace4/pipeline` 使用 LHM++。当前标准字段和完整协方差表达尽量对齐；原读取器若把 SH 当 RGB 或写死 30fps，仍需消费端适配。保留原 25fps，root 水平位移已分离，不能重复去除。没有保证米制 Waymo 世界配准、地面接触、旧 SMPL-X 任意动作绑定或零改动插入。连续时间需要匹配 runtime；单独 PLY 只是姿态快照。

## 本轮迁移验证（2026-10-24）

旧 N04 的 24 视角 × 61 帧共 **1,464 张 RGBA 逐文件哈希一致**，四个分组独立进程恢复原分割执行语义；初始化点云、颜色、时刻、相机、边界和归一化也一致。证据位于当前批次 `CODE_VALIDATION.json` 与 `checks/`。

H0001 的49张原始裁剪逐帧哈希一致，蓝色背包掩码已补回，没有改运动张量或相机。H0001/H0002均完成24路121帧生成、52999次更新、NPZ及打包runtime独立回放；九个时刻三个视角的视觉抽查保留了H0002抬臂中段的明显模糊和拖影。用户明确选择继续批量、逐资产记录缺陷、严重失败不算合格。当前显式300条输入来自156个场景，151条61帧、149条121帧；输入仅做首帧外观复核并复用已有全窗口骨架筛查，不能表述为300条质量合格人物。

两例统一五列视频已完整解码验证。批量队列已启动；变更调度器时旧tmux会话退出连带结束了七个未完成的生成进程，原日志和未完成目录保留，七个对象另有`retry_after_batch`会话在主队列结束后补跑。没有把中断记录改成成功；当前运行队列不再重启。运行记录位于工作目录`reports/interrupted_generation.json`，合格数与技术完成数始终分开。

本轮曾遇到：GVHMR 拷贝源码被误识别为外层仓库 SHA；通过对照固定版本的 216 个文件建立 `VENDORED_SOURCE.json` 解决；分割检查 JSON 的 numpy bool 序列化失败已修正；相机发布阶段只因字段命名/路径结构不同而误判，现逐项严格校验物理相机值，并保留初次失败日志。没有通过放宽质量指标或重跑选优解决。

## 历史回归记录（2026-10-23，输出已按要求删除）


N04重新执行的24路61帧Turbo MP4与旧输出**逐文件SHA256相同**；三人的61/61/121帧包及特征尺寸契约通过。没有声称N06/N11也重跑了DiT。三个旧完整A资产经提取运行时回放均通过。

后端三条冷启动均实际完成52999次优化。H={0,8,16}，每视角13帧，共39观测，352×640；参考是4DAnyone生成结果，非真实Waymo未知视角真值。

|对象/协议|旧前景PSNR|新前景PSNR|变化dB|判定|
|---|---:|---:|---:|---|
|N04 冷启动|26.33037|26.35300|+0.02263|通过数值回归|
|N06 冷启动|20.42270|20.50389|+0.08119|通过数值回归|
|N11 冷启动|22.34956|22.22290|-0.12666|未通过|
|N11 固定33k起点|22.34956|22.33501|-0.01454|通过数值回归|

训练前容差：前景PSNR最多下降0.05dB、SSIM最多下降0.0002、LPIPS最多增加0.0005；没有放宽门槛。N11固定起点对照的SSIM差约-0.00007393、LPIPS差约-0.0004624，均通过。N11冷启动失败不能被该对照或平均值掩盖，**不是三人从头重训全部通过**。

同卡同seed的两个新代码800步控制，首次增密后分别1690/1688点，原函数为1689点，首步损失和采样相同；说明训练存在数值/结构分叉，不能据此证明所有终态差异都来自它。固定旧33k状态后的续训另行验证，不选best或更换seed。

N04/N06新冷启动包、N11固定起点包各39观测的无原图/无原模型新进程NPZ回放，RGB最大误差均0；61/61/121帧预览均完整解码。清理前查看了v00首/中/末帧，新旧N04大致相近；N06抬腿弱化、N11肤色污染和手脸模糊仍存在。数值回归不是人体外观全面合格。上述新输出、图片、视频和逐次报告现已按用户要求删除；历史数值仅在此留存。

试迁移视频后的数据准备时，三人的旧包归一化可逐项复得原相机/点/颜色；但N04的24视角单进程分割只复得1342/1464张，v13/v19的alpha出现差异，初始化1672点而非1796点。因此该自动衔接未被采用，未用它替换冻结后端输入，也未保留这次试接代码和输出。将来应先恢复原分组进程语义并重新验证，不能直接接入训练。


上述历史“未迁入”的分割衔接已在本轮按四组执行语义补齐，结果见前一节；历史 N11 冷启动未通过事实保留，不宣称所有对象完全等价。

仓库保存代码、配置、上游许可和本说明，不保存模型、数据、缓存或实验输出。上游固定来源和新迁入源文件清单见 `THIRD_PARTY_NOTICES.md`、`vendor/preparation_sources.json`。维护仓库：<https://github.com/jincenou/human_cvpr>。

## 插入与轨迹播放估计（2026-10-25）

用户确认按当前工程精度交付，不再以严格接地、防滑步或无缝循环作为批量门槛。每包的 `insertion.json.practical_playback` 集中记录尺度依据、X前/Y左/Z上、近似地面偏移、动作、根运动与朝向处理、步速/接触估计来源、残余局限；只新增一个约十几KB的 `placement_playback.npz`，不新增成品图片或视频。严格诊断记录保留，不因工程可用而改成已验证。

`playback.py` 从最终高斯提取根运动；对应不足时使用躯干几何中心，不能可靠去除的残余朝向继续保留在动作里。脚部候选不足时，接触/步速以源条件骨骼估计并明确标注，不冒充最终高斯接触真值。米制比例仍是人体模型尺度，不是实测个人身高。所有对象默认单次播放，未确认自然衔接的不自动循环。

使用当前代码环境：

```python
from pathlib import Path
import numpy as np
from playback import sample

package = Path('/mnt/ojc/workplace1/asset_library/human/H0107')
t = np.arange(61) / 25  # 从播放开始计的秒数，不能超过原动作有效区间
positions = np.column_stack((0.7 * t, np.zeros(len(t)), np.zeros(len(t))))
yaw = np.zeros(len(t))  # 弧度，绕标准 +Z
world_from_asset, sample_frame = sample(package, t, positions, yaw,
                                        retime_by_distance=False)
```

用 `sample_frame` 查询包内连续 runtime；若使用其 native 坐标，再右乘包内 `asset_from_native` 一次。推荐 `placement.camera_in_native_frame` 将世界相机转回原生坐标，保留完整SH与协方差。不要重复应用旧UP_FIX、重复缩放或再次移除源根位移。`retime_by_distance=True` 使用完整路径前缀与估计步速调节播放时钟；这只能近似减轻滑步，不生成新的行走动作，超出单段动作会明确报错，不静默循环/停帧。

新资产在 `batch.package` 自动补齐；旧资产可执行 `playback.py --all-complete`，使用有界异步子进程。已完成108包补齐及数值读回；H0001/H0107/H0066/H0144各三时刻的真实runtime坐标回放RGB MAE=0，验证接口变换，不代表足底贴地或外观完全正确。试验进程、逐包核验保留在工作目录，不复制进每个资产文件夹。

插入排除标记：已知运动异常且暂不修复的资产在 `insertion.json`、`manifest.json`、`motion_metadata.json` 与总索引中设置 `skip_insertion=true`，原因写入 `insertion_exclusion_reason`。下游选择资产时须过滤此标记；元数据存在不等于允许插入。2026-10-25 按用户复核将 H0144 标记为跳过（明显加速，接地/滑步未解决），保留高斯、网络与视频供回看。其他未确认对象不会仅因严格接地指标未通过而被批量排除；未标记也不表示质量认证。

队列历史状态核对：设置本批次的 `HUMAN_RUN_ROOT` / `HUMAN_ASSET_ROOT` 后运行 `python run.py batch reconcile`。只处理旧失败记录，取得对应任务锁并验证当前完整包、资产/源输入/runtime 哈希及放置契约后，才恢复为技术完成待视觉验收；旧失败嵌入 `previous_failure` 保留。活跃任务和不完整包不改状态，不补跑训练、不授予质量通过。2026-10-25 首次核对59条旧失败，其中10条完整包通过核验，49条仍未完成并保留原状；汇总在批次 `reports/job_reconciliation.json`。

已确认问题的完整资产可执行 `python run.py batch exclude Hxxxx --reason '实际观察到的问题与检查范围'`。命令取得任务锁、核对完整包后，更新资产入口与总索引的跳过标记，保留高斯、runtime和原视频；`batch gallery --all` 更新展示页中的原因。H0633 已因开头两个重建视角存在明显灰白半透明条块而标记跳过；后续帧减轻，不将其解释为全段持续失败，也未为此重训或裁剪。

批量进度可执行 `python run.py batch progress`，或在独立 tmux 中运行 `python run.py batch progress --watch --interval 60`。先设置本批次的 `HUMAN_RUN_ROOT` 和 `HUMAN_ASSET_ROOT`；结果只写工作目录 `PROGRESS_LATEST.json`。它结合当前完成标记、已有回放/五列视频验收记录、元数据哈希及可见进程统计，区分待开始、真实活跃进程、失败和仅有历史运行状态的对象，不修改任务状态或资产。它不会重新计算所有大权重/高斯文件哈希，也不把技术完成当作外观、接地或循环验收。监视器在连续两次未发现该批次进程时退出；退出表示监视结束，不能据此宣称全部资产完成。

## 已发布资产只读复核

`audit.py` 将复核证据写到独立工作目录，不修改资产库。设置本批次的 `HUMAN_RUN_ROOT` / `HUMAN_ASSET_ROOT` 后，先用 `audit.py freeze --assets "$HUMAN_ASSET_ROOT" --output <新的复核目录>` 固定资产清单、完成标记和视频哈希，再运行 `audit.py run --output <同一复核目录>`。已存在的协议不覆盖；执行时检查审计代码哈希，避免中途改变规则。

技术检查重新校验完整包和动态资产/runtime 哈希，检查坐标与侧车契约及已有回放记录，并完整解码五列视频。它不重新执行全部 GPU 渲染，也不授予质量合格。每个对象只保存一张五时刻抽样图；不复制成品视频。技术子进程并发上限 3，视觉请求并发上限 2。

当前视觉初筛复用本机已有 `http://127.0.0.1:8082/v1` 的 `Qwen/Qwen3-VL-8B-Instruct`，由独立 `vllm_vlm` 环境的 OpenAI SDK 发起异步 `client.chat.completions.create` 请求，并使用 `response_format={"type":"json_object"}`、严格字段校验。脚本不启动或修改模型服务；端点不可用、JSON不合法或响应截断均明确记失败，不替代为质量通过。

**VLM 仅是未验证的问题线索。** 2026-10-26 首次复核已发现 H0002 抬臂模糊漏检、H0026 肩包消失误报及 H0236 整腿消失误报。无问题标签不代表合格；抽帧也不能认证全时序抖动、脚部接地、防滑步或循环。问题对象应直接查看画面，并在需要时比较同视角生成与重建，区分前端已有问题和后端新增损伤。生成参考的重建指标不是实拍新视角真值。

### 一次针对性修复

已确认短头尾异常可用 `frontend.py subset` 同步截取全部24路视频、骨架、根运动和时间映射，然后按原固定预算重新重建；不能仅剪成品 demo。裁切逐帧验证 RGB 无损。原生编解码在有界异步子进程中执行，避免两个 Python 工作线程的 PyAV 上下文销毁互相等待。源视频不修改。

携带物明确漏分割时，复用 `run.py input restore/accessory/bind` 恢复原像素并合并物体 mask，必须直接检查修正输入才绑定；随后重新生成和重建。绑定使用实际片长验证运动信息，支持61帧而不强制121帧，骨架张量必须保持原哈希。

大型交互物出画检查：批次内 `ATTACHMENT_POLICY.json` 设置 `mode: enforce` 后，输入准入及每次新 diffusion 启动前都会检查三时刻原观测/实际白底输入，拦截只剩把手、长杆等残片而车体或大物体主体被裁掉的对象；完整普通背包、肩包和手提包继续保留。旧后台控制器启动的新子进程也会执行这项检查，正在运行的训练不因此中断。检查使用既有 `vllm_vlm` 环境的 OpenAI SDK 和本地视觉服务，JSON 输出与源输入/规则哈希绑定；接口失败、歧义或旧缓存不能作为允许生成的证据。没有启用该批次策略时，既有 Waymo 路径保持原行为。

2026-10-26：nuScenes HN0026 原整帧是推购物车，输入将车体裁出画面，只留长把手，三个抽查时刻均如此；生成与重建继承并拉长了残片。按用户要求删除其成品、生成和重建输出及资产索引，仅保留原输入和问题记录。它不通过删除购物车来伪装成无交互行走，也不再重训；其他合格候选补位。新检查在 HN0026 上拒绝，在 HN0004 完整背包、HN0019 完整肩包对照上允许。这是少量实例验证，不能保证自动视觉检查不漏检。`intake maintain` 会在现有补选结束后复查新规则、移除未开始生成的被拦对象并补选。

修复使用独立 `HUMAN_RUN_ROOT`，`HUMAN_ASSET_ROOT` 必须位于该工作目录内，先保存 `REPAIR_PLAN.json`。执行 `python run.py batch repair --cases Hxxxx Hyyyy --gpus 0 1 2 3 4 5 6 7 --min-free-mib 26000`；每卡至多一个修复对象，按显存和磁盘余量接续，共享卡不终止其他任务。禁止 `--all`，不自动重复失败任务，不将候选直接发布为正式资产。修复完成后仍须直接比较保留片段的画面和独立回放；没有通过则按用户授权淘汰并同步索引，不开展反复参数扫描。
