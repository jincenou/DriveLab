# 上游代码

本仓库保留实际运行依赖的上游源码；各文件原有版权声明和许可继续适用，不因本项目的适配而改变。

|路径|来源与固定记录|许可位置|
|---|---|---|
|`vendor/4danyone`|https://github.com/ant-research/4DAnyone ，`9cc2aa230fe5d364da2c5634ec3dabadadcb0d60`|`vendor/4danyone/LICENSE`|
|`vendor/4danyone/third_party/GVHMR`|https://github.com/zju3dv/GVHMR ，`6ec3ca39336c50492c0fae65fba2fb831fc7d866`|该目录 `LICENSE` 及各源文件声明|
|`vendor/grid4d`|https://github.com/JiaweiXu8/Grid4D ，本地适配记录 `2e08174f0a989cb19b51a80aeedd0d155524e6a5`|各文件声明；Gaussian Splatting 派生部分见 `vendor/grid4d/LICENSE.md`|
|`vendor/metrics_loss.py`|历史评测实现；保留文件中的原有声明|文件头声明|
|`vendor/rife`|原 G0 使用的 RIFE 推理实现；本地源文件与 SHA256 见 `vendor/preparation_sources.json`|`vendor/rife/LICENSE`|
|`vendor/segmentation/sam2`|原流程使用的 SAM2 运行源码和配置，逐文件复制，未重新安装|`vendor/segmentation/LICENSE_sam2`|
|`vendor/ftgs_io`|原流程 FreeTime 数据读取与归一化；仅精简包初始化以避免加载无关模块，不执行其重建器|`vendor/ftgs_io/LICENSE`|
|`vendor/sam_transfer_guard.py`|已有 SAM2 CPU/GPU 传输同步修复，逐文件哈希保留|文件声明|
|`vendor/4danyone/fdanyone/vendor/birefnet`|已使用 checkpoint 的 BiRefNet 模型定义原样复制，显式本地导入；只在外部读取模型张量|该目录 `LICENSE`|

Grid4D 保留文件还包含 Gaussian Splatting、Deformable-3D-Gaussians 和哈希编码相关上游代码，来源说明见 `vendor/grid4d/README.md` 与文件头。这里没有为上游代码重新授予统一的宽松许可。模型权重、SMPL/SMPL-X 文件和原始数据不随本仓库分发，需按各自条件取得。
