# SD-Trainer

SD-Trainer（LoRA-scripts）是一个面向 Stable Diffusion 训练的图形化工具集，提供 LoRA / DreamBooth 训练、环境安装、分布式协同与训练诊断能力。

本项目基于 `kohya-ss/sd-scripts` 训练生态，重点提供更稳定的工程化训练流程与易用的 GUI 体验。

## 项目定位

- 日常训练：安装、启动、训练、resume
- 多机：主从机协同、状态续接、网络与兼容性检查。
- 建议使用 Ampere 架构及以上，Torch 2.10 以及 Cuda 12.8 以上版本。

## 近期功能整合（重点）

### 1. Resume 流程强化

- resume 可以稳定从 `*-state` 目录恢复模型、优化器、调度器、dataloader 与随机状态。
- 阶段分辨率训练下，resume 会基于 `plan_id + step` 判断应该从哪个阶段继续，避免串阶段或错位恢复。
- resume 后的训练进度显示更一致，降低 steps/epochs 混乱。

使用建议：

- resume 时保持核心训练参数一致（数据集、分辨率、batch、梯度累加等）。
- 仅填写 `network_weights` 属于“加载权重重新开训”，不等同于 resume。

### 2. 阶段分辨率训练（512 -> 768 -> 1024）

- 启用后以 `1024,1024` 作为基准分辨率自动拆分为三阶段训练。
- 支持 512/768/1024 占比自定义（0%~100%，总和不大于 100%）。
- 自动计算每阶段 batch、epoch、steps、保存频率与采样频率，并在 GUI 中实时预览。
- 切换阶段时会按需要重建缓存。

### 3. 梯度累加语义与等效 batch 统一

当前规则：

- 当 `gradient_accumulation_steps = 1`：三阶段都保持 1。
- 当 `gradient_accumulation_steps > 1`：三阶段都保持与 1024 基准一致。


### 4. 采样与保存频率联动优化

- 阶段分辨率下，ckpt 与 sample 频率按以下规则缩放：
  - `1024 = x`
  - `768 = ceil(1.78x)`
  - `512 = ceil(4x)`
- 频率会纳入阶段 epochs 取整逻辑，降低“该保存却没保存”的风险。

### 5. Sample 稳定性与显存清理增强

- sample 流程增加了更完整的资源回收和异常保护。
- sample OOM 时跳过当前 sample 并继续训练，而不是中断训练。
- 在阶段/轮次边界避免多余 sample 触发，减少无效显存抖动。

### 6. TensorBoard 体验优化

- 训练记录命名更可读，便于区分同模型多次训练。
- resume 会尽量接入同一条训练记录，提升曲线连续性。
- 无 ckpt 的记录会被清理。

### 7. GUI 交互增强

- 在 GUI 页面按 `Ctrl+S` 会触发“保存参数”，不再是浏览器保存网页。
- 新增“一键检测 Batch Size”按钮：按当前配置做真实短跑探测，给出推荐 batch。

## 环境与安装

### 依赖要求

- NVIDIA Ampere 及以上架构 GPU 与 CUDA 12.8 +
- 良好的网络环境
- 可选：`iperf3`（用于集群互联带宽测试）

### Python 策略

- 项目使用内置 Python 3.10（embedded）与项目内 `venv`。
- 训练脚本与 GUI 启动脚本默认都走项目环境，不依赖系统 Python。

### Windows

- 安装：运行 `install.ps1`。
- 启动 GUI：运行 `run_gui.ps1`。

### Linux

- 安装：运行 `install.bash`。
- 启动 GUI：运行 `run_gui.sh`。

默认访问地址为本机 `127.0.0.1:28000`，监听 `0.0.0.0`，同局域网设备也可访问。

## 集群兼容性与网络测试

项目已整合统一检查入口：

- `cluster_compat_check.sh`
- `cluster_compat_check.ps1`

支持内容：

- 基础环境检查
- 单机 NCCL 兼容检查
- 多机 NCCL 兼容检查（主从协同）
- `iperf3` 网格互联带宽测试与结果汇总

## 新显卡与后端建议

- 在 Torch 2.10 + Blackwell 架构场景，默认建议使用 SDPA 路径。
- 实际可用 batch 仍需结合模型、分辨率、缓存策略和系统环境实测。

## 常见建议

- resume 优先使用 `resume=*-state`。
- 阶段分辨率适合在总训练预算接近时提升细节固化质量，但建议先小规模验证。
- 多机训练前先做兼容性与网络测试，以及先完成一个小规模训练，避免正式任务中断。

## 免责声明

本项目用于模型训练流程管理与工程辅助。请在遵守相关法律法规、平台协议与数据合规要求的前提下使用。
