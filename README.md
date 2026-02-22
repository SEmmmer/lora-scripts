<div align="center">

<img src="https://github.com/SEmmmer/lora-scripts/assets/36563862/3b177f4a-d92a-4da4-85c8-a0d163061a40" width="200" height="200" alt="SD-Trainer" style="border-radius: 25px">

# SD-Trainer

_✨ Enjoy Stable Diffusion Train！ ✨_

</div>

<p align="center">
  <a href="https://github.com/SEmmmer/lora-scripts" style="margin: 2px;">
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/SEmmmer/lora-scripts">
  </a>
  <a href="https://github.com/SEmmmer/lora-scripts" style="margin: 2px;">
    <img alt="GitHub forks" src="https://img.shields.io/github/forks/SEmmmer/lora-scripts">
  </a>
  <a href="https://raw.githubusercontent.com/SEmmmer/lora-scripts/main/LICENSE" style="margin: 2px;">
    <img src="https://img.shields.io/github/license/SEmmmer/lora-scripts" alt="license">
  </a>
  <a href="https://github.com/SEmmmer/lora-scripts/releases" style="margin: 2px;">
    <img src="https://img.shields.io/github/v/release/SEmmmer/lora-scripts?color=blueviolet&include_prereleases" alt="release">
  </a>
</p>

<p align="center">
  <a href="https://github.com/SEmmmer/lora-scripts/releases">Download</a>
  ·
  <a href="https://github.com/SEmmmer/lora-scripts/blob/main/README.md">Documents</a>
</p>

LoRA-scripts (a.k.a SD-Trainer)

LoRA & Dreambooth training GUI & scripts preset & one-key training environment for [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts.git).

## ✨NEW: Train WebUI

The **REAL** Stable Diffusion Training Studio. Everything in one WebUI.

Follow the installation guide below to install the GUI, then run `run_gui.ps1` (Windows) or `run_gui.sh` (Linux).

![image](https://github.com/SEmmmer/lora-scripts/assets/36563862/d3fcf5ad-fb8f-4e1d-81f9-c903376c19c6)

| Tensorboard | WD 1.4 Tagger | Tag Editor |
| ------------ | ------------ | ------------ |
| ![image](https://github.com/SEmmmer/lora-scripts/assets/36563862/b2ac5c36-3edf-43a6-9719-cb00b757fc76) | ![image](https://github.com/SEmmmer/lora-scripts/assets/36563862/9504fad1-7d77-46a7-a68f-91fbbdbc7407) | ![image](https://github.com/SEmmmer/lora-scripts/assets/36563862/4597917b-caa8-4e90-b950-8b01738996f2) |

## Usage

### Required Dependencies

- Git
- NVIDIA driver and CUDA-capable GPU
- Internet access for dependency download
- `iperf3` (optional but recommended for mesh bandwidth checks in cluster compatibility test)

> Python installation is not required manually. Installer uses embedded Python 3.10 and creates `venv`.

### Clone repo with submodules

```sh
git clone --recurse-submodules https://github.com/SEmmmer/lora-scripts
```

## ✨ SD-Trainer GUI

### Windows

#### Installation

Run `install.ps1` to install embedded Python + create `venv` + install dependencies.

If you are in mainland China, use `install-cn.ps1`.

#### Start GUI

```powershell
.\run_gui.ps1
```

Then open `http://127.0.0.1:28000`.

### Linux

#### Installation

```bash
bash install.bash
```

#### Start GUI

```bash
bash run_gui.sh
```

Then open `http://127.0.0.1:28000`.

## Legacy training with scripts

### Windows

- Edit `train.ps1`, then run `./train.ps1`
- Edit `train_by_toml.ps1`, then run `./train_by_toml.ps1`

### Linux

- Edit `train.sh`, then run `bash train.sh`
- Edit `train_by_toml.sh`, then run `bash train_by_toml.sh`

`train*.sh` / `train*.ps1` use the project `venv` Python directly. Manual activation is not required.

## Cluster compatibility check (single + multi-node + mesh iperf3)

Unified checker script:

- `cluster_compat_check.py` (core)
- `cluster_compat_check.sh` (Linux launcher, runs in project `venv`)
- `cluster_compat_check.ps1` (Windows launcher, runs in project `venv`)

### Interactive full flow (recommended)

Linux:

```bash
bash cluster_compat_check.sh
```

Windows:

```powershell
.\cluster_compat_check.ps1
```

Flow:

1. Run environment check (Python/driver/torch/NCCL availability/network brief). No `nvcc` check.
2. Run single-node NCCL compatibility check.
3. Ask whether to continue multi-node compatibility check.
4. If `host` role is selected, input cluster size and test parameters; host starts waiting for workers.
5. Workers input host IP/hostname/domain and connect, then enter waiting state.
6. Host confirms and types `start`; NCCL distributed test starts.
7. Output NCCL compatibility result table.
8. Run `iperf3` pairwise mesh tests and output a bandwidth table.

### Manual mode examples

Env check only:

```bash
bash cluster_compat_check.sh --mode check-env
```

Single-node NCCL only:

```bash
bash cluster_compat_check.sh --mode single
```

Host mode:

```bash
bash cluster_compat_check.sh --mode host --cluster-size 2 --master-addr 192.168.50.219 --master-port 29500 --control-port 29610
```

Worker mode:

```bash
bash cluster_compat_check.sh --mode worker --host 192.168.50.219 --control-port 29610
```

All compatibility checks are now unified in `cluster_compat_check.py`.

## TensorBoard

Windows helper script:

```powershell
.\tensorboard.ps1
```

Starts TensorBoard at `http://127.0.0.1:6006` by default.

## Program arguments

| Parameter Name                | Type  | Default Value | Description                                      |
|-------------------------------|-------|---------------|--------------------------------------------------|
| `--host`                      | str   | "0.0.0.0"     | Hostname for the server                          |
| `--port`                      | int   | 28000         | Port to run the server                           |
| `--listen`                    | bool  | false         | Enable listening mode for the server             |
| `--skip-prepare-environment`  | bool  | false         | Skip the environment preparation step            |
| `--disable-tensorboard`       | bool  | false         | Disable TensorBoard                              |
| `--disable-tageditor`         | bool  | false         | Disable tag editor                               |
| `--tensorboard-host`          | str   | "0.0.0.0"     | Host to run TensorBoard                          |
| `--tensorboard-port`          | int   | 6006          | Port to run TensorBoard                          |
| `--localization`              | str   |               | Localization settings for the interface          |
| `--dev`                       | bool  | false         | Developer mode to disable some checks            |
