
import asyncio
import copy
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import toml

from mikazuki.app.models import APIResponse
from mikazuki.log import log
from mikazuki.tasks import tm
from mikazuki.launch_utils import base_dir_path


LEGACY_DEFAULT_SYNC_CONFIG_KEYS = (
    "train_batch_size,gradient_accumulation_steps,max_train_epochs,"
    "learning_rate,unet_lr,text_encoder_lr,resolution,optimizer_type,"
    "network_dim,network_alpha,save_every_n_epochs,save_model_as,mixed_precision,"
    "staged_resolution_ratio_512,staged_resolution_ratio_768,staged_resolution_ratio_1024"
)
DEFAULT_SYNC_CONFIG_KEYS = "*"
DEFAULT_SYNC_ASSET_KEYS = "pretrained_model_name_or_path,train_data_dir,reg_data_dir,vae,resume"
WORKER_REQUIRED_SYNC_CONFIG_KEYS = (
    "model_train_type",
    "v2",
    "lr_scheduler_num_cycles",
    "clip_skip",
)
WORKER_SYNC_CONFIG_FALLBACK_WHEN_MAIN_MISSING = {
    # Keep worker behavior deterministic when main toml omits optional keys.
    "v2": False,
    "lr_scheduler_num_cycles": 1,
}
WORKER_SYNC_CONFIG_CLEAR_WHEN_MAIN_MISSING = ("clip_skip",)
WORKER_REQUIRED_SYNC_ASSET_KEYS = ("resume",)
MODEL_TRAIN_TYPE_TO_TRAINER_FILE = {
    "sd-lora": "./scripts/stable/train_network.py",
    "sdxl-lora": "./scripts/stable/sdxl_train_network.py",
}
WORKER_OUTPUT_MARKER = "THIS_IS_WORKER_NODE_CHECK_MAIN_OUTPUTS"
DATASET_DIR_KEYS = ("train_data_dir", "reg_data_dir")
CKPT_EXTENSIONS = {".safetensors", ".ckpt", ".pt"}
TB_EVENT_FILE_GLOB = "events.out.tfevents.*"
STATE_REQUIRED_FILES = ("train_state.json", "optimizer.bin", "scheduler.bin")
STATE_MODEL_FILE_CANDIDATES = ("model.safetensors", "pytorch_model.bin", "model.bin")
MIXED_RESOLUTION_ENABLE_KEY = "enable_mixed_resolution_training"
MIXED_RESOLUTION_PHASE_SIDES = (512, 768, 1024)
MIXED_RESOLUTION_RATIO_CONFIG_KEYS = {
    512: "staged_resolution_ratio_512",
    768: "staged_resolution_ratio_768",
    1024: "staged_resolution_ratio_1024",
}
MIXED_RESOLUTION_RATIO_DEFAULTS = {
    512: 40.0,
    768: 30.0,
    1024: 30.0,
}
MIXED_RESOLUTION_SAMPLE_EPOCH_FACTORS = {
    512: 4.0,
    768: 1.78,
    1024: 1.0,
}
MIXED_RESOLUTION_RESUME_SENTINEL = "__MIXED_AUTO_RESUME__"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def _parse_csv(value, default_csv: str):
    raw = str(value if value is not None else default_csv)
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_sync_config_keys(value):
    keys = _parse_csv(value, DEFAULT_SYNC_CONFIG_KEYS)
    lowered = {k.strip().lower() for k in keys}
    if any(k in {"*", "__all__", "all"} for k in lowered):
        return ["*"]

    legacy = {x.strip().lower() for x in LEGACY_DEFAULT_SYNC_CONFIG_KEYS.split(",")}
    if {k.strip().lower() for k in keys} == legacy:
        log.info("[sync-config] detected legacy key list, auto-upgrade to full sync mode")
        return ["*"]

    return keys


def _resolve_trainer_file_from_runtime_config(runtime_train_config: dict, fallback_trainer_file: str) -> str:
    if not isinstance(runtime_train_config, dict):
        return fallback_trainer_file

    model_train_type = str(runtime_train_config.get("model_train_type", "") or "").strip().lower()
    if not model_train_type:
        return fallback_trainer_file

    return MODEL_TRAIN_TYPE_TO_TRAINER_FILE.get(model_train_type, fallback_trainer_file)


def _parse_resolution_pair(value: str) -> Optional[tuple[int, int]]:
    raw = str(value or "").strip().lower().replace("x", ",")
    if not raw:
        return None

    parts = [x.strip() for x in raw.split(",") if x.strip()]
    try:
        if len(parts) == 1:
            side = int(parts[0])
            if side <= 0:
                return None
            return side, side
        width = int(parts[0])
        height = int(parts[1])
    except Exception:
        return None

    if width <= 0 or height <= 0:
        return None
    return width, height


def _ceil_to_multiple(value: int, base: int) -> int:
    if base <= 0:
        return value
    return int(math.ceil(value / base) * base)


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def _lcm(a: int, b: int) -> int:
    if a <= 0:
        return max(1, b)
    if b <= 0:
        return max(1, a)
    return abs(a * b) // _gcd(a, b)


def _scale_epoch_interval(base_value: int, factor: float) -> int:
    return max(1, int(math.ceil(max(1, int(base_value)) * float(factor))))


def _load_staged_phase_ratios(config: dict) -> tuple[list[tuple[int, float, float]], str]:
    configured = []
    ratio_sum = 0.0

    for side in MIXED_RESOLUTION_PHASE_SIDES:
        key = MIXED_RESOLUTION_RATIO_CONFIG_KEYS[side]
        default_percent = MIXED_RESOLUTION_RATIO_DEFAULTS[side]
        raw_value = config.get(key, default_percent)
        try:
            percent = float(raw_value)
        except Exception:
            return [], f"{key} 无法解析为数字: {raw_value}"

        if not math.isfinite(percent):
            return [], f"{key} 不是有效数字: {raw_value}"
        if percent < 0 or percent > 100:
            return [], f"{key} 超出范围: {percent}（仅允许 0~100）"

        ratio_sum += percent
        configured.append((side, percent / 100.0, percent))

    if ratio_sum > 100 + 1e-9:
        return [], (
            "阶段分辨率占比总和不能大于 100："
            f"{MIXED_RESOLUTION_RATIO_CONFIG_KEYS[512]} + "
            f"{MIXED_RESOLUTION_RATIO_CONFIG_KEYS[768]} + "
            f"{MIXED_RESOLUTION_RATIO_CONFIG_KEYS[1024]} = {ratio_sum:.4f}"
        )

    active = [item for item in configured if item[2] > 0]
    if not active:
        return [], "阶段分辨率占比总和为 0，至少需要一个阶段占比大于 0"

    return active, ""


def _count_images_recursive(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0

    count = 0
    for file in path.rglob("*"):
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            count += 1
    return count


def _count_train_images_with_repeats(config: dict, repo_root: Path) -> int:
    train_data_dir = str(config.get("train_data_dir", "") or "").strip()
    if not train_data_dir:
        return 0

    train_root = _resolve_local_path(train_data_dir, repo_root)
    if not train_root.exists() or not train_root.is_dir():
        return 0

    repeat_subsets = []
    try:
        subdirs = sorted([p for p in train_root.iterdir() if p.is_dir()])
    except Exception:
        subdirs = []

    for subdir in subdirs:
        match = re.match(r"^(\d+)_", subdir.name)
        if not match:
            continue
        repeats = max(1, int(match.group(1)))
        image_count = _count_images_recursive(subdir)
        repeat_subsets.append((repeats, image_count, subdir))

    if repeat_subsets:
        total = sum(repeats * image_count for repeats, image_count, _ in repeat_subsets)
        return total

    # fallback: no repeat-style subset folder found
    return _count_images_recursive(train_root)


def _resolve_per_device_batch_from_global(global_batch: int, world_size: int) -> Tuple[bool, int, str]:
    ws = max(1, int(world_size))
    gb = int(global_batch)
    if gb <= 0:
        return False, 0, "train_batch_size 必须大于 0"

    if ws == 1:
        return True, gb, ""

    if gb < ws:
        return (
            False,
            0,
            f"为保持等效全局 batch 不变，train_batch_size(={gb}) 不能小于 world_size(={ws})。"
            "请增大 batch 或减少并行卡数。",
        )

    if gb % ws != 0:
        return (
            False,
            0,
            f"为保持等效全局 batch 不变，train_batch_size(={gb}) 必须能被 world_size(={ws}) 整除。"
            "请调整 batch 或并行卡数。",
        )

    return True, gb // ws, ""


def _build_mixed_resolution_plan(
    config: dict,
    toml_path: str,
    trainer_file: str,
    *,
    num_processes_for_epoch_calc: int = 1,
    save_every_n_epochs_default: int = 1,
):
    if not _to_bool(config.get(MIXED_RESOLUTION_ENABLE_KEY), False):
        return None, ""

    trainer_name = Path(str(trainer_file)).name
    if trainer_name not in {"train_network.py", "sdxl_train_network.py"}:
        return None, f"当前训练脚本不支持阶段分辨率训练: {trainer_name}"

    resolution = _parse_resolution_pair(str(config.get("resolution", "") or ""))
    if resolution is None:
        return None, f"无法解析训练分辨率: {config.get('resolution')}"
    width, height = resolution
    if width != height:
        return None, "阶段分辨率训练目前仅支持正方形分辨率（如 1024,1024）"

    base_side = int(width)
    if base_side != 1024:
        return None, "当前阶段分辨率流程固定为 512 -> 768 -> 1024，基础分辨率必须设置为 1024,1024"
    base_pixels = base_side * base_side

    try:
        base_epochs = int(config.get("max_train_epochs"))
    except Exception:
        base_epochs = 0
    if base_epochs <= 0:
        return None, "启用阶段分辨率训练时，max_train_epochs 必须大于 0"

    normalized_num_processes_for_epoch_calc = max(1, int(num_processes_for_epoch_calc))

    try:
        base_global_batch = int(config.get("train_batch_size"))
    except Exception:
        base_global_batch = 0
    if base_global_batch <= 0:
        return None, "启用阶段分辨率训练时，train_batch_size 必须大于 0"
    ok_batch, base_per_device_batch, batch_error = _resolve_per_device_batch_from_global(
        base_global_batch, normalized_num_processes_for_epoch_calc
    )
    if not ok_batch:
        return None, f"启用阶段分辨率训练时批大小配置不合法: {batch_error}"

    try:
        base_gradient_accumulation_steps = int(config.get("gradient_accumulation_steps", 1) or 1)
    except Exception:
        base_gradient_accumulation_steps = 1
    if base_gradient_accumulation_steps <= 0:
        base_gradient_accumulation_steps = 1

    try:
        save_every_n_epochs = int(config.get("save_every_n_epochs", save_every_n_epochs_default) or save_every_n_epochs_default)
    except Exception:
        save_every_n_epochs = save_every_n_epochs_default
    if save_every_n_epochs <= 0:
        save_every_n_epochs = 1

    preview_enabled = _to_bool(config.get("enable_preview", False), False)
    sample_prompts = str(config.get("sample_prompts", "") or "").strip()

    raw_sample_every_n_epochs = config.get("sample_every_n_epochs", None)
    base_sample_every_n_epochs = None
    try:
        if raw_sample_every_n_epochs is not None and str(raw_sample_every_n_epochs).strip() != "":
            parsed_sample_every_n_epochs = int(raw_sample_every_n_epochs)
            if parsed_sample_every_n_epochs > 0:
                base_sample_every_n_epochs = parsed_sample_every_n_epochs
    except Exception:
        base_sample_every_n_epochs = None

    use_sample_epoch_schedule = bool(sample_prompts and base_sample_every_n_epochs is not None)

    configured_phases, ratio_error = _load_staged_phase_ratios(config)
    if ratio_error:
        return None, ratio_error
    configured_ratio_sum_percent = sum(item[2] for item in configured_phases)

    plan_signature_payload = {
        "trainer_name": trainer_name,
        "base_resolution": [base_side, base_side],
        "base_epochs": int(base_epochs),
        "base_batch_global": int(base_global_batch),
        "base_batch_per_device": int(base_per_device_batch),
        "world_size": int(normalized_num_processes_for_epoch_calc),
        "base_gradient_accumulation_steps": int(base_gradient_accumulation_steps),
        "save_every_n_epochs": int(save_every_n_epochs),
        "use_sample_epoch_schedule": bool(use_sample_epoch_schedule),
        "base_sample_every_n_epochs": int(base_sample_every_n_epochs) if base_sample_every_n_epochs is not None else None,
        "configured_phases": [
            {
                "side": int(side),
                "ratio": float(ratio),
                "ratio_percent": float(ratio_percent),
            }
            for side, ratio, ratio_percent in configured_phases
        ],
    }
    plan_id = hashlib.sha1(
        json.dumps(plan_signature_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]

    repo_root = base_dir_path()
    total_train_images = _count_train_images_with_repeats(config, repo_root)
    if total_train_images <= 0:
        return None, "无法统计训练图像数量，无法生成阶段分辨率训练计划"
    # Keep staged plan topology-agnostic: epoch/sample targets should not drift
    # when switching between single-machine and multi-machine runs.
    global_epoch_train_images = max(1, int(total_train_images))

    plan_base = Path(toml_path)
    phase_configs = []
    cumulative_epochs = 0
    cumulative_steps = 0
    previous_side = None

    for idx, (side, ratio, ratio_percent) in enumerate(configured_phases, start=1):
        target_pixels = side * side
        global_batch_this_phase = max(1, int(math.floor(base_global_batch * (base_pixels / target_pixels))))
        ok_phase_batch, per_device_batch_this_phase, phase_batch_error = _resolve_per_device_batch_from_global(
            global_batch_this_phase, normalized_num_processes_for_epoch_calc
        )
        if not ok_phase_batch:
            return (
                None,
                f"阶段 {idx} ({side}x{side}) 批大小配置不合法: {phase_batch_error}"
                f"（当前阶段全局 batch={global_batch_this_phase}）",
            )

        sample_factor = float(MIXED_RESOLUTION_SAMPLE_EPOCH_FACTORS.get(side, base_pixels / target_pixels))
        # Keep gradient-accumulation semantics anchored to 1024 baseline:
        # - base grad_accum <= 1: keep 1
        # - base grad_accum > 1: keep the same x across all phases
        gradient_accumulation_steps_this_phase = (
            1 if base_gradient_accumulation_steps <= 1 else int(base_gradient_accumulation_steps)
        )
        gradient_accumulation_factor = (
            float(gradient_accumulation_steps_this_phase) / float(max(1, base_gradient_accumulation_steps))
        )
        save_every_n_epochs_this_phase = _scale_epoch_interval(save_every_n_epochs, sample_factor)
        sample_every_n_epochs_this_phase = (
            _scale_epoch_interval(base_sample_every_n_epochs, sample_factor) if use_sample_epoch_schedule else None
        )
        epoch_rounding_multiple = int(save_every_n_epochs_this_phase)
        if use_sample_epoch_schedule and sample_every_n_epochs_this_phase is not None:
            epoch_rounding_multiple = _lcm(epoch_rounding_multiple, sample_every_n_epochs_this_phase)

        effective_batch_ratio = (
            (global_batch_this_phase * gradient_accumulation_steps_this_phase)
            / (base_global_batch * base_gradient_accumulation_steps)
        )
        # Raw formula: ceil(base_epochs * phase_ratio * (phase_effective_batch / base_effective_batch))
        raw_epochs_this_phase = int(math.ceil(base_epochs * ratio * effective_batch_ratio))
        # Actual formula: ceil_to_multiple(raw_epochs, lcm(save_every_n_epochs, sample_every_n_epochs_phase))
        epochs_this_phase = _ceil_to_multiple(max(1, raw_epochs_this_phase), epoch_rounding_multiple)
        batches_per_epoch = max(1, int(math.ceil(global_epoch_train_images / global_batch_this_phase)))
        steps_per_epoch = max(1, int(math.ceil(batches_per_epoch / gradient_accumulation_steps_this_phase)))
        steps_this_phase = int(epochs_this_phase * steps_per_epoch)
        cumulative_epochs += int(epochs_this_phase)
        cumulative_steps += int(steps_this_phase)

        phase_toml_path = plan_base.with_name(f"{plan_base.stem}-staged-phase{idx}.toml")
        phase_config = copy.deepcopy(config)
        phase_config[MIXED_RESOLUTION_ENABLE_KEY] = False
        phase_config["resolution"] = f"{side},{side}"
        phase_config["train_batch_size"] = int(per_device_batch_this_phase)
        phase_config["gradient_accumulation_steps"] = int(gradient_accumulation_steps_this_phase)
        phase_config["max_train_steps"] = int(cumulative_steps)
        phase_config.pop("max_train_epochs", None)
        phase_config.pop("resume_epoch_offset", None)
        phase_config["save_state"] = True
        phase_config["staged_plan_id"] = plan_id
        phase_config["staged_phase_index"] = int(idx)
        phase_config["staged_phase_target_max_train_steps"] = int(cumulative_steps)
        phase_config["save_every_n_epochs"] = int(save_every_n_epochs_this_phase)
        if use_sample_epoch_schedule and sample_every_n_epochs_this_phase is not None:
            phase_config["sample_every_n_epochs"] = int(sample_every_n_epochs_this_phase)
        else:
            phase_config.pop("sample_every_n_epochs", None)
        if idx > 1:
            phase_config["resume"] = MIXED_RESOLUTION_RESUME_SENTINEL

        with open(phase_toml_path, "w", encoding="utf-8") as f:
            toml.dump(phase_config, f)

        raw_formula = (
            f"ceil({base_epochs} * ({ratio_percent:g} / 100) * "
            f"(({global_batch_this_phase}*{gradient_accumulation_steps_this_phase}) / "
            f"({base_global_batch}*{base_gradient_accumulation_steps})))"
        )
        if use_sample_epoch_schedule and sample_every_n_epochs_this_phase is not None:
            actual_formula = (
                "ceil_to_multiple(raw_epochs, "
                f"lcm(save_every_n_epochs={save_every_n_epochs_this_phase}, "
                f"sample_every_n_epochs={sample_every_n_epochs_this_phase})={epoch_rounding_multiple})"
            )
        else:
            actual_formula = (
                "ceil_to_multiple(raw_epochs, "
                f"save_every_n_epochs={save_every_n_epochs_this_phase})"
            )

        phase_configs.append(
            {
                "phase_index": idx,
                "toml_path": str(phase_toml_path),
                "plan_id": plan_id,
                "resolution_side": int(side),
                "resolution": f"{side},{side}",
                "ratio": float(ratio),
                "ratio_percent": float(ratio_percent),
                "effective_batch_ratio": float(effective_batch_ratio),
                "raw_epochs": int(raw_epochs_this_phase),
                "epochs": int(epochs_this_phase),
                "raw_epochs_formula": raw_formula,
                "actual_epochs_formula": actual_formula,
                "batches_per_epoch": int(batches_per_epoch),
                "steps_per_epoch": int(steps_per_epoch),
                "phase_steps": int(steps_this_phase),
                "target_max_train_steps": int(cumulative_steps),
                "target_epoch_end": int(cumulative_epochs),
                "batch_size": int(global_batch_this_phase),
                "batch_size_global": int(global_batch_this_phase),
                "batch_size_per_device": int(per_device_batch_this_phase),
                "gradient_accumulation_steps_factor": float(gradient_accumulation_factor),
                "gradient_accumulation_steps": int(gradient_accumulation_steps_this_phase),
                "save_every_n_epochs_factor": float(sample_factor),
                "save_every_n_epochs": int(save_every_n_epochs_this_phase),
                "sample_every_n_epochs_factor": float(sample_factor),
                "sample_every_n_epochs": int(sample_every_n_epochs_this_phase) if sample_every_n_epochs_this_phase is not None else None,
                "epoch_rounding_multiple": int(epoch_rounding_multiple),
                "clear_cache_before_start": previous_side is not None and previous_side != side,
            }
        )
        previous_side = side

    plan = {
        "enabled": True,
        "plan_id": plan_id,
        "plan_signature_payload": plan_signature_payload,
        "base_config_toml": str(toml_path),
        "trainer_file": str(trainer_file),
        "phase_count": len(phase_configs),
        "save_every_n_epochs": int(save_every_n_epochs),
        "configured_ratio_sum_percent": float(configured_ratio_sum_percent),
        "configured_phase_ratios_percent": {
            str(side): float(percent) for side, _, percent in configured_phases
        },
        "preview_enabled": bool(preview_enabled),
        "use_sample_epoch_schedule": bool(use_sample_epoch_schedule),
        "base_sample_every_n_epochs": int(base_sample_every_n_epochs) if base_sample_every_n_epochs is not None else None,
        "sample_every_n_epochs_rule": "1024=x, 768=ceil(1.78x), 512=ceil(4x)",
        "save_every_n_epochs_rule": "1024=x, 768=ceil(1.78x), 512=ceil(4x)",
        "base_resolution": f"{base_side},{base_side}",
        "base_batch_size": int(base_global_batch),
        "base_batch_size_global": int(base_global_batch),
        "base_batch_size_per_device": int(base_per_device_batch),
        "base_gradient_accumulation_steps": int(base_gradient_accumulation_steps),
        "base_epochs": int(base_epochs),
        "world_size": int(normalized_num_processes_for_epoch_calc),
        "total_train_images_with_repeats": int(total_train_images),
        "num_processes_for_epoch_calc": int(normalized_num_processes_for_epoch_calc),
        "per_process_train_images_with_repeats": int(global_epoch_train_images),
        "global_epoch_train_images_with_repeats": int(global_epoch_train_images),
        "total_mixed_epochs": int(cumulative_epochs),
        "total_mixed_steps": int(cumulative_steps),
        "gradient_accumulation_steps_rule": "x=1 -> all phases keep 1; x>1 -> all phases keep x (anchored to 1024 baseline)",
        "phases": phase_configs,
    }
    return plan, ""


def _list_local_network_interfaces() -> list[str]:
    net_root = Path("/sys/class/net")
    if not net_root.exists():
        return []
    try:
        return sorted([p.name for p in net_root.iterdir() if p.is_dir()])
    except Exception:
        return []


def _parse_ifname_candidates(value: str) -> list[str]:
    if not value:
        return []

    result = []
    for token in str(value).split(","):
        name = token.strip()
        if not name or name.startswith("^"):
            continue
        result.append(name)
    return result


def _pick_training_mesh_iface(nccl_socket_ifname: str, gloo_socket_ifname: str, main_process_ip: str) -> str:
    interfaces = _list_local_network_interfaces()
    if not interfaces:
        return ""
    interface_set = set(interfaces)

    # Highest priority: explicitly configured NCCL/GLOO interface.
    for name in _parse_ifname_candidates(nccl_socket_ifname) + _parse_ifname_candidates(gloo_socket_ifname):
        if name in interface_set:
            return name

    # Fallback: infer outgoing device to main process IPv4.
    if main_process_ip and ":" not in str(main_process_ip):
        try:
            route = subprocess.run(
                ["ip", "-4", "route", "get", str(main_process_ip)],
                text=True,
                capture_output=True,
                check=False,
            )
            if route.returncode == 0:
                parts = route.stdout.strip().split()
                if "dev" in parts:
                    dev_idx = parts.index("dev") + 1
                    if dev_idx < len(parts):
                        route_iface = parts[dev_idx]
                        if route_iface in interface_set:
                            return route_iface
        except Exception:
            pass

    # Final fallback: first non-loopback interface.
    for iface in interfaces:
        if iface != "lo":
            return iface

    return ""


def _validate_socket_ifname(name: str, env_key: str) -> Tuple[bool, str]:
    if not name:
        return True, ""

    interfaces = _list_local_network_interfaces()
    if not interfaces:
        return True, ""

    if name in interfaces:
        return True, ""

    return False, (
        f"{env_key} 配置为 '{name}'，但本机不存在该网卡。"
        f"可用网卡: {', '.join(interfaces)}。"
        f"请改成正确网卡名，或留空让系统自动选择。"
    )


def _get_dataset_dirs_from_toml(toml_path: str):
    repo_root = base_dir_path()
    config = toml.load(toml_path)
    dataset_dirs = []
    seen = set()

    for key in DATASET_DIR_KEYS:
        value = config.get(key)
        if not isinstance(value, str):
            continue
        value = value.strip()
        if not value:
            continue

        local_path = _resolve_local_path(value, repo_root)
        local_norm = str(local_path)
        if local_norm in seen:
            continue
        seen.add(local_norm)
        dataset_dirs.append((key, value, local_path))

    return dataset_dirs


def _count_local_dataset_files_without_npz(local_dir: Path) -> int:
    if not local_dir.exists():
        return 0
    if not local_dir.is_dir():
        return -1

    count = 0
    for path in local_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() != ".npz":
            count += 1
    return count


def _count_remote_dataset_files_without_npz(
    remote_host: str,
    ssh_port: int,
    remote_dir: str,
    *,
    use_password_auth: bool = False,
    ssh_password: str = "",
) -> int:
    remote_cmd = (
        f"if [ -d {shlex.quote(remote_dir)} ]; then "
        f"find {shlex.quote(remote_dir)} -type f ! -iname '*.npz' | wc -l; "
        "else echo -1; fi"
    )
    result = _ssh(
        remote_host,
        ssh_port,
        remote_cmd,
        f"[dataset-sync] count remote files {remote_dir}",
        use_password_auth=use_password_auth,
        ssh_password=ssh_password,
    )
    if result is None:
        return -2

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return -2

    try:
        return int(lines[-1])
    except Exception:
        return -2


def _sync_dataset_dir_from_main(
    remote_host: str,
    ssh_port: int,
    remote_dir: str,
    local_dir: Path,
    *,
    use_password_auth: bool = False,
    ssh_password: str = "",
) -> Tuple[bool, str]:
    if shutil.which("rsync") is None:
        return False, "缺少 rsync，无法执行数据集同步"

    local_dir.mkdir(parents=True, exist_ok=True)
    ssh_exec = " ".join(["ssh", "-p", str(ssh_port), *_ssh_options(use_password_auth)])
    rsync_cmd = [
        "rsync",
        "-a",
        "--partial",
        "--delete",
        "--exclude",
        "*.npz",
        "--exclude",
        "*.NPZ",
        "-e",
        ssh_exec,
        f"{remote_host}:{remote_dir.rstrip('/')}/",
        f"{str(local_dir)}/",
    ]
    rsync_cmd = _with_sshpass(
        rsync_cmd,
        use_password_auth,
        ssh_password,
        f"[dataset-sync] rsync {remote_dir} -> {local_dir}",
    )
    if rsync_cmd is None:
        return False, "无法构建密码认证 rsync 命令"

    if _run_cmd(rsync_cmd, f"[dataset-sync] rsync {remote_dir} -> {local_dir}") is None:
        return False, f"数据集同步失败: {remote_dir}"
    return True, ""


def _sync_datasets_when_count_mismatch_from_main(
    toml_path: str,
    remote_host: str,
    ssh_port: int,
    remote_repo_root: str,
    *,
    use_password_auth: bool = False,
    ssh_password: str = "",
) -> Tuple[bool, str]:
    dataset_dirs = _get_dataset_dirs_from_toml(toml_path)
    if not dataset_dirs:
        log.info("[dataset-sync] no dataset dir found in toml, skip count sync")
        return True, ""

    for key, raw_value, local_dir in dataset_dirs:
        remote_dir = _resolve_remote_path(raw_value, remote_repo_root)
        remote_type = _remote_path_type(
            remote_host,
            ssh_port,
            remote_dir,
            use_password_auth=use_password_auth,
            ssh_password=ssh_password,
        )
        if remote_type == "missing":
            return False, f"主机数据集目录不存在: {key} -> {remote_dir}"
        if remote_type != "dir":
            return False, f"主机数据集路径不是目录: {key} -> {remote_dir} ({remote_type})"

        local_count = _count_local_dataset_files_without_npz(local_dir)
        if local_count < 0:
            return False, f"本地数据集路径不是目录: {local_dir}"
        remote_count = _count_remote_dataset_files_without_npz(
            remote_host,
            ssh_port,
            remote_dir,
            use_password_auth=use_password_auth,
            ssh_password=ssh_password,
        )
        if remote_count < 0:
            return False, f"无法统计主机数据集文件数量: {remote_dir}"

        log.info(
            f"[dataset-sync] {key}: local_count={local_count}, remote_count={remote_count}, "
            f"local_dir={local_dir}, remote_dir={remote_dir}"
        )
        if local_count == remote_count:
            log.info(f"[dataset-sync] {key}: file count already matched, skip sync")
            continue

        log.warning(
            f"[dataset-sync] {key}: count mismatch detected, syncing dataset from main "
            f"(local={local_count}, remote={remote_count})"
        )
        ok, message = _sync_dataset_dir_from_main(
            remote_host,
            ssh_port,
            remote_dir,
            local_dir,
            use_password_auth=use_password_auth,
            ssh_password=ssh_password,
        )
        if not ok:
            return False, message

        local_after = _count_local_dataset_files_without_npz(local_dir)
        if local_after != remote_count:
            return (
                False,
                f"数据集同步后文件数仍不一致: {key}, local_after={local_after}, remote={remote_count}",
            )
        log.info(f"[dataset-sync] {key}: sync completed, count={local_after}")

    return True, ""


def _clear_dataset_npz_cache(toml_path: str) -> Tuple[bool, str]:
    dataset_dirs = _get_dataset_dirs_from_toml(toml_path)
    if not dataset_dirs:
        log.info("[cache-reset] no dataset dir found in toml, skip npz cleanup")
        return True, ""

    total_removed = 0
    for key, _, local_dir in dataset_dirs:
        if not local_dir.exists():
            log.info(f"[cache-reset] {key}: dataset dir not found, skip npz cleanup: {local_dir}")
            continue
        if not local_dir.is_dir():
            return False, f"数据集路径不是目录，无法清理 npz: {local_dir}"

        removed = 0
        for npz_file in local_dir.rglob("*.npz"):
            try:
                npz_file.unlink()
                removed += 1
            except Exception as e:
                return False, f"删除缓存失败: {npz_file} ({e})"

        total_removed += removed
        log.info(f"[cache-reset] {key}: removed {removed} npz files under {local_dir}")

    log.info(f"[cache-reset] removed total npz files: {total_removed}")
    return True, ""


def _resolve_local_path(path_value: str, repo_root: Path) -> Path:
    p = Path(path_value).expanduser()
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def _resolve_remote_path(path_value: str, remote_repo_root: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return str(Path(remote_repo_root) / path_value)


def _is_tensorboard_logging_enabled(config: dict) -> bool:
    output_dir = str(config.get("output_dir", "") or "").strip()
    return bool(output_dir)


def _resolve_tensorboard_logging_root(config: dict, repo_root: Path) -> Optional[Path]:
    output_dir = str(config.get("output_dir", "") or "").strip()
    if not output_dir:
        return None
    return _resolve_local_path(output_dir, repo_root)


def _sanitize_tensorboard_component(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", str(value or "").strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-_.")
    return cleaned or "model"


def _resolve_tensorboard_model_name(config: dict) -> str:
    value = str(config.get("output_name", "") or "").strip()
    if value:
        return _sanitize_tensorboard_component(value)
    return "model"


def _read_resume_tensorboard_dir_from_state(config: dict, repo_root: Path) -> Optional[Path]:
    resume_path = str(config.get("resume", "") or "").strip()
    if not resume_path:
        return None

    local_resume_dir = _resolve_local_path(resume_path, repo_root)
    if not local_resume_dir.exists() or not local_resume_dir.is_dir():
        return None

    train_state_file = local_resume_dir / "train_state.json"
    if not train_state_file.exists():
        return None

    try:
        with open(train_state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        log.warning(f"[tensorboard] failed to read state file: {train_state_file}")
        return None

    if not isinstance(data, dict):
        return None

    logging_dir = data.get("logging_dir")
    if not isinstance(logging_dir, str) or not logging_dir.strip():
        return None

    logging_dir_path = Path(logging_dir.strip()).expanduser()
    if not logging_dir_path.is_absolute():
        logging_dir_path = (repo_root / logging_dir_path).resolve()
    if not logging_dir_path.exists() or not logging_dir_path.is_dir():
        return None

    output_root = _resolve_tensorboard_logging_root(config, repo_root)
    if output_root is not None:
        try:
            logging_dir_path.relative_to(output_root)
        except ValueError:
            return None
    return logging_dir_path


def _find_latest_tensorboard_run(logging_root: Path, model_name: str) -> Optional[Path]:
    if not logging_root.exists() or not logging_root.is_dir():
        return None

    pattern = re.compile(rf"^\d{{4}}-\d{{2}}-\d{{2}}_{re.escape(model_name)}_(\d+)$")
    candidates = []
    for entry in logging_root.iterdir():
        if not entry.is_dir():
            continue
        match = pattern.fullmatch(entry.name)
        if not match:
            continue
        try:
            stat = entry.stat()
            candidates.append((stat.st_mtime, int(match.group(1)), entry))
        except Exception:
            continue

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _build_next_tensorboard_run(logging_root: Path, model_name: str) -> Path:
    logging_root.mkdir(parents=True, exist_ok=True)

    date_prefix = datetime.now().strftime("%Y-%m-%d")
    run_prefix = f"{date_prefix}_{model_name}_"
    pattern = re.compile(rf"^{re.escape(run_prefix)}(\d+)$")
    max_index = 0

    for entry in logging_root.iterdir():
        if not entry.is_dir():
            continue
        match = pattern.fullmatch(entry.name)
        if not match:
            continue
        try:
            max_index = max(max_index, int(match.group(1)))
        except Exception:
            continue

    return logging_root / f"{run_prefix}{max_index + 1}"


def _resolve_tensorboard_run_dir_from_config(config: dict, repo_root: Path) -> Optional[Path]:
    if not _is_tensorboard_logging_enabled(config):
        return None

    logging_root = _resolve_tensorboard_logging_root(config, repo_root)
    if logging_root is None:
        return None
    model_name = _resolve_tensorboard_model_name(config)

    resume_logging_dir = _read_resume_tensorboard_dir_from_state(config, repo_root)
    if resume_logging_dir is not None:
        return resume_logging_dir

    resume_path = str(config.get("resume", "") or "").strip()
    if resume_path:
        latest_run = _find_latest_tensorboard_run(logging_root, model_name)
        if latest_run is not None:
            return latest_run

    return _build_next_tensorboard_run(logging_root, model_name)


def _enforce_tb_only_config(config: dict) -> bool:
    if not isinstance(config, dict):
        return False

    changed = False
    output_dir = str(config.get("output_dir", "./output") or "./output").strip()
    if not output_dir:
        output_dir = "./output"
    if config.get("output_dir") != output_dir:
        config["output_dir"] = output_dir
        changed = True

    if config.get("log_with") != "tensorboard":
        config["log_with"] = "tensorboard"
        changed = True

    if config.get("logging_dir") != output_dir:
        config["logging_dir"] = output_dir
        changed = True

    for key in ("wandb_api_key", "wandb_run_name", "log_tracker_config", "log_prefix", "log_tracker_name"):
        if key in config:
            config.pop(key, None)
            changed = True

    return changed


def _snapshot_tensorboard_event_files(run_dir: Optional[Path]) -> dict:
    snapshot = {}
    if run_dir is None or not run_dir.exists():
        return snapshot

    for event_file in run_dir.rglob(TB_EVENT_FILE_GLOB):
        if not event_file.is_file():
            continue
        try:
            stat = event_file.stat()
        except Exception:
            continue
        snapshot[str(event_file.resolve())] = (stat.st_size, stat.st_mtime)
    return snapshot


def _list_checkpoint_files_for_run(config: dict, repo_root: Path) -> list[Path]:
    output_dir = _resolve_local_path(str(config.get("output_dir", "./output") or "./output"), repo_root)
    if not output_dir.exists() or not output_dir.is_dir():
        return []

    output_name = str(config.get("output_name", "") or "").strip()
    files = {}
    for ext in CKPT_EXTENSIONS:
        pattern = f"{output_name}*{ext}" if output_name else f"*{ext}"
        for ckpt_file in output_dir.glob(pattern):
            if ckpt_file.is_file():
                files[str(ckpt_file.resolve())] = ckpt_file
    return list(files.values())


def _has_new_checkpoint_since(config: dict, repo_root: Path, started_at: float) -> bool:
    output_dir = _resolve_local_path(str(config.get("output_dir", "./output") or "./output"), repo_root)
    output_name = str(config.get("output_name", "") or "").strip()

    for ckpt_file in _list_checkpoint_files_for_run(config, repo_root):
        try:
            if ckpt_file.stat().st_mtime >= started_at:
                return True
        except Exception:
            continue

    if output_dir.exists() and output_dir.is_dir():
        for child in output_dir.iterdir():
            if not child.is_dir():
                continue
            if output_name and not child.name.startswith(output_name):
                continue
            if child.name.endswith("-state"):
                continue
            marker_file = child / "model_index.json"
            if not marker_file.is_file():
                continue
            try:
                if max(child.stat().st_mtime, marker_file.stat().st_mtime) >= started_at:
                    return True
            except Exception:
                continue

    return False


def _list_existing_training_artifacts_for_run(config: dict, repo_root: Path) -> list[Path]:
    output_dir = _resolve_local_path(str(config.get("output_dir", "./output") or "./output"), repo_root)
    if not output_dir.exists() or not output_dir.is_dir():
        return []

    output_name = str(config.get("output_name", "") or "").strip()
    artifacts = {}

    for ckpt_file in _list_checkpoint_files_for_run(config, repo_root):
        if ckpt_file.is_file():
            artifacts[str(ckpt_file.resolve())] = ckpt_file

    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        if output_name and not child.name.startswith(output_name):
            continue

        if child.name.endswith("-state"):
            if (child / "train_state.json").is_file():
                artifacts[str(child.resolve())] = child
            continue

        if (child / "model_index.json").is_file():
            artifacts[str(child.resolve())] = child

    return sorted(artifacts.values(), key=lambda p: p.name)


def _check_resume_state_dir_complete(state_dir: Path) -> Tuple[bool, str]:
    for name in STATE_REQUIRED_FILES:
        if not (state_dir / name).is_file():
            return False, f"missing {name}"

    has_model_file = any((state_dir / name).is_file() for name in STATE_MODEL_FILE_CANDIDATES)
    if not has_model_file:
        has_model_file = any(state_dir.glob("pytorch_model*.bin"))
    if not has_model_file:
        has_model_file = any(state_dir.glob("model*.safetensors"))
    if not has_model_file:
        return False, "missing model state file"

    return True, ""


def _validate_resume_launch_guard(config: dict, repo_root: Path) -> Tuple[bool, str]:
    artifacts = _list_existing_training_artifacts_for_run(config, repo_root)
    if not artifacts:
        return True, ""

    output_dir = _resolve_local_path(str(config.get("output_dir", "./output") or "./output"), repo_root)
    resume_path_raw = str(config.get("resume", "") or "").strip()

    if not resume_path_raw:
        sample_names = ", ".join(p.name for p in artifacts[:3])
        return (
            False,
            "检测到输出目录已存在历史训练结果，当前未填写 resume state 路径，已阻止启动。"
            f" output_dir={output_dir}，匹配到 {len(artifacts)} 个结果（例如: {sample_names}）。"
            "请填写同一输出目录下的 state 路径后重试。"
        )

    resume_dir = _resolve_local_path(resume_path_raw, repo_root)
    if not resume_dir.exists() or not resume_dir.is_dir():
        return (
            False,
            "resume 路径不存在或不是目录，已阻止启动。"
            f" resume={resume_dir}"
        )

    train_state_file = resume_dir / "train_state.json"
    if not train_state_file.is_file():
        return (
            False,
            "resume 路径不是有效的 state 目录（缺少 train_state.json），已阻止启动。"
            f" resume={resume_dir}"
        )

    complete, incomplete_reason = _check_resume_state_dir_complete(resume_dir)
    if not complete:
        return (
            False,
            "resume 路径指向的 state 目录不完整，已阻止启动。"
            f" resume={resume_dir}，原因: {incomplete_reason}。"
            "请先确保从主机同步完整 state（model/optimizer/scheduler）后再重试。"
        )

    try:
        resume_dir.relative_to(output_dir)
    except ValueError:
        return (
            False,
            "检测到 output 与 resume 不属于同一个输出目录，已阻止启动。"
            f" output_dir={output_dir}，resume={resume_dir}"
        )

    return True, ""


def _cleanup_tensorboard_records_without_checkpoint(run_dir: Optional[Path], existed_before: bool, event_snapshot: dict):
    if run_dir is None or not run_dir.exists():
        return

    if not existed_before:
        try:
            shutil.rmtree(run_dir)
            log.info(f"[tensorboard] removed run dir because no checkpoint was produced: {run_dir}")
        except Exception as e:
            log.warning(f"[tensorboard] failed to remove run dir without checkpoint: {run_dir} ({e})")
        return

    existing_keys = set(event_snapshot.keys())
    removed_files = 0
    for event_file in run_dir.rglob(TB_EVENT_FILE_GLOB):
        if not event_file.is_file():
            continue
        resolved = str(event_file.resolve())
        if resolved in existing_keys:
            continue
        try:
            event_file.unlink()
            removed_files += 1
        except Exception as e:
            log.warning(f"[tensorboard] failed to remove event file without checkpoint: {event_file} ({e})")

    for dir_path in sorted([p for p in run_dir.rglob("*") if p.is_dir()], reverse=True):
        try:
            dir_path.rmdir()
        except OSError:
            continue

    if removed_files > 0:
        log.info(f"[tensorboard] removed {removed_files} new event file(s) because no checkpoint was produced")


def _run_cmd(cmd: list, desc: str):
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        err = result.stderr.strip() or "<empty>"
        log.error(f"{desc} failed: code={result.returncode}, stderr={err}")
        return None
    return result


def _ssh_options(use_password_auth: bool):
    options = ["-o", "StrictHostKeyChecking=accept-new"]
    if use_password_auth:
        options += [
            "-o", "PubkeyAuthentication=no",
            "-o", "PreferredAuthentications=password,keyboard-interactive",
        ]
    return options


def _with_sshpass(cmd: list, use_password_auth: bool, ssh_password: str, desc: str):
    if not use_password_auth:
        return cmd

    if not ssh_password:
        log.error(f"{desc} failed: password auth is enabled but ssh password is empty")
        return None

    if shutil.which("sshpass") is None:
        log.error(f"{desc} failed: `sshpass` is required for password auth, please install sshpass")
        return None

    return ["sshpass", "-p", ssh_password, *cmd]


def _ssh(
    remote_host: str,
    ssh_port: int,
    remote_cmd: str,
    desc: str,
    *,
    use_password_auth: bool = False,
    ssh_password: str = "",
):
    cmd = ["ssh", "-p", str(ssh_port), *_ssh_options(use_password_auth), remote_host, remote_cmd]
    cmd = _with_sshpass(cmd, use_password_auth, ssh_password, desc)
    if cmd is None:
        return None
    return _run_cmd(cmd, desc)


def _read_remote_text_file(
    remote_host: str,
    ssh_port: int,
    remote_path: str,
    *,
    use_password_auth: bool = False,
    ssh_password: str = "",
) -> Tuple[Optional[str], str]:
    read_cmd = [
        "ssh",
        "-p",
        str(ssh_port),
        *_ssh_options(use_password_auth),
        remote_host,
        f"cat {shlex.quote(remote_path)}",
    ]
    read_cmd = _with_sshpass(read_cmd, use_password_auth, ssh_password, f"[sync-config] read remote file {remote_path}")
    if read_cmd is None:
        return None, "password auth command build failed"

    result = subprocess.run(read_cmd, text=True, capture_output=True)
    if result.returncode != 0:
        err = result.stderr.strip() or "<empty>"
        return None, err
    return result.stdout, ""


def _remote_path_type(
    remote_host: str,
    ssh_port: int,
    remote_path: str,
    *,
    use_password_auth: bool = False,
    ssh_password: str = "",
) -> str:
    probe_cmd = (
        f"if [ -d {shlex.quote(remote_path)} ]; then echo dir; "
        f"elif [ -f {shlex.quote(remote_path)} ]; then echo file; "
        "else echo missing; fi"
    )
    result = _ssh(
        remote_host,
        ssh_port,
        probe_cmd,
        f"[sync] probing remote path {remote_path}",
        use_password_auth=use_password_auth,
        ssh_password=ssh_password,
    )
    if result is None:
        return "error"
    value = result.stdout.strip().splitlines()
    return value[-1] if value else "error"


def _copy_remote_path(
    remote_host: str,
    ssh_port: int,
    remote_path: str,
    local_path: Path,
    path_type: str,
    *,
    use_password_auth: bool = False,
    ssh_password: str = "",
) -> bool:
    src = f"{remote_host}:{remote_path.rstrip('/')}/" if path_type == "dir" else f"{remote_host}:{remote_path}"

    if shutil.which("rsync"):
        if path_type == "dir":
            local_path.mkdir(parents=True, exist_ok=True)
            dst = f"{str(local_path)}/"
        else:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            dst = str(local_path)

        ssh_exec = " ".join(["ssh", "-p", str(ssh_port), *_ssh_options(use_password_auth)])
        rsync_cmd = ["rsync", "-a", "--partial", "-e", ssh_exec, src, dst]
        rsync_cmd = _with_sshpass(rsync_cmd, use_password_auth, ssh_password, f"[sync] rsync {remote_path} -> {local_path}")
        if rsync_cmd is None:
            return False
        if _run_cmd(rsync_cmd, f"[sync] rsync {remote_path} -> {local_path}") is not None:
            return True
        log.warning("[sync] rsync failed, fallback to scp")

    scp_base_cmd = ["scp", "-P", str(ssh_port), *_ssh_options(use_password_auth)]
    if path_type == "dir":
        local_path.parent.mkdir(parents=True, exist_ok=True)
        scp_cmd = [*scp_base_cmd, "-r", src.rstrip("/"), str(local_path.parent)]
    else:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        scp_cmd = [*scp_base_cmd, src, str(local_path)]
    scp_cmd = _with_sshpass(scp_cmd, use_password_auth, ssh_password, f"[sync] scp {remote_path} -> {local_path}")
    if scp_cmd is None:
        return False
    return _run_cmd(scp_cmd, f"[sync] scp {remote_path} -> {local_path}") is not None


def _get_latest_remote_toml(
    remote_host: str,
    ssh_port: int,
    remote_repo_root: str,
    *,
    use_password_auth: bool = False,
    ssh_password: str = "",
) -> Optional[str]:
    autosave_dir = str(Path(remote_repo_root) / "config" / "autosave")
    remote_cmd = f"ls -1t {shlex.quote(autosave_dir)}/*.toml 2>/dev/null | head -n1"
    result = _ssh(
        remote_host,
        ssh_port,
        remote_cmd,
        "[sync-config] detect latest main toml",
        use_password_auth=use_password_auth,
        ssh_password=ssh_password,
    )
    if result is None:
        return None
    path = result.stdout.strip()
    return path or None


def _sync_config_from_main(
    toml_path: str,
    remote_host: str,
    ssh_port: int,
    remote_repo_root: str,
    sync_main_toml: str,
    sync_keys: list,
    *,
    use_password_auth: bool = False,
    ssh_password: str = "",
) -> Tuple[bool, str]:
    local_config = toml.load(toml_path)

    candidate_paths = []
    if sync_main_toml:
        candidate_paths.append(_resolve_remote_path(sync_main_toml, remote_repo_root))

    latest_toml_path = _get_latest_remote_toml(
        remote_host,
        ssh_port,
        remote_repo_root,
        use_password_auth=use_password_auth,
        ssh_password=ssh_password,
    )
    if latest_toml_path:
        candidate_paths.append(latest_toml_path)

    local_toml_name = Path(toml_path).name
    candidate_paths.extend(
        [
            str(Path(remote_repo_root) / "config" / "autosave" / "distributed-main-latest.toml"),
            str(Path(remote_repo_root) / "config" / "autosave" / local_toml_name),
            str(Path(remote_repo_root) / "config" / "default.toml"),
            str(Path(remote_repo_root) / "config" / "lora.toml"),
        ]
    )

    # Deduplicate while preserving order.
    dedup_paths = []
    seen = set()
    for p in candidate_paths:
        if p not in seen:
            dedup_paths.append(p)
            seen.add(p)

    if len(dedup_paths) == 0:
        return False, f"无法构建主机 toml 候选路径。remote_host={remote_host}, remote_repo_root={remote_repo_root}"

    main_config = None
    used_toml_path = None
    errors = []
    for candidate in dedup_paths:
        path_type = _remote_path_type(
            remote_host,
            ssh_port,
            candidate,
            use_password_auth=use_password_auth,
            ssh_password=ssh_password,
        )
        if path_type != "file":
            errors.append(f"{candidate} ({path_type})")
            continue

        text, read_err = _read_remote_text_file(
            remote_host,
            ssh_port,
            candidate,
            use_password_auth=use_password_auth,
            ssh_password=ssh_password,
        )
        if text is None:
            errors.append(f"{candidate} (read failed: {read_err})")
            continue

        try:
            parsed = toml.loads(text)
        except Exception as e:
            errors.append(f"{candidate} (toml parse failed: {e})")
            continue

        main_config = parsed
        used_toml_path = candidate
        break

    if main_config is None:
        return (
            False,
            "无法读取主机 toml 配置。请检查 sync_main_repo_dir / sync_main_toml / SSH 密码权限。"
            f"remote_host={remote_host}, remote_repo_root={remote_repo_root}, attempts={'; '.join(errors)}",
        )

    log.info(f"[sync-config] use main toml: {used_toml_path}")

    sync_all = any(str(k).strip().lower() in {"*", "__all__", "all"} for k in sync_keys)
    keys_to_sync = list(main_config.keys()) if sync_all else sync_keys
    if sync_all:
        log.info(f"[sync-config] full sync mode enabled: syncing all {len(keys_to_sync)} top-level keys")
    else:
        seen_keys = {str(k).strip().lower() for k in keys_to_sync}
        for required_key in WORKER_REQUIRED_SYNC_CONFIG_KEYS:
            if required_key.lower() in seen_keys:
                continue
            if required_key not in main_config:
                continue
            keys_to_sync.append(required_key)
            seen_keys.add(required_key.lower())
            log.info(f"[sync-config] append required key for worker launch consistency: {required_key}")

    changed = 0
    for key in keys_to_sync:
        if key not in main_config:
            log.warning(f"[sync-config] key not found on main config: {key}")
            continue
        old_val = local_config.get(key)
        new_val = main_config.get(key)
        if old_val != new_val:
            local_config[key] = new_val
            changed += 1
            log.info(f"[sync-config] {key}: {old_val} -> {new_val}")
        else:
            log.info(f"[sync-config] {key}: unchanged ({new_val})")

    for key, fallback in WORKER_SYNC_CONFIG_FALLBACK_WHEN_MAIN_MISSING.items():
        if key in main_config or key not in local_config:
            continue
        old_val = local_config.get(key)
        if old_val != fallback:
            local_config[key] = fallback
            changed += 1
            log.info(f"[sync-config] {key}: main missing, fallback applied {old_val} -> {fallback}")

    for key in WORKER_SYNC_CONFIG_CLEAR_WHEN_MAIN_MISSING:
        if key in main_config or key not in local_config:
            continue
        old_val = local_config.pop(key)
        changed += 1
        log.info(f"[sync-config] {key}: main missing, cleared stale local value ({old_val})")

    if changed > 0:
        with open(toml_path, "w", encoding="utf-8") as f:
            f.write(toml.dumps(local_config))
        log.info(f"[sync-config] wrote {changed} updated key(s) to {toml_path}")
    else:
        log.info("[sync-config] no key changes required")

    return True, ""


def _sync_missing_assets_from_main(
    toml_path: str,
    remote_host: str,
    ssh_port: int,
    remote_repo_root: str,
    asset_keys: list,
    *,
    use_password_auth: bool = False,
    ssh_password: str = "",
) -> Tuple[bool, str]:
    local_repo_root = base_dir_path()
    try:
        config = toml.load(toml_path)
    except Exception as e:
        return False, f"读取本地训练配置失败: {toml_path} ({e})"

    for key in asset_keys:
        value = config.get(key)
        if not isinstance(value, str) or value.strip() == "":
            continue

        local_path = _resolve_local_path(value, local_repo_root)
        force_refresh_if_exists = str(key).strip().lower() == "resume"
        if local_path.exists() and not force_refresh_if_exists:
            log.info(f"[sync-assets] local exists, skip copy: {key} -> {local_path}")
            continue

        remote_path = _resolve_remote_path(value, remote_repo_root)
        path_type = _remote_path_type(
            remote_host,
            ssh_port,
            remote_path,
            use_password_auth=use_password_auth,
            ssh_password=ssh_password,
        )
        if path_type == "missing":
            return False, f"主机路径不存在，无法同步: {key} -> {remote_path}"
        if path_type == "error":
            return False, f"无法探测主机路径类型: {key} -> {remote_path}"

        if local_path.exists() and force_refresh_if_exists:
            log.info(f"[sync-assets] local exists, force refresh from main: {key} -> {local_path}")
        else:
            log.info(f"[sync-assets] local missing, start sync: {key} -> {local_path}")
        if not _copy_remote_path(
            remote_host,
            ssh_port,
            remote_path,
            local_path,
            path_type,
            use_password_auth=use_password_auth,
            ssh_password=ssh_password,
        ):
            return False, f"同步失败: {key} -> {remote_path}"
        if not local_path.exists():
            return False, f"同步后本地仍不存在: {key} -> {local_path}"
        log.info(f"[sync-assets] synced: {key} -> {local_path}")

    return True, ""


def _ensure_main_distributed_autosave(toml_path: str, machine_rank: int, num_machines: int) -> Tuple[bool, str]:
    if num_machines <= 1 or machine_rank != 0:
        return True, ""

    src = Path(toml_path)
    if not src.exists():
        return False, f"主机分布式 autosave 源文件不存在: {src}"

    autosave_dir = base_dir_path() / "config" / "autosave"
    autosave_dir.mkdir(parents=True, exist_ok=True)

    latest_file = autosave_dir / "distributed-main-latest.toml"
    timestamp_file = autosave_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-distributed-main.toml"
    try:
        shutil.copy2(src, latest_file)
        shutil.copy2(src, timestamp_file)
    except Exception as e:
        return False, f"主机分布式 autosave 写入失败: {e}"

    log.info(f"[sync-config] main distributed autosave updated: {latest_file}")
    log.info(f"[sync-config] main distributed autosave snapshot: {timestamp_file}")
    return True, ""


def _enforce_distributed_output_policy(toml_path: str, machine_rank: int) -> Tuple[bool, str]:
    repo_root = base_dir_path()
    try:
        config = toml.load(toml_path)
    except Exception as e:
        return False, f"读取训练配置失败: {toml_path} ({e})"

    if machine_rank > 0:
        output_dir = _resolve_local_path(str(config.get("output_dir", "./output")), repo_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        marker_path = output_dir / WORKER_OUTPUT_MARKER
        marker_path.touch(exist_ok=True)
        log.info(f"[output-policy] worker marker created: {marker_path}")
        log.info("[output-policy] worker native save is enabled (no checkpoint/save_state override)")

    return True, ""


def run_train(toml_path: str,
              trainer_file: str = "./scripts/train_network.py",
              gpu_ids: Optional[list] = None,
              cpu_threads: Optional[int] = 2,
              distributed_config: Optional[dict] = None):
    log.info(f"Training started with config file / 训练开始，使用配置文件: {toml_path}")
    base_accelerate_args = [
        sys.executable, "-m", "accelerate.commands.launch",  # use -m to avoid python script executable error
        "--num_cpu_threads_per_process", str(cpu_threads),  # cpu threads
        "--quiet",  # silence accelerate error message
    ]

    customize_env = os.environ.copy()
    customize_env["ACCELERATE_DISABLE_RICH"] = "1"
    customize_env["PYTHONUNBUFFERED"] = "1"
    customize_env["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"
    customize_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    distributed_config = distributed_config or {}
    sync_from_main_settings = distributed_config.get("sync_from_main_settings")
    if not isinstance(sync_from_main_settings, dict):
        sync_from_main_settings = {}

    def get_sync_value(key, default):
        value = distributed_config.get(key, None)
        if value is not None:
            return value
        return sync_from_main_settings.get(key, default)

    enable_distributed_training = _to_bool(distributed_config.get("enable_distributed_training"), False)
    num_machines = int(distributed_config.get("num_machines", 1) or 1)
    machine_rank = int(distributed_config.get("machine_rank", 0) or 0)
    main_process_ip = distributed_config.get("main_process_ip")
    main_process_port = int(distributed_config.get("main_process_port", 29500) or 29500)
    nccl_socket_ifname = str(distributed_config.get("nccl_socket_ifname", "") or "").strip()
    gloo_socket_ifname = str(distributed_config.get("gloo_socket_ifname", "") or "").strip()
    sync_config_from_main = _to_bool(get_sync_value("sync_config_from_main", True), True)
    sync_config_keys_from_main = _parse_sync_config_keys(get_sync_value("sync_config_keys_from_main", None))
    sync_missing_assets_from_main = _to_bool(get_sync_value("sync_missing_assets_from_main", True), True)
    sync_asset_keys = _parse_csv(get_sync_value("sync_asset_keys", None), DEFAULT_SYNC_ASSET_KEYS)
    detected_repo_root = os.getcwd()
    sync_main_repo_dir = str(get_sync_value("sync_main_repo_dir", detected_repo_root) or detected_repo_root)
    sync_main_toml = str(
        get_sync_value("sync_main_toml", "./config/autosave/distributed-main-latest.toml")
        or "./config/autosave/distributed-main-latest.toml"
    ).strip()
    sync_ssh_user = str(get_sync_value("sync_ssh_user", "") or "").strip()
    sync_ssh_port = int(get_sync_value("sync_ssh_port", 22) or 22)
    sync_use_password_auth = _to_bool(get_sync_value("sync_use_password_auth", True), True)
    clear_dataset_npz_before_train = _to_bool(distributed_config.get("clear_dataset_npz_before_train"), False)
    sync_ssh_password = str(
        get_sync_value("sync_ssh_password", "") or os.environ.get("MIKAZUKI_SYNC_SSH_PASSWORD", "")
    ).strip()
    num_processes_per_machine = distributed_config.get("num_processes")
    if num_processes_per_machine is None:
        num_processes_per_machine = len(gpu_ids) if gpu_ids else 1
    else:
        num_processes_per_machine = int(num_processes_per_machine)

    # If distributed mode is disabled, always run as single machine.
    if not enable_distributed_training:
        num_machines = 1
        machine_rank = 0
        main_process_ip = ""
        nccl_socket_ifname = ""
        gloo_socket_ifname = ""

    total_num_processes = num_processes_per_machine * num_machines

    if num_machines < 1:
        return APIResponse(status="error", message="num_machines 必须 >= 1")
    if num_processes_per_machine < 1:
        return APIResponse(status="error", message="num_processes 必须 >= 1")
    if num_machines > 1 and not main_process_ip:
        return APIResponse(status="error", message="多机训练时 main_process_ip 不能为空")
    if machine_rank < 0 or machine_rank >= num_machines:
        return APIResponse(status="error", message="machine_rank 超出范围，请检查 machine_rank 与 num_machines")
    if num_machines > 1:
        ok, message = _validate_socket_ifname(nccl_socket_ifname, "NCCL_SOCKET_IFNAME")
        if not ok:
            return APIResponse(status="error", message=message)
        ok, message = _validate_socket_ifname(gloo_socket_ifname, "GLOO_SOCKET_IFNAME")
        if not ok:
            return APIResponse(status="error", message=message)

    if nccl_socket_ifname:
        customize_env["NCCL_SOCKET_IFNAME"] = nccl_socket_ifname
    if gloo_socket_ifname:
        customize_env["GLOO_SOCKET_IFNAME"] = gloo_socket_ifname

    ok, message = _ensure_main_distributed_autosave(
        toml_path=toml_path,
        machine_rank=machine_rank,
        num_machines=num_machines,
    )
    if not ok:
        return APIResponse(status="error", message=f"主机分布式 autosave 失败: {message}")

    is_worker = num_machines > 1 and machine_rank > 0
    remote_host = ""
    if is_worker:
        seen_asset_keys = {str(k).strip().lower() for k in sync_asset_keys}
        for required_asset_key in WORKER_REQUIRED_SYNC_ASSET_KEYS:
            if required_asset_key.lower() in seen_asset_keys:
                continue
            sync_asset_keys.append(required_asset_key)
            seen_asset_keys.add(required_asset_key.lower())
            log.info(f"[sync-assets] append required key for worker resume consistency: {required_asset_key}")

        remote_host = f"{sync_ssh_user}@{main_process_ip}" if sync_ssh_user else str(main_process_ip)
        if sync_use_password_auth and not sync_ssh_password:
            return APIResponse(
                status="error",
                message="已启用密码认证同步，但未提供密码。请在分布式设置填写 sync_ssh_password 或设置环境变量 MIKAZUKI_SYNC_SSH_PASSWORD。",
            )
        if sync_config_from_main:
            log.info("[sync-config] worker sync from main is enabled")
            ok, message = _sync_config_from_main(
                toml_path=toml_path,
                remote_host=remote_host,
                ssh_port=sync_ssh_port,
                remote_repo_root=sync_main_repo_dir,
                sync_main_toml=sync_main_toml,
                sync_keys=sync_config_keys_from_main,
                use_password_auth=sync_use_password_auth,
                ssh_password=sync_ssh_password,
            )
            if not ok:
                return APIResponse(status="error", message=f"配置同步失败: {message}")

        if sync_missing_assets_from_main:
            log.info("[sync-assets] worker missing-assets sync from main is enabled")
            ok, message = _sync_missing_assets_from_main(
                toml_path=toml_path,
                remote_host=remote_host,
                ssh_port=sync_ssh_port,
                remote_repo_root=sync_main_repo_dir,
                asset_keys=sync_asset_keys,
                use_password_auth=sync_use_password_auth,
                ssh_password=sync_ssh_password,
            )
            if not ok:
                return APIResponse(status="error", message=f"资产同步失败: {message}")

        log.info("[dataset-sync] worker checking dataset count mismatch with main")
        ok, message = _sync_datasets_when_count_mismatch_from_main(
            toml_path=toml_path,
            remote_host=remote_host,
            ssh_port=sync_ssh_port,
            remote_repo_root=sync_main_repo_dir,
            use_password_auth=sync_use_password_auth,
            ssh_password=sync_ssh_password,
        )
        if not ok:
            return APIResponse(status="error", message=f"数据集同步失败: {message}")

    if clear_dataset_npz_before_train:
        log.info("[cache-reset] clearing dataset npz cache before launch (enabled by config)")
        ok, message = _clear_dataset_npz_cache(toml_path=toml_path)
        if not ok:
            return APIResponse(status="error", message=f"缓存清理失败: {message}")
    else:
        log.info("[cache-reset] skipped dataset npz cleanup (clear_dataset_npz_before_train=false)")

    if is_worker:
        ok, message = _enforce_distributed_output_policy(toml_path=toml_path, machine_rank=machine_rank)
        if not ok:
            return APIResponse(status="error", message=f"输出策略应用失败: {message}")
    else:
        log.info("[output-policy] skipped (single-machine or main node)")

    repo_root = base_dir_path()
    runtime_train_config = {}
    try:
        runtime_train_config = toml.load(toml_path)
        if _enforce_tb_only_config(runtime_train_config):
            with open(toml_path, "w", encoding="utf-8") as f:
                toml.dump(runtime_train_config, f)
            log.info("[tensorboard] forced TB-only logging and set logging_dir to output_dir for this run")
    except Exception as e:
        log.warning(f"[runtime-config] failed to parse training config before launch: {toml_path} ({e})")

    resolved_trainer_file = _resolve_trainer_file_from_runtime_config(runtime_train_config, trainer_file)
    if resolved_trainer_file != trainer_file:
        log.info(
            "[sync-config] trainer script synced from main config: "
            f"{trainer_file} -> {resolved_trainer_file}"
        )
        trainer_file = resolved_trainer_file

    world_size_for_batch = max(1, int(total_num_processes))
    staged_resolution_enabled = bool(
        runtime_train_config and _to_bool(runtime_train_config.get(MIXED_RESOLUTION_ENABLE_KEY), False)
    )
    launch_train_batch_override = None

    if runtime_train_config:
        configured_global_batch = _to_int(runtime_train_config.get("train_batch_size", 1), 1)
        ok_batch, per_device_batch, batch_error = _resolve_per_device_batch_from_global(
            configured_global_batch, world_size_for_batch
        )
        if not ok_batch:
            return APIResponse(status="error", message=f"训练批大小配置错误: {batch_error}")
        launch_train_batch_override = int(per_device_batch)

        grad_accum_steps = _to_int(runtime_train_config.get("gradient_accumulation_steps", 1), 1)
        if grad_accum_steps <= 0:
            grad_accum_steps = 1
        world_effective_batch = int(configured_global_batch) * int(grad_accum_steps)
        log.info(
            "[batch-semantics] user_global_batch=%s world_size=%s per_device_batch=%s grad_accum=%s world_effective_batch=%s",
            int(configured_global_batch),
            int(world_size_for_batch),
            int(per_device_batch),
            int(grad_accum_steps),
            int(world_effective_batch),
        )

    if runtime_train_config:
        guard_ok, guard_message = _validate_resume_launch_guard(runtime_train_config, repo_root)
        if not guard_ok:
            log.warning(f"[resume-guard] {guard_message}")
            return APIResponse(status="error", message=guard_message)

    tensorboard_run_dir = _resolve_tensorboard_run_dir_from_config(runtime_train_config, repo_root) if runtime_train_config else None
    tensorboard_run_dir_existed_before = bool(tensorboard_run_dir and tensorboard_run_dir.exists())
    tensorboard_event_snapshot = _snapshot_tensorboard_event_files(tensorboard_run_dir)
    if tensorboard_run_dir is not None:
        log.info(
            f"[tensorboard] resolved run dir: {tensorboard_run_dir} "
            f"(resume_merge={'yes' if tensorboard_run_dir_existed_before else 'no'})"
        )

    if gpu_ids:
        customize_env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        log.info(f"Using GPU(s) / 使用 GPU: {gpu_ids}")

    launch_args = []
    if total_num_processes > 1 or num_machines > 1:
        launch_args += ["--multi_gpu", "--num_processes", str(total_num_processes)]
        if num_machines > 1:
            launch_args += [
                "--num_machines", str(num_machines),
                "--machine_rank", str(machine_rank),
                "--main_process_ip", str(main_process_ip),
                "--main_process_port", str(main_process_port),
            ]
            log.info(
                f"Distributed launch enabled / 启用跨机分布式: "
                f"num_machines={num_machines}, machine_rank={machine_rank}, "
                f"main_process_ip={main_process_ip}, main_process_port={main_process_port}, "
                f"num_processes_per_machine={num_processes_per_machine}, total_num_processes={total_num_processes}"
            )
        if sys.platform == "win32":
            customize_env["USE_LIBUV"] = "0"
            launch_args += ["--rdzv_backend", "c10d"]

    args = None
    mixed_resolution_plan = None
    if staged_resolution_enabled:
        mixed_resolution_plan, mixed_plan_error = _build_mixed_resolution_plan(
            runtime_train_config,
            toml_path,
            trainer_file,
            num_processes_for_epoch_calc=world_size_for_batch,
        )
        if mixed_plan_error:
            return APIResponse(status="error", message=f"阶段分辨率训练配置错误: {mixed_plan_error}")

        if mixed_resolution_plan is not None:
            mixed_resolution_plan["cpu_threads"] = int(cpu_threads)
            mixed_resolution_plan["launch_args"] = list(launch_args)
            mixed_resolution_plan["repo_root"] = str(repo_root)

            mixed_plan_file = Path(toml_path).with_name(f"{Path(toml_path).stem}-staged-plan.json")
            with open(mixed_plan_file, "w", encoding="utf-8") as f:
                json.dump(mixed_resolution_plan, f, ensure_ascii=False, indent=2)

            log.info(
                "[staged-resolution] enabled: "
                f"base={mixed_resolution_plan['base_resolution']}, "
                f"world_size={mixed_resolution_plan.get('world_size')}, "
                f"base_batch_global={mixed_resolution_plan.get('base_batch_size_global', mixed_resolution_plan.get('base_batch_size'))}, "
                f"base_batch_per_device={mixed_resolution_plan.get('base_batch_size_per_device', mixed_resolution_plan.get('base_batch_size'))}, "
                f"base_grad_accum={mixed_resolution_plan.get('base_gradient_accumulation_steps')}, "
                f"base_epochs={mixed_resolution_plan['base_epochs']}, "
                f"ratio_sum_percent={mixed_resolution_plan.get('configured_ratio_sum_percent')}, "
                f"base_save_every_n_epochs={mixed_resolution_plan.get('save_every_n_epochs')}, "
                f"preview_enabled={'yes' if mixed_resolution_plan.get('preview_enabled') else 'no'}, "
                f"sample_schedule_enabled={'yes' if mixed_resolution_plan.get('use_sample_epoch_schedule') else 'no'}, "
                f"base_sample_every_n_epochs={mixed_resolution_plan.get('base_sample_every_n_epochs')}, "
                f"total_epochs={mixed_resolution_plan['total_mixed_epochs']}, "
                f"total_steps={mixed_resolution_plan['total_mixed_steps']}"
            )
            for phase in mixed_resolution_plan["phases"]:
                log.info(
                    "[staged-resolution] phase %s: res=%s ratio_percent=%s batch_global=%s batch_per_device=%s grad_accum=%s "
                    "save_every_n_epochs=%s sample_every_n_epochs=%s "
                    "raw_epochs=%s epochs=%s rounding_multiple=%s batches/epoch=%s steps/epoch=%s "
                    "phase_steps=%s target_max_steps=%s target_epoch_end=%s cache_rebuild=%s toml=%s",
                    phase["phase_index"],
                    phase["resolution"],
                    phase.get("ratio_percent"),
                    phase.get("batch_size_global", phase.get("batch_size")),
                    phase.get("batch_size_per_device", phase.get("batch_size")),
                    phase.get("gradient_accumulation_steps"),
                    phase.get("save_every_n_epochs"),
                    phase.get("sample_every_n_epochs"),
                    phase.get("raw_epochs"),
                    phase["epochs"],
                    phase.get("epoch_rounding_multiple"),
                    phase.get("batches_per_epoch"),
                    phase["steps_per_epoch"],
                    phase["phase_steps"],
                    phase["target_max_train_steps"],
                    phase["target_epoch_end"],
                    "yes" if phase["clear_cache_before_start"] else "no",
                    phase["toml_path"],
                )
            args = [
                sys.executable,
                "-m",
                "mikazuki.mixed_resolution_runner",
                "--plan-file",
                str(mixed_plan_file),
            ]

    if args is None:
        launch_cli_overrides = []
        if launch_train_batch_override is not None:
            launch_cli_overrides += ["--train_batch_size", str(int(launch_train_batch_override))]

        args = list(base_accelerate_args) + list(launch_args) + [
            trainer_file,
            "--config_file",
            toml_path,
        ] + launch_cli_overrides

    mesh_iface = ""
    if num_machines > 1:
        mesh_iface = _pick_training_mesh_iface(nccl_socket_ifname, gloo_socket_ifname, str(main_process_ip or ""))
        if mesh_iface:
            customize_env["MIKAZUKI_MESH_NET_IFACE"] = mesh_iface
            log.debug(
                f"[mesh-net] enabled in trainer postfix: iface={mesh_iface}, "
                "window=30s, metric=iops, format='r/w:2.1k/2.2k'"
            )
        else:
            log.warning("[mesh-net] distributed training detected but unable to resolve local training interface")

    if not (task := tm.create_task(args, customize_env)):
        return APIResponse(status="error", message="Failed to create task / 无法创建训练任务")

    def _run():
        try:
            run_started_at = time.time()
            task.execute()
            result = task.communicate()

            checkpoint_generated = _has_new_checkpoint_since(runtime_train_config, repo_root, run_started_at) if runtime_train_config else False
            if tensorboard_run_dir is not None and not checkpoint_generated:
                _cleanup_tensorboard_records_without_checkpoint(
                    tensorboard_run_dir,
                    tensorboard_run_dir_existed_before,
                    tensorboard_event_snapshot,
                )
            elif tensorboard_run_dir is not None:
                log.info(f"[tensorboard] checkpoint detected, keep run dir: {tensorboard_run_dir}")

            if result.returncode != 0:
                log.error(f"Training failed / 训练失败")
            else:
                log.info(f"Training finished / 训练完成")
        except Exception as e:
            log.error(f"An error occurred when training / 训练出现致命错误: {e}")

    coro = asyncio.to_thread(_run)
    asyncio.create_task(coro)

    return APIResponse(status="success", message=f"Training started / 训练开始 ID: {task.task_id}")
