
import asyncio
import copy
import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import threading
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
WORKER_REQUIRED_SYNC_CONFIG_KEYS = ("model_train_type",)
MODEL_TRAIN_TYPE_TO_TRAINER_FILE = {
    "sd-lora": "./scripts/stable/train_network.py",
    "sdxl-lora": "./scripts/stable/sdxl_train_network.py",
    "sd-dreambooth": "./scripts/stable/train_db.py",
    "sdxl-finetune": "./scripts/stable/sdxl_train.py",
}
WORKER_OUTPUT_MARKER = "THIS_IS_WORKER_NODE_CHECK_MAIN_OUTPUTS"
DATASET_DIR_KEYS = ("train_data_dir", "reg_data_dir")
MESH_NET_MONITOR_INTERVAL_SECONDS = 10
CKPT_EXTENSIONS = {".safetensors", ".ckpt", ".pt"}
TB_EVENT_FILE_GLOB = "events.out.tfevents.*"
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
BATCH_PROBE_MAX_CANDIDATE = 512
BATCH_PROBE_MAX_TRIALS = 7
BATCH_PROBE_TIMEOUT_SECONDS = 600
BATCH_PROBE_OOM_PATTERNS = [
    re.compile(r"cuda out of memory", re.IGNORECASE),
    re.compile(r"cudnn_status_alloc_failed", re.IGNORECASE),
    re.compile(r"out of memory", re.IGNORECASE),
    re.compile(r"allocator.*memory", re.IGNORECASE),
]
BATCH_PROBE_MODEL_MEMORY_PROFILE = {
    # (base_overhead_mib, per_sample_mib_at_1024)
    "sdxl_train_network.py": (2600.0, 1450.0),
    "sdxl_train.py": (3200.0, 1800.0),
    "train_network.py": (1700.0, 780.0),
    "train_db.py": (2100.0, 980.0),
}
BATCH_PROBE_SHARED_MEM_DELTA_THRESHOLD_MIB = 256.0
BATCH_PROBE_SHARED_MEM_ABS_THRESHOLD_MIB = 512.0
BATCH_PROBE_PROCESS_DEDICATED_MIN_MIB = 1024.0
BATCH_PROBE_NEW_PROCESS_MIN_DEDICATED_MIB = 512.0
BATCH_PROBE_DEDICATED_NEAR_FULL_RATIO = 0.92
BATCH_PROBE_DEDICATED_NEAR_FULL_FREE_MIB = 768.0
BATCH_PROBE_MEMORY_SAMPLING_INTERVAL_SECONDS = 0.35
BATCH_PROBE_SHARED_ADAPTER_DELTA_MIN_MIB = 256.0


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


def _build_mixed_resolution_plan(
    config: dict,
    toml_path: str,
    trainer_file: str,
    *,
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

    try:
        base_batch = int(config.get("train_batch_size"))
    except Exception:
        base_batch = 0
    if base_batch <= 0:
        return None, "启用阶段分辨率训练时，train_batch_size 必须大于 0"

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
        "base_batch": int(base_batch),
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

    plan_base = Path(toml_path)
    phase_configs = []
    cumulative_epochs = 0
    cumulative_steps = 0
    previous_side = None

    for idx, (side, ratio, ratio_percent) in enumerate(configured_phases, start=1):
        target_pixels = side * side
        batch_this_phase = max(1, int(math.floor(base_batch * (base_pixels / target_pixels))))

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
            (batch_this_phase * gradient_accumulation_steps_this_phase)
            / (base_batch * base_gradient_accumulation_steps)
        )
        # Raw formula: ceil(base_epochs * phase_ratio * (phase_effective_batch / base_effective_batch))
        raw_epochs_this_phase = int(math.ceil(base_epochs * ratio * effective_batch_ratio))
        # Actual formula: ceil_to_multiple(raw_epochs, lcm(save_every_n_epochs, sample_every_n_epochs_phase))
        epochs_this_phase = _ceil_to_multiple(max(1, raw_epochs_this_phase), epoch_rounding_multiple)
        batches_per_epoch = max(1, int(math.ceil(total_train_images / batch_this_phase)))
        steps_per_epoch = max(1, int(math.ceil(batches_per_epoch / gradient_accumulation_steps_this_phase)))
        steps_this_phase = int(epochs_this_phase * steps_per_epoch)

        cumulative_epochs += int(epochs_this_phase)
        cumulative_steps += int(steps_this_phase)

        phase_toml_path = plan_base.with_name(f"{plan_base.stem}-staged-phase{idx}.toml")
        phase_config = copy.deepcopy(config)
        phase_config[MIXED_RESOLUTION_ENABLE_KEY] = False
        phase_config["resolution"] = f"{side},{side}"
        phase_config["train_batch_size"] = int(batch_this_phase)
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
            f"(({batch_this_phase}*{gradient_accumulation_steps_this_phase}) / "
            f"({base_batch}*{base_gradient_accumulation_steps})))"
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
                "batch_size": int(batch_this_phase),
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
        "base_batch_size": int(base_batch),
        "base_gradient_accumulation_steps": int(base_gradient_accumulation_steps),
        "base_epochs": int(base_epochs),
        "total_train_images_with_repeats": int(total_train_images),
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


def _read_network_iface_stats(iface_name: str) -> Optional[dict]:
    stat_dir = Path("/sys/class/net") / iface_name / "statistics"
    try:
        return {
            "rx_bytes": int((stat_dir / "rx_bytes").read_text(encoding="utf-8").strip()),
            "tx_bytes": int((stat_dir / "tx_bytes").read_text(encoding="utf-8").strip()),
            "rx_packets": int((stat_dir / "rx_packets").read_text(encoding="utf-8").strip()),
            "tx_packets": int((stat_dir / "tx_packets").read_text(encoding="utf-8").strip()),
        }
    except Exception:
        return None


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


def _mesh_network_monitor_loop(
    stop_event: threading.Event,
    iface_name: str,
    machine_rank: int,
    num_machines: int,
    interval_seconds: int,
):
    interval_seconds = max(1, int(interval_seconds))
    begin_stats = _read_network_iface_stats(iface_name)
    if begin_stats is None:
        log.warning(f"[mesh-net] monitor disabled: cannot read interface stats for {iface_name}")
        return

    begin_time = time.time()
    log.info(
        f"[mesh-net] monitor started: iface={iface_name}, rank={machine_rank}/{num_machines - 1}, "
        f"interval={interval_seconds}s"
    )

    while not stop_event.wait(interval_seconds):
        now_stats = _read_network_iface_stats(iface_name)
        if now_stats is None:
            log.warning(f"[mesh-net] stat read failed for iface={iface_name}, skip this round")
            continue

        elapsed = max(time.time() - begin_time, 1e-6)
        in_bytes = max(now_stats["rx_bytes"] - begin_stats["rx_bytes"], 0)
        out_bytes = max(now_stats["tx_bytes"] - begin_stats["tx_bytes"], 0)
        in_packets = max(now_stats["rx_packets"] - begin_stats["rx_packets"], 0)
        out_packets = max(now_stats["tx_packets"] - begin_stats["tx_packets"], 0)

        in_iops = in_packets / elapsed
        out_iops = out_packets / elapsed
        in_gb = in_bytes / 1_000_000_000
        out_gb = out_bytes / 1_000_000_000
        in_gb_s = in_gb / elapsed
        out_gb_s = out_gb / elapsed

        log.info(
            f"[mesh-net] iface={iface_name} avg_in={in_iops:.1f} iops / {in_gb_s:.4f} GB/s (total {in_gb:.3f} GB), "
            f"avg_out={out_iops:.1f} iops / {out_gb_s:.4f} GB/s (total {out_gb:.3f} GB)"
        )

    final_stats = _read_network_iface_stats(iface_name)
    if final_stats is None:
        log.info(f"[mesh-net] monitor stopped: iface={iface_name}")
        return

    elapsed = max(time.time() - begin_time, 1e-6)
    in_bytes = max(final_stats["rx_bytes"] - begin_stats["rx_bytes"], 0)
    out_bytes = max(final_stats["tx_bytes"] - begin_stats["tx_bytes"], 0)
    in_packets = max(final_stats["rx_packets"] - begin_stats["rx_packets"], 0)
    out_packets = max(final_stats["tx_packets"] - begin_stats["tx_packets"], 0)
    in_iops = in_packets / elapsed
    out_iops = out_packets / elapsed
    in_gb = in_bytes / 1_000_000_000
    out_gb = out_bytes / 1_000_000_000
    in_gb_s = in_gb / elapsed
    out_gb_s = out_gb / elapsed
    log.info(
        f"[mesh-net] monitor stopped: iface={iface_name}, "
        f"avg_in={in_iops:.1f} iops / {in_gb_s:.4f} GB/s (total {in_gb:.3f} GB), "
        f"avg_out={out_iops:.1f} iops / {out_gb_s:.4f} GB/s (total {out_gb:.3f} GB)"
    )


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
    logging_dir = str(config.get("logging_dir", "") or "").strip()
    if not logging_dir:
        return False

    log_with = config.get("log_with")
    if log_with is None:
        return True
    return str(log_with).strip().lower() in {"tensorboard", "all"}


def _sanitize_tensorboard_component(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", str(value or "").strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-_.")
    return cleaned or "model"


def _resolve_tensorboard_model_name(config: dict) -> str:
    for key in ("output_name", "log_prefix"):
        value = str(config.get(key, "") or "").strip()
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

    logging_root = _resolve_local_path(str(config.get("logging_dir", "./logs") or "./logs"), repo_root)
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


def _find_first_remote_file(
    remote_host: str,
    ssh_port: int,
    candidates: list,
    *,
    use_password_auth: bool = False,
    ssh_password: str = "",
) -> Optional[str]:
    for path in candidates:
        path_type = _remote_path_type(
            remote_host,
            ssh_port,
            path,
            use_password_auth=use_password_auth,
            ssh_password=ssh_password,
        )
        if path_type == "file":
            return path
    return None


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
        if local_path.exists():
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


def _batch_probe_is_oom(log_text: str) -> bool:
    if not log_text:
        return False
    return any(pattern.search(log_text) for pattern in BATCH_PROBE_OOM_PATTERNS)


def _to_bool_from_config(config: dict, key: str, default=False) -> bool:
    return _to_bool(config.get(key, default), default)


def _is_wsl_platform() -> bool:
    if sys.platform != "linux":
        return False
    if os.environ.get("WSL_INTEROP"):
        return True
    try:
        version = Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
        return "microsoft" in version or "wsl" in version
    except Exception:
        return False


def _detect_runtime_environment_label() -> str:
    if _is_wsl_platform():
        return "wsl"
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "linux":
        return "linux"
    return "unknown"


def _should_probe_shared_gpu_memory() -> bool:
    return sys.platform == "win32" or _is_wsl_platform()


def _to_mib_from_windows_counter(raw_value) -> Optional[float]:
    try:
        value = float(raw_value)
    except Exception:
        return None
    if value < 0:
        return None
    # Win32_PerfFormattedData_GPUPerformanceCounters counters are in bytes.
    return float(value / 1024.0 / 1024.0)


def _extract_pid_from_gpu_counter_name(name: str) -> Optional[int]:
    raw = str(name or "").strip()
    if not raw:
        return None

    patterns = [
        re.compile(r"(?:^|_)pid[_-]?(\d+)(?:_|$)", re.IGNORECASE),
        re.compile(r"(?:^|_)pid(\d+)(?:_|$)", re.IGNORECASE),
    ]
    for pattern in patterns:
        matched = pattern.search(raw)
        if matched:
            try:
                return int(matched.group(1))
            except Exception:
                continue
    return None


def _query_windows_gpu_process_memory_mib() -> dict:
    info = {
        "ok": False,
        "processes": {},
        "error": "",
        "raw_count": 0,
    }
    if not _should_probe_shared_gpu_memory():
        info["error"] = "unsupported_platform"
        return info

    ps_exe = "powershell" if sys.platform == "win32" else "powershell.exe"
    ps_script = (
        "$ErrorActionPreference='Stop';"
        "$items=Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory "
        "| Where-Object { $_.Name -and $_.Name -notlike '*_Total' };"
        "if(-not $items){'[]';exit 0};"
        "$items|Select-Object Name,DedicatedUsage,SharedUsage,LocalUsage,NonLocalUsage|ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            [ps_exe, "-NoProfile", "-Command", ps_script],
            text=True,
            capture_output=True,
            timeout=6,
            check=False,
        )
    except Exception as e:
        info["error"] = str(e)
        return info

    if result.returncode != 0:
        info["error"] = (result.stderr or result.stdout or "").strip()[:500]
        return info

    raw = (result.stdout or "").strip().lstrip("\ufeff")
    if not raw:
        info["error"] = "empty_powershell_output"
        return info
    try:
        data = json.loads(raw)
    except Exception as e:
        info["error"] = f"json_parse_failed: {e}"
        return info

    rows = data if isinstance(data, list) else [data]
    if not isinstance(rows, list):
        info["error"] = "unexpected_payload_type"
        return info

    processes = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        pid = _extract_pid_from_gpu_counter_name(str(item.get("Name", "") or ""))
        if pid is None:
            continue

        dedicated_mib = _to_mib_from_windows_counter(item.get("DedicatedUsage"))
        if dedicated_mib is None:
            dedicated_mib = _to_mib_from_windows_counter(item.get("LocalUsage"))
        shared_mib = _to_mib_from_windows_counter(item.get("SharedUsage"))
        if shared_mib is None:
            shared_mib = _to_mib_from_windows_counter(item.get("NonLocalUsage"))

        if dedicated_mib is None and shared_mib is None:
            continue

        existing = processes.get(pid)
        ded_value = float(dedicated_mib or 0.0)
        shared_value = float(shared_mib or 0.0)
        if existing is None:
            processes[pid] = {
                "pid": int(pid),
                "dedicated_mib": ded_value,
                "shared_mib": shared_value,
                "counter_name": str(item.get("Name", "") or ""),
            }
        else:
            existing["dedicated_mib"] = max(float(existing.get("dedicated_mib", 0.0)), ded_value)
            existing["shared_mib"] = max(float(existing.get("shared_mib", 0.0)), shared_value)

    if not processes:
        info["error"] = "no_process_rows_with_pid"
        return info

    info["ok"] = True
    info["processes"] = processes
    info["raw_count"] = len(rows)
    return info


def _query_windows_gpu_adapter_memory_mib() -> dict:
    info = {
        "ok": False,
        "shared_mib": None,
        "dedicated_mib": None,
        "adapter_name": "",
        "error": "",
    }
    if not _should_probe_shared_gpu_memory():
        info["error"] = "unsupported_platform"
        return info

    ps_exe = "powershell" if sys.platform == "win32" else "powershell.exe"
    ps_script = (
        "$ErrorActionPreference='Stop';"
        "$items=Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory "
        "| Where-Object { $_.Name -notlike '*_Total' };"
        "if(-not $items){'{}';exit 0};"
        "$best=$items|Sort-Object -Property DedicatedUsage -Descending|Select-Object -First 1 Name,DedicatedUsage,SharedUsage;"
        "$best|ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            [ps_exe, "-NoProfile", "-Command", ps_script],
            text=True,
            capture_output=True,
            timeout=6,
            check=False,
        )
    except Exception as e:
        info["error"] = str(e)
        return info

    if result.returncode != 0:
        info["error"] = (result.stderr or result.stdout or "").strip()[:500]
        return info

    raw = (result.stdout or "").strip().lstrip("\ufeff")
    if not raw:
        info["error"] = "empty_powershell_output"
        return info

    try:
        data = json.loads(raw)
    except Exception as e:
        info["error"] = f"json_parse_failed: {e}"
        return info

    if isinstance(data, list):
        if not data:
            info["error"] = "no_adapter_rows"
            return info
        data = data[0]
    if not isinstance(data, dict):
        info["error"] = "unexpected_payload_type"
        return info

    shared_mib = _to_mib_from_windows_counter(data.get("SharedUsage"))
    dedicated_mib = _to_mib_from_windows_counter(data.get("DedicatedUsage"))
    if shared_mib is None and dedicated_mib is None:
        info["error"] = "counter_values_unavailable"
        return info

    info.update(
        {
            "ok": True,
            "shared_mib": float(shared_mib or 0.0),
            "dedicated_mib": float(dedicated_mib or 0.0),
            "adapter_name": str(data.get("Name", "") or ""),
        }
    )
    return info


def _query_gpu_compute_process_memory_mib(gpu_ids: Optional[list] = None) -> dict:
    info = {
        "ok": False,
        "processes": {},
        "error": "",
    }
    cmd = ["nvidia-smi"]
    if gpu_ids:
        try:
            cmd.extend(["-i", str(int(gpu_ids[0]))])
        except Exception:
            pass
    cmd.extend(
        [
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=8)
    except Exception as e:
        info["error"] = str(e)
        return info

    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        if "no running processes found" in stderr.lower():
            info["ok"] = True
            return info
        info["error"] = stderr[:500]
        return info

    rows = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    processes = {}
    for line in rows:
        if "no running processes found" in line.lower():
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except Exception:
            continue

        if len(parts) == 2:
            proc_name = ""
            used_raw = parts[1]
        else:
            proc_name = ",".join(parts[1:-1]).strip()
            used_raw = parts[-1]

        dedicated_known = True
        try:
            dedicated_mib = float(used_raw)
            if not math.isfinite(dedicated_mib) or dedicated_mib < 0:
                dedicated_known = False
                dedicated_mib = None
        except Exception:
            dedicated_known = False
            dedicated_mib = None

        existing = processes.get(pid)
        if existing is None:
            processes[pid] = {
                "pid": int(pid),
                "process_name": proc_name,
                "dedicated_mib": (float(dedicated_mib) if dedicated_mib is not None else None),
                "dedicated_known": bool(dedicated_known),
                "dedicated_raw": str(used_raw),
            }
        else:
            if dedicated_mib is not None:
                prev = existing.get("dedicated_mib")
                if prev is None:
                    existing["dedicated_mib"] = float(dedicated_mib)
                else:
                    existing["dedicated_mib"] = max(float(prev), float(dedicated_mib))
                existing["dedicated_known"] = True
            else:
                existing["dedicated_known"] = bool(existing.get("dedicated_known", False))
                existing["dedicated_raw"] = str(existing.get("dedicated_raw", "") or str(used_raw))

    info["ok"] = True
    info["processes"] = processes
    return info


def _is_dedicated_vram_near_full(total_mib, used_mib) -> bool:
    try:
        total = float(total_mib)
        used = float(used_mib)
    except Exception:
        return False

    if not math.isfinite(total) or not math.isfinite(used):
        return False
    if total <= 0 or used <= 0:
        return False

    free_mib = max(0.0, total - used)
    if used / total >= BATCH_PROBE_DEDICATED_NEAR_FULL_RATIO:
        return True
    if free_mib <= BATCH_PROBE_DEDICATED_NEAR_FULL_FREE_MIB:
        return True
    return False


def _query_gpu_memory_info(gpu_ids: Optional[list] = None) -> dict:
    info = {
        "ok": False,
        "index": None,
        "name": "",
        "total_mib": None,
        "used_mib": None,
        "free_mib": None,
        "error": "",
    }
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=8)
        if result.returncode != 0:
            info["error"] = (result.stderr or result.stdout or "").strip()[:500]
            return info
        lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
        if not lines:
            info["error"] = "nvidia-smi returned no gpu rows"
            return info

        rows = []
        for line in lines:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                rows.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "total_mib": int(parts[2]),
                        "used_mib": int(parts[3]),
                    }
                )
            except Exception:
                continue
        if not rows:
            info["error"] = "failed to parse nvidia-smi output"
            return info

        preferred_index = None
        if gpu_ids:
            try:
                preferred_index = int(gpu_ids[0])
            except Exception:
                preferred_index = None

        selected = None
        if preferred_index is not None:
            for row in rows:
                if row["index"] == preferred_index:
                    selected = row
                    break
        if selected is None:
            selected = rows[0]

        free_mib = max(0, int(selected["total_mib"]) - int(selected["used_mib"]))
        info.update(
            {
                "ok": True,
                "index": int(selected["index"]),
                "name": str(selected["name"]),
                "total_mib": int(selected["total_mib"]),
                "used_mib": int(selected["used_mib"]),
                "free_mib": int(free_mib),
            }
        )
        return info
    except Exception as e:
        info["error"] = str(e)
        return info


def _estimate_batch_probe_range(
    probe_base_config: dict,
    trainer_file: str,
    *,
    start_batch: int,
    hard_cap: int,
    gpu_ids: Optional[list] = None,
) -> dict:
    memory = _query_gpu_memory_info(gpu_ids=gpu_ids)
    resolution = _parse_resolution_pair(str(probe_base_config.get("resolution", "") or ""))
    if not memory.get("ok") or resolution is None:
        return {
            "ok": False,
            "range_low": 1,
            "range_high": int(max(1, min(hard_cap, start_batch))),
            "suggested_start_batch": int(max(1, min(hard_cap, start_batch))),
            "search_hard_cap": int(max(1, hard_cap)),
            "memory": memory,
            "reason": "gpu_memory_or_resolution_unavailable",
        }

    trainer_name = Path(str(trainer_file)).name
    base_overhead_mib, per_sample_1024_mib = BATCH_PROBE_MODEL_MEMORY_PROFILE.get(
        trainer_name, (2200.0, 1100.0)
    )
    width, height = resolution
    area_factor = max(0.1, float(width * height) / float(1024 * 1024))
    per_sample_mib = float(per_sample_1024_mib) * area_factor
    overhead_mib = float(base_overhead_mib)

    if not _to_bool_from_config(probe_base_config, "gradient_checkpointing", False):
        per_sample_mib *= 1.25
        overhead_mib *= 1.08
    if str(probe_base_config.get("mixed_precision", "fp16") or "fp16").strip().lower() == "no":
        per_sample_mib *= 1.18
    if not _to_bool_from_config(probe_base_config, "cache_latents", True):
        per_sample_mib *= 1.08
    if not _to_bool_from_config(probe_base_config, "cache_text_encoder_outputs", True):
        per_sample_mib *= 1.05

    total_mib = int(memory.get("total_mib") or 0)
    free_now_mib = int(memory.get("free_mib") or 0)

    if total_mib <= 10 * 1024:
        util = 0.78
        reserve_mib = 1000
    elif total_mib <= 12 * 1024:
        util = 0.82
        reserve_mib = 1200
    elif total_mib <= 16 * 1024:
        util = 0.86
        reserve_mib = 1400
    else:
        util = 0.90
        reserve_mib = 1800

    budget_by_total = int(total_mib * util)
    budget_by_current_free = int(free_now_mib * 0.95)
    usable_budget = max(256, min(budget_by_total, budget_by_current_free) - reserve_mib)
    effective_budget = max(0.0, float(usable_budget) - float(overhead_mib))

    estimated_high = int(math.floor(effective_budget / max(1.0, per_sample_mib)))
    estimated_high = max(1, min(int(hard_cap), estimated_high))
    estimated_low = max(1, int(math.floor(estimated_high * 0.55)))

    # Keep search wide enough for estimation bias while still reducing probe time.
    search_hard_cap = min(int(hard_cap), max(int(start_batch), estimated_high + 2, int(math.ceil(estimated_high * 1.5))))
    suggested_start = int(start_batch)
    if suggested_start < estimated_low:
        suggested_start = estimated_low
    if suggested_start > estimated_high:
        suggested_start = estimated_high
    if suggested_start > search_hard_cap:
        suggested_start = search_hard_cap

    return {
        "ok": True,
        "range_low": int(estimated_low),
        "range_high": int(estimated_high),
        "suggested_start_batch": int(max(1, suggested_start)),
        "search_hard_cap": int(max(1, search_hard_cap)),
        "memory": memory,
        "assumptions": {
            "trainer": trainer_name,
            "resolution": [int(width), int(height)],
            "area_factor": float(round(area_factor, 4)),
            "per_sample_mib": float(round(per_sample_mib, 2)),
            "overhead_mib": float(round(overhead_mib, 2)),
            "usable_budget_mib": int(usable_budget),
        },
    }


def _compute_batch_probe_safe_recommendation(best_batch: int, gpu_memory: dict) -> int:
    best_batch = max(1, int(best_batch))
    if best_batch <= 1:
        return 1

    total_mib = int(gpu_memory.get("total_mib") or 0)
    # Keep stronger headroom on 8~10GB cards where fluctuations are common.
    if total_mib > 0 and total_mib <= 10 * 1024:
        return max(1, best_batch - 1)
    if total_mib > 0 and total_mib <= 16 * 1024:
        if best_batch <= 3:
            return max(1, best_batch - 1)
        return max(1, int(math.floor(best_batch * 0.85)))

    return max(1, int(math.floor(best_batch * 0.9)))


def _batch_probe_tail(log_text: str, *, max_lines: int = 60, max_chars: int = 4000) -> str:
    if not log_text:
        return ""
    lines = log_text.splitlines()
    tail = "\n".join(lines[-max_lines:])
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


def _prepare_batch_probe_base_config(config: dict, trainer_file: str) -> Tuple[Optional[dict], str]:
    probe_config = copy.deepcopy(config)

    pretrained_model = str(probe_config.get("pretrained_model_name_or_path", "") or "").strip()
    if not pretrained_model:
        return None, "缺少底模路径（pretrained_model_name_or_path）"

    train_data_dir = str(probe_config.get("train_data_dir", "") or "").strip()
    if not train_data_dir:
        return None, "缺少训练数据集路径（train_data_dir）"

    resolution_raw = str(probe_config.get("resolution", "") or "").strip()
    if _parse_resolution_pair(resolution_raw) is None:
        return None, f"训练分辨率无效：{resolution_raw}"

    try:
        base_batch = int(probe_config.get("train_batch_size", 1))
    except Exception:
        base_batch = 1
    if base_batch <= 0:
        base_batch = 1

    trainer_name = Path(str(trainer_file)).name
    if trainer_name in {"train_network.py", "sdxl_train_network.py"}:
        if not str(probe_config.get("network_module", "") or "").strip():
            probe_config["network_module"] = "networks.lora"
        if probe_config.get("network_dim") in (None, ""):
            probe_config["network_dim"] = 32
        if probe_config.get("network_alpha") in (None, ""):
            probe_config["network_alpha"] = 32

    probe_config["train_batch_size"] = int(base_batch)
    probe_config[MIXED_RESOLUTION_ENABLE_KEY] = False
    probe_config["max_data_loader_n_workers"] = 0
    probe_config["persistent_data_loader_workers"] = False

    # Keep the real training path but force it to a single short step.
    probe_config["max_train_steps"] = 1
    probe_config.pop("max_train_epochs", None)
    probe_config.pop("resume", None)

    # Disable periodic save/sample; final model save by trainer is cleaned after trial.
    probe_config["save_every_n_epochs"] = 10**9
    probe_config.pop("save_every_n_steps", None)
    probe_config["save_state"] = False
    probe_config["save_state_on_train_end"] = False
    probe_config.pop("save_last_n_epochs_state", None)
    probe_config.pop("save_last_n_steps_state", None)
    probe_config.pop("save_last_n_epochs", None)
    probe_config.pop("save_last_n_steps", None)
    probe_config["enable_preview"] = False
    probe_config.pop("sample_prompts", None)
    probe_config.pop("sample_every_n_epochs", None)
    probe_config.pop("sample_sampler", None)

    # Avoid external tracker failures in probe mode.
    if str(probe_config.get("log_with", "") or "").strip().lower() == "wandb":
        probe_config["log_with"] = "tensorboard"
    probe_config.pop("wandb_api_key", None)

    return probe_config, ""


def _run_single_batch_probe(
    probe_base_config: dict,
    trainer_file: str,
    *,
    batch_size: int,
    trial_index: int,
    gpu_ids: Optional[list] = None,
    cpu_threads: int = 2,
) -> dict:
    repo_root = base_dir_path()
    trial_root = (Path("/tmp") / "mikazuki-batch-probe" / datetime.now().strftime("%Y%m%d-%H%M%S-%f") / f"trial-{trial_index:02d}-b{batch_size}").resolve()
    output_dir = trial_root / "output"
    logging_dir = trial_root / "logs"
    toml_path = trial_root / "probe.toml"

    trial_root.mkdir(parents=True, exist_ok=True)

    probe_config = copy.deepcopy(probe_base_config)
    probe_config["train_batch_size"] = int(batch_size)
    probe_config["output_dir"] = str(output_dir)
    probe_config["logging_dir"] = str(logging_dir)
    probe_config["output_name"] = f"batch-probe-b{batch_size}"

    with open(toml_path, "w", encoding="utf-8") as f:
        f.write(toml.dumps(probe_config))

    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_cpu_threads_per_process",
        str(max(1, int(cpu_threads))),
        "--quiet",
        str(trainer_file),
        "--config_file",
        str(toml_path),
    ]

    env = os.environ.copy()
    env["ACCELERATE_DISABLE_RICH"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    started_at = time.time()
    output_text = ""
    status = "error"
    reason = ""
    return_code = -1
    runtime_env_label = _detect_runtime_environment_label()
    runtime_platform = platform.platform()
    runtime_machine = platform.machine()
    runtime_python = platform.python_version()

    def _fmt_mib(value) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.1f}MiB"
        except Exception:
            return str(value)

    shared_probe_enabled = _should_probe_shared_gpu_memory()
    shared_mem_probe = {}
    shared_mem_baseline_mib = None
    shared_mem_post_mib = None
    shared_mem_delta_mib = None
    probe_shared_peak_mib = None
    probe_shared_delta_peak_mib = None
    probe_dedicated_peak_mib = None
    probe_dedicated_peak_from_device_delta_mib = None
    probe_device_used_peak_mib = None
    probe_device_total_mib = None
    probe_device_used_baseline_mib = None
    probe_dedicated_near_full = False
    probe_attribution_mode = "none"
    tracked_probe_pids = set()
    probe_has_unknown_dedicated = False
    adapter_shared_baseline_mib = None
    adapter_shared_peak_mib = None
    adapter_shared_delta_peak_mib = None

    log.info(
        "[batch-probe] trial=%s batch=%s start env=%s sys_platform=%s os=%s machine=%s py=%s trainer=%s resolution=%s gpu_ids=%s",
        trial_index,
        int(batch_size),
        runtime_env_label,
        sys.platform,
        runtime_platform,
        runtime_machine,
        runtime_python,
        str(Path(trainer_file).name),
        str(probe_config.get("resolution", "")),
        ",".join(map(str, gpu_ids)) if gpu_ids else "auto",
    )
    log.info(
        "[batch-probe] trial=%s config grad_ckpt=%s mixed_precision=%s cache_latents=%s cache_text=%s",
        trial_index,
        str(probe_config.get("gradient_checkpointing", "")),
        str(probe_config.get("mixed_precision", "")),
        str(probe_config.get("cache_latents", "")),
        str(probe_config.get("cache_text_encoder_outputs", "")),
    )

    baseline_gpu_memory = _query_gpu_memory_info(gpu_ids=gpu_ids)
    if baseline_gpu_memory.get("ok"):
        probe_device_total_mib = float(baseline_gpu_memory.get("total_mib") or 0.0)
        probe_device_used_peak_mib = float(baseline_gpu_memory.get("used_mib") or 0.0)
        probe_device_used_baseline_mib = float(baseline_gpu_memory.get("used_mib") or 0.0)
        log.info(
            "[batch-probe] trial=%s baseline gpu=%s(%s) used=%s total=%s free=%s",
            trial_index,
            str(baseline_gpu_memory.get("name", "")),
            str(baseline_gpu_memory.get("index", "")),
            _fmt_mib(baseline_gpu_memory.get("used_mib")),
            _fmt_mib(baseline_gpu_memory.get("total_mib")),
            _fmt_mib(baseline_gpu_memory.get("free_mib")),
        )
    else:
        log.info(
            "[batch-probe] trial=%s baseline gpu query failed: %s",
            trial_index,
            str(baseline_gpu_memory.get("error", "")),
        )

    baseline_compute_memory = _query_gpu_compute_process_memory_mib(gpu_ids=gpu_ids)
    baseline_compute_map = baseline_compute_memory.get("processes", {}) if baseline_compute_memory.get("ok") else {}

    baseline_shared_process_memory = {
        "ok": False,
        "processes": {},
        "error": "disabled",
    }
    baseline_shared_map = {}
    if shared_probe_enabled:
        baseline_shared_process_memory = _query_windows_gpu_process_memory_mib()
        if baseline_shared_process_memory.get("ok"):
            baseline_shared_map = baseline_shared_process_memory.get("processes", {})
        baseline_adapter_memory = _query_windows_gpu_adapter_memory_mib()
        if baseline_adapter_memory.get("ok"):
            adapter_shared_baseline_mib = float(baseline_adapter_memory.get("shared_mib") or 0.0)
            adapter_shared_peak_mib = float(adapter_shared_baseline_mib)

    def _sample_probe_memory(launcher_pid: int):
        nonlocal probe_device_total_mib
        nonlocal probe_device_used_peak_mib
        nonlocal probe_dedicated_peak_mib
        nonlocal probe_shared_peak_mib
        nonlocal probe_shared_delta_peak_mib
        nonlocal probe_attribution_mode
        nonlocal probe_has_unknown_dedicated
        nonlocal adapter_shared_peak_mib
        nonlocal adapter_shared_delta_peak_mib

        device_memory = _query_gpu_memory_info(gpu_ids=gpu_ids)
        if device_memory.get("ok"):
            used_mib = float(device_memory.get("used_mib") or 0.0)
            total_mib = float(device_memory.get("total_mib") or 0.0)
            if probe_device_total_mib is None:
                probe_device_total_mib = total_mib
            if probe_device_used_peak_mib is None:
                probe_device_used_peak_mib = used_mib
            else:
                probe_device_used_peak_mib = max(probe_device_used_peak_mib, used_mib)

        compute_memory = _query_gpu_compute_process_memory_mib(gpu_ids=gpu_ids)
        current_compute_map = compute_memory.get("processes", {}) if compute_memory.get("ok") else {}

        candidate_pids = set()
        if launcher_pid in current_compute_map:
            candidate_pids.add(int(launcher_pid))

        if baseline_compute_memory.get("ok"):
            for pid, item in current_compute_map.items():
                if pid in baseline_compute_map:
                    continue
                dedicated_known = bool(item.get("dedicated_known", True))
                dedicated_mib_raw = item.get("dedicated_mib")
                dedicated_mib = float(dedicated_mib_raw or 0.0) if dedicated_mib_raw is not None else 0.0
                if (not dedicated_known) or dedicated_mib >= BATCH_PROBE_NEW_PROCESS_MIN_DEDICATED_MIB:
                    candidate_pids.add(int(pid))

        if candidate_pids:
            tracked_probe_pids.update(candidate_pids)
            if probe_attribution_mode == "none":
                probe_attribution_mode = "nvidia-compute"

        if tracked_probe_pids and current_compute_map:
            dedicated_now = 0.0
            has_known_dedicated_sample = False
            for pid in tracked_probe_pids:
                item = current_compute_map.get(pid)
                if item is None:
                    continue
                dedicated_known = bool(item.get("dedicated_known", True))
                dedicated_mib_raw = item.get("dedicated_mib")
                if dedicated_known and dedicated_mib_raw is not None:
                    has_known_dedicated_sample = True
                    dedicated_now += max(0.0, float(dedicated_mib_raw))
                else:
                    probe_has_unknown_dedicated = True
            if has_known_dedicated_sample:
                if probe_dedicated_peak_mib is None:
                    probe_dedicated_peak_mib = dedicated_now
                else:
                    probe_dedicated_peak_mib = max(probe_dedicated_peak_mib, dedicated_now)

        if not shared_probe_enabled:
            return

        adapter_memory = _query_windows_gpu_adapter_memory_mib()
        if adapter_memory.get("ok"):
            adapter_shared_now = float(adapter_memory.get("shared_mib") or 0.0)
            if adapter_shared_peak_mib is None:
                adapter_shared_peak_mib = adapter_shared_now
            else:
                adapter_shared_peak_mib = max(adapter_shared_peak_mib, adapter_shared_now)

            if adapter_shared_baseline_mib is not None:
                adapter_delta_now = max(0.0, adapter_shared_now - adapter_shared_baseline_mib)
                if adapter_shared_delta_peak_mib is None:
                    adapter_shared_delta_peak_mib = adapter_delta_now
                else:
                    adapter_shared_delta_peak_mib = max(adapter_shared_delta_peak_mib, adapter_delta_now)

        shared_process_memory = _query_windows_gpu_process_memory_mib()
        if not shared_process_memory.get("ok"):
            return
        current_shared_map = shared_process_memory.get("processes", {})

        if not tracked_probe_pids and launcher_pid in current_shared_map:
            tracked_probe_pids.add(int(launcher_pid))
            if probe_attribution_mode == "none":
                probe_attribution_mode = "launcher-fallback"
        if not tracked_probe_pids and baseline_shared_process_memory.get("ok"):
            for pid, item in current_shared_map.items():
                if pid in baseline_shared_map:
                    continue
                dedicated_mib = float(item.get("dedicated_mib") or 0.0)
                if dedicated_mib >= BATCH_PROBE_NEW_PROCESS_MIN_DEDICATED_MIB:
                    tracked_probe_pids.add(int(pid))
            if tracked_probe_pids and probe_attribution_mode == "none":
                probe_attribution_mode = "counter-delta-fallback"
        if not tracked_probe_pids:
            return

        shared_now = 0.0
        baseline_shared = 0.0
        for pid in tracked_probe_pids:
            current_item = current_shared_map.get(pid)
            if current_item is not None:
                shared_now += max(0.0, float(current_item.get("shared_mib") or 0.0))
            baseline_item = baseline_shared_map.get(pid)
            if baseline_item is not None:
                baseline_shared += max(0.0, float(baseline_item.get("shared_mib") or 0.0))

        delta_shared = max(0.0, shared_now - baseline_shared)
        if probe_shared_peak_mib is None:
            probe_shared_peak_mib = shared_now
        else:
            probe_shared_peak_mib = max(probe_shared_peak_mib, shared_now)
        if probe_shared_delta_peak_mib is None:
            probe_shared_delta_peak_mib = delta_shared
        else:
            probe_shared_delta_peak_mib = max(probe_shared_delta_peak_mib, delta_shared)

    process = None
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        launcher_pid = int(process.pid)
        deadline = started_at + BATCH_PROBE_TIMEOUT_SECONDS

        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                try:
                    process.kill()
                except Exception:
                    pass
                stdout_text, stderr_text = process.communicate(timeout=5)
                output_text = (stdout_text or "") + "\n" + (stderr_text or "")
                return_code = int(process.returncode or -1)
                status = "timeout"
                reason = f"timeout>{BATCH_PROBE_TIMEOUT_SECONDS}s"
                log.info("[batch-probe] trial=%s timeout reached, process killed", trial_index)
                break

            try:
                stdout_text, stderr_text = process.communicate(
                    timeout=max(0.05, min(BATCH_PROBE_MEMORY_SAMPLING_INTERVAL_SECONDS, remaining))
                )
                output_text = (stdout_text or "") + "\n" + (stderr_text or "")
                return_code = int(process.returncode or 0)
                if return_code == 0:
                    status = "success"
                    reason = "ok"
                elif _batch_probe_is_oom(output_text):
                    status = "oom"
                    reason = "oom"
                else:
                    status = "error"
                    reason = f"non-oom error (code={return_code})"
                log.info(
                    "[batch-probe] trial=%s process exited code=%s status=%s",
                    trial_index,
                    return_code,
                    status,
                )
                break
            except subprocess.TimeoutExpired:
                _sample_probe_memory(launcher_pid)

        _sample_probe_memory(launcher_pid)
    except Exception as e:
        if process is not None:
            try:
                process.kill()
            except Exception:
                pass
        status = "error"
        reason = f"probe exception: {e}"
    finally:
        try:
            shutil.rmtree(trial_root, ignore_errors=True)
        except Exception:
            pass

    tracked_probe_pids_list = sorted(int(pid) for pid in tracked_probe_pids)
    shared_metrics_source = "none"
    use_process_shared_metrics = bool(tracked_probe_pids_list and probe_shared_peak_mib is not None)
    if use_process_shared_metrics:
        process_shared_baseline = float(
            sum(max(0.0, float((baseline_shared_map.get(pid) or {}).get("shared_mib") or 0.0)) for pid in tracked_probe_pids_list)
        )
        process_shared_post = float(max(0.0, probe_shared_peak_mib))
        process_shared_delta = float(
            max(0.0, probe_shared_delta_peak_mib or (process_shared_post - process_shared_baseline))
        )
        adapter_delta_for_switch = None
        if adapter_shared_peak_mib is not None and adapter_shared_baseline_mib is not None:
            adapter_delta_for_switch = float(
                max(
                    0.0,
                    adapter_shared_delta_peak_mib
                    if adapter_shared_delta_peak_mib is not None
                    else (adapter_shared_peak_mib - adapter_shared_baseline_mib),
                )
            )
        if (
            adapter_delta_for_switch is not None
            and process_shared_delta < 1.0
            and process_shared_post < 1.0
            and adapter_delta_for_switch >= BATCH_PROBE_SHARED_ADAPTER_DELTA_MIN_MIB
        ):
            use_process_shared_metrics = False
        else:
            shared_mem_baseline_mib = process_shared_baseline
            shared_mem_post_mib = process_shared_post
            shared_mem_delta_mib = process_shared_delta
            shared_metrics_source = "process"

    if not use_process_shared_metrics and tracked_probe_pids_list and adapter_shared_peak_mib is not None and adapter_shared_baseline_mib is not None:
        # WSL may expose Linux compute PIDs but Windows-side shared-memory counters use host PIDs.
        # In that case, use adapter shared-memory delta as fallback only when probe PIDs are known.
        shared_mem_baseline_mib = float(max(0.0, adapter_shared_baseline_mib))
        shared_mem_post_mib = float(max(0.0, adapter_shared_peak_mib))
        shared_mem_delta_mib = float(
            max(
                0.0,
                adapter_shared_delta_peak_mib
                if adapter_shared_delta_peak_mib is not None
                else (shared_mem_post_mib - shared_mem_baseline_mib),
            )
        )
        shared_metrics_source = "adapter_fallback"

    if probe_device_used_peak_mib is not None and probe_device_used_baseline_mib is not None:
        probe_dedicated_peak_from_device_delta_mib = float(
            max(0.0, probe_device_used_peak_mib - probe_device_used_baseline_mib)
        )

    probe_dedicated_near_full = _is_dedicated_vram_near_full(probe_device_total_mib, probe_device_used_peak_mib)
    probe_shared_triggered = (
        shared_mem_post_mib is not None
        and shared_mem_delta_mib is not None
        and (
            shared_mem_post_mib >= BATCH_PROBE_SHARED_MEM_ABS_THRESHOLD_MIB
            or shared_mem_delta_mib >= BATCH_PROBE_SHARED_MEM_DELTA_THRESHOLD_MIB
        )
    )
    probe_dedicated_significant = False
    if probe_dedicated_peak_mib is not None:
        probe_dedicated_significant = probe_dedicated_peak_mib >= BATCH_PROBE_PROCESS_DEDICATED_MIN_MIB
    elif probe_has_unknown_dedicated and probe_dedicated_peak_from_device_delta_mib is not None:
        probe_dedicated_significant = probe_dedicated_peak_from_device_delta_mib >= BATCH_PROBE_PROCESS_DEDICATED_MIN_MIB

    probe_shared_attributed = bool(tracked_probe_pids_list)

    if (
        status == "success"
        and probe_shared_attributed
        and probe_shared_triggered
        and probe_dedicated_significant
        and probe_dedicated_near_full
    ):
        status = "shared_mem"
        reason = (
            "probe_shared_gpu_memory_detected"
            f"(baseline={shared_mem_baseline_mib:.1f}MiB,post={shared_mem_post_mib:.1f}MiB,"
            f"delta={shared_mem_delta_mib:.1f}MiB,probe_dedicated_peak={float(probe_dedicated_peak_mib or 0.0):.1f}MiB,"
            f"gpu_peak={float(probe_device_used_peak_mib or 0.0):.1f}/{float(probe_device_total_mib or 0.0):.1f}MiB)"
        )

    shared_mem_probe = {
        "enabled": bool(shared_probe_enabled),
        "process_counter_ok": bool(baseline_shared_process_memory.get("ok")) if shared_probe_enabled else False,
        "process_counter_error": str(baseline_shared_process_memory.get("error", "")) if shared_probe_enabled else "disabled",
        "compute_counter_ok": bool(baseline_compute_memory.get("ok")),
        "compute_counter_error": str(baseline_compute_memory.get("error", "")),
        "attribution_mode": probe_attribution_mode,
        "shared_metrics_source": shared_metrics_source,
        "tracked_pids": tracked_probe_pids_list,
        "shared_peak_mib": probe_shared_peak_mib,
        "shared_delta_peak_mib": probe_shared_delta_peak_mib,
        "adapter_shared_baseline_mib": adapter_shared_baseline_mib,
        "adapter_shared_peak_mib": adapter_shared_peak_mib,
        "adapter_shared_delta_peak_mib": adapter_shared_delta_peak_mib,
        "probe_dedicated_peak_mib": probe_dedicated_peak_mib,
        "probe_dedicated_peak_from_device_delta_mib": probe_dedicated_peak_from_device_delta_mib,
        "probe_has_unknown_dedicated": bool(probe_has_unknown_dedicated),
        "device_used_peak_mib": probe_device_used_peak_mib,
        "device_used_baseline_mib": probe_device_used_baseline_mib,
        "device_total_mib": probe_device_total_mib,
        "dedicated_near_full": bool(probe_dedicated_near_full),
        "shared_triggered": bool(probe_shared_triggered),
    }

    return {
        "batch_size": int(batch_size),
        "status": status,
        "reason": reason,
        "return_code": int(return_code),
        "elapsed_sec": round(max(0.0, time.time() - started_at), 3),
        "log_tail": _batch_probe_tail(output_text),
        "shared_memory_baseline_mib": shared_mem_baseline_mib,
        "shared_memory_post_mib": shared_mem_post_mib,
        "shared_memory_delta_mib": shared_mem_delta_mib,
        "shared_memory_probe": shared_mem_probe or {},
        "probe_dedicated_peak_mib": probe_dedicated_peak_mib,
        "probe_dedicated_peak_from_device_delta_mib": probe_dedicated_peak_from_device_delta_mib,
        "probe_has_unknown_dedicated": bool(probe_has_unknown_dedicated),
        "probe_device_used_peak_mib": probe_device_used_peak_mib,
        "probe_device_used_baseline_mib": probe_device_used_baseline_mib,
        "probe_device_total_mib": probe_device_total_mib,
        "probe_dedicated_near_full": bool(probe_dedicated_near_full),
        "probe_shared_attributed": bool(probe_shared_attributed),
        "probe_attribution_mode": probe_attribution_mode,
        "probe_shared_metrics_source": shared_metrics_source,
        "probe_tracked_pids": tracked_probe_pids_list,
    }


def probe_recommended_batch_size(
    config: dict,
    trainer_file: str,
    *,
    gpu_ids: Optional[list] = None,
    cpu_threads: int = 2,
    max_trials: int = BATCH_PROBE_MAX_TRIALS,
    hard_cap: int = BATCH_PROBE_MAX_CANDIDATE,
) -> dict:
    probe_base_config, prep_error = _prepare_batch_probe_base_config(config, trainer_file)
    if probe_base_config is None:
        return {"ok": False, "message": prep_error, "data": {"trials": []}}

    runtime_env_label = _detect_runtime_environment_label()
    log.info(
        "[batch-probe] environment detected: env=%s sys_platform=%s kernel=%s shared_probe_enabled=%s",
        runtime_env_label,
        sys.platform,
        platform.platform(),
        _should_probe_shared_gpu_memory(),
    )

    try:
        start_batch = int(probe_base_config.get("train_batch_size", 1))
    except Exception:
        start_batch = 1
    start_batch = max(1, start_batch)
    hard_cap = max(start_batch, int(hard_cap))
    max_trials = max(1, int(max_trials))

    estimate = _estimate_batch_probe_range(
        probe_base_config,
        trainer_file,
        start_batch=start_batch,
        hard_cap=hard_cap,
        gpu_ids=gpu_ids,
    )
    if estimate.get("ok"):
        start_batch = int(estimate.get("suggested_start_batch", start_batch) or start_batch)
        hard_cap = int(max(start_batch, estimate.get("search_hard_cap", hard_cap) or hard_cap))
        # With memory estimation available, cap trial count for faster response.
        max_trials = min(max_trials, 5)

    trials = []
    trial_index = 0

    def run_candidate(candidate_batch: int) -> dict:
        nonlocal trial_index
        trial_index += 1
        result = _run_single_batch_probe(
            probe_base_config,
            trainer_file,
            batch_size=int(candidate_batch),
            trial_index=trial_index,
            gpu_ids=gpu_ids,
            cpu_threads=cpu_threads,
        )
        trials.append(result)
        log.info(
            "[batch-probe] trial=%s batch=%s status=%s reason=%s elapsed=%.3fs",
            trial_index,
            result["batch_size"],
            result["status"],
            result["reason"],
            result["elapsed_sec"],
        )
        return result

    first = run_candidate(start_batch)
    if first["status"] == "timeout":
        return {
            "ok": False,
            "message": "batch 检测超时，请检查模型/数据路径或减少复杂配置后重试。",
            "data": {
                "trials": trials,
                "start_batch_size": start_batch,
                "resolution": str(probe_base_config.get("resolution", "")),
                "estimated_range": {
                    "low": int(estimate.get("range_low", 1) or 1),
                    "high": int(estimate.get("range_high", 1) or 1),
                },
                "gpu_memory": estimate.get("memory", {}),
            },
        }
    if first["status"] == "error":
        return {
            "ok": False,
            "message": "batch 检测失败（非显存错误），请先修复当前训练配置。",
            "data": {
                "trials": trials,
                "start_batch_size": start_batch,
                "resolution": str(probe_base_config.get("resolution", "")),
                "estimated_range": {
                    "low": int(estimate.get("range_low", 1) or 1),
                    "high": int(estimate.get("range_high", 1) or 1),
                },
                "gpu_memory": estimate.get("memory", {}),
            },
        }

    best_batch = 0
    if first["status"] == "oom":
        est_low = int(estimate.get("range_low", 1) or 1)
        est_high = int(estimate.get("range_high", start_batch - 1) or (start_batch - 1))
        low = max(1, min(start_batch - 1, est_low))
        high = max(low, min(start_batch - 1, est_high))
        if low > high:
            low, high = 1, max(1, start_batch - 1)
        while low <= high and trial_index < max_trials:
            mid = (low + high) // 2
            result = run_candidate(mid)
            if result["status"] == "success":
                best_batch = mid
                low = mid + 1
            elif result["status"] in {"oom", "shared_mem"}:
                high = mid - 1
            else:
                return {
                    "ok": False,
                    "message": "batch 检测中遇到非显存错误，已中止。",
                    "data": {
                        "trials": trials,
                        "start_batch_size": start_batch,
                        "resolution": str(probe_base_config.get("resolution", "")),
                        "estimated_range": {
                            "low": int(estimate.get("range_low", 1) or 1),
                            "high": int(estimate.get("range_high", 1) or 1),
                        },
                        "gpu_memory": estimate.get("memory", {}),
                    },
                }
    else:
        best_batch = start_batch
        failure_batch = None
        current = start_batch

        while trial_index < max_trials and current < hard_cap:
            next_batch = min(hard_cap, max(current + 1, current * 2))
            if next_batch <= current:
                break
            result = run_candidate(next_batch)
            if result["status"] == "success":
                best_batch = next_batch
                current = next_batch
                continue
            if result["status"] in {"oom", "shared_mem"}:
                failure_batch = next_batch
                break
            return {
                "ok": False,
                "message": "batch 检测中遇到非显存错误，已中止。",
                "data": {
                    "trials": trials,
                    "start_batch_size": start_batch,
                    "resolution": str(probe_base_config.get("resolution", "")),
                    "estimated_range": {
                        "low": int(estimate.get("range_low", 1) or 1),
                        "high": int(estimate.get("range_high", 1) or 1),
                    },
                    "gpu_memory": estimate.get("memory", {}),
                },
            }

        if failure_batch is not None and trial_index < max_trials:
            low = best_batch + 1
            high = failure_batch - 1
            while low <= high and trial_index < max_trials:
                mid = (low + high) // 2
                result = run_candidate(mid)
                if result["status"] == "success":
                    best_batch = mid
                    low = mid + 1
                elif result["status"] in {"oom", "shared_mem"}:
                    high = mid - 1
                else:
                    return {
                        "ok": False,
                        "message": "batch 检测中遇到非显存错误，已中止。",
                        "data": {
                            "trials": trials,
                            "start_batch_size": start_batch,
                            "resolution": str(probe_base_config.get("resolution", "")),
                            "estimated_range": {
                                "low": int(estimate.get("range_low", 1) or 1),
                                "high": int(estimate.get("range_high", 1) or 1),
                            },
                            "gpu_memory": estimate.get("memory", {}),
                        },
                    }

    if best_batch <= 0:
        return {
            "ok": False,
            "message": "batch 检测失败：batch_size=1 仍触发显存不足或配置错误。",
            "data": {
                "trials": trials,
                "start_batch_size": start_batch,
                "resolution": str(probe_base_config.get("resolution", "")),
                "estimated_range": {
                    "low": int(estimate.get("range_low", 1) or 1),
                    "high": int(estimate.get("range_high", 1) or 1),
                },
                "gpu_memory": estimate.get("memory", {}),
            },
        }

    recommended_batch = _compute_batch_probe_safe_recommendation(
        best_batch=best_batch,
        gpu_memory=estimate.get("memory", {}),
    )
    has_shared_mem_hits = any((trial.get("status") == "shared_mem") for trial in trials)
    shared_mem_hit_count = sum(1 for trial in trials if trial.get("status") == "shared_mem")
    shared_mem_suffix = ""
    if has_shared_mem_hits:
        shared_mem_suffix = f"（检测到进程级共享显存命中 {shared_mem_hit_count} 次，且独显接近满载，已按效率安全策略降档）"

    return {
        "ok": True,
        "message": (
            f"检测完成：推荐 batch_size={recommended_batch} "
            f"(最大稳定 batch={best_batch}，已自动预留安全余量){shared_mem_suffix}"
        ),
        "data": {
            "recommended_batch_size": int(recommended_batch),
            "max_stable_batch_size": int(best_batch),
            "has_shared_memory_hits": bool(has_shared_mem_hits),
            "shared_memory_hit_count": int(shared_mem_hit_count),
            "start_batch_size": int(start_batch),
            "resolution": str(probe_base_config.get("resolution", "")),
            "max_trials": int(max_trials),
            "hard_cap": int(hard_cap),
            "estimated_range": {
                "low": int(estimate.get("range_low", 1) or 1),
                "high": int(estimate.get("range_high", 1) or 1),
            },
            "gpu_memory": estimate.get("memory", {}),
            "estimate_assumptions": estimate.get("assumptions", {}),
            "trials": trials,
        },
    }


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
    mesh_net_monitor_interval_seconds = _to_int(
        distributed_config.get("mesh_net_monitor_interval_seconds", MESH_NET_MONITOR_INTERVAL_SECONDS),
        MESH_NET_MONITOR_INTERVAL_SECONDS,
    )
    if mesh_net_monitor_interval_seconds < 1:
        mesh_net_monitor_interval_seconds = MESH_NET_MONITOR_INTERVAL_SECONDS
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
    except Exception as e:
        log.warning(f"[runtime-config] failed to parse training config before launch: {toml_path} ({e})")

    resolved_trainer_file = _resolve_trainer_file_from_runtime_config(runtime_train_config, trainer_file)
    if resolved_trainer_file != trainer_file:
        log.info(
            "[sync-config] trainer script synced from main config: "
            f"{trainer_file} -> {resolved_trainer_file}"
        )
        trainer_file = resolved_trainer_file

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
    if runtime_train_config and _to_bool(runtime_train_config.get(MIXED_RESOLUTION_ENABLE_KEY), False):
        mixed_resolution_plan, mixed_plan_error = _build_mixed_resolution_plan(
            runtime_train_config,
            toml_path,
            trainer_file,
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
                f"base_batch={mixed_resolution_plan['base_batch_size']}, "
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
                    "[staged-resolution] phase %s: res=%s ratio_percent=%s batch=%s grad_accum=%s "
                    "save_every_n_epochs=%s sample_every_n_epochs=%s "
                    "raw_epochs=%s epochs=%s rounding_multiple=%s batches/epoch=%s steps/epoch=%s "
                    "phase_steps=%s target_max_steps=%s target_epoch_end=%s cache_rebuild=%s toml=%s",
                    phase["phase_index"],
                    phase["resolution"],
                    phase.get("ratio_percent"),
                    phase["batch_size"],
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
                log.info(
                    "[staged-resolution] phase %s formulas: raw='%s', actual='%s'",
                    phase["phase_index"],
                    phase.get("raw_epochs_formula"),
                    phase.get("actual_epochs_formula"),
                )

            args = [
                sys.executable,
                "-m",
                "mikazuki.mixed_resolution_runner",
                "--plan-file",
                str(mixed_plan_file),
            ]

    if args is None:
        args = list(base_accelerate_args) + list(launch_args) + [
            trainer_file,
            "--config_file",
            toml_path,
        ]

    if not (task := tm.create_task(args, customize_env)):
        return APIResponse(status="error", message="Failed to create task / 无法创建训练任务")

    mesh_iface = ""
    if num_machines > 1:
        mesh_iface = _pick_training_mesh_iface(nccl_socket_ifname, gloo_socket_ifname, str(main_process_ip or ""))
        if mesh_iface:
            log.info(
                f"[mesh-net] enabled for distributed training: iface={mesh_iface}, "
                f"interval={mesh_net_monitor_interval_seconds}s"
            )
        else:
            log.warning("[mesh-net] distributed training detected but unable to resolve local training interface")

    def _run():
        mesh_stop_event = None
        mesh_thread = None
        try:
            run_started_at = time.time()
            task.execute()
            if num_machines > 1 and mesh_iface:
                mesh_stop_event = threading.Event()
                mesh_thread = threading.Thread(
                    target=_mesh_network_monitor_loop,
                    args=(
                        mesh_stop_event,
                        mesh_iface,
                        machine_rank,
                        num_machines,
                        mesh_net_monitor_interval_seconds,
                    ),
                    daemon=True,
                )
                mesh_thread.start()
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
        finally:
            if mesh_stop_event is not None:
                mesh_stop_event.set()
            if mesh_thread is not None and mesh_thread.is_alive():
                mesh_thread.join(timeout=2)

    coro = asyncio.to_thread(_run)
    asyncio.create_task(coro)

    return APIResponse(status="success", message=f"Training started / 训练开始 ID: {task.task_id}")
