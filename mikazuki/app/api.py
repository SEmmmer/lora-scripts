import asyncio
import hashlib
import json
import os
import re
import random

from glob import glob
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import toml
from fastapi import APIRouter, BackgroundTasks, Request

import mikazuki.process as process
from mikazuki import launch_utils
from mikazuki.app.config import app_config
from mikazuki.app.models import (APIResponse, APIResponseFail,
                                 APIResponseSuccess, TaggerInterrogateRequest)
from mikazuki.log import log
from mikazuki.tagger.interrogator import (available_interrogators,
                                          on_interrogate)
from mikazuki.tasks import tm
from mikazuki.utils import train_utils
from mikazuki.utils.devices import printable_devices
from mikazuki.utils.tk_window import (open_directory_selector,
                                      open_file_selector)

router = APIRouter()

avaliable_scripts = [
    "networks/extract_lora_from_models.py",
    "networks/extract_lora_from_dylora.py",
    "networks/merge_lora.py",
    "tools/merge_models.py",
]
cli_arg_key_pattern = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
script_search_roots = [
    launch_utils.base_dir_path() / "scripts" / "stable",
    launch_utils.base_dir_path() / "scripts" / "dev",
    launch_utils.base_dir_path() / "scripts",
]

avaliable_schemas = []
avaliable_presets = []

trainer_mapping = {
    "sd-lora": "./scripts/stable/train_network.py",
    "sdxl-lora": "./scripts/stable/sdxl_train_network.py",

    "sd-dreambooth": "./scripts/stable/train_db.py",
    "sdxl-finetune": "./scripts/stable/sdxl_train.py",

    "sd3-lora": "./scripts/dev/sd3_train_network.py",
    "flux-lora": "./scripts/dev/flux_train_network.py",
    "flux-finetune": "./scripts/dev/flux_train.py",
}


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


def _read_iface_speed_mbps(iface_name: str) -> Optional[int]:
    speed_file = Path("/sys/class/net") / iface_name / "speed"
    try:
        raw = speed_file.read_text(encoding="utf-8").strip()
        value = int(raw)
        if value > 0:
            return value
    except Exception:
        return None
    return None


def _get_network_interface_options():
    options = [{"value": "", "label": "自动选择"}]
    net_root = Path("/sys/class/net")
    if not net_root.exists():
        return options

    for iface in sorted([p.name for p in net_root.iterdir() if p.is_dir()]):
        speed = _read_iface_speed_mbps(iface)
        if speed is None:
            label = f"{iface} (速率未知)"
        else:
            label = f"{iface} ({speed} Mbps)"
        options.append({"value": iface, "label": label})

    return options


async def load_schemas():
    avaliable_schemas.clear()

    schema_dir = os.path.join(os.getcwd(), "mikazuki", "schema")
    schemas = os.listdir(schema_dir)
    network_interface_options = _get_network_interface_options()
    detected_repo_root = os.getcwd()

    def lambda_hash(x):
        return hashlib.md5(x.encode()).hexdigest()

    for schema_name in schemas:
        with open(os.path.join(schema_dir, schema_name), encoding="utf-8") as f:
            content = f.read()
            runtime_prelude = (
                "window.__MIKAZUKI__ = window.__MIKAZUKI__ || {};\n"
                f"window.__MIKAZUKI__.NETWORK_INTERFACE_OPTIONS = {json.dumps(network_interface_options, ensure_ascii=False)};\n"
                f"window.__MIKAZUKI__.SYNC_MAIN_REPO_DIR_DEFAULT = {json.dumps(detected_repo_root, ensure_ascii=False)};\n"
            )
            content = runtime_prelude + content
            avaliable_schemas.append({
                "name": schema_name.rstrip(".ts"),
                "schema": content,
                "hash": lambda_hash(content)
            })


async def load_presets():
    avaliable_presets.clear()

    preset_dir = os.path.join(os.getcwd(), "config", "presets")
    presets = os.listdir(preset_dir)

    for preset_name in presets:
        with open(os.path.join(preset_dir, preset_name), encoding="utf-8") as f:
            content = f.read()
            avaliable_presets.append(toml.loads(content))


def _resolve_available_script(script_name: str) -> Optional[Path]:
    rel_path = Path(script_name.strip())
    if rel_path.is_absolute() or ".." in rel_path.parts:
        return None

    for root in script_search_roots:
        candidate = root / rel_path
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def get_sample_prompts(config: dict) -> Tuple[Optional[str], str]:
    # backward compatibility
    if "sample_prompts" in config and "positive_prompts" not in config:
        return None, config["sample_prompts"]

    train_data_dir = config["train_data_dir"]
    sub_dir = [dir for dir in glob(os.path.join(train_data_dir, '*')) if os.path.isdir(dir)]

    positive_prompts = config.pop('positive_prompts', None)
    negative_prompts = config.pop('negative_prompts', '')
    sample_width = config.pop('sample_width', 512)
    sample_height = config.pop('sample_height', 512)
    sample_cfg = config.pop('sample_cfg', 7)
    sample_seed = config.pop('sample_seed', 2333)
    sample_steps = config.pop('sample_steps', 24)
    randomly_choice_prompt = config.pop('randomly_choice_prompt', False)

    if randomly_choice_prompt:
        if len(sub_dir) != 1:
            raise ValueError('训练数据集下有多个子文件夹，无法启用随机选取 Prompt 功能')

        txt_files = glob(os.path.join(sub_dir[0], '*.txt'))
        if not txt_files:
            raise ValueError('训练数据集路径没有 txt 文件')
        try:
            sample_prompt_file = random.choice(txt_files)
            with open(sample_prompt_file, 'r', encoding='utf-8') as f:
                positive_prompts = f.read()
        except IOError:
            log.error(f"读取 {sample_prompt_file} 文件失败")

    return positive_prompts, f'{positive_prompts} --n {negative_prompts}  --w {sample_width} --h {sample_height} --l {sample_cfg}  --s {sample_steps}  --d {sample_seed}'


@router.post("/run")
async def create_toml_file(request: Request):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    toml_file = os.path.join(os.getcwd(), f"config", "autosave", f"{timestamp}.toml")
    json_data = await request.body()

    config: dict = json.loads(json_data.decode("utf-8"))
    train_utils.fix_config_types(config)

    gpu_ids = config.pop("gpu_ids", None)
    enable_distributed_training = config.pop("enable_distributed_training", None)
    num_machines = config.pop("num_machines", 1)
    machine_rank = config.pop("machine_rank", 0)
    sync_from_main_settings = config.pop("sync_from_main_settings", {})
    if not isinstance(sync_from_main_settings, dict):
        sync_from_main_settings = {}

    def pop_sync_value(key, default):
        if key in config:
            return config.pop(key)
        return sync_from_main_settings.get(key, default)

    if enable_distributed_training is None:
        distributed_enabled = _to_int(num_machines, 1) > 1
    else:
        distributed_enabled = _to_bool(enable_distributed_training, False)
    distributed_config = {
        "enable_distributed_training": distributed_enabled,
        "num_processes": config.pop("num_processes", None),
        "num_machines": num_machines,
        "machine_rank": machine_rank,
        "main_process_ip": config.pop("main_process_ip", ""),
        "main_process_port": config.pop("main_process_port", 29500),
        "nccl_socket_ifname": config.pop("nccl_socket_ifname", ""),
        "gloo_socket_ifname": config.pop("gloo_socket_ifname", ""),
        "sync_use_password_auth": pop_sync_value("sync_use_password_auth", True),
        "sync_ssh_password": pop_sync_value("sync_ssh_password", ""),
        "sync_config_from_main": pop_sync_value("sync_config_from_main", True),
        "sync_config_keys_from_main": pop_sync_value("sync_config_keys_from_main", "train_batch_size,gradient_accumulation_steps,max_train_epochs,learning_rate,unet_lr,text_encoder_lr,resolution,optimizer_type,network_dim,network_alpha,save_every_n_epochs,save_model_as,mixed_precision"),
        "sync_missing_assets_from_main": pop_sync_value("sync_missing_assets_from_main", True),
        "sync_asset_keys": pop_sync_value("sync_asset_keys", "pretrained_model_name_or_path,train_data_dir,reg_data_dir,vae,resume"),
        "sync_main_repo_dir": pop_sync_value("sync_main_repo_dir", os.getcwd()),
        "sync_main_toml": pop_sync_value("sync_main_toml", "./config/autosave/distributed-main-latest.toml"),
        "sync_ssh_user": pop_sync_value("sync_ssh_user", ""),
        "sync_ssh_port": pop_sync_value("sync_ssh_port", 22),
    }
    is_worker = distributed_enabled and _to_int(distributed_config.get("num_machines"), 1) > 1 and _to_int(distributed_config.get("machine_rank"), 0) > 0
    skip_local_path_validation = is_worker and _to_bool(distributed_config.get("sync_missing_assets_from_main"), True)
    if skip_local_path_validation:
        log.info("Worker mode detected with asset sync enabled, skip local model/data validation before sync.")

    suggest_cpu_threads = 8 if len(train_utils.get_total_images(config["train_data_dir"])) > 200 else 2
    model_train_type = config.pop("model_train_type", "sd-lora")
    trainer_file = trainer_mapping[model_train_type]

    if model_train_type != "sdxl-finetune" and not skip_local_path_validation:
        if not train_utils.validate_data_dir(config["train_data_dir"]):
            return APIResponseFail(message="训练数据集路径不存在或没有图片，请检查目录。")

    if not skip_local_path_validation:
        validated, message = train_utils.validate_model(config["pretrained_model_name_or_path"], model_train_type)
        if not validated:
            return APIResponseFail(message=message)

    if "prompt_file" in config and config["prompt_file"].strip() != "":
        prompt_file = config["prompt_file"].strip()
        if not os.path.exists(prompt_file):
            return APIResponseFail(message=f"Prompt 文件 {prompt_file} 不存在，请检查路径。")
        config["sample_prompts"] = prompt_file
    else:
        try:
            positive_prompt, sample_prompts_arg = get_sample_prompts(config=config)

            if positive_prompt is not None and train_utils.is_promopt_like(sample_prompts_arg):
                sample_prompts_file = os.path.join(os.getcwd(), f"config", "autosave", f"{timestamp}-promopt.txt")
                with open(sample_prompts_file, "w", encoding="utf-8") as f:
                    f.write(sample_prompts_arg)
                config["sample_prompts"] = sample_prompts_file
                log.info(f"Wrote prompts to file {sample_prompts_file}")

        except ValueError as e:
            log.error(f"Error while processing prompts: {e}")
            return APIResponseFail(message=str(e))

    with open(toml_file, "w", encoding="utf-8") as f:
        f.write(toml.dumps(config))

    result = process.run_train(
        toml_file,
        trainer_file,
        gpu_ids,
        suggest_cpu_threads,
        distributed_config=distributed_config,
    )

    return result


@router.post("/run_script")
async def run_script(request: Request, background_tasks: BackgroundTasks):
    try:
        payload = json.loads((await request.body()).decode("utf-8"))
    except json.JSONDecodeError:
        return APIResponseFail(message="Invalid JSON payload")
    if not isinstance(payload, dict):
        return APIResponseFail(message="Invalid payload type")

    script_name = str(payload.get("script_name", "")).strip()
    if script_name not in avaliable_scripts:
        return APIResponseFail(message="Script not found")

    script_path = _resolve_available_script(script_name)
    if script_path is None:
        return APIResponseFail(message=f"Script not found on disk: {script_name}")

    cmd = [launch_utils.python_bin, str(script_path)]
    skipped_params = []
    for k, v in payload.items():
        if k == "script_name":
            continue

        key = str(k).strip()
        if key == "":
            continue
        if not cli_arg_key_pattern.match(key):
            skipped_params.append(key)
            continue

        if isinstance(v, bool):
            if v:
                cmd.append(f"--{key}")
            continue

        if v is None:
            skipped_params.append(key)
            continue

        if isinstance(v, str):
            value = v.strip()
            if value == "":
                skipped_params.append(key)
                continue
            cmd.extend([f"--{key}", value])
            continue

        if isinstance(v, (dict, list, tuple, set)):
            skipped_params.append(key)
            continue

        cmd.extend([f"--{key}", str(v)])

    if skipped_params:
        log.warning(f"/run_script skipped invalid/empty params: {', '.join(skipped_params)}")

    background_tasks.add_task(launch_utils.run, cmd, shell=False)
    return APIResponseSuccess()


@router.post("/interrogate")
async def run_interrogate(req: TaggerInterrogateRequest, background_tasks: BackgroundTasks):
    interrogator = available_interrogators.get(req.interrogator_model, available_interrogators["wd14-convnextv2-v2"])
    background_tasks.add_task(
        on_interrogate,
        image=None,
        batch_input_glob=req.path,
        batch_input_recursive=req.batch_input_recursive,
        batch_output_dir="",
        batch_output_filename_format="[name].[output_extension]",
        batch_output_action_on_conflict=req.batch_output_action_on_conflict,
        batch_remove_duplicated_tag=True,
        batch_output_save_json=False,
        interrogator=interrogator,
        threshold=req.threshold,
        character_threshold=req.character_threshold,
        add_rating_tag=req.add_rating_tag,
        add_model_tag=req.add_model_tag,
        additional_tags=req.additional_tags,
        exclude_tags=req.exclude_tags,
        sort_by_alphabetical_order=False,
        add_confident_as_weight=False,
        replace_underscore=req.replace_underscore,
        replace_underscore_excludes=req.replace_underscore_excludes,
        escape_tag=req.escape_tag,
        unload_model_after_running=True
    )
    return APIResponseSuccess()


@router.get("/pick_file")
async def pick_file(picker_type: str):
    if picker_type == "folder":
        coro = asyncio.to_thread(open_directory_selector, "")
    elif picker_type == "model-file":
        file_types = [("checkpoints", "*.safetensors;*.ckpt;*.pt"), ("all files", "*.*")]
        coro = asyncio.to_thread(open_file_selector, "", "Select file", file_types)
    else:
        return APIResponseFail(message="Invalid picker type")

    result = await coro
    if result == "":
        return APIResponseFail(message="用户取消选择")

    return APIResponseSuccess(data={
        "path": result
    })


@router.get("/get_files")
async def get_files(pick_type) -> APIResponse:
    pick_preset = {
        "model-file": {
            "type": "file",
            "path": "./sd-models",
            "filter": "(.safetensors|.ckpt|.pt)"
        },
        "model-saved-file": {
            "type": "file",
            "path": "./output",
            "filter": "(.safetensors|.ckpt|.pt)"
        },
        "train-dir": {
            "type": "folder",
            "path": "./train",
            "filter": None
        },
    }

    folder_blacklist = [".ipynb_checkpoints", ".DS_Store"]

    def list_path_or_files(preset_info):
        path = Path(preset_info["path"])
        file_type = preset_info["type"]
        regex_filter = preset_info["filter"]
        result_list = []

        if file_type == "file":
            if regex_filter:
                pattern = re.compile(regex_filter)
                files = [f for f in path.glob("**/*") if f.is_file() and pattern.search(f.name)]
            else:
                files = [f for f in path.glob("**/*") if f.is_file()]
            for file in files:
                result_list.append({
                    "path": str(file.resolve().absolute()).replace("\\", "/"),
                    "name": file.name,
                    "size": f"{round(file.stat().st_size / (1024**3),2)} GB"
                })
        elif file_type == "folder":
            folders = [f for f in path.iterdir() if f.is_dir()]
            for folder in folders:
                if folder.name in folder_blacklist:
                    continue
                result_list.append({
                    "path": str(folder.resolve().absolute()).replace("\\", "/"),
                    "name": folder.name,
                    "size": 0
                })

        return result_list

    if pick_type not in pick_preset:
        return APIResponseFail(message="Invalid request")

    dirs = list_path_or_files(pick_preset[pick_type])
    return APIResponseSuccess(data={
        "files": dirs
    })


@router.get("/tasks", response_model_exclude_none=True)
async def get_tasks() -> APIResponse:
    return APIResponseSuccess(data={
        "tasks": tm.dump()
    })


@router.get("/tasks/terminate/{task_id}", response_model_exclude_none=True)
async def terminate_task(task_id: str):
    tm.terminate_task(task_id)
    return APIResponseSuccess()


@router.get("/graphic_cards")
async def list_avaliable_cards() -> APIResponse:
    if not printable_devices:
        return APIResponse(status="pending")

    return APIResponseSuccess(data={
        "cards": printable_devices
    })


@router.get("/schemas/hashes")
async def list_schema_hashes() -> APIResponse:
    if os.environ.get("MIKAZUKI_SCHEMA_HOT_RELOAD", "0") == "1":
        log.info("Hot reloading schemas")
        await load_schemas()

    return APIResponseSuccess(data={
        "schemas": [
            {
                "name": schema["name"],
                "hash": schema["hash"]
            }
            for schema in avaliable_schemas
        ]
    })


@router.get("/schemas/all")
async def get_all_schemas() -> APIResponse:
    return APIResponseSuccess(data={
        "schemas": avaliable_schemas
    })


@router.get("/presets")
async def get_presets() -> APIResponse:
    if os.environ.get("MIKAZUKI_SCHEMA_HOT_RELOAD", "0") == "1":
        log.info("Hot reloading presets")
        await load_presets()

    return APIResponseSuccess(data={
        "presets": avaliable_presets
    })


@router.get("/config/saved_params")
async def get_saved_params() -> APIResponse:
    saved_params = app_config["saved_params"]
    return APIResponseSuccess(data=saved_params)
