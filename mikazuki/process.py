
import asyncio
import os
import shlex
import shutil
import subprocess
import sys
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
    "network_dim,network_alpha,save_every_n_epochs,save_model_as,mixed_precision"
)
DEFAULT_SYNC_CONFIG_KEYS = "*"
DEFAULT_SYNC_ASSET_KEYS = "pretrained_model_name_or_path,train_data_dir,reg_data_dir,vae,resume"
WORKER_OUTPUT_MARKER = "THIS_IS_WORKER_NODE_CHECK_MAIN_OUTPUTS"
DATASET_DIR_KEYS = ("train_data_dir", "reg_data_dir")
SYNCED_MAIN_TOML_LATEST_NAME = "distributed-main-synced-latest.toml"


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


def _list_local_network_interfaces() -> list[str]:
    net_root = Path("/sys/class/net")
    if not net_root.exists():
        return []
    try:
        return sorted([p.name for p in net_root.iterdir() if p.is_dir()])
    except Exception:
        return []


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


def _store_synced_main_toml_locally(main_toml_text: str) -> Tuple[bool, str, Optional[str]]:
    autosave_dir = base_dir_path() / "config" / "autosave"
    autosave_dir.mkdir(parents=True, exist_ok=True)

    latest_file = autosave_dir / SYNCED_MAIN_TOML_LATEST_NAME
    snapshot_file = autosave_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-distributed-main-synced.toml"
    try:
        latest_file.write_text(main_toml_text, encoding="utf-8")
        snapshot_file.write_text(main_toml_text, encoding="utf-8")
    except Exception as e:
        return False, f"写入本地同步 toml 失败: {e}", None

    log.info(f"[sync-config] synced main toml to local: {latest_file}")
    log.info(f"[sync-config] synced main toml snapshot: {snapshot_file}")
    return True, "", str(latest_file)


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
    resolved_sync_main_toml = _resolve_remote_path(sync_main_toml, remote_repo_root) if sync_main_toml else ""
    default_distributed_main_toml = str(Path(remote_repo_root) / "config" / "autosave" / "distributed-main-latest.toml")
    latest_toml_path = _get_latest_remote_toml(
        remote_host,
        ssh_port,
        remote_repo_root,
        use_password_auth=use_password_auth,
        ssh_password=ssh_password,
    )

    # Always try the newest autosave first, then fallback to configured path.
    if latest_toml_path:
        candidate_paths.append(latest_toml_path)
    if resolved_sync_main_toml:
        candidate_paths.append(resolved_sync_main_toml)

    local_toml_name = Path(toml_path).name
    candidate_paths.extend(
        [
            default_distributed_main_toml,
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
    local_synced_main_toml = None
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

        ok, store_msg, local_synced_path = _store_synced_main_toml_locally(text)
        if not ok:
            errors.append(f"{candidate} (sync local failed: {store_msg})")
            continue

        try:
            parsed = toml.load(local_synced_path)
        except Exception as e:
            errors.append(f"{candidate} (toml parse failed: {e})")
            continue

        main_config = parsed
        local_synced_main_toml = local_synced_path
        used_toml_path = candidate
        break

    if main_config is None:
        return (
            False,
            "无法读取主机 toml 配置。请检查 sync_main_repo_dir / sync_main_toml / SSH 密码权限。"
            f"remote_host={remote_host}, remote_repo_root={remote_repo_root}, attempts={'; '.join(errors)}",
        )

    log.info(f"[sync-config] use main toml: {used_toml_path}")
    if local_synced_main_toml:
        log.info(f"[sync-config] read config from local synced toml: {local_synced_main_toml}")

    sync_all = any(str(k).strip().lower() in {"*", "__all__", "all"} for k in sync_keys)
    keys_to_sync = list(main_config.keys()) if sync_all else sync_keys
    if sync_all:
        log.info(f"[sync-config] full sync mode enabled: syncing all {len(keys_to_sync)} top-level keys")

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
    config = toml.load(toml_path)

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
    config = toml.load(toml_path)
    changed = False

    max_train_epochs = _to_int(config.get("max_train_epochs"), 1)
    if max_train_epochs < 1:
        max_train_epochs = 1

    if machine_rank > 0:
        # Worker node should not create checkpoints. Keep save interval beyond this run.
        target_save_every = max_train_epochs + 1
        if _to_int(config.get("save_every_n_epochs"), 1) != target_save_every:
            old = config.get("save_every_n_epochs")
            config["save_every_n_epochs"] = target_save_every
            changed = True
            log.info(
                f"[output-policy] worker disable checkpoint save: "
                f"save_every_n_epochs {old} -> {target_save_every}"
            )

        if _to_bool(config.get("save_state"), False):
            config["save_state"] = False
            changed = True
            log.info("[output-policy] worker disable save_state: True -> False")

        if "save_last_n_epochs_state" in config and _to_int(config.get("save_last_n_epochs_state"), 0) != 0:
            old = config.get("save_last_n_epochs_state")
            config["save_last_n_epochs_state"] = 0
            changed = True
            log.info(f"[output-policy] worker disable save_last_n_epochs_state: {old} -> 0")

        if "sample_every_n_epochs" in config:
            sample_every = _to_int(config.get("sample_every_n_epochs"), 0)
            target_sample_every = max_train_epochs + 1
            if sample_every > 0 and sample_every != target_sample_every:
                config["sample_every_n_epochs"] = target_sample_every
                changed = True
                log.info(
                    f"[output-policy] worker reduce preview outputs: "
                    f"sample_every_n_epochs {sample_every} -> {target_sample_every}"
                )

    if changed:
        with open(toml_path, "w", encoding="utf-8") as f:
            f.write(toml.dumps(config))
        log.info(f"[output-policy] wrote enforced policy to {toml_path}")

    if machine_rank > 0:
        output_dir = _resolve_local_path(str(config.get("output_dir", "./output")), repo_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        marker_path = output_dir / WORKER_OUTPUT_MARKER
        marker_path.touch(exist_ok=True)
        log.info(f"[output-policy] worker marker created: {marker_path}")

    return True, ""


def run_train(toml_path: str,
              trainer_file: str = "./scripts/train_network.py",
              gpu_ids: Optional[list] = None,
              cpu_threads: Optional[int] = 2,
              distributed_config: Optional[dict] = None):
    log.info(f"Training started with config file / 训练开始，使用配置文件: {toml_path}")
    args = [
        sys.executable, "-m", "accelerate.commands.launch",  # use -m to avoid python script executable error
        "--num_cpu_threads_per_process", str(cpu_threads),  # cpu threads
        "--quiet",  # silence accelerate error message
        trainer_file,
        "--config_file", toml_path,
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
    sync_main_repo_dir = str(get_sync_value("sync_main_repo_dir", base_dir_path()) or base_dir_path())
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

    ok, message = _enforce_distributed_output_policy(toml_path=toml_path, machine_rank=machine_rank)
    if not ok:
        return APIResponse(status="error", message=f"输出策略应用失败: {message}")

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
        args[3:3] = launch_args

    if not (task := tm.create_task(args, customize_env)):
        return APIResponse(status="error", message="Failed to create task / 无法创建训练任务")

    def _run():
        try:
            task.execute()
            result = task.communicate()
            if result.returncode != 0:
                log.error(f"Training failed / 训练失败")
            else:
                log.info(f"Training finished / 训练完成")
        except Exception as e:
            log.error(f"An error occurred when training / 训练出现致命错误: {e}")

    coro = asyncio.to_thread(_run)
    asyncio.create_task(coro)

    return APIResponse(status="success", message=f"Training started / 训练开始 ID: {task.task_id}")
