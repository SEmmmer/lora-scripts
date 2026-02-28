import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import toml

from mikazuki.log import log


MIXED_RESOLUTION_RESUME_SENTINEL = "__MIXED_AUTO_RESUME__"
DATASET_DIR_KEYS = ("train_data_dir", "reg_data_dir")
STATE_REQUIRED_FILES = ("train_state.json", "optimizer.bin", "scheduler.bin")
STATE_MODEL_FILE_CANDIDATES = ("model.safetensors", "pytorch_model.bin", "model.bin")


def _resolve_local_path(path_value: str, repo_root: Path) -> Path:
    p = Path(path_value).expanduser()
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def _clear_dataset_npz_cache_by_config(config: dict, repo_root: Path):
    total_removed = 0
    for key in DATASET_DIR_KEYS:
        value = str(config.get(key, "") or "").strip()
        if not value:
            continue
        local_dir = _resolve_local_path(value, repo_root)
        if not local_dir.exists() or not local_dir.is_dir():
            continue

        removed = 0
        for npz_file in local_dir.rglob("*.npz"):
            try:
                npz_file.unlink()
                removed += 1
            except Exception as e:
                raise RuntimeError(f"删除缓存失败: {npz_file} ({e})") from e

        total_removed += removed
        log.info(f"[staged-resolution] cache reset: {key}, removed={removed}, dir={local_dir}")

    log.info(f"[staged-resolution] cache reset finished, total npz removed={total_removed}")


def _safe_int(value, default=-1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _check_state_dir_complete(state_dir: Path) -> tuple[bool, str]:
    for name in STATE_REQUIRED_FILES:
        if not (state_dir / name).is_file():
            return False, f"missing {name}"

    has_model_file = any((state_dir / name).is_file() for name in STATE_MODEL_FILE_CANDIDATES)
    if not has_model_file:
        has_model_file = bool(list(state_dir.glob("pytorch_model*.bin")))
    if not has_model_file:
        has_model_file = bool(list(state_dir.glob("model*.safetensors")))
    if not has_model_file:
        return False, "missing model state file"

    return True, ""


def _build_state_candidate_from_dir(state_dir: Path) -> tuple[Optional[dict[str, Any]], str]:
    if not state_dir.exists() or not state_dir.is_dir():
        return None, "not a directory"

    complete, incomplete_reason = _check_state_dir_complete(state_dir)
    if not complete:
        return None, incomplete_reason

    state_file = state_dir / "train_state.json"
    step_num = -1
    epoch_num = -1
    sample_num = -1
    state_data: dict[str, Any] = {}
    if state_file.exists():
        try:
            state_data = json.loads(state_file.read_text(encoding="utf-8"))
            step_num = _safe_int(state_data.get("current_step", -1), -1)
            epoch_num = _safe_int(state_data.get("current_epoch", -1), -1)
            sample_num = _safe_int(state_data.get("current_global_samples", -1), -1)
        except Exception:
            pass

    if step_num < 0 or epoch_num < 0:
        # fallback for legacy folders without readable train_state
        match = re.search(r"-(\d+)-state$", state_dir.name)
        epoch_num = _safe_int(match.group(1), -1) if match else -1

    try:
        mtime = state_dir.stat().st_mtime
    except Exception:
        mtime = 0

    plan_id = None
    phase_target_global_samples = -1
    if isinstance(state_data, dict):
        plan_id_raw = str(state_data.get("staged_plan_id", "") or "").strip()
        plan_id = plan_id_raw if plan_id_raw else None
        phase_target_global_samples = _safe_int(state_data.get("staged_phase_target_global_samples", -1), -1)

    return (
        {
            "path": state_dir,
            "step_num": int(step_num),
            "epoch_num": int(epoch_num),
            "sample_num": int(sample_num),
            "phase_target_global_samples": int(phase_target_global_samples),
            "mtime": float(mtime),
            "plan_id": plan_id,
        },
        "",
    )


def _collect_state_candidates(config: dict, repo_root: Path) -> list[dict[str, Any]]:
    output_dir = _resolve_local_path(str(config.get("output_dir", "./output") or "./output"), repo_root)
    if not output_dir.exists() or not output_dir.is_dir():
        return []

    output_name = str(config.get("output_name", "") or "").strip()
    candidates: list[dict[str, Any]] = []
    for entry in output_dir.glob("*-state"):
        if not entry.is_dir():
            continue
        if output_name and not entry.name.startswith(f"{output_name}-"):
            continue
        candidate, reason = _build_state_candidate_from_dir(entry)
        if candidate is None:
            log.warning(f"[staged-resolution] skip incomplete state dir: {entry} ({reason})")
            continue
        candidates.append(candidate)

    candidates.sort(
        key=lambda x: (
            x.get("sample_num", -1),
            x["step_num"],
            x["epoch_num"],
            x["mtime"],
        ),
        reverse=True,
    )
    return candidates


def _select_latest_state_candidate(
    config: dict,
    repo_root: Path,
    *,
    max_samples: Optional[int] = None,
    min_samples_exclusive: Optional[int] = None,
    max_step: Optional[int] = None,
    min_step_exclusive: Optional[int] = None,
    plan_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    candidates = _collect_state_candidates(config, repo_root)
    if not candidates:
        return None

    def _filter(items, strict_plan: bool):
        result = []
        for item in items:
            step_num = int(item.get("step_num", -1))
            sample_num = int(item.get("sample_num", -1))
            state_plan_id = item.get("plan_id")

            if max_samples is not None and sample_num < 0:
                # Sample-bounded matching requires sample progress metadata.
                continue
            if max_samples is not None and sample_num > int(max_samples):
                continue
            if min_samples_exclusive is not None and sample_num <= int(min_samples_exclusive):
                continue
            if (max_step is not None or min_step_exclusive is not None) and step_num < 0:
                continue
            if max_step is not None and step_num > int(max_step):
                continue
            if min_step_exclusive is not None and step_num <= int(min_step_exclusive):
                continue
            if sample_num < 0 and step_num < 0:
                # Need at least one numeric progress metric.
                continue

            if plan_id:
                if strict_plan:
                    if state_plan_id != plan_id:
                        continue
                else:
                    # Backward compatibility: keep legacy states without plan_id.
                    if state_plan_id is not None and state_plan_id != plan_id:
                        continue

            result.append(item)
        return result

    if plan_id:
        strict = _filter(candidates, strict_plan=True)
        if strict:
            return strict[0]

    relaxed = _filter(candidates, strict_plan=False)
    if relaxed:
        return relaxed[0]

    return None


def _load_toml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return toml.load(f)


def _write_toml(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        toml.dump(data, f)


def _build_phase_command(trainer_file: str, toml_file: str, cpu_threads: int, launch_args: list[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_cpu_threads_per_process",
        str(cpu_threads),
        "--quiet",
        *launch_args,
        trainer_file,
        "--config_file",
        toml_file,
    ]


def _infer_resume_context(
    plan: dict,
    phases: list[dict[str, Any]],
    first_phase_config: dict,
    repo_root: Path,
    *,
    explicit_resume: bool = False,
) -> dict[str, Any]:
    if not phases:
        return {
            "start_pos": 0,
            "completed": False,
            "resume_state_dir": "",
            "resume_step": None,
            "resume_samples": None,
            "intra_phase_resume": False,
            "apply_boundary_offset": False,
        }

    total_target_max_steps = int(phases[-1].get("target_max_train_steps", 0) or 0)
    total_target_global_samples = int(phases[-1].get("target_global_train_samples", 0) or 0)
    use_sample_progress = total_target_global_samples > 0
    plan_id = str(plan.get("plan_id", "") or "").strip() or None

    latest_state = None
    resume_value = str(first_phase_config.get("resume", "") or "").strip()
    if explicit_resume and resume_value and resume_value != MIXED_RESOLUTION_RESUME_SENTINEL:
        explicit_resume_path = _resolve_local_path(resume_value, repo_root)
        if explicit_resume_path.exists() and explicit_resume_path.is_dir():
            explicit_state, explicit_reason = _build_state_candidate_from_dir(explicit_resume_path)
            if explicit_state is None:
                log.warning(
                    "[staged-resolution] explicit resume state cannot be parsed for phase inference: "
                    f"{explicit_resume_path} ({explicit_reason}), fallback to auto detection"
                )
            else:
                state_plan_id = explicit_state.get("plan_id")
                if plan_id and state_plan_id is not None and state_plan_id != plan_id:
                    log.warning(
                        "[staged-resolution] explicit resume state plan_id mismatch: "
                        f"state={state_plan_id}, current={plan_id}. continue with explicit state as requested."
                    )
                latest_state = explicit_state
                log.info(
                    "[staged-resolution] explicit resume state is used for phase inference: "
                    f"{explicit_resume_path}"
                )
        else:
            log.info(
                "[staged-resolution] explicit resume is not a local state directory, "
                "fallback to auto state detection"
            )

    if latest_state is None:
        latest_state = _select_latest_state_candidate(
            first_phase_config,
            repo_root,
            max_samples=total_target_global_samples if use_sample_progress else None,
            min_samples_exclusive=None,
            max_step=(None if use_sample_progress else (total_target_max_steps if total_target_max_steps > 0 else None)),
            min_step_exclusive=None,
            plan_id=plan_id,
        )
    if latest_state is None and use_sample_progress:
        # Backward compatibility for legacy states that don't contain sample metadata.
        latest_state = _select_latest_state_candidate(
            first_phase_config,
            repo_root,
            max_samples=None,
            min_samples_exclusive=None,
            max_step=total_target_max_steps if total_target_max_steps > 0 else None,
            min_step_exclusive=None,
            plan_id=plan_id,
        )
    if latest_state is None:
        return {
            "start_pos": 0,
            "completed": False,
            "resume_state_dir": "",
            "resume_step": None,
            "resume_samples": None,
            "intra_phase_resume": False,
            "apply_boundary_offset": False,
        }

    step_num = int(latest_state.get("step_num", -1))
    sample_num = int(latest_state.get("sample_num", -1))
    state_dir = str(latest_state["path"])

    if use_sample_progress and sample_num >= 0:
        progress_value = sample_num
        target_key = "target_global_train_samples"
    else:
        progress_value = step_num
        target_key = "target_max_train_steps"

    if progress_value < 0:
        # Cannot map phase robustly without a step, but resume from this state is still safer than full restart.
        return {
            "start_pos": 0,
            "completed": False,
            "resume_state_dir": state_dir,
            "resume_step": int(step_num) if step_num >= 0 else None,
            "resume_samples": int(sample_num) if sample_num >= 0 else None,
            "intra_phase_resume": True,
            "apply_boundary_offset": False,
        }

    prev_target = 0
    start_pos = None
    for idx, phase in enumerate(phases):
        target = int(phase.get(target_key, 0) or 0)
        if progress_value < target:
            start_pos = idx
            break
        prev_target = target

    if start_pos is None:
        if not explicit_resume:
            return {
                "start_pos": 0,
                "completed": False,
                "resume_state_dir": "",
                "resume_step": step_num,
                "resume_samples": sample_num if sample_num >= 0 else None,
                "intra_phase_resume": False,
                "apply_boundary_offset": False,
                "ignore_existing_completed_state": True,
            }
        return {
            "start_pos": len(phases),
            "completed": True,
            "resume_state_dir": state_dir,
            "resume_step": step_num,
            "resume_samples": sample_num if sample_num >= 0 else None,
            "intra_phase_resume": False,
            "apply_boundary_offset": False,
        }

    prev_target_for_start = int(phases[start_pos - 1].get(target_key, 0) or 0) if start_pos > 0 else 0
    intra_phase_resume = progress_value > prev_target_for_start
    apply_boundary_offset = bool(start_pos > 0 and not intra_phase_resume)

    return {
        "start_pos": int(start_pos),
        "completed": False,
        "resume_state_dir": state_dir,
        "resume_step": int(step_num),
        "resume_samples": int(sample_num) if sample_num >= 0 else None,
        "intra_phase_resume": bool(intra_phase_resume),
        "apply_boundary_offset": bool(apply_boundary_offset),
    }


def _run_mixed_plan(plan: dict) -> int:
    trainer_file = str(plan["trainer_file"])
    cpu_threads = int(plan.get("cpu_threads", 2) or 2)
    launch_args = [str(x) for x in plan.get("launch_args", [])]
    phases = list(plan.get("phases", []))
    repo_root = Path(plan.get("repo_root", ".")).resolve()

    if not phases:
        log.error("[staged-resolution] no phase in plan")
        return 2

    phase_tomls = []
    for phase in phases:
        phase_toml = Path(str(phase["toml_path"])).resolve()
        if not phase_toml.exists():
            log.error(f"[staged-resolution] phase toml not found: {phase_toml}")
            return 2
        phase_tomls.append(phase_toml)

    first_phase_config = _load_toml(phase_tomls[0])
    explicit_resume = bool(str(first_phase_config.get("resume", "") or "").strip())
    resume_ctx = _infer_resume_context(
        plan,
        phases,
        first_phase_config,
        repo_root,
        explicit_resume=explicit_resume,
    )
    if resume_ctx.get("ignore_existing_completed_state"):
        log.info(
            "[staged-resolution] found existing state at/over final target progress, "
            "but no explicit resume is set. start from phase 1 as a fresh staged run."
        )
    start_pos = int(resume_ctx.get("start_pos", 0) or 0)
    if resume_ctx.get("completed"):
        log.info(
            "[staged-resolution] detected existing state reaches or exceeds final target progress. "
            "all phases are considered completed."
        )
        return 0

    auto_resume_state_dir = ""
    prev_resolution = str(phases[start_pos - 1].get("resolution", "") or "").strip() if start_pos > 0 else None
    if resume_ctx.get("resume_state_dir"):
        log.info(
            f"[staged-resolution] restart detected: start_phase={start_pos + 1}, "
            f"resume_state={resume_ctx.get('resume_state_dir')}, resume_step={resume_ctx.get('resume_step')}, "
            f"resume_samples={resume_ctx.get('resume_samples')}, "
            f"intra_phase_resume={'yes' if resume_ctx.get('intra_phase_resume') else 'no'}, "
            f"boundary_offset={'yes' if resume_ctx.get('apply_boundary_offset') else 'no'}"
        )

    for pos in range(start_pos, len(phases)):
        phase = phases[pos]
        phase_index = int(phase["phase_index"])
        phase_toml = phase_tomls[pos]

        phase_config = _load_toml(phase_toml)
        phase_resolution = str(phase.get("resolution", phase_config.get("resolution", "")) or "").strip()

        should_clear_cache = bool(phase.get("clear_cache_before_start", False))
        if should_clear_cache:
            if pos == start_pos and resume_ctx.get("intra_phase_resume"):
                # Same phase retry: keep existing cache to avoid unnecessary rebuild.
                pass
            else:
                prev_resolution_for_switch = prev_resolution
                if not prev_resolution_for_switch and pos > 0:
                    prev_resolution_for_switch = str(phases[pos - 1].get("resolution", "") or "").strip()
                if prev_resolution_for_switch and phase_resolution != prev_resolution_for_switch:
                    log.info(
                        f"[staged-resolution] phase {phase_index}: resolution switched "
                        f"{prev_resolution_for_switch} -> {phase_resolution}, reset dataset npz cache"
                    )
                    _clear_dataset_npz_cache_by_config(phase_config, repo_root)

        if pos == start_pos and resume_ctx.get("resume_state_dir"):
            phase_config["resume"] = str(resume_ctx["resume_state_dir"])
            if resume_ctx.get("apply_boundary_offset"):
                phase_config["resume_epoch_offset"] = 1
            else:
                phase_config.pop("resume_epoch_offset", None)
            _write_toml(phase_toml, phase_config)
            log.info(
                f"[staged-resolution] phase {phase_index}: restart resume from "
                f"{resume_ctx.get('resume_state_dir')} (resume_epoch_offset={phase_config.get('resume_epoch_offset', 0)})"
            )
        else:
            resume_value = str(phase_config.get("resume", "") or "").strip()
            if resume_value == MIXED_RESOLUTION_RESUME_SENTINEL:
                if not auto_resume_state_dir:
                    log.error(
                        f"[staged-resolution] phase {phase_index}: resume sentinel found but previous phase state is missing"
                    )
                    return 2
                phase_config["resume"] = auto_resume_state_dir
                phase_config["resume_epoch_offset"] = 1
                _write_toml(phase_toml, phase_config)
                log.info(f"[staged-resolution] phase {phase_index}: auto resume from {auto_resume_state_dir}")

        log.info(
            f"[staged-resolution] phase {phase_index}/{len(phases)} start: "
            f"res={phase.get('resolution')} ratio_percent={phase.get('ratio_percent')} "
            f"batch={phase.get('batch_size')} "
            f"grad_accum={phase.get('gradient_accumulation_steps')} "
            f"save_every_n_epochs={phase.get('save_every_n_epochs')} "
            f"sample_every_n_epochs={phase.get('sample_every_n_epochs')} "
            f"epochs={phase.get('epochs')} "
            f"batches_per_epoch={phase.get('batches_per_epoch')} "
            f"phase_steps={phase.get('phase_steps')} target_max_steps={phase.get('target_max_train_steps')} "
            f"target_global_samples={phase.get('target_global_train_samples')} "
            f"target_epoch_end={phase.get('target_epoch_end')}"
        )
        log.info(
            f"[staged-resolution] phase {phase_index} formulas: "
            f"raw='{phase.get('raw_epochs_formula')}', actual='{phase.get('actual_epochs_formula')}'"
        )
        cmd = _build_phase_command(trainer_file, str(phase_toml), cpu_threads, launch_args)
        proc = subprocess.Popen(cmd, env=os.environ.copy())
        return_code = proc.wait()
        if return_code != 0:
            log.error(f"[staged-resolution] phase {phase_index} failed with code={return_code}")
            return return_code

        prev_resolution = phase_resolution
        is_last_phase = pos >= len(phases) - 1
        if is_last_phase:
            log.info(f"[staged-resolution] phase {phase_index} finished (last phase)")
            continue

        phase_config_after = _load_toml(phase_toml)
        phase_target_global_samples = int(phase.get("target_global_train_samples", 0) or 0)
        prev_phase_target_global_samples = (
            int(phases[pos - 1].get("target_global_train_samples", 0) or 0) if pos > 0 else 0
        )
        use_sample_bounds = phase_target_global_samples > 0
        phase_target_max_steps = int(phase.get("target_max_train_steps", 0) or 0)
        prev_phase_target_max_steps = int(phases[pos - 1].get("target_max_train_steps", 0) or 0) if pos > 0 else 0
        latest_state = _select_latest_state_candidate(
            phase_config_after,
            repo_root,
            max_samples=phase_target_global_samples if use_sample_bounds else None,
            min_samples_exclusive=prev_phase_target_global_samples if (use_sample_bounds and prev_phase_target_global_samples > 0) else None,
            max_step=(None if use_sample_bounds else (phase_target_max_steps if phase_target_max_steps > 0 else None)),
            min_step_exclusive=(None if use_sample_bounds else (prev_phase_target_max_steps if prev_phase_target_max_steps > 0 else None)),
            plan_id=str(plan.get("plan_id", "") or "").strip() or None,
        )
        if latest_state is None and use_sample_bounds:
            latest_state = _select_latest_state_candidate(
                phase_config_after,
                repo_root,
                max_samples=None,
                min_samples_exclusive=None,
                max_step=phase_target_max_steps if phase_target_max_steps > 0 else None,
                min_step_exclusive=prev_phase_target_max_steps if prev_phase_target_max_steps > 0 else None,
                plan_id=str(plan.get("plan_id", "") or "").strip() or None,
            )
        if latest_state is None:
            log.error(
                "[staged-resolution] cannot find latest state after phase "
                f"{phase_index}. 请确认 save_state 已启用并且 save_every_n_epochs 配置正确。"
            )
            return 2

        auto_resume_state_dir = str(latest_state["path"])
        log.info(
            f"[staged-resolution] phase {phase_index} finished, "
            f"resume state={auto_resume_state_dir}, resume_step={latest_state.get('step_num')}, "
            f"resume_samples={latest_state.get('sample_num')}, "
            "next phase will apply resume_epoch_offset=1"
        )

    log.info("[staged-resolution] all phases finished")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Staged-resolution phase runner")
    parser.add_argument("--plan-file", required=True, help="Path to staged-resolution plan json")
    args = parser.parse_args()

    plan_file = Path(args.plan_file).resolve()
    if not plan_file.exists():
        log.error(f"[staged-resolution] plan file not found: {plan_file}")
        sys.exit(2)

    try:
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
    except Exception as e:
        log.error(f"[staged-resolution] failed to parse plan file: {plan_file} ({e})")
        sys.exit(2)

    try:
        code = _run_mixed_plan(plan)
    except Exception as e:
        log.error(f"[staged-resolution] runner fatal error: {e}")
        code = 2
    sys.exit(int(code))


if __name__ == "__main__":
    main()
