
import asyncio
import os
import sys
from typing import Optional

from mikazuki.app.models import APIResponse
from mikazuki.log import log
from mikazuki.tasks import tm
from mikazuki.launch_utils import base_dir_path


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
    num_machines = int(distributed_config.get("num_machines", 1) or 1)
    machine_rank = int(distributed_config.get("machine_rank", 0) or 0)
    main_process_ip = distributed_config.get("main_process_ip")
    main_process_port = int(distributed_config.get("main_process_port", 29500) or 29500)
    nccl_socket_ifname = str(distributed_config.get("nccl_socket_ifname", "") or "").strip()
    gloo_socket_ifname = str(distributed_config.get("gloo_socket_ifname", "") or "").strip()
    num_processes_per_machine = distributed_config.get("num_processes")
    if num_processes_per_machine is None:
        num_processes_per_machine = len(gpu_ids) if gpu_ids else 1
    else:
        num_processes_per_machine = int(num_processes_per_machine)
    total_num_processes = num_processes_per_machine * num_machines

    if num_machines < 1:
        return APIResponse(status="error", message="num_machines 必须 >= 1")
    if num_processes_per_machine < 1:
        return APIResponse(status="error", message="num_processes 必须 >= 1")
    if num_machines > 1 and not main_process_ip:
        return APIResponse(status="error", message="多机训练时 main_process_ip 不能为空")
    if machine_rank < 0 or machine_rank >= num_machines:
        return APIResponse(status="error", message="machine_rank 超出范围，请检查 machine_rank 与 num_machines")
    if nccl_socket_ifname:
        customize_env["NCCL_SOCKET_IFNAME"] = nccl_socket_ifname
    if gloo_socket_ifname:
        customize_env["GLOO_SOCKET_IFNAME"] = gloo_socket_ifname

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
