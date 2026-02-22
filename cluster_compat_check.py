#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional


DEFAULT_CONTROL_PORT = 29610
DEFAULT_MASTER_PORT = 29500
DEFAULT_NCCL_TIMEOUT_SECONDS = 180
DEFAULT_IPERF_DURATION_SECONDS = 5
DEFAULT_IPERF_PORT_BASE = 5301


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def _run_command(cmd: list[str], timeout: int = 20) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError as e:
        return 127, "", str(e)
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "").strip() if isinstance(e.stdout, str) else ""
        err = (e.stderr or "").strip() if isinstance(e.stderr, str) else ""
        if not err:
            err = "command timeout"
        return 124, out, err
    except Exception as e:  # pragma: no cover
        return 1, "", repr(e)


def _run_optional_output(cmd: list[str], timeout: int = 20) -> str:
    code, out, err = _run_command(cmd, timeout=timeout)
    if code == 0:
        return out if out else "(ok)"
    detail = err or out or "(empty)"
    return f"(failed code={code}: {detail})"


def _first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _detect_host_ip_guess() -> str:
    # Try route-based local source address first.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and ip != "127.0.0.1":
                return ip
    except Exception:
        pass

    # Fallback to hostname resolve.
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip:
            return ip
    except Exception:
        pass
    return "127.0.0.1"


def _prompt_text(prompt: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{prompt}{suffix}: ").strip()
    if value == "" and default is not None:
        return default
    return value


def _prompt_int(prompt: str, default: int, min_value: Optional[int] = None) -> int:
    while True:
        raw = _prompt_text(prompt, str(default))
        try:
            value = int(raw)
        except Exception:
            print(f"输入无效: {raw}, 请输入整数")
            continue
        if min_value is not None and value < min_value:
            print(f"输入无效: {value}, 需 >= {min_value}")
            continue
        return value


def _prompt_yes_no(prompt: str, default_yes: bool = False) -> bool:
    suffix = "Y/n" if default_yes else "y/N"
    raw = input(f"{prompt} [{suffix}]: ").strip().lower()
    if raw == "":
        return default_yes
    return raw in {"y", "yes", "1", "true"}


def _format_table(headers: list[str], rows: list[list[Any]]) -> str:
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def _fmt_row(row: list[Any]) -> str:
        return "| " + " | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)) + " |"

    line = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    out = [line, _fmt_row(headers), line]
    for row in rows:
        out.append(_fmt_row(row))
    out.append(line)
    return "\n".join(out)


def run_check_env() -> None:
    _print_section("Python")
    print("exe:", sys.executable)
    print("version:", sys.version.split()[0])

    _print_section("Host")
    print("hostname:", socket.gethostname())
    print("platform:", platform.platform())

    _print_section("NVIDIA")
    print("nvidia-smi -L:", _run_optional_output(["nvidia-smi", "-L"]))
    code, out, err = _run_command(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        timeout=20,
    )
    if code == 0:
        print("driver:", _first_non_empty_line(out) or "(empty)")
    else:
        print("driver:", f"(failed code={code}: {err or out or '(empty)'})")

    _print_section("Torch")
    try:
        import torch
        import torch.distributed as dist

        print("torch:", torch.__version__)
        print("torch_file:", torch.__file__)
        print("cuda:", torch.version.cuda)
        print("cuda_available:", torch.cuda.is_available())
        print("gpu_count:", torch.cuda.device_count())
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            print("gpu0:", torch.cuda.get_device_name(0))
            try:
                print("nccl_version:", torch.cuda.nccl.version())
            except Exception as e:
                print("nccl_version:", f"(error: {e!r})")
        print("dist_nccl_available:", dist.is_nccl_available())
        print("dist_gloo_available:", dist.is_gloo_available())
    except Exception as e:
        print("torch_import_failed:", repr(e))

    _print_section("Network (brief)")
    if platform.system().lower().startswith("win"):
        print(_run_optional_output(["ipconfig"]))
    else:
        print(_run_optional_output(["ip", "-br", "a"]))


def _run_nccl_distributed(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    start_time = time.time()
    result: dict[str, Any] = {
        "ok": False,
        "rank": rank,
        "world_size": world_size,
        "hostname": socket.gethostname(),
        "sum": None,
        "expected_sum": float(world_size * (world_size - 1) / 2),
        "elapsed_seconds": 0.0,
        "error": "",
    }

    try:
        import torch
        import torch.distributed as dist
    except Exception as e:
        result["error"] = f"import torch failed: {e!r}"
        result["elapsed_seconds"] = round(time.time() - start_time, 3)
        return result

    initialized = False
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        if torch.cuda.device_count() < 1:
            raise RuntimeError("no CUDA GPU detected")

        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)

        torch.cuda.set_device(0)
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=timeout_seconds),
        )
        initialized = True

        x = torch.tensor([float(rank)], device="cuda")
        dist.all_reduce(x)
        reduced = float(x.item())
        dist.barrier()

        result["sum"] = reduced
        result["ok"] = abs(reduced - result["expected_sum"]) < 1e-4
        if not result["ok"]:
            result["error"] = f"all_reduce sum mismatch: got={reduced}, expected={result['expected_sum']}"
    except Exception as e:
        result["error"] = repr(e)
    finally:
        try:
            if initialized and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

    result["elapsed_seconds"] = round(time.time() - start_time, 3)
    return result


def run_nccl_single() -> dict[str, Any]:
    return _run_nccl_distributed(
        rank=0,
        world_size=1,
        master_addr="127.0.0.1",
        master_port=DEFAULT_MASTER_PORT + 1,
        timeout_seconds=60,
    )


def run_nccl_from_env() -> dict[str, Any]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.environ.get("MASTER_PORT", str(DEFAULT_MASTER_PORT)))
    timeout_seconds = int(os.environ.get("NCCL_TEST_TIMEOUT_SECONDS", str(DEFAULT_NCCL_TIMEOUT_SECONDS)))
    return _run_nccl_distributed(
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
        timeout_seconds=timeout_seconds,
    )


def _print_nccl_result(result: dict[str, Any], label: str) -> None:
    status = "PASS" if result.get("ok") else "FAIL"
    print(
        f"{label} status={status}, rank={result.get('rank')}/{result.get('world_size') - 1}, "
        f"host={result.get('hostname')}, sum={result.get('sum')}, "
        f"expected={result.get('expected_sum')}, elapsed={result.get('elapsed_seconds')}s"
    )
    if result.get("error"):
        print(f"{label} error={result.get('error')}")


@dataclass
class JsonChannel:
    sock: socket.socket

    def __post_init__(self) -> None:
        self.reader = self.sock.makefile("r", encoding="utf-8")
        self.writer = self.sock.makefile("w", encoding="utf-8")

    def send(self, payload: dict[str, Any]) -> None:
        self.writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.writer.flush()

    def recv(self, timeout: Optional[float] = None) -> dict[str, Any]:
        previous_timeout = self.sock.gettimeout()
        try:
            self.sock.settimeout(timeout)
            line = self.reader.readline()
        except socket.timeout:
            raise TimeoutError("socket read timeout")
        finally:
            self.sock.settimeout(previous_timeout)

        if not line:
            raise EOFError("peer closed connection")
        return json.loads(line)

    def close(self) -> None:
        try:
            self.reader.close()
        except Exception:
            pass
        try:
            self.writer.close()
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass


@dataclass
class WorkerConnection:
    rank: int
    hostname: str
    advertised_ip: str
    peer_ip: str
    gpu_count: int
    channel: JsonChannel


@dataclass
class NodeInfo:
    rank: int
    name: str
    ip: str
    is_host: bool
    worker: Optional[WorkerConnection] = None


class IperfServerManager:
    def __init__(self, label: str):
        self.label = label
        self._servers: dict[int, subprocess.Popen[str]] = {}

    def start(self, port: int) -> dict[str, Any]:
        if port in self._servers:
            return {"ok": False, "error": f"iperf3 server already exists on port {port}"}
        try:
            proc = subprocess.Popen(
                ["iperf3", "-s", "-1", "-p", str(port), "-J"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            return {"ok": False, "error": repr(e)}
        self._servers[port] = proc
        return {"ok": True}

    def wait(self, port: int, timeout_seconds: int) -> dict[str, Any]:
        proc = self._servers.pop(port, None)
        if proc is None:
            return {"ok": False, "error": f"iperf3 server not found on port {port}"}
        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            return {
                "ok": False,
                "returncode": 124,
                "stdout": stdout or "",
                "stderr": (stderr or "") + " timeout",
            }
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": stdout or "",
            "stderr": stderr or "",
        }

    def shutdown_all(self) -> None:
        for port, proc in list(self._servers.items()):
            try:
                proc.kill()
                proc.communicate(timeout=2)
            except Exception:
                pass
            self._servers.pop(port, None)


def _extract_iperf_bps(output_text: str) -> Optional[float]:
    try:
        data = json.loads(output_text)
    except Exception:
        return None

    end = data.get("end", {})
    candidates: list[float] = []
    for key in ("sum_received", "sum_sent"):
        value = end.get(key)
        if isinstance(value, dict):
            bps = value.get("bits_per_second")
            if isinstance(bps, (int, float)):
                candidates.append(float(bps))
    if candidates:
        return max(candidates)

    streams = end.get("streams")
    if isinstance(streams, list):
        for item in streams:
            sender = item.get("sender") if isinstance(item, dict) else None
            receiver = item.get("receiver") if isinstance(item, dict) else None
            for side in (sender, receiver):
                if isinstance(side, dict):
                    bps = side.get("bits_per_second")
                    if isinstance(bps, (int, float)):
                        candidates.append(float(bps))
    if candidates:
        return max(candidates)
    return None


def _check_iperf3_available() -> dict[str, Any]:
    code, out, err = _run_command(["iperf3", "--version"], timeout=10)
    if code != 0:
        return {"ok": False, "detail": err or out or f"exit={code}"}
    return {"ok": True, "detail": _first_non_empty_line(out) or "iperf3 ok"}


def _run_iperf_client(target_host: str, port: int, duration_seconds: int) -> dict[str, Any]:
    code, out, err = _run_command(
        ["iperf3", "-c", target_host, "-p", str(port), "-t", str(duration_seconds), "-J"],
        timeout=duration_seconds + 30,
    )
    if code != 0:
        return {"ok": False, "error": err or out or f"exit={code}", "bps": None}
    bps = _extract_iperf_bps(out)
    if bps is None:
        return {"ok": False, "error": "cannot parse iperf3 json throughput", "bps": None}
    return {"ok": True, "bps": bps, "error": ""}


def _worker_command_loop(channel: JsonChannel) -> int:
    iperf_manager = IperfServerManager("worker")
    try:
        while True:
            msg = channel.recv(timeout=None)
            msg_type = str(msg.get("type", ""))
            if msg_type == "start_nccl":
                result = _run_nccl_distributed(
                    rank=int(msg["rank"]),
                    world_size=int(msg["world_size"]),
                    master_addr=str(msg["master_addr"]),
                    master_port=int(msg["master_port"]),
                    timeout_seconds=int(msg.get("timeout_seconds", DEFAULT_NCCL_TIMEOUT_SECONDS)),
                )
                channel.send({"type": "nccl_result", **result})
            elif msg_type == "check_iperf3":
                channel.send({"type": "check_iperf3_result", **_check_iperf3_available()})
            elif msg_type == "iperf_server_start":
                port = int(msg["port"])
                channel.send({"type": "iperf_server_start_result", **iperf_manager.start(port)})
            elif msg_type == "iperf_server_wait":
                port = int(msg["port"])
                timeout_seconds = int(msg.get("timeout_seconds", 40))
                channel.send({"type": "iperf_server_wait_result", **iperf_manager.wait(port, timeout_seconds)})
            elif msg_type == "iperf_client_run":
                target_host = str(msg["target_host"])
                port = int(msg["port"])
                duration_seconds = int(msg.get("duration_seconds", DEFAULT_IPERF_DURATION_SECONDS))
                channel.send({"type": "iperf_client_result", **_run_iperf_client(target_host, port, duration_seconds)})
            elif msg_type == "shutdown":
                iperf_manager.shutdown_all()
                channel.send({"type": "shutdown_ack", "ok": True})
                return 0
            else:
                channel.send({"type": "error", "ok": False, "error": f"unknown command: {msg_type}"})
    finally:
        iperf_manager.shutdown_all()


def run_worker_mode(host: str, control_port: int) -> int:
    print(f"[worker] connecting to host {host}:{control_port} ...")
    try:
        sock = socket.create_connection((host, control_port), timeout=30)
    except Exception as e:
        print(f"[worker] connection failed: {e!r}")
        return 1
    channel = JsonChannel(sock)
    try:
        local_ip = sock.getsockname()[0]
        gpu_count = -1
        try:
            import torch

            gpu_count = int(torch.cuda.device_count())
        except Exception:
            pass

        channel.send(
            {
                "type": "register",
                "hostname": socket.gethostname(),
                "advertised_ip": local_ip,
                "gpu_count": gpu_count,
            }
        )
        reg = channel.recv(timeout=60)
        if str(reg.get("type", "")) != "registered":
            print(f"[worker] register failed: {reg}")
            return 1

        rank = reg.get("rank")
        world_size = reg.get("world_size")
        print(f"[worker] connected. assigned rank={rank}/{int(world_size) - 1}.")
        print("[worker] 当前正在等待主机开始测试。请继续在下一台从机触发，或回到主机开始测试。")
        return _worker_command_loop(channel)
    finally:
        channel.close()


def _send_and_expect(worker: WorkerConnection, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    worker.channel.send(payload)
    return worker.channel.recv(timeout=timeout)


def _node_check_iperf(node: NodeInfo) -> dict[str, Any]:
    if node.is_host:
        return _check_iperf3_available()
    assert node.worker is not None
    response = _send_and_expect(node.worker, {"type": "check_iperf3"}, timeout=15)
    return {"ok": bool(response.get("ok")), "detail": str(response.get("detail", response.get("error", "")))}


def _node_start_iperf_server(node: NodeInfo, manager: IperfServerManager, port: int) -> dict[str, Any]:
    if node.is_host:
        return manager.start(port)
    assert node.worker is not None
    response = _send_and_expect(node.worker, {"type": "iperf_server_start", "port": port}, timeout=15)
    return {"ok": bool(response.get("ok")), "error": str(response.get("error", ""))}


def _node_wait_iperf_server(node: NodeInfo, manager: IperfServerManager, port: int, timeout_seconds: int) -> dict[str, Any]:
    if node.is_host:
        return manager.wait(port, timeout_seconds)
    assert node.worker is not None
    response = _send_and_expect(
        node.worker,
        {"type": "iperf_server_wait", "port": port, "timeout_seconds": timeout_seconds},
        timeout=timeout_seconds + 10,
    )
    return {
        "ok": bool(response.get("ok")),
        "error": str(response.get("error", "")),
        "stdout": str(response.get("stdout", "")),
        "stderr": str(response.get("stderr", "")),
    }


def _node_run_iperf_client(node: NodeInfo, target_host: str, port: int, duration_seconds: int) -> dict[str, Any]:
    if node.is_host:
        return _run_iperf_client(target_host, port, duration_seconds)
    assert node.worker is not None
    response = _send_and_expect(
        node.worker,
        {
            "type": "iperf_client_run",
            "target_host": target_host,
            "port": port,
            "duration_seconds": duration_seconds,
        },
        timeout=duration_seconds + 40,
    )
    return {
        "ok": bool(response.get("ok")),
        "bps": response.get("bps"),
        "error": str(response.get("error", "")),
    }


def _run_iperf_direction(
    client_node: NodeInfo,
    server_node: NodeInfo,
    manager: IperfServerManager,
    port: int,
    duration_seconds: int,
) -> dict[str, Any]:
    start_res = _node_start_iperf_server(server_node, manager, port)
    if not start_res.get("ok"):
        return {"ok": False, "bps": None, "error": f"server start failed: {start_res.get('error', '')}"}

    time.sleep(0.8)
    client_res = _node_run_iperf_client(client_node, server_node.ip, port, duration_seconds)
    server_res = _node_wait_iperf_server(server_node, manager, port, duration_seconds + 30)

    if not client_res.get("ok"):
        return {"ok": False, "bps": None, "error": f"client failed: {client_res.get('error', '')}"}
    if not server_res.get("ok"):
        return {"ok": False, "bps": None, "error": f"server failed: {server_res.get('error', '') or server_res.get('stderr', '')}"}

    bps = client_res.get("bps")
    if not isinstance(bps, (int, float)):
        return {"ok": False, "bps": None, "error": "no throughput parsed"}
    return {"ok": True, "bps": float(bps), "error": ""}


def _run_mesh_iperf_tests(nodes: list[NodeInfo], duration_seconds: int) -> None:
    _print_section("iperf3 mesh compatibility")

    availability_rows: list[list[Any]] = []
    all_ok = True
    for node in nodes:
        res = _node_check_iperf(node)
        ok = bool(res.get("ok"))
        all_ok = all_ok and ok
        availability_rows.append([node.rank, node.name, node.ip, "PASS" if ok else "FAIL", res.get("detail", "")])

    print(_format_table(["Rank", "Node", "IP", "iperf3", "Detail"], availability_rows))
    if not all_ok:
        print("存在节点缺少 iperf3，跳过 mesh 互联测速。")
        return

    manager = IperfServerManager("host")
    try:
        rows: list[list[Any]] = []
        port_cursor = DEFAULT_IPERF_PORT_BASE
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_a = nodes[i]
                node_b = nodes[j]

                print(f"[iperf3] testing pair rank{node_a.rank} <-> rank{node_b.rank} ...")
                a_to_b = _run_iperf_direction(node_a, node_b, manager, port_cursor, duration_seconds)
                b_to_a = _run_iperf_direction(node_b, node_a, manager, port_cursor + 1, duration_seconds)
                port_cursor += 2

                a_to_b_gbps = (a_to_b["bps"] / 1e9) if a_to_b.get("ok") and a_to_b.get("bps") else None
                b_to_a_gbps = (b_to_a["bps"] / 1e9) if b_to_a.get("ok") and b_to_a.get("bps") else None

                gbps_values = [x for x in (a_to_b_gbps, b_to_a_gbps) if isinstance(x, (int, float))]
                avg_gbps = sum(gbps_values) / len(gbps_values) if gbps_values else None
                status = "PASS" if a_to_b.get("ok") and b_to_a.get("ok") else "FAIL"
                detail = ""
                if not a_to_b.get("ok"):
                    detail += f"A->B: {a_to_b.get('error', '')} "
                if not b_to_a.get("ok"):
                    detail += f"B->A: {b_to_a.get('error', '')}"

                rows.append(
                    [
                        f"{node_a.rank}-{node_b.rank}",
                        f"{node_a.name}({node_a.ip})",
                        f"{node_b.name}({node_b.ip})",
                        f"{a_to_b_gbps:.3f}" if a_to_b_gbps is not None else "N/A",
                        f"{b_to_a_gbps:.3f}" if b_to_a_gbps is not None else "N/A",
                        f"{avg_gbps:.3f}" if avg_gbps is not None else "N/A",
                        status,
                        detail.strip(),
                    ]
                )

        print(
            _format_table(
                ["Pair", "NodeA", "NodeB", "A->B Gbps", "B->A Gbps", "Avg Gbps", "Status", "Detail"],
                rows,
            )
        )
    finally:
        manager.shutdown_all()


def run_host_mode(
    cluster_size: int,
    master_addr: str,
    master_port: int,
    control_port: int,
    nccl_timeout_seconds: int,
    iperf_duration_seconds: int,
) -> int:
    if cluster_size < 2:
        print("cluster_size 必须 >= 2")
        return 1

    try:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("0.0.0.0", control_port))
        listener.listen(cluster_size)
    except Exception as e:
        print(f"[host] start listener failed: {e!r}")
        return 1

    print(f"[host] 控制通道已启动: 0.0.0.0:{control_port}")
    print(f"[host] NCCL 将使用 MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
    print(f"[host] 集群大小={cluster_size}，正在等待 {cluster_size - 1} 台从机连接...")
    print("[host] 请在从机运行本脚本并选择 worker，输入主机地址后会自动进入等待。")

    workers: list[WorkerConnection] = []
    try:
        while len(workers) < cluster_size - 1:
            conn, addr = listener.accept()
            channel = JsonChannel(conn)
            try:
                register = channel.recv(timeout=30)
                if str(register.get("type", "")) != "register":
                    channel.send({"type": "error", "error": "first message must be register"})
                    channel.close()
                    continue

                rank = len(workers) + 1
                hostname = str(register.get("hostname", f"worker-{rank}"))
                advertised_ip = str(register.get("advertised_ip", "")) or addr[0]
                gpu_count = int(register.get("gpu_count", -1))
                worker = WorkerConnection(
                    rank=rank,
                    hostname=hostname,
                    advertised_ip=advertised_ip,
                    peer_ip=addr[0],
                    gpu_count=gpu_count,
                    channel=channel,
                )
                workers.append(worker)
                channel.send({"type": "registered", "rank": rank, "world_size": cluster_size})
                print(
                    f"[host] worker connected: rank={rank}, host={hostname}, "
                    f"peer_ip={addr[0]}, advertised_ip={advertised_ip}, gpu_count={gpu_count} "
                    f"({len(workers)}/{cluster_size - 1})"
                )
            except Exception as e:
                print(f"[host] worker register failed from {addr}: {e!r}")
                channel.close()

        print("[host] 所有从机已连接。")
        print("        请确认所有从机都显示为等待状态，然后回到主机输入 start 开始测试。")
        while True:
            command = input("输入 start 开始测试: ").strip().lower()
            if command in {"start", "s", ""}:
                break
            print("请输入 start")

        for worker in workers:
            worker.channel.send(
                {
                    "type": "start_nccl",
                    "rank": worker.rank,
                    "world_size": cluster_size,
                    "master_addr": master_addr,
                    "master_port": master_port,
                    "timeout_seconds": nccl_timeout_seconds,
                }
            )

        local_result = _run_nccl_distributed(
            rank=0,
            world_size=cluster_size,
            master_addr=master_addr,
            master_port=master_port,
            timeout_seconds=nccl_timeout_seconds,
        )

        worker_results: list[dict[str, Any]] = []
        for worker in workers:
            try:
                response = worker.channel.recv(timeout=nccl_timeout_seconds + 90)
                if str(response.get("type", "")) != "nccl_result":
                    response = {
                        "ok": False,
                        "rank": worker.rank,
                        "world_size": cluster_size,
                        "hostname": worker.hostname,
                        "sum": None,
                        "expected_sum": float(cluster_size * (cluster_size - 1) / 2),
                        "elapsed_seconds": 0.0,
                        "error": f"unexpected response: {response}",
                    }
            except Exception as e:
                response = {
                    "ok": False,
                    "rank": worker.rank,
                    "world_size": cluster_size,
                    "hostname": worker.hostname,
                    "sum": None,
                    "expected_sum": float(cluster_size * (cluster_size - 1) / 2),
                    "elapsed_seconds": 0.0,
                    "error": f"recv failed: {e!r}",
                }
            worker_results.append(response)

        _print_section("NCCL compatibility result")
        all_results = [local_result, *worker_results]
        all_results.sort(key=lambda item: int(item.get("rank", 0)))
        rows = []
        for item in all_results:
            rows.append(
                [
                    item.get("rank"),
                    item.get("hostname"),
                    "PASS" if item.get("ok") else "FAIL",
                    item.get("sum"),
                    item.get("expected_sum"),
                    item.get("elapsed_seconds"),
                    item.get("error", ""),
                ]
            )
        print(_format_table(["Rank", "Node", "Status", "Sum", "Expected", "Elapsed(s)", "Detail"], rows))

        nodes: list[NodeInfo] = [NodeInfo(rank=0, name=socket.gethostname(), ip=master_addr, is_host=True)]
        for worker in workers:
            nodes.append(
                NodeInfo(
                    rank=worker.rank,
                    name=worker.hostname,
                    ip=worker.advertised_ip,
                    is_host=False,
                    worker=worker,
                )
            )
        nodes.sort(key=lambda n: n.rank)
        _run_mesh_iperf_tests(nodes, iperf_duration_seconds)

        return 0 if all(item.get("ok") for item in all_results) else 2
    finally:
        for worker in workers:
            try:
                worker.channel.send({"type": "shutdown"})
                worker.channel.recv(timeout=5)
            except Exception:
                pass
            worker.channel.close()
        try:
            listener.close()
        except Exception:
            pass


def run_full_interactive(args: argparse.Namespace) -> int:
    _print_section("check env")
    run_check_env()

    _print_section("single-node compatibility (NCCL world_size=1)")
    single_result = run_nccl_single()
    _print_nccl_result(single_result, "[single]")

    if not _prompt_yes_no("单机兼容性测试已完成，是否继续测试节点间兼容性（多机）？", default_yes=False):
        return 0 if single_result.get("ok") else 2

    while True:
        role = _prompt_text("请选择角色 host / worker", "host").strip().lower()
        if role in {"host", "worker"}:
            break
        print("角色输入无效，请输入 host 或 worker。")

    if role == "host":
        cluster_size = args.cluster_size or _prompt_int("请输入集群大小（总节点数）", 2, min_value=2)
        control_port = args.control_port or _prompt_int("请输入控制端口", DEFAULT_CONTROL_PORT, min_value=1)
        master_port = args.master_port or _prompt_int("请输入 NCCL MASTER_PORT", DEFAULT_MASTER_PORT, min_value=1)
        default_master_addr = args.master_addr or _detect_host_ip_guess()
        master_addr = _prompt_text("请输入主机 MASTER_ADDR（从机可访问）", default_master_addr)
        timeout_seconds = args.nccl_timeout_seconds or _prompt_int(
            "请输入 NCCL 测试超时时间（秒）",
            DEFAULT_NCCL_TIMEOUT_SECONDS,
            min_value=10,
        )
        iperf_duration_seconds = args.iperf_duration_seconds or _prompt_int(
            "请输入 iperf3 每轮测试时长（秒）",
            DEFAULT_IPERF_DURATION_SECONDS,
            min_value=1,
        )
        return run_host_mode(
            cluster_size=cluster_size,
            master_addr=master_addr,
            master_port=master_port,
            control_port=control_port,
            nccl_timeout_seconds=timeout_seconds,
            iperf_duration_seconds=iperf_duration_seconds,
        )

    host = args.host or _prompt_text("请输入主机 IP/hostname/域名")
    control_port = args.control_port or _prompt_int("请输入主机控制端口", DEFAULT_CONTROL_PORT, min_value=1)
    return run_worker_mode(host=host, control_port=control_port)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA 集群兼容性检查（env + NCCL + iperf3 mesh）")
    parser.add_argument(
        "--mode",
        choices=["full", "check-env", "single", "dist-env", "host", "worker"],
        default="full",
        help="运行模式",
    )
    parser.add_argument("--host", default="", help="worker 模式下连接的主机地址")
    parser.add_argument("--cluster-size", type=int, default=0, help="host 模式下集群总节点数")
    parser.add_argument("--control-port", type=int, default=0, help="控制端口")
    parser.add_argument("--master-addr", default="", help="NCCL MASTER_ADDR")
    parser.add_argument("--master-port", type=int, default=0, help="NCCL MASTER_PORT")
    parser.add_argument("--nccl-timeout-seconds", type=int, default=0, help="NCCL 超时秒数")
    parser.add_argument("--iperf-duration-seconds", type=int, default=0, help="iperf3 每轮时长（秒）")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.mode == "check-env":
        run_check_env()
        return 0

    if args.mode == "single":
        result = run_nccl_single()
        _print_nccl_result(result, "[single]")
        return 0 if result.get("ok") else 2

    if args.mode == "dist-env":
        result = run_nccl_from_env()
        _print_nccl_result(result, "[dist-env]")
        return 0 if result.get("ok") else 2

    if args.mode == "host":
        cluster_size = args.cluster_size if args.cluster_size > 0 else 2
        control_port = args.control_port if args.control_port > 0 else DEFAULT_CONTROL_PORT
        master_port = args.master_port if args.master_port > 0 else DEFAULT_MASTER_PORT
        master_addr = args.master_addr or _detect_host_ip_guess()
        timeout_seconds = args.nccl_timeout_seconds if args.nccl_timeout_seconds > 0 else DEFAULT_NCCL_TIMEOUT_SECONDS
        iperf_duration = args.iperf_duration_seconds if args.iperf_duration_seconds > 0 else DEFAULT_IPERF_DURATION_SECONDS
        return run_host_mode(
            cluster_size=cluster_size,
            master_addr=master_addr,
            master_port=master_port,
            control_port=control_port,
            nccl_timeout_seconds=timeout_seconds,
            iperf_duration_seconds=iperf_duration,
        )

    if args.mode == "worker":
        host = args.host or _prompt_text("请输入主机 IP/hostname/域名")
        control_port = args.control_port if args.control_port > 0 else DEFAULT_CONTROL_PORT
        return run_worker_mode(host=host, control_port=control_port)

    return run_full_interactive(args)


if __name__ == "__main__":
    raise SystemExit(main())
