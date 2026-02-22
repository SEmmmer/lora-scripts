import argparse
import importlib.util
import locale
import os
import platform
import socket
import subprocess
import sys

from mikazuki.launch_utils import (base_dir_path, catch_exception, git_tag,
                                   prepare_environment, check_port_avaliable, find_avaliable_ports)
from mikazuki.log import log

parser = argparse.ArgumentParser(description="GUI for stable diffusion training")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=28000, help="Port to run the server on")
parser.add_argument("--listen", action="store_true", help="Backward-compatible alias for --host 0.0.0.0")
parser.add_argument("--skip-prepare-environment", action="store_true")
parser.add_argument("--disable-tensorboard", action="store_true")
parser.add_argument("--disable-tageditor", action="store_true")
parser.add_argument("--disable-auto-mirror", action="store_true")
parser.add_argument("--tensorboard-host", type=str, default="0.0.0.0", help="Host to run the tensorboard on")
parser.add_argument("--tensorboard-port", type=int, default=6006, help="Port to run the tensorboard")
parser.add_argument("--localization", type=str)
parser.add_argument("--dev", action="store_true")


def resolve_lan_ip() -> str | None:
    # Prefer the active outbound interface address.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
    except OSError:
        pass

    # Fallback to host-resolved IPv4 addresses.
    try:
        host = socket.gethostname()
        for item in socket.getaddrinfo(host, None, family=socket.AF_INET):
            ip = item[4][0]
            if ip and not ip.startswith("127."):
                return ip
    except OSError:
        pass

    return None


def is_ipv6_host(host: str) -> bool:
    h = (host or "").strip()
    return h.startswith("[") or ":" in h


@catch_exception
def run_tensorboard():
    if importlib.util.find_spec("tensorboard.main") is None:
        log.warning(
            "tensorboard module not found, skip tensorboard. "
            "Install dependencies in venv first (e.g. run install script)."
        )
        return

    log.info("Starting tensorboard...")
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mikazuki.tensorboard_launcher",
            "--logdir",
            "logs",
            "--host",
            args.tensorboard_host,
            "--port",
            str(args.tensorboard_port),
        ]
    )


@catch_exception
def run_tag_editor():
    log.info("Starting tageditor...")
    script_candidates = [
        base_dir_path() / "mikazuki/dataset-tag-editor/scripts/launch.py",
        base_dir_path() / "mikazuki/dataset-tag-editor/launch.py",
    ]
    script_path = next((p for p in script_candidates if p.exists()), None)
    if script_path is None:
        log.warning(
            "Tageditor script not found. Checked: "
            + ", ".join(str(p) for p in script_candidates)
            + ". Skip starting tageditor."
        )
        return

    cmd = [
        sys.executable,
        script_path,
        "--port", "28001",
        "--shadow-gradio-output",
        "--root-path", "/proxy/tageditor"
    ]
    if args.localization:
        cmd.extend(["--localization", args.localization])
    else:
        l = locale.getlocale()[0]
        if l and l.startswith("zh"):
            cmd.extend(["--localization", "zh-Hans"])
    subprocess.Popen(cmd)


def launch():
    log.info("Starting SD-Trainer Mikazuki GUI...")
    log.info(f"Base directory: {base_dir_path()}, Working directory: {os.getcwd()}")
    log.info(f"{platform.system()} Python {platform.python_version()} {sys.executable}")

    if not args.skip_prepare_environment:
        prepare_environment(disable_auto_mirror=args.disable_auto_mirror)

    if not check_port_avaliable(args.port):
        avaliable = find_avaliable_ports(30000, 30000+20)
        if avaliable is not None:
            log.warning(f"Port {args.port} is unavailable, using fallback port {avaliable}.")
            args.port = avaliable
        else:
            log.error(f"Port {args.port} is unavailable and fallback search failed. Abort launch.")
            sys.exit(1)

    log.info(f"SD-Trainer Version: {git_tag(base_dir_path())}")

    if args.listen:
        if args.host != "0.0.0.0" or args.tensorboard_host != "0.0.0.0":
            log.warning("--listen is set, force host and tensorboard-host to 0.0.0.0")
        args.host = "0.0.0.0"
        args.tensorboard_host = "0.0.0.0"

    if is_ipv6_host(args.host) or is_ipv6_host(args.tensorboard_host):
        log.error("IPv6 is disabled in this project. Please use IPv4 host (e.g. 0.0.0.0 / 127.0.0.1).")
        sys.exit(1)

    os.environ["MIKAZUKI_HOST"] = args.host
    os.environ["MIKAZUKI_PORT"] = str(args.port)
    os.environ["MIKAZUKI_TENSORBOARD_HOST"] = args.tensorboard_host
    os.environ["MIKAZUKI_TENSORBOARD_PORT"] = str(args.tensorboard_port)
    os.environ["MIKAZUKI_DEV"] = "1" if args.dev else "0"

    if not args.disable_tageditor:
        run_tag_editor()

    if not args.disable_tensorboard:
        run_tensorboard()

    import uvicorn
    if args.host == "0.0.0.0":
        local_url = f"http://127.0.0.1:{args.port}"
        lan_ip = resolve_lan_ip()
        lan_url = f"http://{lan_ip}:{args.port}" if lan_ip else f"http://<your-lan-ip>:{args.port}"
        log.info("Server started and listening on 0.0.0.0.")
        log.info(f"Please visit (Local): {local_url}")
        log.info(f"Please visit (LAN):   {lan_url}")
        log.info("Any device on the same LAN can access this service.")
    else:
        log.info(f"Server started. Please visit: http://{args.host}:{args.port}")
    uvicorn.run("mikazuki.app:app", host=args.host, port=args.port, log_level="error", reload=args.dev)


if __name__ == "__main__":
    args, _ = parser.parse_known_args()
    launch()
