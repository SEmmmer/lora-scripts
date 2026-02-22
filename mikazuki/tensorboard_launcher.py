import os
import socket
import sys
import types

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata

try:
    from packaging.version import parse as parse_version
except ModuleNotFoundError:  # pragma: no cover
    from setuptools._vendor.packaging.version import parse as parse_version


class _EntryPointProxy:
    def __init__(self, entry_point):
        self._entry_point = entry_point

    def resolve(self):
        return self._entry_point.load()


def _ensure_pkg_resources_for_tensorboard() -> bool:
    # tensorboard==2.10.x imports pkg_resources unconditionally, but some
    # newer setuptools builds no longer ship that module.
    try:
        import pkg_resources  # noqa: F401
        return False
    except ModuleNotFoundError:
        pass

    shim = types.ModuleType("pkg_resources")

    def iter_entry_points(group, name=None):
        entry_points = importlib_metadata.entry_points()
        if hasattr(entry_points, "select"):
            selected = entry_points.select(group=group)
        else:
            selected = entry_points.get(group, [])

        for ep in selected:
            if name is None or ep.name == name:
                yield _EntryPointProxy(ep)

    shim.iter_entry_points = iter_entry_points
    shim.parse_version = parse_version
    sys.modules["pkg_resources"] = shim
    return True


def _resolve_lan_ip() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
    except OSError:
        pass

    try:
        host = socket.gethostname()
        for item in socket.getaddrinfo(host, None, family=socket.AF_INET):
            ip = item[4][0]
            if ip and not ip.startswith("127."):
                return ip
    except OSError:
        pass

    return None


def _ensure_default_load_fast(argv: list[str]) -> list[str]:
    for arg in argv[1:]:
        if arg == "--load_fast" or arg.startswith("--load_fast="):
            return argv
    return [*argv, "--load_fast=false"]


def _patch_tensorboard_messages():
    import absl.logging
    from tensorboard import main_lib, program, version

    def quiet_global_init():
        # Keep TensorBoard's runtime behavior but suppress the noisy
        # "TensorFlow installation not found" startup banner.
        os.environ["GCS_READ_CACHE_DISABLED"] = "1"
        absl.logging.set_verbosity(absl.logging.WARNING)

    original_print_serving_message = program.WerkzeugServer.print_serving_message

    def patched_print_serving_message(self):
        host = getattr(self._flags, "host", None)
        if host not in ("0.0.0.0", "::"):
            original_print_serving_message(self)
            return

        path_prefix = self._flags.path_prefix.rstrip("/")
        path_suffix = f"{path_prefix}/" if path_prefix else "/"

        local_url = f"http://127.0.0.1:{self.server_port}{path_suffix}"
        lan_ip = _resolve_lan_ip()
        lan_url = (
            f"http://{lan_ip}:{self.server_port}{path_suffix}"
            if lan_ip
            else f"http://<your-lan-ip>:{self.server_port}{path_suffix}"
        )

        sys.stderr.write(
            f"TensorBoard {version.VERSION} started and listening on {host}.\n"
        )
        sys.stderr.write(f"Please visit (Local): {local_url}\n")
        sys.stderr.write(f"Please visit (LAN):   {lan_url}\n")
        sys.stderr.write("Any device on the same LAN can access this service.\n")
        sys.stderr.flush()

    main_lib.global_init = quiet_global_init
    program.WerkzeugServer.print_serving_message = patched_print_serving_message


def main():
    shim_enabled = _ensure_pkg_resources_for_tensorboard()
    if shim_enabled:
        print(
            "pkg_resources is unavailable in current environment; "
            "using compatibility shim for TensorBoard.",
            file=sys.stderr,
        )

    _patch_tensorboard_messages()

    from tensorboard import main as tensorboard_main

    argv = ["tensorboard", *sys.argv[1:]]
    sys.argv = _ensure_default_load_fast(argv)
    tensorboard_main.run_main()


if __name__ == "__main__":
    main()
