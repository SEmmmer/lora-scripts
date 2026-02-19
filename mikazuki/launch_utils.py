import locale
import os
import platform
import re
import shutil
import subprocess
import sys
import socket
import sysconfig
from typing import List
from pathlib import Path
from typing import Optional

try:
    import pkg_resources
except ModuleNotFoundError:
    pkg_resources = None

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata

from mikazuki.log import log

python_bin = sys.executable


def base_dir_path():
    return Path(__file__).parents[1].absolute()


def find_windows_git():
    possible_paths = ["git\\bin\\git.exe", "git\\cmd\\git.exe", "Git\\mingw64\\libexec\\git-core\\git.exe", "C:\\Program Files\\Git\\cmd\\git.exe"]
    for path in possible_paths:
        if os.path.exists(path):
            return path


def prepare_git():
    if shutil.which("git"):
        return True

    log.info("Finding git...")

    if sys.platform == "win32":
        git_path = find_windows_git()

        if git_path is not None:
            log.info(f"Git not found, but found git in {git_path}, add it to PATH")
            os.environ["PATH"] += os.pathsep + os.path.dirname(git_path)
            return True
        else:
            return False
    else:
        log.error("git not found, please install git first")
        return False


def prepare_submodules():
    frontend_dist_path = base_dir_path() / "frontend" / "dist"
    frontend_index_path = base_dir_path() / "frontend" / "index.html"
    tag_editor_script_path = base_dir_path() / "mikazuki" / "dataset-tag-editor" / "scripts" / "launch.py"
    tag_editor_alt_script_path = base_dir_path() / "mikazuki" / "dataset-tag-editor" / "launch.py"

    def repo_missing(path: Path):
        return not path.exists() or not any(path.iterdir())

    frontend_ready = os.path.exists(frontend_dist_path) or os.path.exists(frontend_index_path)
    tag_editor_ready = os.path.exists(tag_editor_script_path) or os.path.exists(tag_editor_alt_script_path)

    if frontend_ready and tag_editor_ready:
        return

    log.info("submodule assets not found, try to recover...")
    log.info("checking git installation...")
    if not prepare_git():
        log.error("git not found, please install git first")
        return

    subprocess.run(["git", "submodule", "sync", "--recursive"], check=False)
    subprocess.run(["git", "submodule", "update", "--init", "--recursive"], check=False)

    frontend_ready = os.path.exists(frontend_dist_path) or os.path.exists(frontend_index_path)
    tag_editor_ready = os.path.exists(tag_editor_script_path) or os.path.exists(tag_editor_alt_script_path)
    if frontend_ready and tag_editor_ready:
        return

    # Fallback for repos where submodule gitlinks are missing.
    fallback_repos = [
        {
            "path": base_dir_path() / "frontend",
            "url": "https://github.com/hanamizuki-ai/lora-gui-dist",
        },
        {
            "path": base_dir_path() / "mikazuki" / "dataset-tag-editor",
            "url": "https://github.com/Akegarasu/dataset-tag-editor",
        },
    ]

    for repo in fallback_repos:
        repo_path = repo["path"]
        if not repo_missing(repo_path):
            continue

        if repo_path.exists():
            try:
                repo_path.rmdir()
            except OSError:
                log.warning(f"skip fallback clone because path is not empty: {repo_path}")
                continue
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"cloning fallback repo: {repo['url']} -> {repo_path}")
        subprocess.run(["git", "clone", "--depth", "1", repo["url"], str(repo_path)], check=False)

    frontend_ready = os.path.exists(frontend_dist_path) or os.path.exists(frontend_index_path)
    tag_editor_ready = os.path.exists(tag_editor_script_path) or os.path.exists(tag_editor_alt_script_path)
    if not frontend_ready:
        log.warning("frontend assets are still missing after recovery. GUI static files will be unavailable.")
    if not tag_editor_ready:
        log.warning("dataset-tag-editor launcher is still missing after recovery. Tageditor will be unavailable.")


def git_tag(path: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", path, "describe", "--tags"]).strip().decode("utf-8")
    except Exception as e:
        return "<none>"


def check_dirs(dirs: List):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def run(command,
        desc: Optional[str] = None,
        errdesc: Optional[str] = None,
        custom_env: Optional[list] = None,
        live: Optional[bool] = True,
        shell: Optional[bool] = None):

    if shell is None:
        shell = False if sys.platform == "win32" else True

    if desc is not None:
        print(desc)

    if live:
        result = subprocess.run(command, shell=shell, env=os.environ if custom_env is None else custom_env)
        if result.returncode != 0:
            raise RuntimeError(f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}""")

        return ""

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            shell=shell, env=os.environ if custom_env is None else custom_env)

    if result.returncode != 0:
        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout) > 0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr) > 0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")


def get_installed_version(pkg_name: str) -> Optional[str]:
    candidate_names = [pkg_name, pkg_name.lower(), pkg_name.replace("_", "-")]

    if pkg_resources is not None:
        for candidate in candidate_names:
            try:
                return pkg_resources.get_distribution(candidate).version
            except Exception:
                continue

    for candidate in candidate_names:
        try:
            return importlib_metadata.version(candidate)
        except importlib_metadata.PackageNotFoundError:
            continue
        except Exception:
            continue

    return None


def is_installed(package, friendly: str = None):
    #
    # This function was adapted from code written by vladimandic: https://github.com/vladmandic/automatic/commits/master
    #

    # Remove brackets and their contents from the line using regular expressions
    # e.g., diffusers[torch]==0.10.2 becomes diffusers==0.10.2
    package = re.sub(r'\[.*?\]', '', package)

    try:
        if friendly:
            pkgs = friendly.split()
        else:
            pkgs = [
                p
                for p in package.split()
                if not p.startswith('-') and not p.startswith('=')
            ]
            pkgs = [
                p.split('/')[-1] for p in pkgs
            ]   # get only package name if installing from URL

        for pkg in pkgs:
            if '>=' in pkg:
                pkg_name, pkg_version = [x.strip() for x in pkg.split('>=')]
            elif '==' in pkg:
                pkg_name, pkg_version = [x.strip() for x in pkg.split('==')]
            else:
                pkg_name, pkg_version = pkg.strip(), None

            version = get_installed_version(pkg_name)
            if version is not None:
                # log.debug(f'Package version found: {pkg_name} {version}')

                if pkg_version is not None:
                    if '>=' in pkg:
                        ok = version >= pkg_version
                    else:
                        ok = version == pkg_version

                    if not ok:
                        log.info(f'Package wrong version: {pkg_name} {version} required {pkg_version}')
                        return False
            else:
                log.warning(f'Package version not found: {pkg_name}')
                return False

        return True
    except ModuleNotFoundError:
        log.warning(f'Package not installed: {pkgs}')
        return False


def validate_requirements(requirements_file: str):
    with open(requirements_file, 'r', encoding='utf8') as f:
        lines = [
            line.strip()
            for line in f.readlines()
            if line.strip() != ''
            and not line.startswith("#")
            and not (line.startswith("-") and not line.startswith("--index-url "))
            and line is not None
            and "# skip_verify" not in line
        ]

        index_url = ""
        for line in lines:
            if line.startswith("--index-url "):
                index_url = line.replace("--index-url ", "")
                continue

            if not is_installed(line):
                if index_url != "":
                    run_pip(f"install {line} --index-url {index_url}", line, live=True)
                else:
                    run_pip(f"install {line}", line, live=True)


def setup_windows_bitsandbytes():
    if sys.platform != "win32":
        return

    # bnb_windows_index = os.environ.get("BNB_WINDOWS_INDEX", "https://jihulab.com/api/v4/projects/140618/packages/pypi/simple")
    bnb_package = "bitsandbytes==0.46.0"
    bnb_path = os.path.join(sysconfig.get_paths()["purelib"], "bitsandbytes")

    installed_bnb = is_installed("bitsandbytes")  # don't check version here
    bnb_cuda_setup = len([f for f in os.listdir(bnb_path) if re.findall(r"libbitsandbytes_cuda.+?\.dll", f)]) != 0

    if not installed_bnb or not bnb_cuda_setup:
        log.error("detected wrong install of bitsandbytes, reinstall it")
        run_pip(f"uninstall bitsandbytes -y", "bitsandbytes", live=True)
        run_pip(f"install {bnb_package}", bnb_package, live=True)


def setup_onnxruntime(
        onnx_version: Optional[str] = None,
        index_url: Optional[str] = None
):
    if sys.platform == "linux":
        libc_ver = platform.libc_ver()
        if libc_ver[0] == "glibc" and libc_ver[1] <= "2.27":
            onnx_version = "1.16.3"

    onnx_version = os.environ.get("ONNXRUNTIME_VERSION", onnx_version)

    if onnx_version and not is_installed(f"onnxruntime-gpu=={onnx_version}"):
        log.info("uninstalling wrong onnxruntime version")
        run_pip(f"uninstall onnxruntime -y", "onnxruntime", live=True)
        run_pip(f"uninstall onnxruntime-gpu -y", "onnxruntime", live=True)

    if not is_installed(f"onnxruntime-gpu"):
        log.info(f"installing onnxruntime")
        pip_install("onnxruntime", onnx_version, index_url=index_url, live=True)
        pip_install("onnxruntime-gpu", onnx_version, index_url=index_url, live=True)


def run_pip(command, desc=None, live=False):
    return run(f'"{python_bin}" -m pip {command}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}", live=live)


def pip_install(package: str, version: Optional[str] = None, index_url: Optional[str] = None, live: bool = True):
    """
    Install a package using pip.
    :param package: The name of the package to install.
    :param version: The version of the package to install (optional).
    :param index_url: The index URL to use for installing the package (optional).
    """
    if version:
        package = f"{package}=={version}"

    command = f"install {package}"

    if index_url:
        command = f"{command} -i {index_url}"

    run_pip(command, desc=f"Installing {package}", live=live)


def check_run(file: str) -> bool:
    result = subprocess.run([python_bin, file], capture_output=True, shell=False)
    log.info(result.stdout.decode("utf-8").strip())
    return result.returncode == 0


def network_gfw_test(timeout=3):
    try:
        import requests
        # requests will auto detect system proxies
        response = requests.get("https://www.google.com", timeout=timeout)
        if response.status_code == 200:
            log.info("Network test passed")
            return True
        else:
            log.error(f"Network test failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        log.error(f"Network test failed: {e}")
        return False


def prepare_environment(disable_auto_mirror: bool = True, prepare_onnxruntime: bool = True):
    if sys.platform == "win32":
        # disable triton on windows
        os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

    if not disable_auto_mirror and not network_gfw_test():
        log.info("use pip & huggingface mirrors")
        os.environ.setdefault("PIP_FIND_LINKS", "https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html")
        os.environ.setdefault("PIP_INDEX_URL", "https://pypi.tuna.tsinghua.edu.cn/simple")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    if not os.environ.get("PATH"):
        os.environ["PATH"] = os.path.dirname(sys.executable)

    prepare_submodules()

    check_dirs(["config/autosave", "logs"])

    # if not check_run("mikazuki/scripts/torch_check.py"):
    #     sys.exit(1)

    validate_requirements("requirements.txt")
    setup_windows_bitsandbytes()

    if prepare_onnxruntime:
        setup_onnxruntime()


def catch_exception(f):
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            log.error(f"An error occurred: {e}")
    return wrapper


def check_port_avaliable(port: int):
    try:
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
        s.close()
        return True
    except:
        return False


def find_avaliable_ports(port_init: int, port_range: int):
    server_ports = range(port_init, port_range)

    for p in server_ports:
        if check_port_avaliable(p):
            return p

    log.error(f"error finding avaliable ports in range: {port_init} -> {port_range}")
    return None
