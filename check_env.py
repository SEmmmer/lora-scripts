import os
import sys
import socket
import subprocess

def run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as e:
        return f"(failed: {e})"

print("=== Python ===")
print("exe:", sys.executable)
print("version:", sys.version.split()[0])

print("\n=== Host ===")
print("hostname:", socket.gethostname())
print("uname:", run(["uname", "-a"]))

print("\n=== NVIDIA ===")
print("nvidia-smi:", run(["bash", "-lc", "nvidia-smi -L || true"]))
print("driver:", run(["bash", "-lc", "nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true"]))
print("nvcc:", run(["bash", "-lc", "nvcc -V || true"]))

print("\n=== Torch ===")
try:
    import torch
    import torch.distributed as dist
    print("torch:", torch.__version__)
    print("torch_file:", torch.__file__)
    print("cuda:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
    print("gpu_count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("gpu0:", torch.cuda.get_device_name(0))
        # NCCL version API exists when compiled with NCCL
        try:
            print("nccl_version:", torch.cuda.nccl.version())
        except Exception as e:
            print("nccl_version: (error)", repr(e))
    print("dist_nccl_available:", dist.is_nccl_available())
    print("dist_gloo_available:", dist.is_gloo_available())
except Exception as e:
    print("torch_import_failed:", repr(e))

print("\n=== Network (brief) ===")
print(run(["bash", "-lc", "ip -br a || true"]))
