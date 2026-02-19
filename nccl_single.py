import torch
import torch.distributed as dist
import datetime, os

def main():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")

    dist.init_process_group("nccl", rank=0, world_size=1, timeout=datetime.timedelta(seconds=60))
    torch.cuda.set_device(0)
    x = torch.tensor([1.0], device="cuda")
    dist.all_reduce(x)
    print("nccl single all_reduce ok, x =", x.item())
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
