import socket, datetime
import torch
import torch.distributed as dist

def main():
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
    rank = dist.get_rank()
    world = dist.get_world_size()

    torch.cuda.set_device(0)
    x = torch.tensor([float(rank)], device="cuda")
    dist.all_reduce(x)
    print(f"[{datetime.datetime.now()}] host={socket.gethostname()} rank={rank}/{world} sum={x.item()}")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
