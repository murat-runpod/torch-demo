import os
import torch
import torch.distributed as dist

def init_distributed():
    """Initialize the distributed training environment"""
    # Initialize the process group
    dist.init_process_group(backend="nccl")
    
    # Get local rank and global rank
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set device for this process
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    print(f"Running on rank {global_rank}/{world_size-1} (local rank: {local_rank})")
    
    return local_rank, global_rank, world_size, device

def cleanup_distributed():
    """Clean up the distributed environment"""
    dist.destroy_process_group()

def main():
    # Initialize distributed environment
    local_rank, global_rank, world_size, device = init_distributed()
    
    # Create a vector filled with this rank's value
    vector_size = 5
    vector = torch.ones(vector_size, device=device) * global_rank
    
    print(f"Rank {global_rank} original vector: {vector}")
    
    # Perform all-reduce operation (sum)
    dist.all_reduce(vector, op=dist.ReduceOp.SUM)
    
    # After all_reduce, all ranks have the same vector: the sum of all original vectors
    print(f"Rank {global_rank} after all_reduce: {vector}")
    
    # Calculate what the expected sum should be
    # Sum of arithmetic sequence 0 to (world_size-1) for each element
    expected_sum = world_size * (world_size - 1) / 2  # Sum of 0 to (world_size-1)
    print(f"Rank {global_rank} expected sum: {expected_sum}")
    
    # Clean up distributed environment when done
    cleanup_distributed()
    
if __name__ == "__main__":
    main()