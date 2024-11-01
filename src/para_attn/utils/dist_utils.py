from typing import Callable

import torch

if torch.__version__ >= "2.5.0":
    from torch.distributed._symmetric_memory import _get_backend_stream, get_symm_mem_workspace
else:
    raise RuntimeError("PyTorch version >= 2.5.0 is required")


def pipelined_dual_all_gather_and_consume(
    shard_1: torch.Tensor,
    shard_2: torch.Tensor,
    shard_consumer: Callable[[torch.Tensor, torch.Tensor, int], None],
    ag_out_1: torch.Tensor,
    ag_out_2: torch.Tensor,
    group_name: str,
) -> None:
    """
    Perform the following logic with micro-pipelined computation and
    communication:

        tensor = all_gather_tensor(shard, gather_dim=1, group=group)
        chunks = tensor.chunk(group.size())
        for src_rank, chunk in enumerate(chunks):
            shard_consumer(chunk, src_rank)

    NOTE:
    - The shard passed to shard consumer will always be contiguous.
    """
    p2p_workspace_size_req_1 = shard_1.numel() * shard_1.element_size()
    p2p_workspace_size_req_2 = shard_2.numel() * shard_2.element_size()
    symm_mem = get_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req_1 + p2p_workspace_size_req_2)
    group_size = symm_mem.world_size
    rank = symm_mem.rank

    backend_stream = _get_backend_stream()
    backend_stream.wait_stream(torch.cuda.current_stream())
    local_p2p_buf_1 = symm_mem.get_buffer(rank, shard_1.shape, shard_1.dtype)
    local_p2p_buf_2 = symm_mem.get_buffer(rank, shard_2.shape, shard_2.dtype)

    chunks_1 = ag_out_1.chunk(group_size)
    chunks_2 = ag_out_2.chunk(group_size)

    # While consuming local shard, copy it to the local p2p buffer
    # in another stream.
    shard_consumer(shard_1, shard_2, rank)
    chunks_1[rank].copy_(shard_1)
    chunks_2[rank].copy_(shard_2)

    with torch.cuda.stream(backend_stream):
        local_p2p_buf_1.copy_(shard_1)
        local_p2p_buf_2.copy_(shard_2)
        symm_mem.barrier(channel=0)
    torch.cuda.current_stream().wait_stream(backend_stream)

    # At this point, all ranks have copied their local shard to
    # their local p2p buffer. Each rank can now copy and consume
    # remote shards.
    for step in range(1, group_size):
        if step % 2 == 0:
            stream = torch.cuda.current_stream()
        else:
            stream = backend_stream
        remote_rank = (step + rank) % group_size
        remote_p2p_buf_1 = symm_mem.get_buffer(remote_rank, shard_1.shape, shard_1.dtype)
        remote_p2p_buf_2 = symm_mem.get_buffer(remote_rank, shard_2.shape, shard_2.dtype)
        with torch.cuda.stream(stream):
            chunks_1[remote_rank].copy_(remote_p2p_buf_1)
            chunks_2[remote_rank].copy_(remote_p2p_buf_2)
            shard_consumer(chunks_1[remote_rank], chunks_2[remote_rank], remote_rank)

    with torch.cuda.stream(backend_stream):
        symm_mem.barrier(channel=group_size % 2)
    torch.cuda.current_stream().wait_stream(backend_stream)
