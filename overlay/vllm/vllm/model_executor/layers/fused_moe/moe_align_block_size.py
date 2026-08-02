# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import triton
from vllm.utils.math_utils import round_up


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Note: In the case of expert_parallel, moe_align_block_size initially
    considers all experts as valid and aligns all tokens appropriately.
    Before the function returns it marks the experts_ids that are not in
    the current GPU rank as -1 so the MoE matmuls could skip those blocks.
    This requires the num_experts input arg to be the num global experts.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.
    - expert_map: A tensor of shape [num_experts] that maps the expert index
        from the global space to the local index space of the current
        expert parallel shard. If the expert is not in the current expert
        parallel shard, the mapping is set to -1.
    - pad_sorted_ids: A flag indicating whether the sorted_token_ids length
        should be padded to a multiple of block_size,
    - ignore_invalid_experts: A flag indicating whether to ignore invalid
        experts. When False, all expert_ids in topk_ids will participate in
        counting and ranking, but invalid experts in expert_ids will be marked
        as -1. When True, all invalid expert_ids in topk_ids will be ignored
        and will not participate in counting or ranking, and there will be no
        -1 in expert_ids.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    ops.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map if ignore_invalid_experts else None,
    )

    if expert_map is not None and not ignore_invalid_experts:
        expert_ids = expert_map[expert_ids]

    sorted_ids = _canonicalize_sorted_ids(
        sorted_ids, expert_ids, block_size, topk_ids.numel()
    )

    return sorted_ids, expert_ids, num_tokens_post_pad


# moet/sm89: the CUDA kernel assigns each token its slot within its expert via
#   rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1)
# (csrc/.../moe_align_sum_kernels.cu:318). atomicAdd returns the OLD value, so
# a token's rank is decided by thread arrival order -- which is not
# reproducible. Same inputs, different `sorted_token_ids` on every call:
# measured 1/1 immediate repeats differ. Everything downstream of it is
# deterministic (the Marlin GEMM 0/1500, _fused_marlin_moe 0/100, the TP
# all-reduce 0/100), and yet the MoE block's output changes when only this
# changes -- so a permutation here is not numerically free.
#
# Consequence at the top: greedy decoding was not reproducible. Same prompt,
# temperature 0, prompt-logprob deltas up to ~3.7 nats, and a different first
# token often enough to change the answer.
#
# The order within an expert carries no meaning -- it is a bucketing, and any
# order computes the same set of (token, expert) pairs. So canonicalize it:
# sort each expert's region by token id. Real ids are all < numel and padding
# slots hold exactly numel, so ascending order keeps the reals packed at the
# front of the region exactly as the kernel left them, only in a defined order.
#
# Off by default (VLLM_DSV4_DETERMINISTIC_MOE=1) because it is an argsort per
# MoE layer per step that upstream does not pay.
# "1" = ascending token id, "2" = descending. Both are canonical; the pair
# exists as a control. If two different fixed orders give different logits,
# then order is not numerically inert here -- which it should be, since the
# bucketing is arbitrary -- and something downstream is order-sensitive.
_DETERMINISTIC_MOE = os.environ.get("VLLM_DSV4_DETERMINISTIC_MOE", "0")


def _canonicalize_sorted_ids(
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: int,
    numel: int,
) -> torch.Tensor:
    if _DETERMINISTIC_MOE not in ("1", "2") or sorted_ids.numel() == 0:
        return sorted_ids
    n = sorted_ids.numel()
    n_blocks = expert_ids.numel()
    if n_blocks * block_size < n:
        # fewer block slots than ids: geometry we do not understand, leave it
        return sorted_ids
    # Sort within each expert's RUN of blocks, not within each block: the
    # nondeterministic rank decides a token's slot across the whole run, so a
    # token moves between blocks of the same expert. Reordering inside a run is
    # safe (every block in it uses the same expert weights, so each (token,
    # expert) pairing is preserved); reordering across runs would not be.
    #
    # Run ids come from where `expert_ids` changes, so this holds whatever the
    # layout is -- no assumption that experts appear once or in sorted order.
    eids = expert_ids.to(torch.int64)
    boundary = torch.ones_like(eids)
    boundary[1:] = (eids[1:] != eids[:-1]).to(eids.dtype)
    run_id = torch.cumsum(boundary, 0) - 1
    # `max_num_tokens_padded` is not block-aligned in general, so the per-block
    # run ids expand to slightly more slots than sorted_ids has -- truncate.
    ids = sorted_ids.to(torch.int64)
    if _DETERMINISTIC_MOE == "2":
        # descending among the real ids, but padding (== numel) must stay at
        # the back of the run or the GEMM would skip real rows.
        ids = torch.where(ids < numel, numel - 1 - ids, ids)
    key = run_id.repeat_interleave(block_size)[:n] * (numel + 1) + ids
    return sorted_ids[torch.argsort(key, stable=True)]


def batched_moe_align_block_size(
    max_tokens_per_batch: int, block_size: int, expert_num_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given num_batches, max_tokens_per_batch, block_size and the number of
    valid-tokens in each batch, prepare sorted_token_ids, expert_ids and
    num_tokens_post_pad. sorted_token_ids, expert_ids and num_tokens_post_pad
    have the same semantics as in moe_align_block_size.

    This function is intended to be a drop in replacement for
    moe_align_batch_size for the batched case.

    Parameters:
    - max_tokens_per_batch (int): Number of tokens in each batch (both
        valid and invalid).
    - block_size (int): block_size to align the data to.
    - expert_num_tokens (torch.Tensor): expert_num_tokens[i], indicates
        the number of valid tokens in batch i.

    Returns:
    - sorted_token_ids (torch.Tensor): Torch tensor of size
        (num_batches * max_tokens_per_batch) indicating the token indices for
        that block.
    - expert_ids (torch.Tensor): Torch tensor of size
        ceil((num_batches * max_tokens_per_batch) / block_size) indicating
        what expert to use for each block.
    - num_tokens_post_pad (torch.Tensor): Torch tensor of size 1
        indicating the number of valid blocks with actual data to
        process. This is represented in terms of num tokens.
    Example:
    Let num_batches=5, max_tokens_per_batch=8, block_size=4, and
    expert_num_tokens=[2, 3, 0, 6, 8]. This expert_num_tokens tensor
    indicates that,
     - The first 2 tokens in the 0th batch are valid and the rest 6 are
     invalid (i.e. in the 2D hidden_states tensor of shape,
     [num_batches * max_tokens_per_batch, K], indices 0, 1 are valid)
     - The first 3 tokens in the 1st batch are valid. i.e. indices 8, 9, 10
     - 0 tokens in the 2nd batch are valid
     - first 6 tokens in the  3rd batch are valid. i.e. indices,
     24, 25, 26, 27, 28, 29
     - so on ...

     In this case,
      sorted_token_ids will be [0, 1, 40, 40,
                                8, 9, 10, 40,
                                24, 25, 26, 27,
                                28, 29, 40, 40,
                                32, 33, 34, 35,
                                36, 37, 38, 39,
                                40, 40, 40, 40,
                                (rest all 40, 40, 40, 40)
                                ...]
      Here, 40 represents an invalid index. as there is no token index 40.
      The gemm kernel using this sorted_token_ids is expected to skip the
      gemm computation when it encounters this invalid index.

      expert_ids will be [0, 1, 3, 3, 4, 5, 5, -1, -1, (rest all -1) ...]
      Here, -1 represents an invalid expert. The gemm kernel using this
      expert_ids is expected to skip the gemm computation when it encounters
      an expert of id -1.

      num_tokens_post_pad will be 24 as sorted_token_ids has valid entries
      until 24.
    """

    B = expert_num_tokens.size(0)
    device = expert_num_tokens.device

    # Round up so each batch can be split to blocks evenly.
    max_num_tokens_padded = B * round_up(max_tokens_per_batch, block_size)

    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=device)
    assert max_num_tokens_padded % block_size == 0
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=device)

    ops.batched_moe_align_block_size(
        max_tokens_per_batch,
        block_size,
        expert_num_tokens,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )

    return sorted_ids, expert_ids, num_tokens_post_pad
