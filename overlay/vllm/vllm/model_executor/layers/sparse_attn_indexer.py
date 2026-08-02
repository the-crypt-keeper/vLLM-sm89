# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import os

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config
from vllm.distributed import get_dcp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    has_deep_gemm,
)
from vllm.utils.import_utils import has_cutedsl
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.attention.ops.dcp_sparse_topk import (
    MAX_FP32_EXACT_ID,
    dcp_gather_topk_scores,
    dcp_local_pos_to_global,
    dcp_merge_global_topk,
)
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32

# ---------------------------------------------------------------------------
# moet/sm89 diagnostic: decode top-k ORACLE PROBE.
#
# Disabled unless VLLM_DSV4_TOPK_ORACLE names an output prefix. When on, each
# qualifying decode row is re-selected in plain PyTorch (torch.topk over that
# row's VALID compressed columns) and every in-tree selector is scored against
# that exact answer. The question it answers: when decode and prefill disagree
# past 2048, is the SELECTION wrong, or are the LOGITS already different?
#
#   VLLM_DSV4_TOPK_ORACLE        output prefix; unset/empty = off
#   VLLM_DSV4_TOPK_ORACLE_MIN    skip rows whose compressed seq_len is below this
#   VLLM_DSV4_TOPK_ORACLE_MAX    stop after this many recorded rows (per rank)
#   VLLM_DSV4_TOPK_ORACLE_LAYER  substring filter on the layer name
#   VLLM_DSV4_TOPK_ORACLE_RADIX  1 = also run persistent_topk into a scratch
#                                buffer and score IT against the oracle
#   VLLM_DSV4_TOPK_ORACLE_DET    1 = recompute the logits and report whether the
#                                kernel is bitwise reproducible on identical input
#
# This syncs the device, allocates, and writes a file per rank on every decode
# step -- a bisection tool, not something to leave on. Requires
# CUDAGRAPH_MODE=NONE: the host-side branching here cannot run inside a
# captured graph, and under capture it would be traced once and never again.
_ORACLE_PATH = os.environ.get("VLLM_DSV4_TOPK_ORACLE", "")
_ORACLE_MIN = int(os.environ.get("VLLM_DSV4_TOPK_ORACLE_MIN", "0"))
_ORACLE_MAX = int(os.environ.get("VLLM_DSV4_TOPK_ORACLE_MAX", "20000"))
_ORACLE_LAYER = os.environ.get("VLLM_DSV4_TOPK_ORACLE_LAYER", "")
_ORACLE_RADIX = os.environ.get("VLLM_DSV4_TOPK_ORACLE_RADIX", "0") == "1"
_ORACLE_DET = os.environ.get("VLLM_DSV4_TOPK_ORACLE_DET", "0") == "1"
# Number of TRAILING query rows of each prefill chunk to probe (0 = off). The
# last row of a prefill sees exactly the context a decode step would see at the
# same length, so its record is directly comparable to a decode record with the
# same seq_len -- which is the whole point: bug #2 is a decode/prefill
# divergence, and this says which side of it is wrong.
_ORACLE_PREFILL = int(os.environ.get("VLLM_DSV4_TOPK_ORACLE_PREFILL", "0"))
# 1 = also dump the RAW selected index vector, in emission order, as "sel_idx".
# The scored `sel` field only ever answers "is this the right SET" -- it is
# computed against torch.topk and reports hit/miss/dup counts, so two runs that
# select the same 512 candidates in a DIFFERENT ORDER both score exact:true and
# look identical. The sparse-MLA attention accumulates candidates in index
# order and float addition is not associative, so order is not free: a permuted
# but equal set shifts the output by ~1 bf16 ULP (measured). This dumps the
# vector itself so two runs can be diffed ELEMENT-WISE rather than as sets.
_ORACLE_RAW = os.environ.get("VLLM_DSV4_TOPK_ORACLE_RAW", "0") == "1"
_oracle_state: dict = {"rows": 0, "step": 0, "fh": None}


def _oracle_active(layer_name) -> bool:
    if not _ORACLE_PATH or _oracle_state["rows"] >= _ORACLE_MAX:
        return False
    if _ORACLE_LAYER and _ORACLE_LAYER not in str(layer_name):
        return False
    return True


def _oracle_fh():
    fh = _oracle_state["fh"]
    if fh is None:
        try:
            rank = torch.distributed.get_rank()
        except Exception:
            rank = os.getpid()
        path = f"{_ORACLE_PATH}.rank{rank}.jsonl"
        fh = open(path, "a", buffering=1)
        _oracle_state["fh"] = fh
        logger.info(
            "DSv4 decode top-k ORACLE PROBE active -> %s "
            "(min_seq=%d max_rows=%d radix=%s det=%s layer=%r)",
            path,
            _ORACLE_MIN,
            _ORACLE_MAX,
            _ORACLE_RADIX,
            _ORACLE_DET,
            _ORACLE_LAYER or "*",
        )
    return fh


def _oracle_score_selection(sel_row, oracle_idx, oracle_vals, row, seq_len):
    """Score one selector's row against the exact top-k for that row."""
    k_eff = oracle_idx.numel()
    sel = sel_row[sel_row >= 0]
    n_pad = int(sel_row.numel() - sel.numel())
    oob = int((sel >= seq_len).sum())
    in_range = sel[sel < seq_len]
    uniq = torch.unique(in_range)
    dup = int(in_range.numel() - uniq.numel())
    oracle_set = torch.zeros(seq_len, dtype=torch.bool, device=row.device)
    oracle_set[oracle_idx] = True
    hit = int(oracle_set[uniq].sum())
    thresh = float(oracle_vals[-1])
    got = float(row[uniq].sum()) if uniq.numel() else 0.0
    want = float(oracle_vals.sum())
    return {
        "k_eff": k_eff,
        "n_sel": int(sel.numel()),
        "n_pad": n_pad,
        "oob": oob,
        "dup": dup,
        "hit": hit,
        "miss": k_eff - hit,
        "exact": hit == k_eff and oob == 0 and dup == 0,
        # score mass actually gathered vs the best possible; 1.0 means the
        # misses were all exact ties and cost the model nothing.
        "mass_ratio": (got / want) if want != 0.0 else None,
        "thresh": thresh,
        "min_sel": float(row[uniq].min()) if uniq.numel() else None,
    }


def _oracle_probe_slots(kv, block_table, b, cols):
    """Read the indexer KV-cache entries backing specific candidate columns.

    Layout is uint8 [num_blocks, block_size, 1, D+4]: D quantized k bytes then
    a 4-byte little-endian f32 dequant scale. A candidate whose logit is NaN
    can only get there through that scale (the fp8 byte decode cannot produce
    NaN), so this says whether the slot was never written, written with a
    degenerate scale, or written fine.
    """
    kvf = kv.reshape(kv.shape[0], kv.shape[1], -1)
    block_size = kvf.shape[1]
    D = kvf.shape[2] - 4
    out = []
    for c in cols:
        blk = c // block_size
        if blk >= block_table.shape[1]:
            out.append({"col": c, "err": "block_table too short"})
            continue
        phys = int(block_table[b, blk])
        if not (0 <= phys < kvf.shape[0]):
            out.append({"col": c, "phys": phys, "err": "phys out of range"})
            continue
        ent = kvf[phys, c % block_size]
        kb = ent[:D]
        sb = ent[D : D + 4].to(torch.int32)
        raw = int(sb[0] | (sb[1] << 8) | (sb[2] << 16) | (sb[3] << 24)) & 0xFFFFFFFF
        # int32 is signed; wrap the raw uint32 before bitcasting.
        signed = raw - 0x100000000 if raw >= 0x80000000 else raw
        scale = float(
            torch.tensor([signed], dtype=torch.int32).view(torch.float32)[0]
        )
        out.append(
            {
                "col": c,
                "phys": phys,
                "slot": c % block_size,
                "scale_bits": f"0x{raw:08x}",
                "scale": None if scale != scale else scale,
                "k_nonzero": int((kb != 0).sum()),
                "k_all_ff": int((kb == 255).sum()),
                # full entry for the first few records: locates the scale
                # empirically instead of trusting the assumed offset.
                "raw": ent.tolist() if _oracle_state["rows"] < 2 else None,
                "kv_shape": list(kv.shape) if _oracle_state["rows"] < 2 else None,
            }
        )
    return out


def _oracle_row(row, seq_len, k_select, sel_row):
    """Per-row record: the row's health, plus the selector scored against exact
    top-k. `row` must already be sliced to this row's VALID candidate span and
    `sel_row` rebased so its indices are relative to that span."""
    k_eff = min(k_select, seq_len)
    oracle_vals, oracle_idx = torch.topk(row, k_eff)
    thresh = float(oracle_vals[-1])
    finite = torch.isfinite(row)
    return {
        "seq_len": seq_len,
        "k": k_select,
        "nan": int(torch.isnan(row).sum()),
        "inf": int(torch.isinf(row).sum()),
        # exact-tie multiplicity at the cut line: a "miss" that swaps two
        # equal scores is not an error, it is a tiebreak difference.
        "ties_at_thresh": int((row == thresh).sum()),
        # WHERE the bad columns are. Recorded as distance from the end of the
        # valid region (seq_len - 1 - col), because the suspect is the most
        # recent compressed block: 0 = last candidate.
        "nan_pos": (seq_len - 1 - torch.nonzero(torch.isnan(row)).flatten()[:8]).tolist(),
        # exactly 0.0 == an all-zero (never-written) compressed cache slot:
        # k == 0 -> every q.k dot is 0 -> relu -> acc lands on exactly 0.0.
        "zero_pos": (seq_len - 1 - torch.nonzero(row == 0.0).flatten()[:8]).tolist(),
        "n_zero": int((row == 0.0).sum()),
        "finite_max": float(row[finite].max()) if int(finite.sum()) else None,
        "finite_min": float(row[finite].min()) if int(finite.sum()) else None,
        "n_finite": int(finite.sum()),
        "sel": _oracle_score_selection(sel_row, oracle_idx, oracle_vals, row, seq_len),
    }


def _oracle_record_prefill(layer_name, logits, ks, ke, sel_buf, topk_tokens):
    """Same measurement on the prefill side. Only the trailing rows of the
    chunk: the last query row of a prefill sees the same context a decode step
    at that length would."""
    import json

    fh = _oracle_fh()
    num_rows = logits.shape[0]
    ks_l = ks.tolist()
    ke_l = ke.tolist()
    for r in range(max(0, num_rows - _ORACLE_PREFILL), num_rows):
        if _oracle_state["rows"] >= _ORACLE_MAX:
            break
        s, e = int(ks_l[r]), int(ke_l[r])
        seq_len = e - s
        if seq_len < _ORACLE_MIN or seq_len <= 0:
            continue
        raw = sel_buf[r]
        sel_row = torch.where(raw >= 0, raw - s, torch.full_like(raw, -1))
        rec = _oracle_row(logits[r, s:e].float(), seq_len, topk_tokens, sel_row)
        rec.update(
            {
                "phase": "prefill",
                "step": _oracle_state["step"],
                "layer": str(layer_name),
                "row": r,
                "n_query_rows": num_rows,
                # ks > 0 means this row's candidate span does not start at
                # token 0 (windowing) -- only ks == 0 rows are directly
                # comparable to a decode row of the same length.
                "ks": s,
                "ke": e,
            }
        )
        if _ORACLE_RAW:
            rec["sel_idx"] = sel_row.tolist()
        fh.write(json.dumps(rec) + "\n")
        _oracle_state["rows"] += 1
    _oracle_state["step"] += 1


def _oracle_record(
    layer_name,
    logits,
    logits_dup,
    seq_lens,
    k_select,
    sel_buf,
    radix_buf,
    num_rows,
    next_n,
    kv=None,
    block_table=None,
):
    import json

    fh = _oracle_fh()
    step = _oracle_state["step"]
    _oracle_state["step"] = step + 1

    sl = seq_lens.reshape(-1)
    if sl.numel() != num_rows:
        sl = sl.repeat_interleave(max(1, num_rows // max(1, sl.numel())))
    n_cols = logits.shape[1]

    det = None
    if logits_dup is not None:
        diff = (logits != logits_dup) & ~(torch.isnan(logits) & torch.isnan(logits_dup))
        det = {
            "ne": int(diff.sum()),
            "max_abs": float((logits - logits_dup).abs().nan_to_num().max()),
        }

    for r in range(num_rows):
        seq_len = int(sl[r])
        if seq_len < _ORACLE_MIN or seq_len <= 0:
            continue
        if _oracle_state["rows"] >= _ORACLE_MAX:
            break
        row = logits[r, :seq_len].float()
        rec = _oracle_row(row, seq_len, k_select, sel_buf[r])
        rec.update(
            {
                "phase": "decode",
                "step": step,
                "layer": str(layer_name),
                "row": r,
                "next_n": next_n,
                "n_cols": n_cols,
            }
        )
        # Does anything beyond the valid region carry a real value? On sm89 the
        # Triton port allocates the logits buffer fresh and -inf-filled every
        # call, so clean_logits=False costs nothing here -- this confirms it.
        if seq_len < n_cols:
            tail = logits[r, seq_len:].float()
            tf = tail[torch.isfinite(tail)]
            rec["tail"] = {"n": int(tail.numel()), "n_finite": int(tf.numel())}
        if kv is not None and block_table is not None:
            bad = [seq_len - 1 - d for d in rec["nan_pos"][:3]]
            zero = [seq_len - 1 - d for d in rec["zero_pos"][:1]]
            # a healthy column for contrast: the argmax is guaranteed non-NaN
            # only if the row has finite values, so pick from the oracle set.
            good = [int(torch.topk(torch.nan_to_num(row, nan=-1e30), 1).indices[0])]
            rec["slots"] = {
                "nan": _oracle_probe_slots(kv, block_table, r // max(1, next_n), bad),
                "zero": _oracle_probe_slots(kv, block_table, r // max(1, next_n), zero),
                "good": _oracle_probe_slots(kv, block_table, r // max(1, next_n), good),
            }
        if radix_buf is not None:
            k_eff = min(k_select, seq_len)
            ov, oi = torch.topk(row, k_eff)
            rec["radix"] = _oracle_score_selection(radix_buf[r], oi, ov, row, seq_len)
        if det is not None:
            rec["logits_det"] = det
        fh.write(json.dumps(rec) + "\n")
        _oracle_state["rows"] += 1


def _assert_cutedsl_dcp_merge_supported(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    k: int,
) -> None:
    # The DCP merge only supports the CuteDSL path (Triton pack kernel + CuteDSL
    # stable-topk selector); there is no PyTorch fallback. The first cut targets
    # Blackwell/Hopper with index_topk in (512, 1024, 2048) (the selector's radix
    # sizing); the Triton pack itself has no shape/topk constraints.
    if not has_cutedsl():
        raise RuntimeError(
            "DCP sparse-indexer merge requires CuteDSL; install it or disable DCP."
        )
    if logits.device.type != "cuda":
        raise RuntimeError("DCP sparse-indexer merge requires CUDA tensors.")
    if logits.dtype != torch.float32 or topk_indices.dtype != torch.int32:
        raise RuntimeError(
            "DCP sparse-indexer merge requires fp32 logits and int32 indices."
        )
    if k not in (512, 1024, 2048):
        raise RuntimeError(
            f"DCP sparse-indexer merge requires index_topk in (512, 1024, 2048); "
            f"got {k}."
        )


def _merge_dcp_topk_global(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
    row_starts: torch.Tensor | None = None,
) -> None:
    """Merge each DCP rank's local top-K into the global top-K.

    ``topk_indices`` are this rank's local top-K positions into its 1/N KV
    shard. A token in the global top-K must also be in its owning rank's local
    top-K (at most ``topk_tokens - 1`` tokens rank globally above it, hence at
    most that many on its own rank), so exchanging only the per-rank local
    candidates is exact -- equivalent to all-gathering the full logit matrix,
    but it ships ``dcp_world_size * topk_tokens`` candidates instead of the whole
    score row. Overwrites ``topk_indices`` with global token ids (``-1`` for
    padding); the attention backend localizes them back to physical slots per
    rank.
    """
    if dcp_world_size <= 1:
        return

    # CuteDSL-only path (no PyTorch fallback): Triton-pack each rank's
    # (score, global_id) candidates on-device, all-gather, then the CuteDSL
    # stable-topk selector.
    _assert_cutedsl_dcp_merge_supported(logits, topk_indices, topk_tokens)
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        pack_dcp_topk_candidates_cutedsl,
        stable_topk_from_gathered_candidates_cutedsl,
    )

    packed = torch.empty(
        (*topk_indices.shape, 2),
        dtype=torch.float32,
        device=topk_indices.device,
    )
    pack_dcp_topk_candidates_cutedsl(
        logits,
        topk_indices,
        packed,
        dcp_rank,
        dcp_world_size,
        cp_interleave,
        row_starts,
    )
    gathered = get_dcp_group().all_gather(packed, dim=1)
    stable_topk_from_gathered_candidates_cutedsl(
        gathered, topk_tokens, out=topk_indices
    )


@triton.jit
def _fused_indexer_q_rope_quant_kernel(
    positions,
    q,
    q_s0,
    q_s1,
    cos_sin_cache,
    cos_sin_s0,
    q_fp8,
    q_fp8_s0,
    q_fp8_s1,
    weights,
    weights_s0,
    weights_s1,
    weights_out,
    weights_out_s0,
    weights_out_s1,
    softmax_scale,
    head_scale,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    is_neox: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offs32 = tl.arange(0, 32)
    offs64 = tl.arange(0, 64)

    pos = tl.load(positions + token)
    cos = tl.load(cos_sin_cache + pos * cos_sin_s0 + offs32).to(tl.float32)
    sin = tl.load(cos_sin_cache + pos * cos_sin_s0 + 32 + offs32).to(tl.float32)
    q_base = q + token * q_s0 + head * q_s1
    out_base = q_fp8 + token * q_fp8_s0 + head * q_fp8_s1

    if is_neox:
        # NeoX layout, x0 = q[0:32], x1 = q[32:64]
        x0 = tl.load(q_base + offs32).to(tl.float32)
        x1 = tl.load(q_base + 32 + offs32).to(tl.float32)
    else:
        # interleaved layout
        # x0 = q[0, 2, 4, ...], x1 = q[1, 3, 5, ...]
        x0 = tl.load(q_base + offs32 * 2).to(tl.float32)
        x1 = tl.load(q_base + offs32 * 2 + 1).to(tl.float32)
    r0 = (x0 * cos - x1 * sin).to(tl.bfloat16).to(tl.float32)
    r1 = (x1 * cos + x0 * sin).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(r0)), tl.max(tl.abs(r1)))

    q_nope = tl.load(q_base + 64 + offs64).to(tl.float32)
    amax = tl.maximum(amax, tl.max(tl.abs(q_nope)))
    scale_raw = tl.maximum(amax, 1e-10) * (1.0 / fp8_max)
    # e8m0 format
    q_scale = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))

    if is_neox:
        tl.store(
            out_base + offs32,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + 32 + offs32,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    else:
        tl.store(
            out_base + offs32 * 2,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + offs32 * 2 + 1,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    tl.store(
        out_base + 64 + offs64,
        tl.clamp(q_nope / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
    )

    weight = tl.load(weights + token * weights_s0 + head * weights_s1).to(tl.float32)
    tl.store(
        weights_out + token * weights_out_s0 + head * weights_out_s1,
        weight * q_scale * softmax_scale * head_scale,
    )


def fused_indexer_q_rope_quant(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert current_platform.is_cuda()
    assert q.dtype == torch.bfloat16
    assert q.shape[-1] == 128
    assert cos_sin_cache.shape[-1] == 64
    assert weights.shape == q.shape[:2]

    q_fp8 = torch.empty_like(q, dtype=current_platform.fp8_dtype())
    weights_out = torch.empty_like(weights, dtype=torch.float32)
    fp8_min, fp8_max = get_fp8_min_max()
    _fused_indexer_q_rope_quant_kernel[(q.shape[0], q.shape[1])](
        positions,
        q,
        q.stride(0),
        q.stride(1),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights,
        weights.stride(0),
        weights.stride(1),
        weights_out,
        weights_out.stride(0),
        weights_out.stride(1),
        softmax_scale,
        head_scale,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        is_neox=is_neox,
        num_warps=1,
    )
    return q_fp8, weights_out


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


def _dcp_merge_topk_into_buffer(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    interleave: int,
    col_offset: torch.Tensor | None = None,
) -> None:
    """Lift local top-k winners to global ids and reduce across the DCP group.

    ``topk_indices`` (int32, ``-1`` pad) hold *local-shard* positions and are
    overwritten in place with the merged *global* top-k (identical on every
    rank). ``col_offset`` rebases per-row positions into the shared prefill
    logits workspace for the score gather; decode logits are already
    request-relative.
    """
    from vllm.distributed.parallel_state import get_dcp_group

    dcp_group = get_dcp_group()
    scores = dcp_gather_topk_scores(logits, topk_indices, col_offset=col_offset)
    global_ids = dcp_local_pos_to_global(
        topk_indices,
        dcp_group.world_size,
        dcp_group.rank_in_group,
        interleave,
    )
    merged = dcp_merge_global_topk(
        global_ids, scores, topk_indices.shape[1], dcp_group
    )
    topk_indices.copy_(merged)


@eager_break_during_capture
def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
    # moet: DCP local-quota mode - per-rank top-(k/N), no cross-rank merge
    dcp_local_topk: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        # Dummy allocation to simulate for peak logits tensor memory during inference.
        # FP8 elements so elements == bytes
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_fp4_cache,
            dcp_world_size,
            cp_kv_cache_interleave_size,
            dcp_local_topk,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

    # The buffer must be pre-filled with -1 (the "no token" sentinel) before the
    # top-k kernels scatter valid indices into it. On the fused deepseek_v32
    # nvidia path, _fused_norm_rope_kernel already cleared the same
    # [:num_tokens, :topk] region earlier in this forward, so skip the redundant
    # fill.
    if not skip_topk_buffer_clear:
        topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        for chunk in prefill_metadata.chunks:
            cu_seqlen_ks = chunk.cu_seqlen_ks
            cu_seqlen_ke = chunk.cu_seqlen_ke
            assert chunk.local_cu_seq_lens is not None
            k_quant = k_quant_full[: chunk.max_local_total_seq_lens]
            k_scale = k_scale_full[: chunk.max_local_total_seq_lens]
            if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.local_cu_seq_lens,
                )

            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            if chunk.local_total_seq_lens == 0:
                logits = q_slice.new_empty((q_slice.shape[0], 0), dtype=torch.float32)
                topk_indices.fill_(-1)
            else:
                # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
                # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
                if use_fp4_cache:
                    q_slice_cast = q_slice.view(torch.int8)
                    k_quant_cast = k_quant.view(torch.int8)
                    k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
                else:
                    q_slice_cast = q_slice
                    k_quant_cast = k_quant
                    k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
                if current_platform.is_xpu():
                    if q_scale_slice is not None:
                        raise RuntimeError("XPU fp8_mqa_logits does not support FP4 Q")
                    logits = torch.ops.vllm.xpu_fp8_mqa_logits(
                        q_slice_cast,
                        k_quant_cast,
                        k_scale_cast,
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                    )
                else:
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )
                num_rows = logits.shape[0]
                ops.top_k_per_row_prefill(
                    logits,
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
                if _ORACLE_PREFILL and _oracle_active(k_cache_prefix):
                    _oracle_record_prefill(
                        k_cache_prefix,
                        logits,
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        topk_indices,
                        topk_tokens,
                    )

            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
                row_starts=chunk.cu_seqlen_ks,
            )

            if dcp_world_size > 1:
                # Local winners -> global ids -> group-wide top-k. The score
                # gather needs workspace columns, so rebase the per-request
                # positions by each row's K start (cu_seqlen_ks).
                _dcp_merge_topk_into_buffer(
                    logits,
                    topk_indices,
                    cp_kv_cache_interleave_size,
                    col_offset=chunk.cu_seqlen_ks,
                )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        if current_platform.is_xpu():
            if padded_q_scale is not None:
                raise RuntimeError("XPU fp8_paged_mqa_logits does not support FP4 Q")
            seq_lens_xpu = (
                seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
            )
            logits = torch.ops.vllm.xpu_fp8_paged_mqa_logits(
                padded_q_quant_cast,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens_xpu,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len,
            )
        else:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
        num_rows = logits.shape[0]

        # ORACLE PROBE (off by default): is the logits kernel itself bitwise
        # reproducible on identical input? Keep the first result before the
        # second call -- the workspace buffer is reused, so the two would alias.
        # (Only the VALID columns are comparable: with clean_logits=False the
        # second pass sees the first pass's output as its stale tail.)
        oracle_logits_dup = None
        if _ORACLE_DET and _oracle_active(k_cache_prefix) and not current_platform.is_xpu():
            oracle_logits_dup = logits.clone()
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )

        # DCP local-quota mode: select each rank's local top-(k/N) and skip
        # the per-layer cross-rank merge entirely (~21 sync points per decode
        # step otherwise). The attention-side LSE merge weights the per-rank
        # partial outputs exactly; only the candidate SET is approximate
        # (union of per-rank quotas instead of the global top-k, a close
        # match under interleave=1 striping). The buffer tail beyond the
        # quota stays -1 and is skipped by the sparse kernel.
        dcp_local_quota = dcp_world_size > 1 and dcp_local_topk
        k_select = topk_tokens // dcp_world_size if dcp_local_quota else topk_tokens
        topk_indices = topk_indices_buffer[:num_padded_tokens, :k_select]

        use_cooperative_topk = (
            current_platform.is_cuda()
            and k_select in (512, 1024, 2048)
            and num_rows <= 32
            and logits.stride(0) % 4 == 0  # TMA 16-byte alignment
            and current_platform.has_device_capability(90)
            # thread-block cluster launch is SM90/SM100-only; consumer
            # Blackwell (SM12x) rejects it with "invalid argument"
            and not current_platform.is_device_capability_family(120)
        )
        use_persistent_topk = current_platform.is_cuda() and k_select in (
            512,
            1024,
            2048,
        )
        # sm89 (Ada) correctness fix, 2026-08-01. DO NOT re-enable the radix
        # selectors (cooperative_topk / persistent_topk) without reading this.
        #
        # They are unsafe here for TWO independent reasons. Fixing only the
        # first moves the crash, it does not remove it -- measured, not argued:
        #
        # (1) SCAN BOUND UNITS -- fixed at source in `mla/indexer.py`
        #     (`if self.compress_ratio > 1: max_seq_len //= ...`). The bound
        #     used to arrive UNCOMPRESSED while `seq_lens` in the same call was
        #     already compressed, so up to 4x too many columns were eligible and
        #     stale data (the logits buffer is built with clean_logits=False)
        #     ranked against real candidates. Symptom: NaN logits row at
        #     absolute position ~2053 -- compressed count 513, i.e. the first
        #     step where real ranking happens -> sampler emits BOS with a null
        #     logprob -> the rest of the generation collapses into token salad.
        #
        # (2) `k_select` IS NEVER CLAMPED to the compressed candidate count, and
        #     this is still unfixed. With (1) corrected and the selectors turned
        #     back on, the NaN simply MOVED DOWN to absolute position 1772 --
        #     compressed count 443 < k=512. The kernel is asked for 512 items
        #     from 443 valid columns and cannot fill them. Before (1) was fixed,
        #     the 4x-too-wide bound accidentally guaranteed >= 512 columns to
        #     draw from, which is precisely why the failure used to start only
        #     above 2048. Verified 2026-08-01: NaN in 1/8 generations.
        #
        # All four reference implementations clamp -- DeepSeek's own
        # inference/model.py:433 `topk(min(index_topk, end_pos // ratio))`,
        # antirez/DS.cpp ds4.c:12886, llama.cpp deepseek4.cpp:613, SGLang
        # dsv4/indexer.py:296. vLLM is the only one that does not.
        #
        # The clamp is PER ROW while these kernels take a scalar k, so a correct
        # re-enable needs a gate like "every row in this decode batch has >= k
        # valid compressed candidates". That needs an exact host-side minimum;
        # `common_attn_metadata.seq_lens_cpu_upper_bound` is an UPPER bound and
        # is therefore the wrong direction for such a gate.
        #
        # `top_k_per_row_decode` takes no scan bound and clamps per row via the
        # compressed `seq_lens`, so it is correct by construction for both
        # regimes. It is slower; correctness first. Eval with this path:
        # 0.985 +/- 0.006, invalid 0.0000 (radix path: 0.952 +/- 0.011,
        # invalid 0.0217).
        #
        # KNOWN RESIDUAL: decode/prefill selection still diverges past 2048 and
        # compounds with depth, stalling termination past ~6K generated tokens
        # (bug #2 in CLAUDE-TP8.md). This path fixed the NaN, not that.
        #
        # NOTE: unconditional on purpose -- worker processes do not inherit an
        # env knob set on the API server, so gating this by env silently no-ops.
        use_cooperative_topk = False
        use_persistent_topk = False

        if use_cooperative_topk:
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.cooperative_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                k_select,
                attn_metadata_narrowed.max_seq_len,
            )
        elif use_persistent_topk:
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                k_select,
                attn_metadata_narrowed.max_seq_len,
            )
        else:
            ops.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                k_select,
            )

        if _oracle_active(k_cache_prefix):
            # Score whatever just ran against exact top-k, and -- when every
            # row can actually satisfy k -- run the radix selector into a
            # scratch buffer on the SAME logits so the two are directly
            # comparable. Scratch, never the live buffer: this must not change
            # what the model attends to.
            oracle_radix = None
            if (
                _ORACLE_RADIX
                and not use_persistent_topk
                and not use_cooperative_topk
                and current_platform.is_cuda()
                and k_select in (512, 1024, 2048)
                and int(seq_lens.min()) >= k_select
            ):
                oracle_radix = torch.full_like(topk_indices, -1)
                (oracle_ws,) = current_workspace_manager().get_simultaneous(
                    ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
                )
                torch.ops._C.persistent_topk(
                    logits,
                    seq_lens,
                    oracle_radix,
                    oracle_ws,
                    k_select,
                    attn_metadata_narrowed.max_seq_len,
                )
            _oracle_record(
                k_cache_prefix,
                logits,
                oracle_logits_dup,
                seq_lens,
                k_select,
                topk_indices,
                oracle_radix,
                num_rows,
                next_n,
                kv=kv_cache,
                block_table=decode_metadata.block_table,
            )

        if dcp_local_quota:
            # Quota tail is dead space for this step; keep the -1 padding
            # convention so downstream conversion and the sparse kernel
            # skip it (the buffer may hold stale ids from earlier steps).
            topk_indices_buffer[:num_padded_tokens, k_select:topk_tokens] = -1
            from vllm.distributed.parallel_state import get_dcp_group

            dcp_group = get_dcp_group()
            topk_indices.copy_(
                dcp_local_pos_to_global(
                    topk_indices,
                    dcp_group.world_size,
                    dcp_group.rank_in_group,
                    cp_kv_cache_interleave_size,
                )
            )
        elif dcp_world_size > 1:
            # Exact mode: decode logits columns are local-shard positions per
            # row; merge before the (optional) unpack so padded rows simply
            # ride along with -inf scores and stay -1.
            _dcp_merge_topk_into_buffer(
                logits,
                topk_indices,
                cp_kv_cache_interleave_size,
            )

        if decode_metadata.global_seq_lens is not None:
            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
            )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
    # moet: DCP local-quota mode - per-rank top-(k/N), no cross-rank merge
    dcp_local_topk: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        # DCP scalars are constant for the run; resolve them here (config is set
        # during model construction) and pass them into the custom op, rather
        # than threading them through per-step metadata.
        parallel_config = get_current_vllm_config().parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        if current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM support in "
                "the current vLLM environment."
            )

        # (dcp_world_size / dcp_rank / cp_kv_cache_interleave_size were
        # resolved above from the current config - upstream's derivation;
        # the pre-0.25 re-derivation block that lived here made
        # get_current_vllm_config/get_dcp_group function-local via late
        # imports and broke the earlier use: UnboundLocalError at first
        # model construction on the merged lineage.)
        # Decode top-k mode under DCP. Default (exact): local top-k, then an
        # all-gather + re-select of the global top-k on every indexer layer
        # (~21 collectives per decode step). Measured on GLM-5.2 @ 4x PRO
        # 6000 they are CONSTANT-cost and off the critical path (steps/s is
        # flat from 8K to 891K context), so exact stays the default.
        # VLLM_DCP_SPARSE_LOCAL_TOPK=1 switches decode to a per-rank local
        # top-(k/N) quota with no collective at all. Only for MTP-off
        # configs: the approximate candidate set decorrelates the shared-
        # index MTP drafter from the target (acceptance 2.9 -> 1.5) and
        # costs more end-to-end than the collectives it saves. Prefill
        # always uses the exact merge.
        self.dcp_local_topk = (
            self.dcp_world_size > 1
            and os.environ.get("VLLM_DCP_SPARSE_LOCAL_TOPK", "0") == "1"
        )
        if self.dcp_world_size > 1:
            assert max_model_len < MAX_FP32_EXACT_ID, (
                "DCP sparse-topk exchanges candidate ids as exact fp32 "
                f"values, which caps max_model_len at {MAX_FP32_EXACT_ID}; "
                f"got {max_model_len}."
            )
            assert not use_fp4_cache, (
                "DCP for the DSA indexer is not wired up for the FP4 "
                "indexer cache."
            )
            if self.dcp_local_topk:
                logger.info_once(
                    "DCP DSA indexer decode top-k: LOCAL quota "
                    "(top-%d per rank, no per-layer collective). "
                    "Not recommended with MTP - the approximate candidate "
                    "set collapses draft acceptance.",
                    topk_tokens // self.dcp_world_size,
                )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_quant, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_quant, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_fp4_cache,
            self.dcp_rank,
            self.dcp_world_size,
            self.cp_kv_cache_interleave_size,
            dcp_local_topk=self.dcp_local_topk,
        )

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return self.forward_cuda(hidden_states, q_fp8, k, weights)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert self.dcp_world_size == 1, (
            "DCP is not supported by the ROCm sparse_attn_indexer path."
        )
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_quant,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
                skip_k_cache_insert=self.skip_k_cache_insert,
            )
        raise RuntimeError(
            "Sparse attention indexer ROCm path is only supported on AITER. "
            "Please enable aiter with VLLM_ROCM_USE_AITER=1"
        )
