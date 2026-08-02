"""Repro v2: engine geometry -- dual cache (plain SWA + PACKED compressed),
extra_sparse_indices, sinks. Sweeps the EXTRA (compressed) candidate count
across 512, which is where the in-engine boundary sits.

Determinism is checked two ways:
  same-tensors    R calls on one set of input tensors (pure kernel race)
  rebuilt-inputs  inputs reallocated between calls, so any read of memory the
                  kernel does not own shows up as a difference (bug #2 shape)
"""
import torch, sys
sys.path.insert(0, "/home/cirrascale/vllm-ds4/moet-fork/venv/lib/python3.12/site-packages")
from vllm.v1.attention.ops.triton_sparse_mla_dsv4 import (
    triton_sparse_mla_dsv4, pack_dsv4_reference_cache, _D, _BPT_PACKED)

dev = torch.device("cuda:0")
R = 5
PAGES, PBS, T, H = 64, 64, 15, 16   # T=15 matches the diverging chunk

def build(n_extra, n_main=128, seed=0):
    torch.manual_seed(seed)
    kv_f32 = torch.randn(PAGES * PBS, _D, device=dev, dtype=torch.float32) * 2.0
    packed = pack_dsv4_reference_cache(kv_f32, PBS)
    plain = torch.randn(PAGES, PBS, _D, device=dev, dtype=torch.bfloat16)
    q = torch.randn(T, H, _D, device=dev, dtype=torch.bfloat16)
    idx = torch.randint(0, PAGES * PBS, (T, n_main), device=dev, dtype=torch.int32)
    lens = torch.full((T,), n_main, device=dev, dtype=torch.int32)
    eidx = torch.randint(0, PAGES * PBS, (T, 1, n_extra), device=dev, dtype=torch.int32)
    elens = torch.full((T,), n_extra, device=dev, dtype=torch.int32)
    sinks = torch.full((H,), -float("inf"), device=dev, dtype=torch.float32)
    sinks[:8] = torch.randn(8, device=dev, dtype=torch.float32)
    return dict(
        query=q,
        swa_kv_cache=plain.unsqueeze(-2),
        sparse_indices=idx,
        compressed_kv_cache=packed.view(PAGES, PBS, 1, _BPT_PACKED),
        swa_topk_lens=lens,
        extra_sparse_indices=eidx,
        extra_sparse_topk_lens=elens,
        bmm1_scale=_D ** -0.5,
        sinks=sinks,
        kv_layout="NHD",
    )

def nbytes(o):
    return o.to(torch.float32).cpu().numpy().tobytes()

print(f"{'n_extra':>8} | {'same-tensors':>13} | {'rebuilt-inputs':>15} | {'max |delta|':>12}")
print("-" * 60)
for n_extra in (128, 256, 448, 511, 512, 513, 516, 640, 1024):
    kw = build(n_extra)
    outs = [triton_sparse_mla_dsv4(**kw).clone() for _ in range(R)]
    same_n = len({nbytes(o) for o in outs})

    reb = []
    for _ in range(R):
        # reallocate everything with the same seed -> identical VALUES, new memory
        torch.cuda.empty_cache()
        kw2 = build(n_extra)
        reb.append(triton_sparse_mla_dsv4(**kw2).clone())
    reb_n = len({nbytes(o) for o in reb})
    maxd = max(float((reb[0].to(torch.float32) - o.to(torch.float32)).abs().max())
               for o in reb[1:])
    flag = "  <-- NONDET" if (same_n > 1 or reb_n > 1) else ""
    print(f"{n_extra:>8} | {same_n:>13} | {reb_n:>15} | {maxd:>12.3e}{flag}")
