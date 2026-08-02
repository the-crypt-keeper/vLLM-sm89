"""Standalone: is triton_sparse_mla_dsv4 deterministic on byte-identical input?

No server. Calls the kernel R times on the SAME tensors and compares outputs
bit-for-bit. Sweeps the number of selected candidates across the k_select=512
boundary, because the in-engine evidence says <=512 is reproducible and >512 is
not.

Also passes an explicitly poisoned `out` buffer on alternate calls: if the
kernel fails to write some element, the poison shows through and the difference
tracks the poison value rather than being a genuine compute difference.
"""
import torch, sys
sys.path.insert(0, "/home/cirrascale/vllm-ds4/moet-fork/venv/lib/python3.12/site-packages")
from vllm.v1.attention.ops.triton_sparse_mla_dsv4 import (
    triton_sparse_mla_dsv4, pack_dsv4_reference_cache, _D, _BPT_PACKED)

dev = torch.device("cuda:0")
R = 5

def build(n_cand, t=4, h=16, pages=64, pbs=64, seed=0):
    torch.manual_seed(seed)
    kv_f32 = torch.randn(pages * pbs, _D, device=dev, dtype=torch.float32) * 2.0
    packed = pack_dsv4_reference_cache(kv_f32, pbs)
    plain = torch.randn(pages, pbs, _D, device=dev, dtype=torch.bfloat16)
    q = torch.randn(t, h, _D, device=dev, dtype=torch.bfloat16)
    idx = torch.randint(0, pages * pbs, (t, n_cand), device=dev, dtype=torch.int32)
    lens = torch.full((t,), n_cand, device=dev, dtype=torch.int32)
    return dict(
        query=q,
        swa_kv_cache=plain.unsqueeze(-2),
        sparse_indices=idx,
        compressed_kv_cache=packed.view(pages, pbs, 1, _BPT_PACKED),
        swa_topk_lens=lens,
        bmm1_scale=_D ** -0.5,
        kv_layout="NHD",
    )

print(f"{'n_cand':>7} | {'distinct outs':>13} | {'max |delta|':>12} | {'poison leak':>11}")
print("-" * 56)
for n_cand in (128, 256, 448, 512, 513, 516, 640, 1024):
    kw = build(n_cand)
    outs = []
    for r in range(R):
        o = triton_sparse_mla_dsv4(**kw)
        outs.append(o.clone())
    dist = len({o.to(torch.float32).cpu().numpy().tobytes() for o in outs})
    maxd = max(float((outs[0].to(torch.float32) - o.to(torch.float32)).abs().max())
               for o in outs[1:]) if R > 1 else 0.0

    # poison probe: hand the kernel a pre-filled buffer; any surviving poison
    # means an element was never written
    POISON = 12345.0
    shape = outs[0].shape
    buf = torch.full(shape, POISON, dtype=torch.bfloat16, device=dev)
    triton_sparse_mla_dsv4(out=buf, **kw)
    leak = int((buf == POISON).sum())

    print(f"{n_cand:>7} | {dist:>13} | {maxd:>12.3e} | {leak:>11}")
