#!/usr/bin/env python3
"""TRELLIS sparse-structure cache A/B: HiCache (Hermite) vs HiCache++ (DMD), 7th matrix cell.

faster-trellis accelerates the sparse-structure (SS) stage with a HiCache (Hermite) velocity
forecast. This isolates the *basis*: at the same SS skip schedule (same speedup), how far does
the generated geometry drift from the uncached baseline under Hermite vs DMD? We patch ONLY the
SS sampler (no token-carving confound), run with a fixed seed so the noise is identical, and
score the Chamfer distance of the output mesh vertices against the vanilla mesh. A lossless cache
keeps that drift near zero; the question is which basis stays lossless at larger intervals.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")

from trellis.pipelines import TrellisImageTo3DPipeline           # noqa: E402
from trellis.pipelines.samplers import hicache                   # noqa: E402

WEIGHTS = os.path.join(HERE, "..", "..", "data", "weights", "TRELLIS")


def load():
    p = TrellisImageTo3DPipeline.from_pretrained(WEIGHTS)
    p.cuda()
    return p


def run_mesh(p, img, seed, ss_steps):
    t0 = time.time()
    out = p.run(img, num_samples=1, seed=seed, formats=["mesh"], preprocess_image=True,
                sparse_structure_sampler_params={"steps": ss_steps, "cfg_strength": 7.5},
                slat_sampler_params={"steps": 25, "cfg_strength": 3.0})
    dt = time.time() - t0
    return out["mesh"][0].vertices.float().detach(), dt


def chamfer(a, b):
    from pytorch3d.loss import chamfer_distance
    return float(chamfer_distance(a[None], b[None])[0].item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", default=[os.path.join(HERE, "assets", "teaser.png"),
                                                    os.path.join(HERE, "assets", "demo.png")])
    ap.add_argument("--intervals", nargs="+", type=int, default=[3, 4, 5, 6])
    ap.add_argument("--ss-steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    imgs = [Image.open(p).convert("RGB") for p in args.images]
    print(f"=== TRELLIS SS-cache A/B on {len(imgs)} images, ss_steps={args.ss_steps} ===")

    p = load()
    refs, t_base = [], []
    for img in imgs:
        v, dt = run_mesh(p, img, args.seed, args.ss_steps)
        refs.append(v); t_base.append(dt)
    print(f"[vanilla] {np.mean(t_base):.2f}s/img, {refs[0].shape[0]} verts")
    del p; torch.cuda.empty_cache()

    rows = []
    for N in args.intervals:
        for backend in ("hermite", "dmd"):
            p = load()
            hicache.enable(p, backend=backend, interval=N, patch_slat=False,
                           patch_sparse_structure=True, first_enhance=3, max_order=2, sigma=0.5)
            drift, secs = [], []
            for img, ref in zip(imgs, refs):
                v, dt = run_mesh(p, img, args.seed, args.ss_steps)
                drift.append(chamfer(v, ref)); secs.append(dt)
            cd, t = float(np.mean(drift)), float(np.mean(secs))
            rows.append((backend, N, cd, t))
            print(f"[{backend}_i{N}] chamfer_vs_vanilla={cd:.6f}  {t:.2f}s/img")
            del p; torch.cuda.empty_cache()

    # rough SS compute count at interval N over ss_steps (first_enhance=3): fe + (steps-fe)/N
    print("\n| SS interval | Hermite drift | DMD drift | ~SS compute / 25 |")
    print("|---:|---:|---:|---:|")
    for N in args.intervals:
        h = next(c for b, n, c, _ in rows if b == "hermite" and n == N)
        d = next(c for b, n, c, _ in rows if b == "dmd" and n == N)
        comp = 3 + int(np.ceil((args.ss_steps - 3) / N))
        print(f"| i{N} | {h:.6f} | {d:.6f} | {comp}/25 |")


if __name__ == "__main__":
    main()
