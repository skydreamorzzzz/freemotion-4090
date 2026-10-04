"""Full-size upstream FreeMotion initialization and synthetic smoke checks.

This is NOT training on InterHuman and produces NO benchmark results.
Run using ../.venv/Scripts/python.exe; paths are resolved from this file.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "FreeMotion"
sys.path.insert(0, str(REPO))
os.chdir(REPO)  # upstream loads normalization statistics relative to cwd
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
from configs import get_config
from models import InterGenSpatialControlNet


def load_torch(path, map_location="cpu"):
    """weights_only on torch>=2.0; plain load on the paper's torch 1.13."""
    if "weights_only" in inspect.signature(torch.load).parameters:
        return torch.load(path, map_location=map_location, weights_only=True)
    return torch.load(path, map_location=map_location)


def strip_prefix(text, prefix):
    return text[len(prefix):] if prefix and text.startswith(prefix) else text


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["single", "multi"], required=True)
    p.add_argument("--seed", type=int, default=20261003)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.frames < 2:
        p.error("frames must be >= 2 for velocity losses")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    out = ROOT / "artifacts" / "random_init"
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = REPO / "configs" / ("model_single.yaml" if args.stage == "single" else "model_inter.yaml")
    cfg = get_config(str(cfg_path))
    metadata = {
        "status": "random_initialization_not_a_trained_baseline",
        "stage": args.stage, "seed": args.seed,
        "upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "config": cfg.dump(), "config_sha256": digest(cfg_path),
        "python": sys.version, "torch": str(torch.__version__),
        "numpy": np.__version__, "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "text_encoder": "upstream pretrained OpenAI CLIP ViT-L/14@336px; not randomized",
        "initialization": "upstream constructors, including zero output head and zero control projections",
        "data": "synthetic mean + 0.05 * std * Gaussian noise; NOT real or physically valid motion",
        "frames": args.frames, "batch_size": 1,
        "normalization_sha256": {n: digest(REPO / "data" / n) for n in ["global_mean.npy", "global_std.npy"]},
    }
    print("Constructing original full-size model", flush=True)
    model = InterGenSpatialControlNet(cfg)
    if args.stage == "multi":
        parent = out / "single_random_init.ckpt"
        ckpt = load_torch(parent, map_location="cpu")
        state = {strip_prefix(k, "model."): v for k, v in ckpt["state_dict"].items()}
        result = model.load_state_dict(state, strict=False)  # upstream copies single branch to control branch
        assert not result.unexpected_keys, result
        assert all(k.startswith("decoder.net.zero_linear.") for k in result.missing_keys), result
        metadata["parent_checkpoint_sha256"] = digest(parent)
        metadata["load_missing_keys"] = list(result.missing_keys)
        del ckpt, state
        gc.collect()
    metadata["parameters"] = sum(p.numel() for p in model.parameters())
    metadata["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    checkpoint = out / f"{args.stage}_random_init.ckpt"
    # Compatible with the state_dict / model. convention used by upstream train/eval.
    # Saved BEFORE any synthetic optimizer step. No optimizer/trainer state is implied.
    torch.save({"state_dict": {"model." + k: v for k, v in model.state_dict().items()},
                "metadata": metadata}, checkpoint)
    metadata["checkpoint"] = checkpoint.name
    metadata["checkpoint_sha256"] = digest(checkpoint)
    print("Saved untrained checkpoint", checkpoint, flush=True)
    # Verify a saved full multi checkpoint without invoking upstream's stage-transfer override.
    saved = load_torch(checkpoint, map_location="cpu")
    torch.nn.Module.load_state_dict(model, {strip_prefix(k, "model."): v for k, v in saved["state_dict"].items()}, strict=True)
    del saved
    gc.collect()
    metadata["strict_checkpoint_reload"] = True
    model.to(args.device).eval()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    lengths = torch.tensor([args.frames], device=args.device, dtype=torch.long)
    generated = []
    persons = 1 if args.stage == "single" else 2
    with torch.no_grad():
        for person in range(persons):
            batch = {"text": ["A person walks forward."],
                     "text_multi_person": ["Two people walk together."],
                     "motion_lens": lengths, "person_num": person + 1,
                     "motion_guidance": torch.stack(generated, dim=2) if generated else None}
            output = model.forward_test(batch)["output"]
            assert output.shape == (1, args.frames, 262)
            assert torch.isfinite(output).all()
            generated.append(output)
    motions = torch.stack(generated, dim=2)
    np.save(out / f"{args.stage}_normalized_smoke.npy", motions.cpu().numpy())
    metadata["sampling"] = {"strategy": cfg.STRATEGY, "shape": list(motions.shape),
                            "finite": True, "max_abs": motions.abs().max().item(),
                            "seconds": time.time() - start}
    print("Sampling passed", metadata["sampling"], flush=True)
    del motions, generated, output, batch
    model.train()
    mean = torch.from_numpy(np.load(REPO / "data/global_mean.npy")).float().to(args.device)
    std = torch.from_numpy(np.load(REPO / "data/global_std.npy")).float().to(args.device)
    synthetic = mean + 0.05 * std * torch.randn(1, args.frames, persons, 262, device=args.device)
    batch = {"text": ["A person walks forward."],
             "text_multi_person": ["Two people walk together."],
             "motion_lens": lengths, "person_num": persons,
             "motions": synthetic.flatten(2)}
    loss, terms = model.compute_loss(batch)
    assert torch.isfinite(loss), terms
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    nonzero = sum(bool(torch.count_nonzero(g).item()) for g in grads)
    metadata["synthetic_backward"] = {"loss_terms": {k: v.item() for k, v in terms.items()},
                                       "finite_gradients": True, "nonzero_gradient_tensors": nonzero}
    if args.stage == "single":
        assert nonzero > 0, "single-stage random model must have a learning signal"
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=2e-5)
        torch.nn.utils.clip_grad_norm_(params, 0.5)
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in params)
        metadata["synthetic_backward"]["optimizer_step"] = "AdamW succeeded; updated weights deliberately not saved"
    else:
        metadata["synthetic_backward"]["warning"] = "Frozen random zero output head blocks control-branch learning; real stage 2 needs trained stage 1"
    if args.device.startswith("cuda"):
        metadata["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
    (out / f"{args.stage}_report.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
