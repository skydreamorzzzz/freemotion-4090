"""Full-data FreeMotion Stage-1 (single) training sized for one 24GB GPU (RTX 4090).

Online counterpart to scripts/train_overnight.py. It keeps the upstream model, loss
and dataset byte-for-byte, but replaces the 8GB microbatch=1 + accumulate-16 loop with
real minibatches, overlaps data preparation with GPU compute, and follows the paper's
schedule (2500 epochs, warmup 10, cosine). FP32 by default (paper-faithful, TF32
matmuls like upstream tools/train.py); pass --bf16 for BF16 autocast matmuls.

Scope: Stage-1 generation module only. This is not a finished paper reproduction and
computes no FID / R-Precision itself.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

# Captured before small_scale_trial (imported below) chdir's into FreeMotion, so a
# relative --output-dir still resolves against the directory the job was launched from.
INVOCATION_CWD = Path.cwd()

from small_scale_trial import (
    ROOT, seed, get_config, SingleHumanDataset,
    InterGenSpatialControlNet, CosineWarmupScheduler, builtin_metadata, evaluate,
)

# Fixed 4-train/4-val 120-frame subset built by small_scale_trial.prepare(); reused as a
# cheap "is the model still improving" signal, not as a paper metric.
VALIDATION_SUBSET = ROOT / "artifacts/small_scale_8gb/subset.npz"


def atomic_json(path, value):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def rng_state():
    n = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [n[0], torch.tensor(n[1].astype(np.int64)), n[2], n[3], n[4]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(r):
    random.setstate(r["python"])
    n = r["numpy"]
    np.random.set_state((n[0], n[1].numpy().astype(np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(r["torch"])
    torch.cuda.set_rng_state_all(r["cuda"])


def collate_single(samples):
    """Batch the 7-tuple returned by SingleHumanDataset.__getitem__.

    Every motion is already padded to max_gt_length=300, so motions stack cleanly. Only
    the target person and its length are kept; the empty partner / spatial slots that the
    single-person path ignores are dropped here instead of carried as zero-width tensors.
    """
    names, texts, motions, lengths = [], [], [], []
    for name, text, _text_multi, motion1, _motion2, length, _spatial in samples:
        names.append(name)
        texts.append(text)
        motions.append(torch.as_tensor(np.asarray(motion1, dtype=np.float32)))
        lengths.append(int(length))
    return {"names": names, "text": texts,
            "motions": torch.stack(motions, dim=0),
            "motion_lens": torch.tensor(lengths, dtype=torch.long)}


def build_dataset(cache):
    cfg = get_config(str(ROOT / "configs/datasets_single_local.yaml")).interhuman.clone()
    cfg.defrost()
    cfg.CACHE = bool(cache)
    cfg.freeze()
    dataset = SingleHumanDataset(cfg)
    # Sort so the sample order (and therefore the deterministic epoch shuffle) is stable
    # across runs, matching the integrity check used by train_overnight.py.
    dataset.data_list.sort(key=lambda x: (int(x["name"].removesuffix("_swap")), bool(x["swap"])))
    n = len(dataset)
    ids = sorted({x["name"].removesuffix("_swap") for x in dataset.data_list}, key=int)
    expected = json.loads((ROOT / "artifacts/local_data_policy.json").read_text(encoding="utf-8"))["eligible"]["train"]
    assert len(ids) == expected and n == 2 * expected, (len(ids), n, expected)
    assert dataset.max_gt_length == 300
    return dataset, ids, n


def load_payload(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def main(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; this entry needs a CUDA GPU")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    # Upstream tools/train.py runs matmuls with reduced FP32 precision; mirror that by
    # default, allow --no-tf32 for strict FP32.
    torch.set_float32_matmul_precision("highest" if args.no_tf32 else "high")

    device_name = torch.cuda.get_device_name()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    # Fixed cost is trainable params + grads + AdamW moments + frozen CLIP (~3.4 GiB);
    # activations grow ~linearly with batch. Estimate is deliberately rough: used as a
    # guard rail, not a guarantee.
    estimate_gib = 3.6 + args.batch_size * 0.32
    print(json.dumps({"gpu": device_name, "free_gib": round(free_bytes / 1024**3, 2),
                      "total_gib": round(total_bytes / 1024**3, 2),
                      "estimated_peak_gib": round(estimate_gib, 2),
                      "batch_size": args.batch_size, "accumulation": args.accumulation,
                      "effective_batch": args.batch_size * args.accumulation,
                      "precision": "bf16-mixed" if args.bf16 else "fp32"}), flush=True)
    if free_bytes < 4 * 1024**3:
        raise RuntimeError(f"Only {free_bytes/1024**3:.2f} GiB GPU memory free; release other workloads first")
    if free_bytes < estimate_gib * 1024**3:
        print(f"WARNING: free memory ({free_bytes/1024**3:.2f} GiB) below the estimate for "
              f"batch={args.batch_size} ({estimate_gib:.2f} GiB); lower --batch-size if it OOMs", flush=True)

    seed(args.seed)
    dataset, ids, n = build_dataset(cache=not args.no_cache)
    if args.batch_size > n:
        raise SystemExit(f"--batch-size {args.batch_size} exceeds dataset size {n}")
    updates_per_epoch = n // args.batch_size
    print(json.dumps({"train_ids": len(ids), "augmented_samples": n,
                      "updates_per_epoch": updates_per_epoch,
                      "epochs": args.epochs, "total_updates": updates_per_epoch * args.epochs}), flush=True)

    model = InterGenSpatialControlNet(get_config("configs/model_single.yaml"))
    source = ROOT / "artifacts/random_init/single_random_init.ckpt"
    resume_file = out / "latest.ckpt"
    if args.resume and not resume_file.exists():
        raise SystemExit(f"--resume requested but {resume_file} does not exist")

    protocol = {
        "stage": "single", "archi": "single", "original_frozen_text_adapter": True,
        "train_ids": len(ids), "augmented_epoch_samples": n,
        "sample_order_sha256": hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest(),
        "frames_cap": 300, "batch_size": args.batch_size, "accumulation": args.accumulation,
        "effective_batch": args.batch_size * args.accumulation,
        "precision": "bf16-mixed" if args.bf16 else "fp32",
        "tf32": not args.no_tf32,
        "lr": args.lr, "weight_decay": args.weight_decay, "gradient_clip": args.grad_clip,
        "warmup_epochs": args.warmup_epochs, "cosine_epochs": args.epochs,
        "scheduler": "CosineWarmupScheduler(warmup=%d, max_iters=%d)" % (args.warmup_epochs, args.epochs),
        "seed": args.seed, "cache": not args.no_cache, "num_workers": args.num_workers,
        "source": str(source),
    }

    if args.resume:
        saved = load_payload(resume_file)
        assert saved["protocol"] == protocol, "Resume protocol differs from the saved run"
    else:
        if resume_file.exists():
            raise SystemExit(f"{resume_file} already exists; use --resume or a new --output-dir")
        saved = load_payload(source)
    model.load_state_dict({k.removeprefix("model."): v for k, v in saved["state_dict"].items()}, strict=True)

    model.cuda()
    assert not any(p.requires_grad for p in model.clipTransEncoder.parameters())
    assert not any(p.requires_grad for p in model.clip_transformer.parameters())
    params = [p for p in model.parameters() if p.requires_grad]
    print(json.dumps({"trainable_parameters": sum(p.numel() for p in params)}), flush=True)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineWarmupScheduler(optimizer, warmup=args.warmup_epochs, max_iters=args.epochs)

    state = {"updates": 0, "epoch": 0, "elapsed_seconds": 0.0, "best_validation_mse": None}
    epoch_generator_state = None
    saved_rng = None
    if args.resume:
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        state = saved["training_state"]
        epoch_generator_state = saved["generator_state"]
        saved_rng = saved["rng_state"]
    del saved
    gc.collect()
    torch.cuda.empty_cache()

    # Restore RNG state (resume) or seed fresh, immediately before the loader/training so
    # nothing in between consumes the generators we track.
    if args.resume:
        restore_rng(saved_rng)
    else:
        seed(args.seed)

    epoch_generator = torch.Generator()
    epoch_generator.manual_seed(args.seed)
    if epoch_generator_state is not None:
        epoch_generator.set_state(epoch_generator_state)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, generator=epoch_generator,
        num_workers=args.num_workers, collate_fn=collate_single, drop_last=True,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )

    atomic_json(out / "protocol.json", protocol)
    atomic_json(out / "train_ids.json", ids)
    validation_data = np.load(VALIDATION_SUBSET) if VALIDATION_SUBSET.exists() else None

    start = time.monotonic()
    previous = state["elapsed_seconds"]
    last_save = start
    epoch_times = []

    def elapsed():
        return previous + time.monotonic() - start

    def status(label):
        value = {"status": label, "pid": os.getpid(), "updates": state["updates"],
                 "epoch": state["epoch"], "epochs_total": args.epochs,
                 "batch_size": args.batch_size, "accumulation": args.accumulation,
                 "elapsed_seconds": elapsed(),
                 "seconds_per_epoch": epoch_times[-1] if epoch_times else None,
                 "eta_seconds": (np.mean(epoch_times) * (args.epochs - state["epoch"])) if epoch_times else None,
                 "lr": optimizer.param_groups[0]["lr"],
                 "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                 "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                 "best_validation_mse": state.get("best_validation_mse"),
                 "last_checkpoint": str(out / "latest.ckpt"), "stop_file": str(out / "STOP")}
        atomic_json(out / "status.json", builtin_metadata(value))
        return value

    def save(label):
        nonlocal last_save
        for p in model.parameters():
            if not torch.isfinite(p).all():
                raise FloatingPointError("Nonfinite model parameter")
        state["elapsed_seconds"] = elapsed()
        payload = {"state_dict": {"model." + k: v.detach().cpu() for k, v in model.state_dict().items()},
                   "optimizer_state_dict": builtin_metadata(optimizer.state_dict()),
                   "scheduler_state_dict": builtin_metadata(scheduler.state_dict()),
                   "training_state": dict(state), "rng_state": rng_state(),
                   "generator_state": epoch_generator.get_state(), "protocol": protocol,
                   "metadata": {"stage": "single", "epochs_done": state["epoch"], "status": label}}
        tmp = out / "latest.ckpt.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, out / "latest.ckpt")
        del payload
        last_save = time.monotonic()
        value = status(label)
        print(json.dumps({**value, "checkpoint_saved": True, "label": label}), flush=True)

    def validate():
        if validation_data is None:
            return False
        saved_rng = rng_state()
        try:
            result = evaluate(model, validation_data, "single")
        finally:
            model.train()
            restore_rng(saved_rng)
        value = result["val"]["mean_normalized_mse"]
        best = state.get("best_validation_mse")
        improved = best is None or value < best
        if improved:
            state["best_validation_mse"] = value
        record = {"updates": state["updates"], "epoch": state["epoch"],
                  "fixed_val_denoising_mse": value, "best_so_far": state["best_validation_mse"],
                  "note": "120-frame fixed diagnostic; not FID or full validation benchmark"}
        with (out / "validation.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        return improved

    status("running")
    reason = "epoch_schedule_complete"
    progress = (out / "progress.jsonl").open("a", encoding="utf-8")
    try:
        for epoch in range(state["epoch"], args.epochs):
            state["epoch"] = epoch
            model.train()
            epoch_start = time.monotonic()
            window_loss = 0.0
            window_count = 0
            micro = 0
            updates_this_epoch = 0
            optimizer.zero_grad(set_to_none=True)
            for batch in loader:
                batch["motions"] = batch["motions"].cuda(non_blocking=True).float()
                batch["motion_lens"] = batch["motion_lens"].cuda(non_blocking=True)
                batch["person_num"] = 1
                batch["text_multi_person"] = None
                batch = model.text_process(batch)  # frozen fp16 CLIP; keep out of autocast
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                    terms = model.decoder.compute_loss(batch)
                loss = terms["total"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, micro {micro}")
                loss.backward()
                window_loss += float(loss.detach())
                window_count += 1
                micro += 1
                if micro < args.accumulation:
                    del loss, terms, batch
                    continue

                if args.accumulation > 1:
                    for p in params:
                        if p.grad is not None:
                            p.grad.mul_(1.0 / window_count)
                norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                state["updates"] += 1
                updates_this_epoch += 1
                micro = 0
                row = {"update": state["updates"], "epoch": epoch,
                       "epoch_fraction": updates_this_epoch / updates_per_epoch,
                       "effective_batch": window_count, "loss": window_loss / window_count,
                       "grad_norm": float(norm), "lr": optimizer.param_groups[0]["lr"],
                       "elapsed_seconds": elapsed(),
                       "allocated_mib": torch.cuda.memory_allocated() / 1024**2,
                       "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2}
                window_loss = 0.0
                window_count = 0
                del loss, terms, batch
                if state["updates"] % args.log_every == 0 or state["updates"] <= 3:
                    progress.write(json.dumps(row) + "\n")
                    progress.flush()
                    print(json.dumps(row), flush=True)
                    if state["updates"] % (args.log_every * 10) == 0:
                        status("running")
                if args.max_updates and state["updates"] >= args.max_updates:
                    reason = "max_updates_reached"
                    break
            if reason == "max_updates_reached":
                break
            gc.collect()

            scheduler.step()
            state["epoch"] = epoch + 1
            epoch_times.append(time.monotonic() - epoch_start)
            improved = validate() if (epoch + 1) % args.val_every == 0 else False
            due = (time.monotonic() - last_save >= args.checkpoint_minutes * 60
                   or (epoch + 1) % args.save_every_epochs == 0)
            if improved or due:
                save("running")
            if improved:
                shutil.copyfile(out / "latest.ckpt", out / "best_validation.ckpt.tmp")
                os.replace(out / "best_validation.ckpt.tmp", out / "best_validation.ckpt")
            if (out / "STOP").exists():
                reason = "stop_file_requested"
                break
            if args.stop_after_hours and elapsed() >= args.stop_after_hours * 3600:
                reason = "time_budget_complete"
                break
    except Exception:
        status("failed_last_checkpoint_preserved")
        raise
    finally:
        progress.close()

    save(reason)
    status("stopped:" + reason)
    print(json.dumps({"stopped": reason, "updates": state["updates"], "epoch": state["epoch"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/full_single_4090")
    parser.add_argument("--batch-size", type=int, default=32, help="minibatch per GPU (default 32)")
    parser.add_argument("--accumulation", type=int, default=1, help="gradient accumulation microbatches")
    parser.add_argument("--epochs", type=int, default=2500, help="paper schedule length")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-5)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bf16", action="store_true", help="BF16 autocast matmuls (~2x faster, slight numeric change)")
    parser.add_argument("--no-tf32", action="store_true", help="strict FP32 instead of TF32 matmuls")
    parser.add_argument("--no-cache", action="store_true", help="read motions from disk each item instead of caching")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-minutes", type=float, default=20)
    parser.add_argument("--save-every-epochs", type=int, default=5)
    parser.add_argument("--val-every", type=int, default=1, help="run the fixed-val diagnostic every N epochs")
    parser.add_argument("--log-every", type=int, default=10, help="write a progress row every N updates")
    parser.add_argument("--max-updates", type=int, help="stop after this many updates (preflight)")
    parser.add_argument("--stop-after-hours", type=float, default=0.0, help="optional safety budget; 0 disables")
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    if args.batch_size < 1 or args.accumulation < 1 or args.epochs < 1:
        parser.error("batch-size, accumulation and epochs must be positive")
    if args.num_workers < 0 or args.log_every < 1 or args.val_every < 1 or args.save_every_epochs < 1:
        parser.error("num-workers must be >=0; log-every/val-every/save-every-epochs must be >=1")
    if not args.output_dir.is_absolute():
        args.output_dir = INVOCATION_CWD / args.output_dir
    args.output_dir = args.output_dir.resolve()
    main(args)
