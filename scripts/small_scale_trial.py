"""Bounded full-size FreeMotion experiment on fixed real-data subsets.

This diagnoses memory, optimization and early samples; it is not a paper benchmark.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "FreeMotion"
sys.path.insert(0, str(REPO))
os.chdir(REPO)
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import torch
from configs import get_config
from datasets.interhuman import SingleHumanDataset, InterHumanDataset
from models import InterGenSpatialControlNet
from models.utils import CosineWarmupScheduler
from utils.utils import MotionNormalizer

OUT = ROOT / "artifacts/small_scale_8gb"
SEED = 20261003


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def builtin_metadata(value):
    """Keep optimizer/scheduler NumPy scalars compatible with weights_only load."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k:builtin_metadata(v) for k,v in value.items()}
    if isinstance(value, list):
        return [builtin_metadata(v) for v in value]
    if isinstance(value, tuple):
        return tuple(builtin_metadata(v) for v in value)
    return value


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT / "subset.npz").exists() and json.loads((OUT / "subset.json").read_text(encoding="utf-8")).get("schema_version") == 2:
        print("Using saved fixed subset", flush=True)
        return
    audit = json.loads((ROOT / "artifacts/data_audit.json").read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    ids = {}
    for split, count in [("train", 32), ("val", 8)]:
        available = set((REPO / "data/split" / f"{split}.txt").read_text().splitlines())
        available -= set(audit["splits"][split]["missing_ids"]) | set(audit["upstream_short_ids"])
        ids[split] = rng.sample(sorted(available, key=int), count)
    assert not set(ids["train"]) & set(ids["val"])
    arrays, manifest = {}, {"schema_version":2,"seed":SEED,"train_ids":ids["train"],"val_ids":ids["val"],
        "frames_cap":120,"subset_policy":"Fixed first sampled caption/person/crop per ID; no resampling augmentation during this diagnostic", "examples":{}}
    for kind in ["single", "multi"]:
        for split in ["train", "val"]:
            seed(SEED + (0 if split == "train" else 100) + (1000 if kind == "multi" else 0))
            cfg = get_config("configs/datasets_single.yaml" if kind == "single" else "configs/datasets_inter.yaml").interhuman.clone()
            cfg.defrost()
            cfg.MODE, cfg.CACHE = split, False
            cfg.freeze()
            dataset = (SingleHumanDataset if kind == "single" else InterHumanDataset)(cfg)
            dataset.max_gt_length = dataset.max_length = 120
            index = {entry["name"]:i for i,entry in enumerate(dataset.data_list) if not entry["swap"]}
            motions, lengths, texts, scenes, partners = [], [], [], [], []
            selected = ids[split] if kind == "single" else ids[split][:8 if split == "train" else 4]
            for motion_id in selected:
                name, text, other_text, a, b, length, _ = dataset[index[motion_id]]
                motion = np.asarray(a, dtype=np.float32)
                if kind == "multi":
                    motion = np.concatenate([motion, np.asarray(b, dtype=np.float32)], axis=-1)
                    # The upstream training dataset returns scene + target-person text,
                    # but not the guidance person's text. Recover its paired row for
                    # later two-person sampling without inventing a second prompt.
                    entry = dataset.data_list[index[motion_id]]
                    candidates = set()
                    for whole, first, second in zip(entry["texts"],entry["texts1"],entry["texts2"]):
                        if whole.strip() == text:
                            if first.strip() == other_text:
                                candidates.add(second.strip())
                            if second.strip() == other_text:
                                candidates.add(first.strip())
                    if len(candidates) != 1:
                        raise ValueError(f"Ambiguous paired person description for {motion_id}")
                    partners.append(candidates.pop())
                else:
                    partners.append("")
                motions.append(motion)
                lengths.append(int(length))
                texts.append(text)
                scenes.append(other_text if isinstance(other_text,str) else "")
            prefix = f"{kind}_{split}"
            arrays[prefix + "_motions"] = np.stack(motions)
            arrays[prefix + "_lengths"] = np.asarray(lengths)
            arrays[prefix + "_texts"] = np.asarray(texts)
            arrays[prefix + "_scenes"] = np.asarray(scenes)
            arrays[prefix + "_partners"] = np.asarray(partners)
            manifest["examples"][prefix] = [{"id":i,"text":t,"additional_condition":s,"partner_text":p,"length":l} for i,t,s,p,l in zip(selected,texts,scenes,partners,lengths)]
            del dataset
    np.savez(OUT / "subset.npz", **arrays)
    write_json(OUT / "subset.json", manifest)
    print("Prepared 32/8 single and 8/4 interaction train/validation examples",flush=True)


def load_batch(data, stage, split, i, device="cuda"):
    prefix = f"{stage}_{split}"
    text, scene = str(data[prefix+"_texts"][i]), str(data[prefix+"_scenes"][i])
    return {"text":[text], "text_multi_person":[scene] if scene else None,
        "motions":torch.from_numpy(data[prefix+"_motions"][i:i+1]).to(device),
        "motion_lens":torch.tensor([int(data[prefix+"_lengths"][i])],device=device),
        "person_num":1 if stage == "single" else 2}


@torch.no_grad()
def evaluate(model, data, stage):
    model.eval()
    result = {}
    for split in ["train","val"]:
        rows = []
        for i in range(4):
            batch = model.text_process(load_batch(data,stage,split,i))
            x = batch["motions"].reshape(1,120,batch["person_num"],262)
            normalized = model.decoder.diffusion.normalizer.forward(x)
            target = normalized[:,:,0]
            mask = model.decoder.generate_src_mask(120,batch["motion_lens"],batch["person_num"]).cuda()
            for timestep in [500,900]:
                # Same corruption for every before/after evaluation, independent of training RNG.
                seed(SEED+10000+i+timestep)
                t = torch.tensor([timestep],device="cuda")
                noisy = model.decoder.diffusion.q_sample(target,t,noise=torch.randn_like(target))
                pred = model.decoder.net(noisy,t,cond=batch["cond"],mask=mask,
                    motion_guidance=normalized[:,:,1:] if stage == "multi" else None)
                length = int(batch["motion_lens"].item())
                mse = (pred[:,:length]-target[:,:length]).square().mean()
                assert torch.isfinite(mse)
                rows.append({"example":i,"t":timestep,"normalized_mse":mse.item()})
        result[split] = {"mean_normalized_mse":float(np.mean([r["normalized_mse"] for r in rows])),"details":rows}
    return result


@torch.no_grad()
def samples(model,data,stage,label):
    model.eval()
    normalizer = MotionNormalizer()
    summaries = []
    for split in ["train","val"]:
        batch = load_batch(data,stage,split,0)
        length = int(batch["motion_lens"].item())
        generated = []
        seed(SEED+20000)
        persons = 1 if stage == "single" else 2
        for person in range(persons):
            text = batch["text"] if stage == "single" else (batch["text_multi_person"] if person==0 else [str(data[f"{stage}_{split}_partners"][0])])
            scene = None if stage == "single" else batch["text"]
            inp = {"text":text,"text_multi_person":scene,
                "motion_lens":batch["motion_lens"],"person_num":person+1,
                "motion_guidance":torch.stack(generated,dim=2) if generated else None}
            output = model.forward_test(inp)["output"]
            assert torch.isfinite(output).all()
            generated.append(output)
        features = normalizer.backward(torch.stack(generated,dim=2).cpu().numpy()[0])
        gt = batch["motions"][0,:length].reshape(length,persons,262).cpu().numpy()
        np.savez(OUT/f"{stage}_{split}_{label}.npz",generated=features,ground_truth=gt)
        joints = features[...,:66].reshape(length,persons,22,3)
        gt_joints = gt[...,:66].reshape(length,persons,22,3)
        summaries.append({"split":split,"frames":length,
            "mean_joint_displacement_per_frame":float(np.linalg.norm(np.diff(joints,axis=0),axis=-1).mean()),
            "gt_mean_joint_displacement_per_frame":float(np.linalg.norm(np.diff(gt_joints,axis=0),axis=-1).mean()),
            "note":"Motion amount only; not semantic accuracy or a paper metric"})
    # Upstream sampling attaches an nn.Module CFG wrapper sharing decoder.net.
    # Remove that temporary alias so checkpoint keys match a fresh constructor.
    if hasattr(model.decoder,"cfg_model"):
        del model.decoder.cfg_model
    return summaries


def run(stage, steps, unfreeze_text_adapter=False):
    if (OUT / f"{stage}_report.json").exists():
        raise SystemExit("Completed experiment exists; keep it intact and choose a new experiment directory for another run")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    free, total = torch.cuda.mem_get_info()
    if free < 5.5 * 1024**3:
        raise RuntimeError(f"Only {free/1024**3:.2f} GiB GPU memory free; release other GPU workloads first")
    seed(SEED)
    torch.set_num_threads(4)
    data = np.load(OUT / "subset.npz")
    cfg = get_config("configs/model_single.yaml" if stage == "single" else "configs/model_inter.yaml")
    model = InterGenSpatialControlNet(cfg)
    source = ROOT / "artifacts/random_init/single_random_init.ckpt" if stage == "single" else OUT / "single_trial.ckpt"
    ckpt = torch.load(source,map_location="cpu",weights_only=True)
    result = model.load_state_dict({k.removeprefix("model."):v for k,v in ckpt["state_dict"].items()}, strict=stage=="single")
    if stage == "multi":
        assert not result.unexpected_keys and all(k.startswith("decoder.net.zero_linear.") for k in result.missing_keys)
    del ckpt
    gc.collect()
    if unfreeze_text_adapter:
        for module in [model.clipTransEncoder, model.clip_ln]:
            module.requires_grad_(True)
    adapter_params=list(model.clipTransEncoder.parameters())+list(model.clip_ln.parameters())
    assert not any(p.requires_grad for p in model.clip_transformer.parameters())
    model.cuda()
    before = evaluate(model,data,stage)
    before_samples = samples(model,data,stage,"before")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,lr=1e-4,weight_decay=2e-5)
    scheduler = CosineWarmupScheduler(optimizer,warmup=10,max_iters=2500 if stage=="single" else 1000)
    seed(SEED)
    count = len(data[f"{stage}_train_motions"])
    rng = random.Random(SEED)
    order = []
    start = time.time()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    history = []
    with (OUT/f"{stage}_progress.jsonl").open("w",encoding="utf-8") as log:
        for step in range(steps):
            if not order:
                if step:
                    scheduler.step()
                order = list(range(count))
                rng.shuffle(order)
            i = order.pop()
            optimizer.zero_grad(set_to_none=True)
            loss, terms = model.compute_loss(load_batch(data,stage,"train",i))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step+1}")
            loss.backward()
            adapter_grad_norm=float(torch.stack([p.grad.detach().norm() for p in adapter_params if p.grad is not None]).norm()) if any(p.grad is not None for p in adapter_params) else 0.0
            norm = torch.nn.utils.clip_grad_norm_(params,0.5,error_if_nonfinite=True)
            optimizer.step()
            row = {"step":step+1,"example":i,"loss":loss.item(),"grad_norm":float(norm),"adapter_grad_norm":adapter_grad_norm,
                "lr":optimizer.param_groups[0]["lr"],"elapsed_seconds":time.time()-start,
                "allocated_mib":torch.cuda.memory_allocated()/1024**2,
                "reserved_mib":torch.cuda.memory_reserved()/1024**2}
            history.append(row)
            log.write(json.dumps(row)+"\n")
            log.flush()
            if (step+1)%20==0 or step==0 or step+1==steps:
                print(json.dumps(row),flush=True)
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError("Nonfinite updated parameter")
    peak = {"allocated_mib":torch.cuda.max_memory_allocated()/1024**2,
            "reserved_mib":torch.cuda.max_memory_reserved()/1024**2}
    torch.save({"state_dict":{"model."+k:v.cpu() for k,v in model.state_dict().items()},
        "optimizer_state_dict":builtin_metadata(optimizer.state_dict()),"scheduler_state_dict":builtin_metadata(scheduler.state_dict()),
        "metadata":{"status":"tiny_subset_trial_not_a_converged_baseline","stage":stage,"steps":steps,"seed":SEED,"config":cfg.dump(),"unfreeze_text_adapter":unfreeze_text_adapter}},OUT/f"{stage}_trial.ckpt")
    del optimizer, scheduler, loss, terms
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    after = evaluate(model,data,stage)
    after_samples = samples(model,data,stage,"after")
    report = {"status":"completed_small_scale_trial_not_paper_reproduction","stage":stage,"steps":steps,
        "unfreeze_text_adapter":unfreeze_text_adapter,"adapter_nonzero_gradient_steps":sum(r["adapter_grad_norm"]>0 for r in history),
        "gpu":torch.cuda.get_device_name(),"batch_size":1,"precision":"float32","frames_cap":120,
        "parameters":sum(p.numel() for p in model.parameters()),"peak_training_memory":peak,
        "training_seconds":history[-1]["elapsed_seconds"],"finite_loss_gradients_parameters":True,
        "before_denoising":before,"after_denoising":after,"before_samples":before_samples,"after_samples":after_samples,
        "first_20_mean_loss":float(np.mean([r["loss"] for r in history[:20]])),
        "last_20_mean_loss":float(np.mean([r["loss"] for r in history[-20:]])),
        "gradient_nonzero_steps":sum(r["grad_norm"]>0 for r in history),
        "limitations":["tiny fixed subset","batch differs from paper","120-frame cap","short training","no FID/R-precision claims"],
        "upstream_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()}
    write_json(OUT/f"{stage}_report.json",report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--stage",choices=["single","multi"],default="single")
    parser.add_argument("--steps",type=int,default=200)
    parser.add_argument("--output-dir",type=Path,default=OUT)
    parser.add_argument("--subset-from",type=Path)
    parser.add_argument("--unfreeze-text-adapter",action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.subset_from:
        source_dir = args.subset_from.resolve()
        for name in ["subset.npz","subset.json"]:
            source_file, target_file = source_dir/name, OUT/name
            if target_file.exists():
                if source_file.read_bytes() != target_file.read_bytes():
                    raise ValueError(f"Existing subset differs: {target_file}")
            else:
                shutil.copy2(source_file,target_file)
    prepare()
    if not args.prepare_only:
        run(args.stage,args.steps,args.unfreeze_text_adapter)
