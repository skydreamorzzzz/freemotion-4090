"""A few real-data stage-1 updates; NOT a trained baseline or benchmark."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "FreeMotion"
sys.path.insert(0, str(REPO))
os.chdir(REPO)
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import torch
from configs import get_config
from datasets.interhuman import SingleHumanDataset
from models import InterGenSpatialControlNet
from models.utils import CosineWarmupScheduler
from small_scale_trial import load_torch, strip_prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    random.seed(20261003)
    np.random.seed(20261003)
    torch.manual_seed(20261003)
    torch.set_num_threads(4)
    data_cfg = get_config("configs/datasets_single.yaml").interhuman.clone()
    data_cfg.defrost()
    data_cfg.CACHE = False  # memory-only change; original on-demand path
    data_cfg.freeze()
    dataset = SingleHumanDataset(data_cfg)
    if not len(dataset):
        raise RuntimeError("No real training records available")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    cfg = get_config("configs/model_single.yaml")
    model = InterGenSpatialControlNet(cfg)
    ckpt = load_torch(ROOT / "artifacts/random_init/single_random_init.ckpt", map_location="cpu")
    model.load_state_dict({strip_prefix(k,"model."):v for k,v in ckpt["state_dict"].items()}, strict=True)
    del ckpt
    gc.collect()
    model.cuda().train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=2e-5)
    scheduler = CosineWarmupScheduler(optimizer, warmup=10, max_iters=2500)
    report = {"status": "real_data_smoke_only_not_converged", "dataset_records_with_mirroring": len(dataset),
              "seed":20261003, "batch_size":1, "cache":False, "workers":0, "steps":[]}
    for index, item in enumerate(loader):
        name, text, text_multi, motion1, motion2, lengths, spatial = item
        assert min(motion2.shape) == 0
        batch = {"text":list(text), "text_multi_person":None, "motions":motion1.float().cuda(),
                 "motion_lens":lengths.long().cuda(), "person_num":1}
        optimizer.zero_grad(set_to_none=True)
        loss, terms = model.compute_loss(batch)
        assert torch.isfinite(loss), terms
        loss.backward()
        grads = [p.grad for p in params if p.grad is not None]
        assert all(torch.isfinite(g).all() for g in grads)
        nonzero = sum(bool(torch.count_nonzero(g).item()) for g in grads)
        assert nonzero > 0
        torch.nn.utils.clip_grad_norm_(params,0.5)
        optimizer.step()
        row = {"index":index, "sample":list(name), "text":list(text), "lengths":lengths.tolist(),
               "tensor_shape":list(motion1.shape), "lr":optimizer.param_groups[0]["lr"],
               "loss_terms":{k:v.item() for k,v in terms.items()}, "nonzero_gradient_tensors":nonzero}
        report["steps"].append(row)
        print(json.dumps(row), flush=True)
        if index+1 >= args.steps:
            break
    report["checkpoint_saved"] = False
    # Exercise low-t geometric terms explicitly; random smoke timesteps may miss them.
    optimizer.zero_grad(set_to_none=True)
    mask = model.decoder.generate_src_mask(batch["motions"].shape[1], batch["motion_lens"], 1).cuda()
    terms = model.decoder.diffusion.training_losses(
        model=model.decoder.net, x_start=batch["motions"],
        t=torch.tensor([500], device="cuda"), mask=mask, t_bar=cfg.T_BAR,
        cond_mask=torch.ones(1,1,device="cuda"),
        model_kwargs={"mask":mask,"cond":batch["cond"],"person_num":1})
    assert all(torch.isfinite(v) for v in terms.values()), terms
    terms["total"].backward()
    assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
    report["fixed_t500_geometry_check"] = {k:v.item() for k,v in terms.items()}
    report["reason"] = "Few-step validation weights must not be confused with the untrained initialization or a trained baseline"
    (ROOT / "artifacts/real_data_smoke.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
