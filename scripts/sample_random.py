"""Sample a stored untrained FreeMotion checkpoint; outputs are NOT meaningful motions."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "FreeMotion"
sys.path.insert(0,str(REPO))
os.chdir(REPO)
from configs import get_config
from models import InterGenSpatialControlNet
from utils.utils import MotionNormalizer
from small_scale_trial import load_torch, strip_prefix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text", action="append", required=True, help="One person description per option, in generation order")
    p.add_argument("--interaction", default=None, help="Optional whole-scene description")
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--seed", type=int, default=20261003)
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/random_sample.npz")
    args = p.parse_args()
    if not 1 <= args.frames <= 300:
        p.error("frames must be in [1,300]")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    stage = "single" if len(args.text) == 1 else "multi"
    cfg = get_config("configs/model_single.yaml" if stage == "single" else "configs/model_inter.yaml")
    model = InterGenSpatialControlNet(cfg)
    checkpoint = ROOT / "artifacts/random_init" / f"{stage}_random_init.ckpt"
    data = load_torch(checkpoint,map_location="cpu")
    torch.nn.Module.load_state_dict(model,{strip_prefix(k,"model."):v for k,v in data["state_dict"].items()},strict=True)
    del data
    model.cuda().eval()
    generated = []
    with torch.no_grad():
        for i,text in enumerate(args.text):
            batch = {"text":[text],"text_multi_person":[args.interaction] if args.interaction else None,
                     "motion_lens":torch.tensor([args.frames],device="cuda"),"person_num":i+1,
                     "motion_guidance":torch.stack(generated,dim=2) if generated else None}
            output = model.forward_test(batch)["output"]
            assert torch.isfinite(output).all()
            generated.append(output)
    normalized = torch.stack(generated,dim=2).cpu().numpy()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output,normalized=normalized,features=MotionNormalizer().backward(normalized))
    args.output.with_suffix(".json").write_text(json.dumps({"status":"untrained_random_checkpoint_not_motion_quality_evidence",
        "texts":args.text,"interaction":args.interaction,"seed":args.seed,"shape":list(normalized.shape),
        "checkpoint":str(checkpoint)},ensure_ascii=False,indent=2),encoding="utf-8")
    print("Saved untrained sampling output",args.output,flush=True)


if __name__ == "__main__":
    main()
