"""Strict-load the official evaluator; check embeddings, not benchmark scores."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "FreeMotion"
sys.path.insert(0,str(REPO))
os.chdir(REPO)
import torch
from configs import get_config
from datasets.evaluator import EvaluatorModelWrapper

torch.set_num_threads(4)
torch.manual_seed(20261003)
model = EvaluatorModelWrapper(get_config("configs/eval_model.yaml"),"cuda")
batch = {"text":["Two people walk together."],"motions":torch.randn(1,16,524,device="cuda"),"motion_lens":torch.tensor([16])}
with torch.no_grad():
    text = model.model.encode_text(batch)["text_emb"]
    motion = model.model.encode_motion(batch)["motion_emb"]
assert torch.isfinite(text).all() and torch.isfinite(motion).all()
report = {"official_checkpoint_strict_load": True, "text_shape":list(text.shape),"motion_shape":list(motion.shape),
          "finite":True, "input":"synthetic; no benchmark metrics computed"}
(ROOT / "artifacts/evaluator_smoke.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(report,flush=True)
