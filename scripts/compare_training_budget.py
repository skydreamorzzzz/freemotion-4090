"""Compare two completed single-stage budgets with identical fixed inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from render_small_scale import animate


def compare(short, long):
    for name in ["subset.npz", "subset.json"]:
        assert (short/name).read_bytes() == (long/name).read_bytes(), name
    a = [json.loads(x) for x in (short/"single_progress.jsonl").read_text().splitlines()]
    b = [json.loads(x) for x in (long/"single_progress.jsonl").read_text().splitlines()]
    assert len(b) > len(a)
    prefix = {
        "compared_steps": len(a),
        "same_sample_order": all(x["example"] == y["example"] for x,y in zip(a,b)),
        "same_learning_rates": all(x["lr"] == y["lr"] for x,y in zip(a,b)),
        "max_loss_difference": max(abs(x["loss"]-y["loss"]) for x,y in zip(a,b)),
    }
    reports = [json.loads((p/"single_report.json").read_text()) for p in [short,long]]
    summary = {"subset_sha256":hashlib.sha256((short/"subset.npz").read_bytes()).hexdigest(),
               "prefix_comparison":prefix,"runs":[]}
    for path, report in zip([short,long],reports):
        summary["runs"].append({"path":str(path),"steps":report["steps"],
            "denoising":report["after_denoising"],"samples":report["after_samples"],
            "peak_training_memory":report["peak_training_memory"]})
    for split in ["train","val"]:
        with np.load(short/f"single_{split}_after.npz") as x, np.load(long/f"single_{split}_after.npz") as y:
            assert np.array_equal(x["ground_truth"],y["ground_truth"])
            animate([y["ground_truth"],x["generated"],y["generated"]],
                [f"{split}: real reference",f"{len(a)} updates",f"{len(b)} updates"],
                long/f"single_{split}_budget_comparison.gif")
    (long/"budget_comparison.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(prefix),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("short",type=Path)
    parser.add_argument("long",type=Path)
    args=parser.parse_args()
    compare(args.short.resolve(),args.long.resolve())
