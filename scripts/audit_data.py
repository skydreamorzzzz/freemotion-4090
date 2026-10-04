"""Audit downloaded official data without changing the original split files."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "FreeMotion/data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    report = {"source": "official_downloads", "split_files_modified": False}
    archive = ROOT / "downloads/motions_processed.zip"
    if args.extract:
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise RuntimeError("Motion ZIP CRC validation failed")
            if not all((DATA / n).resolve().is_relative_to(DATA.resolve()) for n in z.namelist()):
                raise RuntimeError("Unsafe ZIP path")
            print("CRC passed; extracting", len(z.infolist()), "entries", flush=True)
            z.extractall(DATA)
        report["motion_archive_crc"] = "passed"
    def ids(folder):
        return {p.stem for p in (DATA / folder).glob("*") if p.is_file() and p.suffix in [".npy", ".txt"]}
    groups = {n: ids(n) for n in ["annots", "separate_annots/text1", "separate_annots/text2", "motions_processed/person1", "motions_processed/person2"]}
    report["counts"] = {n: len(v) for n, v in groups.items()}
    complete = set.intersection(*groups.values())
    report["complete_paired_ids"] = len(complete)
    report["splits"] = {}
    splits = {}
    for split in ["train", "val", "test"]:
        path = DATA / "split" / (split + ".txt")
        lines = path.read_text().splitlines()
        splits[split] = set(lines)
        report["splits"][split] = {"listed": len(lines), "unique": len(set(lines)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "complete": len(set(lines) & complete), "missing_ids": sorted(set(lines) - complete),
            "missing_by_asset": {n: sorted(set(lines) - v) for n, v in groups.items()}}
    report["split_overlap"] = {a + "_" + b: sorted(splits[a] & splits[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    bad = []
    upstream_short_ids = []
    encoding_fallbacks = []
    def text_lines(path):
        raw = path.read_bytes()
        try:
            return raw.decode("utf-8").splitlines()
        except UnicodeDecodeError:
            # Observed A1 AF in two official annotations decodes to U+2019.
            lines = raw.decode("gb18030").splitlines()
            encoding_fallbacks.append(str(path.relative_to(DATA)))
            return lines
    for motion_id in sorted(complete):
        counts = [len(text_lines(DATA / folder / (motion_id + ".txt"))) for folder in ["annots", "separate_annots/text1", "separate_annots/text2"]]
        if len(set(counts)) != 1 or not counts[0]:
            bad.append({"id": motion_id, "issue": "text_line_counts", "counts": counts})
        arrays = [np.load(DATA / "motions_processed" / person / (motion_id + ".npy"), mmap_mode="r") for person in ["person1", "person2"]]
        shapes = [list(a.shape) for a in arrays]
        if any(a.ndim != 2 or a.shape[1] < 312 or not np.isfinite(a).all() for a in arrays) or arrays[0].shape[0] != arrays[1].shape[0]:
            bad.append({"id": motion_id, "issue": "motion_shapes_finiteness_or_min_length", "shapes": shapes})
        elif arrays[0].shape[0] < 15:
            upstream_short_ids.append(motion_id)
    report["issues"] = bad
    report["upstream_short_ids"] = upstream_short_ids
    for split, data in report["splits"].items():
        data["eligible_after_upstream_min_length"] = len((splits[split] & complete) - set(upstream_short_ids))
    report["gb18030_fallback_files"] = encoding_fallbacks
    report["note"] = "Missing official IDs remain missing; no replacement data or resplitting. Numeric/text line checks do not prove semantic annotation alignment."
    out = ROOT / "artifacts/data_audit.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "complete_paired_ids":len(complete), "splits": {k: {"listed":v["listed"], "complete":v["complete"]} for k,v in report["splits"].items()}, "issues":len(bad)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
