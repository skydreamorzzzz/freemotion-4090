"""Download official InterHuman annotations/splits from the recorded Drive manifest."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import requests
import threading

LOCAL = threading.local()

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "FreeMotion" / "data"


def fetch(entry):
    path = DATA / entry["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    def retrieve(file_id):
        if not hasattr(LOCAL, "session"):
            LOCAL.session = requests.Session()
        response = LOCAL.session.get("https://drive.google.com/uc",
                                     params={"export": "download", "id": file_id}, timeout=(15, 30))
        response.raise_for_status()
        content = response.content
        if not content or b"<html" in content.lower() or b"<!doctype" in content.lower():
            raise ValueError(f"Not a text annotation: {entry['path']}")
        return content
    if len(entry["source_ids"]) > 1:
        # Drive permits distinct files with the same name. Never race/overwrite them.
        variants = ROOT / "downloads/annotation_variants"
        variants.mkdir(exist_ok=True)
        hashes = []
        for file_id in entry["source_ids"]:
            variant = variants / (file_id + ".txt")
            if not variant.exists():
                variant.write_bytes(retrieve(file_id))
            content = variant.read_bytes()
            hashes.append(hashlib.sha256(content).hexdigest())
        if len(set(hashes)) != 1:
            raise ValueError(f"Conflicting same-name official files: {entry['path']}; preserved by Drive ID")
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != hashes[0]:
            raise ValueError(f"Existing file differs from official variants: {entry['path']}")
        if not path.exists():
            path.write_bytes(content)
    elif not path.exists():
        partial = path.with_suffix(path.suffix + ".part")
        content = retrieve(entry["source_ids"][0])
        partial.write_bytes(content)
        partial.replace(path)
    return {**entry, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    initial = ROOT / "downloads/interhuman_manifest.json"
    if initial.exists():
        entries = json.loads(initial.read_text(encoding="utf-8"))
    else:
        # IDs verified from the official InterHuman folder. Annotation IDs are
        # obtained by paginating the same public folder, never fabricated.
        entries = [
            {"path":"split/train.txt","id":"1b0Y6uwNK5cyU4Itzavr3FeHiy8VrF-Vj"},
            {"path":"split/val.txt","id":"1I1i1V2fRmYArIM51kXM2OC8dCznvY_1H"},
            {"path":"split/test.txt","id":"1xRe3m1Pl4UEpgqNgl304yYFXygpTRyHW"},
            {"path":"LICENSE.md","id":"1Pu0iqiUn1ANtaNPoWG5_GuOMQsybgnRw"},
        ]
    full = ROOT / "downloads/annots_full_manifest.json"
    if full.exists():
        entries = [e for e in entries if not e["path"].startswith("annots/")] + json.loads(full.read_text(encoding="utf-8"))
    else:
        raise SystemExit("Run list_public_annotations.py first; the embedded-folder snapshot is incomplete")
    selected = [e for e in entries if e["path"].startswith(("annots/", "split/")) or e["path"] == "LICENSE.md"]
    grouped = defaultdict(set)
    for entry in selected:
        grouped[entry["path"]].add(entry["id"])
    selected = [{"path":path,"source_ids":sorted(ids)} for path,ids in grouped.items()]
    selected.sort(key=lambda e: (not e["path"].startswith("split/"), e["path"]))
    results, errors = [], []
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, e): e for e in selected}
        for i, f in enumerate(as_completed(futures), 1):
            try:
                results.append(f.result())
            except Exception as error:
                errors.append({**futures[f], "error": str(error)})
            if i % 100 == 0 or i == len(selected):
                report = {"expected": len(selected), "complete": len(results), "errors": errors,
                          "seconds": time.time() - start, "files": results}
                (ROOT / "downloads/annotations_download_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"{i}/{len(selected)} processed, {len(errors)} failures, {time.time()-start:.1f}s", flush=True)
    if errors:
        raise SystemExit("Incomplete download; see annotations_download_report.json and rerun to retry")


if __name__ == "__main__":
    main()
