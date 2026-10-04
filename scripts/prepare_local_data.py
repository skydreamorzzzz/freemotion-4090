"""Declare missing official assets explicitly via the upstream ignore-list mechanism."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
download = json.loads((ROOT / "downloads/annotations_download_report.json").read_text())
if download["errors"] or download["complete"] != download["expected"]:
    raise SystemExit("Finish annotation download before producing a missing-asset list")
audit = json.loads((ROOT / "artifacts/data_audit.json").read_text(encoding="utf-8"))
if audit["issues"]:
    raise SystemExit("Resolve data-audit issues before preparing a run")
missing = sorted({i for split in audit["splits"].values() for i in split["missing_ids"]}, key=int)
target = ROOT / "FreeMotion/data/split/ignore_list.txt"
content = "".join(i + "\n" for i in missing)
if target.exists() and target.read_text() != content:
    raise SystemExit("Existing ignore_list differs; inspect before changing it")
target.write_text(content, encoding="utf-8")
report = {"origin":"new_local_missing_asset_exclusions_not_an_official_ignore_list",
          "reason":"Official downloaded assets are incomplete for these original split IDs; never synthesize labels",
          "excluded_ids":missing, "original_split_files_modified":False,
          "eligible":{k:v["eligible_after_upstream_min_length"] for k,v in audit["splits"].items()}}
(ROOT / "artifacts/local_data_policy.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
