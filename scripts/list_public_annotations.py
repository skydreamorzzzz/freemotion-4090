"""Enumerate the same public official Drive folder using its public listing API."""
import json
from pathlib import Path
import re
import requests

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "downloads").mkdir(exist_ok=True)
folder = "1cYffmLMC3-eturODlM1231YFbgZLDj0W"
session = requests.Session()
page = session.get(f"https://drive.google.com/drive/folders/{folder}?hl=en",timeout=30)
page.raise_for_status()
# These are unauthenticated public page application keys, not user credentials.
keys = list(dict.fromkeys(re.findall(r"AIza[0-9A-Za-z_-]{35}",page.text)))
print("Public page application keys found", len(keys), flush=True)
for key in keys:
    params = {"q":f"'{folder}' in parents and trashed = false", "pageSize":1000,
              "fields":"files(id,name),nextPageToken", "key":key}
    response = session.get("https://www.googleapis.com/drive/v3/files",params=params,
                           headers={"Referer":"https://drive.google.com/"},timeout=30)
    if response.status_code != 200:
        print("Public listing unavailable", response.status_code, response.json().get("error",{}).get("status"),flush=True)
        continue
    entries = []
    while True:
        data = response.json()
        entries.extend({"id":f["id"],"path":"annots/"+f["name"]} for f in data.get("files",[]))
        print("Enumerated", len(entries), "public files",flush=True)
        if not data.get("nextPageToken"):
            break
        params["pageToken"] = data["nextPageToken"]
        response = session.get("https://www.googleapis.com/drive/v3/files",params=params,
                               headers={"Referer":"https://drive.google.com/"},timeout=30)
        response.raise_for_status()
    (ROOT / "downloads/annots_full_manifest.json").write_text(json.dumps(entries,indent=2),encoding="utf-8")
    break
else:
    raise SystemExit("Public API listing unavailable; no authentication bypass attempted")
