"""Download the official processed-motion archive via Drive's public confirmation form."""
from pathlib import Path
import time
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
target = ROOT / "downloads/motions_processed.zip"
partial = target.with_suffix(".zip.part")
if target.exists():
    raise SystemExit("Archive already exists; validate it before downloading again")
s = requests.Session()
r = s.get("https://drive.google.com/uc", params={"export": "download", "id": "1qpFij33tVZhUkk9-kDaDy1FuQb0LJSEf"}, timeout=30)
r.raise_for_status()
form = BeautifulSoup(r.text, "html.parser").find("form", id="download-form")
if form is None:
    raise RuntimeError("Expected public Drive download confirmation form")
params = {i["name"]: i.get("value", "") for i in form.select("input[name]")}
start = partial.stat().st_size if partial.exists() else 0
headers = {"Range": f"bytes={start}-"} if start else {}
print("Connecting to official download, resume offset", start, flush=True)
with s.get(form["action"], params=params, headers=headers, stream=True, timeout=(30, 120)) as response:
    response.raise_for_status()
    if "text/html" in response.headers.get("Content-Type", ""):
        raise RuntimeError("Drive returned HTML instead of the archive")
    if start and response.status_code != 206:
        start = 0  # server ignored Range: restart this partial only
    mode = "ab" if start else "wb"
    total = int(response.headers.get("Content-Length", 0))
    print("Response", response.status_code, "remaining bytes", total, flush=True)
    downloaded = 0
    last = time.monotonic()
    with partial.open(mode) as f:
        for block in response.iter_content(1024 * 1024):
            f.write(block)
            downloaded += len(block)
            if time.monotonic() - last >= 30:
                print("Downloaded bytes", start + downloaded, flush=True)
                last = time.monotonic()
    if total and downloaded != total:
        raise RuntimeError("Truncated response; rerun to resume")
partial.replace(target)
print("Archive download complete", target.stat().st_size, flush=True)
