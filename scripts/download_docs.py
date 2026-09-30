"""Download the regulation PDFs listed in data/manifest.json into data/raw/.

    python scripts/download_docs.py          # skips files that already exist
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    docs = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))["documents"]
    failed = 0
    for d in docs:
        out = RAW / d["file"]
        if out.exists():
            print(f"exists  {d['file']}")
            continue
        req = urllib.request.Request(d["source_url"], headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                out.write_bytes(r.read())
            print(f"ok      {d['file']} ({out.stat().st_size // 1024} KB)")
        except Exception as e:  # noqa: BLE001 - report and continue with the rest
            failed += 1
            print(f"FAILED  {d['file']}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
