"""Zip the project for Google Colab -> dist/regrag_colab.zip

    python scripts/make_colab_bundle.py

Includes code, manifest, test set, saved evaluation results and figures. Excludes secrets (.env),
the virtualenv, downloaded PDFs and built indexes (the Colab notebook re-downloads and rebuilds them).
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "regrag_colab.zip"

INCLUDE = ["regrag", "eval", "scripts", "tests", "notebooks", "docs/figures", ".streamlit",
           "data/manifest.json", "app.py", "README.md", "requirements.txt", "requirements-colab.txt",
           ".env.example"]
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints"}
SKIP_NAMES = {".env"}
SKIP_SUFFIXES = {".pyc", ".log"}


def files():
    for item in INCLUDE:
        p = ROOT / item
        for f in ([p] if p.is_file() else sorted(p.rglob("*"))):
            if (f.is_file() and f.name not in SKIP_NAMES and f.suffix not in SKIP_SUFFIXES
                    and not SKIP_PARTS & set(f.parts)):
                yield f


def main():
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        n = 0
        for f in files():
            z.write(f, f.relative_to(ROOT).as_posix())
            n += 1
    names = zipfile.ZipFile(OUT).namelist()
    assert ".env" not in names, "refusing to ship secrets"
    print(f"wrote {OUT} ({n} files, {OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
