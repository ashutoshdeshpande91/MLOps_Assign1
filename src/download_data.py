"""Download the UCI Heart Disease (Cleveland) dataset into data/raw/.

Usage:
    python src/download_data.py
"""
import io
import sys
import zipfile
from pathlib import Path

import requests

RAW_DIR = Path("data/raw")
RAW_FILE = RAW_DIR / "processed.cleveland.data"

PRIMARY_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
FALLBACK_ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"


def download_data(force: bool = False) -> Path:
    """Download the raw file if it does not already exist. Returns its path."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists() and not force:
        print(f"[skip] {RAW_FILE} already exists")
        return RAW_FILE

    try:
        print(f"Downloading from {PRIMARY_URL}")
        resp = requests.get(PRIMARY_URL, timeout=30)
        resp.raise_for_status()
        RAW_FILE.write_bytes(resp.content)
    except requests.RequestException as err:
        print(f"Primary URL failed ({err}); trying fallback zip...")
        resp = requests.get(FALLBACK_ZIP_URL, timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            RAW_FILE.write_bytes(zf.read("processed.cleveland.data"))

    print(f"Saved to {RAW_FILE}")
    return RAW_FILE


if __name__ == "__main__":
    try:
        download_data(force="--force" in sys.argv)
    except Exception as exc:  # fail loudly so CI shows a clear error
        print(f"Download failed: {exc}", file=sys.stderr)
        sys.exit(1)
