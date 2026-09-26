"""Download the PaySim dataset from Kaggle into data/raw/.

Requires the Kaggle CLI and an API token (~/.kaggle/kaggle.json):
    pip install kaggle
    python scripts/download_data.py

Alternatively download it manually from
https://www.kaggle.com/datasets/ealaxi/paysim1 and place
PS_20174392719_1491204439457_log.csv in data/raw/.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile

from fraudguard.config import RAW_DATA_DIR, RAW_DATA_FILE

KAGGLE_DATASET = "ealaxi/paysim1"


def main() -> int:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_FILE.exists():
        print(f"Dataset already present: {RAW_DATA_FILE}")
        return 0

    print(f"Downloading {KAGGLE_DATASET} via Kaggle CLI ...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(RAW_DATA_DIR)],
            check=True,
        )
    except FileNotFoundError:
        print(
            "Kaggle CLI not found. Install with `pip install kaggle` and configure ~/.kaggle/kaggle.json"
        )
        return 1

    for zip_path in RAW_DATA_DIR.glob("*.zip"):
        print(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DATA_DIR)
        zip_path.unlink()

    if RAW_DATA_FILE.exists():
        print(f"Done: {RAW_DATA_FILE} ({RAW_DATA_FILE.stat().st_size / 1e6:.0f} MB)")
        return 0
    print("Download finished but expected CSV not found; check data/raw/.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
