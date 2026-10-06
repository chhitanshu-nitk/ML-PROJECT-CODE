"""Clean the raw CIC-IDS2017 CSV and cache it as float32 .npy files.

    python scripts/01_preprocess.py                       # uses config.yaml
    python scripts/01_preprocess.py --nrows 200000        # quick test on the first 200k rows
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from ids.data import preprocess
from ids.utils import load_config

p = argparse.ArgumentParser()
p.add_argument("--config")
p.add_argument("--raw-csv", help="override data.raw_csv")
p.add_argument("--processed-dir", help="override data.processed_dir")
p.add_argument("--nrows", type=int)
a = p.parse_args()

cfg = load_config(a.config)
if a.raw_csv:
    cfg["data"]["raw_csv"] = str(Path(a.raw_csv).resolve())
if a.processed_dir:
    cfg["data"]["processed_dir"] = str(Path(a.processed_dir).resolve())
preprocess(cfg, nrows=a.nrows)
