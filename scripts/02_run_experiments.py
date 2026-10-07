"""Run the experiments (baseline vs proposal, binary + multiclass, DT/RF/ET/XGB).

    python scripts/02_run_experiments.py --protocol paper  --name paper_protocol
    python scripts/02_run_experiments.py --protocol custom --name custom_strict
    python scripts/02_run_experiments.py --folds 3 --models DT ET --max-per-class 50000 --name quick
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from ids.experiment import run
from ids.utils import load_config, resolve

p = argparse.ArgumentParser()
p.add_argument("--config")
p.add_argument("--protocol", choices=["paper", "custom"])
p.add_argument("--tasks", nargs="+", choices=["binary", "multiclass"])
p.add_argument("--modes", nargs="+", choices=["baseline", "proposal"])
p.add_argument("--models", nargs="+", choices=["DT", "RF", "ET", "XGB"])
p.add_argument("--folds", type=int, help="k for stratified CV (1 = single hold-out split)")
p.add_argument("--max-per-class", type=int, help="override oversampling cap for both tasks")
p.add_argument("--sample-frac", type=float, help="use only this fraction of the data (debug)")
p.add_argument("--processed-dir")
p.add_argument("--name", default=None, help="results sub-folder name")
a = p.parse_args()

cfg = load_config(a.config)
if a.processed_dir:
    cfg["data"]["processed_dir"] = str(Path(a.processed_dir).resolve())
e = cfg["experiment"]
protocol = a.protocol or e["protocol"]
name = a.name or protocol
run(cfg,
    protocol=protocol,
    tasks=a.tasks or e["tasks"],
    modes=a.modes or e["modes"],
    models=a.models or e["models"],
    folds=a.folds if a.folds is not None else e["folds"],
    out_dir=resolve("results") / name,
    sample_frac=a.sample_frac,
    max_per_class=a.max_per_class)
