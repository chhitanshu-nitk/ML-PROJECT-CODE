#!/usr/bin/env bash
# Full pipeline: preprocess once -> paper reproduction -> custom strict run
set -e
python scripts/01_preprocess.py
python scripts/02_run_experiments.py --protocol paper  --name paper_full
python scripts/02_run_experiments.py --protocol custom --name custom_strict
