#!/usr/bin/env bash
# Full pipeline: clean data -> paper-protocol run -> leak-free run
set -e
python scripts/01_preprocess.py
python scripts/02_run_experiments.py --protocol paper  --name paper_protocol
python scripts/02_run_experiments.py --protocol strict --name strict_protocol
