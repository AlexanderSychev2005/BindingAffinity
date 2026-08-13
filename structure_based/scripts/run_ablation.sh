#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

PY=.venv/Scripts/python.exe
SUMMARY=structure_based/runs/ablation_summary.log
> "$SUMMARY"

run_one () {
  local name="$1"
  local data_root="$2"
  echo "=== $name ===" | tee -a "$SUMMARY"
  "$PY" structure_based/scripts/train_gign.py \
    --data-root "$data_root" \
    --run-dir "structure_based/runs/gign_${name}" \
    > "structure_based/runs/gign_${name}.log" 2>&1
  tail -n 1 "structure_based/runs/gign_${name}.log" >> "$SUMMARY"
}

run_one CL1 structure_based/data
run_one CL2 structure_based/data/ablation_CL2
run_one CL3 structure_based/data/ablation_CL3
run_one ALL structure_based/data/ablation_ALL

echo "=== all runs finished ===" | tee -a "$SUMMARY"
