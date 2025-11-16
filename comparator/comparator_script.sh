#!/usr/bin/env bash
set -euo pipefail

# 1. Run the CPU and GPU versions
./cpu_solver        > cpu_results.csv
./gpu_solver        > gpu_results.csv

# 2. Optionally skip the header (if you have one)
tail -n +2 cpu_results.csv > cpu_body.csv
tail -n +2 gpu_results.csv > gpu_body.csv

# 3. Compare
if diff -u cpu_body.csv gpu_body.csv >/dev/null; then
  echo "✅ Outputs match exactly"
  exit 0
else
  echo "❌ Outputs differ! See unified diff below:"
  diff -u cpu_body.csv gpu_body.csv
  exit 1
fi

