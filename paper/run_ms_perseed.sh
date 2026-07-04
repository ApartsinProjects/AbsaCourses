#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY=/c/Python314/python
O=/e/Projects/Submitted/CourseABSA/paper/outputs
export PYTHONUTF8=1
for s in 17 23; do
  echo "[perseed] === starting seed $s in fresh process ===" >> "$O/multiseed.log"
  $PY multiseed_validation.py --seeds $s >> "$O/multiseed.log" 2>&1
  echo "[perseed] seed $s exit=$?" >> "$O/multiseed.log"
done
echo "MULTISEED_DONE" >> "$O/multiseed.log"
