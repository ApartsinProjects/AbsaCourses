#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY=/c/Python314/python
O=/e/Projects/Submitted/CourseABSA/paper/outputs
export PYTHONUTF8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[msorch] waiting for exp3b to finish..."
while ! grep -q EXP3B_DONE "$O/orchestrator3b.log" 2>/dev/null; do sleep 30; done
sleep 10
echo "[msorch] running multi-seed validation (3 seeds, paired deltas)..."
$PY multiseed_validation.py > "$O/multiseed.log" 2>&1; echo "[msorch] multiseed exit=$?"
echo "MULTISEED_DONE"
