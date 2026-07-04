#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY=/c/Python314/python
O=/e/Projects/Submitted/CourseABSA/paper/outputs
export PYTHONUTF8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[ms2] waiting for independent baseline generation..."
while ! grep -q 'generated ' "$O/indep_gen.log" 2>/dev/null; do sleep 20; done
sleep 5
echo "[ms2] running corrected multi-seed validation..."
$PY multiseed_validation.py > "$O/multiseed.log" 2>&1; echo "[ms2] multiseed exit=$?"
echo "MULTISEED_DONE"
