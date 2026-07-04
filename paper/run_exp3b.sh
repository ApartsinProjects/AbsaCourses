#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY=/c/Python314/python
O=/e/Projects/Submitted/CourseABSA/paper/outputs
export PYTHONUTF8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[orch3b] waiting for exp2b_matched to finish (GPU)..."
while [ ! -f "$O/exp2b_matched.json" ]; do sleep 30; done
sleep 10
echo "[orch3b] running exp3b (NLI aspect-query)..."
$PY exp3b_aspect_query_nli.py > "$O/exp3b.log" 2>&1; echo "[orch3b] exp3b exit=$?"
echo "EXP3B_DONE"
