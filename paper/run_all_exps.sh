#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY=/c/Python314/python
O=/e/Projects/Submitted/CourseABSA/paper/outputs
export PYTHONUTF8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[all] exp2..."; $PY exp2_sentence_train.py > "$O/exp2.log" 2>&1; echo "[all] exp2 exit=$?"
echo "[all] exp3..."; $PY exp3_aspect_query.py   > "$O/exp3.log" 2>&1; echo "[all] exp3 exit=$?"
echo "[all] exp4b..."; $PY exp4_train.py         > "$O/exp4b.log" 2>&1; echo "[all] exp4b exit=$?"
echo "[all] exp5b..."; $PY exp5_train.py         > "$O/exp5_train.log" 2>&1; echo "[all] exp5b exit=$?"
echo "ALL_EXP_DONE"
