#!/usr/bin/env bash
# Orchestrate exp2 -> exp3 -> exp4b sequentially after exp1 frees the GPU.
set -u
cd "$(dirname "$0")"
PY=/c/Python314/python
O=/e/Projects/Submitted/CourseABSA/paper/outputs
export PYTHONUTF8=1

echo "[orch] waiting for exp1 to finish (GPU)..."
while ! grep -q EXP1_DONE "$O/exp1_full.log" 2>/dev/null; do sleep 20; done
echo "[orch] exp1 done. running exp2..."
$PY exp2_sentence_train.py > "$O/exp2.log" 2>&1; echo "[orch] exp2 exit=$?"

echo "[orch] running exp3..."
$PY exp3_aspect_query.py > "$O/exp3.log" 2>&1; echo "[orch] exp3 exit=$?"

echo "[orch] waiting for exp4 generation to complete..."
while ! grep -q 'generated ' "$O/exp4_gen.log" 2>/dev/null; do sleep 20; done
echo "[orch] exp4 gen done. running exp4_train..."
$PY exp4_train.py > "$O/exp4b.log" 2>&1; echo "[orch] exp4b exit=$?"

echo "ALL_EXP_DONE"
