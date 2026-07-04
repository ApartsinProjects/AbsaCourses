#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY=/c/Python314/python
O=/e/Projects/Submitted/CourseABSA/paper/outputs
export PYTHONUTF8=1
echo "[orch5] waiting for main orchestrator (ALL_EXP_DONE)..."
while ! grep -q ALL_EXP_DONE "$O/orchestrator.log" 2>/dev/null; do sleep 20; done
echo "[orch5] waiting for exp5 generation..."
while ! grep -q 'generated ' "$O/exp5_gen.log" 2>/dev/null; do sleep 20; done
echo "[orch5] running exp5_train..."
$PY exp5_train.py > "$O/exp5_train.log" 2>&1; echo "[orch5] exp5_train exit=$?"
echo "EXP5_DONE"
