#!/usr/bin/env python3
"""RunPod driver for C1 (relabel-and-retrain, synthetic-only, transfer metric).

Runs the reproduce-gate (herath seed42) FIRST; only scales to the full
4-seed x 2-target sweep if the baseline reproduces the 0.4593 synthetic-only
transfer reference (within REPRO_TOL). All outputs go to results/ (the only
dir the runner uploads back). Emits the [train]/=== DONE === monitor markers.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()


def log(m):
    print(m, flush=True)


# --- 1. Pin the dependency combo that worked on Modal (the vast run died on a
#        transformers BertModel import from a version mismatch). Do NOT touch torch.
log("[train] installing pinned deps (transformers==4.46.0, accelerate==1.1.1, sklearn, pandas)...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "transformers==4.46.0", "accelerate==1.1.1", "scikit-learn", "pandas"],
    check=True,
)

import torch  # noqa: E402
assert torch.cuda.is_available(), "CUDA not available. This script requires a GPU."
log(f"[train] GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
import transformers  # noqa: E402
log(f"[train] transformers {transformers.__version__}")
# fail fast + loud if the BertModel import path is broken (the vast failure mode)
from transformers import AutoModel  # noqa: E402
_ = AutoModel.from_pretrained("bert-base-uncased")
log("[train] bert-base-uncased import OK")

# --- 2. Make absa_model_comparison + absa_data_io importable the way c1_worker expects
#        (C1_BASE/paper). Files were uploaded flat into HERE.
os.environ["C1_BASE"] = str(HERE)
paperdir = HERE / "paper"
paperdir.mkdir(exist_ok=True)
for f in ("absa_model_comparison.py", "absa_data_io.py"):
    src = HERE / f
    if src.exists() and not (paperdir / f).exists():
        shutil.copy(src, paperdir / f)
sys.path.insert(0, str(HERE))

import c1_worker  # noqa: E402

# The engine writes a GPU lock under paper/benchmark_outputs/; run() does NOT
# create it (the worker's main() normally calls ensure_dirs()). We call run()
# directly, so create the dirs here or training dies on a missing-lock FileNotFound.
c1_worker.eng.configure_console_encoding()
c1_worker.eng.ensure_dirs()

RES = HERE / "results"
RES.mkdir(exist_ok=True)
CLEAN = HERE / "cleaned_corpus_10k.jsonl"
HER = HERE / "herath_mapped_real_reviews_2829.jsonl"
EDU = HERE / "edurabsa_test_mapped.jsonl"


def do(seed, target):
    out = RES / f"{target}_seed{seed}"
    return c1_worker.run(seed, target, out, CLEAN, HER, EDU)


# --- 3. Reproduce-gate: herath seed 42 first ---
log("[train] 1/8 smoke: herath seed42 (reproduce gate vs 0.4593)...")
r = do(42, "herath")
b = r["baseline"]["detection_micro_f1"]
log(f"[train] herath s42: baseline_det_f1={b:.4f} reproduce_ok={r['reproduce_ok']} "
    f"sentmse_delta={r['paired_delta_treatment_minus_baseline']['sentiment_mse_detected']}")
if not r["reproduce_ok"]:
    log(f"[train] REPRODUCE GATE FAILED (baseline {b:.4f} not ~0.4593). Not scaling.")
    log("[train] === DONE ===")
    sys.exit(0)

# --- 4. Scale: remaining seeds x targets ---
specs = [("herath", 17), ("herath", 23), ("herath", 41),
         ("edurabsa", 42), ("edurabsa", 17), ("edurabsa", 23), ("edurabsa", 41)]
for i, (tgt, seed) in enumerate(specs, start=2):
    r = do(seed, tgt)
    d = r["paired_delta_treatment_minus_baseline"]
    log(f"[train] {i}/8 {tgt} s{seed}: det_delta={d['detection_micro_f1']} "
        f"sentmse_delta={d['sentiment_mse_detected']} "
        f"(base_det={r['baseline']['detection_micro_f1']:.4f})")

log("[train] === DONE ===")
