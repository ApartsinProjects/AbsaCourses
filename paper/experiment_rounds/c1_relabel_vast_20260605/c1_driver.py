"""C1 driver: run baseline+treatment synthetic-only transfer for given seeds and targets.

Runs INSIDE the vast.ai container (cwd=/app). For each (seed, target) it calls the
c1_worker.run() in-process and writes results under results/<target>_seed<seed>/.

Output convention for gpu2vast monitoring: [train] phase markers, step lines, === DONE ===.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

assert torch.cuda.is_available(), "CUDA not available. This script requires a GPU."
DEVICE = torch.device("cuda")
print(f"[train] GPU: {torch.cuda.get_device_name(0)}"); sys.stdout.flush()

# Base dir holds paper/ (engine), code/ (worker+driver), data/ (inputs).
# Defaults to the current working directory (the vast container runs from /workspace/data).
BASE = Path(os.environ.get("C1_BASE", ".")).resolve()
sys.path.insert(0, str(BASE / "paper"))
sys.path.insert(0, str(BASE / "code"))  # c1_worker.py

import c1_worker as c1  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

c1.eng.configure_console_encoding()
c1.eng.ensure_dirs()  # creates benchmark_outputs/ so the gpu_training_lock path exists

RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)
DATA = BASE / "data"

SEEDS = [int(s) for s in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["42"])]
TARGETS = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["herath", "edurabsa"])

writer = SummaryWriter(log_dir="runs")
writer.add_text("phase", f"start: seeds={SEEDS} targets={TARGETS}", 0)
writer.flush()

print(f"[train] C1 relabel-and-retrain. seeds={SEEDS} targets={TARGETS}"); sys.stdout.flush()
print(f"[train] cleaned corpus: {DATA/'cleaned_corpus_10k.jsonl'}"); sys.stdout.flush()

all_results = []
total = len(SEEDS) * len(TARGETS)
step = 0
t_start = time.time()
for target in TARGETS:
    for seed in SEEDS:
        step += 1
        print(f"[train] {step}/{total} target={target} seed={seed} START"); sys.stdout.flush()
        writer.add_text("phase", f"run_start: {step}/{total} target={target} seed={seed}", step)
        out_dir = RESULTS / f"{target}_seed{seed}"
        t0 = time.time()
        res = c1.run(
            seed=seed, target=target, out_dir=out_dir,
            cleaned_jsonl=DATA / "cleaned_corpus_10k.jsonl",
            herath_jsonl=DATA / "herath_mapped_real_reviews_2829.jsonl",
            edurabsa_jsonl=DATA / "edurabsa_test_mapped.jsonl",
        )
        res["elapsed_seconds"] = round(time.time() - t0, 1)
        (out_dir / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        all_results.append(res)
        b = res["baseline"]; tr = res["treatment"]; d = res["paired_delta_treatment_minus_baseline"]
        print(f"[train] {step}/{total} target={target} seed={seed} DONE "
              f"baseline[detF1={b['detection_micro_f1']:.4f} sentMSE={b['sentiment_mse_detected']:.4f}] "
              f"treatment[detF1={tr['detection_micro_f1']:.4f} sentMSE={tr['sentiment_mse_detected']:.4f}] "
              f"delta[detF1={d['detection_micro_f1']:+.4f} sentMSE={d['sentiment_mse_detected']:+.4f}] "
              f"reproduce_ok={res['reproduce_ok']} ({res['elapsed_seconds']:.0f}s)"); sys.stdout.flush()
        writer.add_scalar(f"{target}/baseline_detF1", b["detection_micro_f1"], seed)
        writer.add_scalar(f"{target}/treatment_detF1", tr["detection_micro_f1"], seed)
        writer.add_scalar(f"{target}/baseline_sentMSE", b["sentiment_mse_detected"], seed)
        writer.add_scalar(f"{target}/treatment_sentMSE", tr["sentiment_mse_detected"], seed)
        writer.flush()

(RESULTS / "all_results.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
import shutil  # noqa: E402
try:
    shutil.copytree("runs", "results/tb_runs", dirs_exist_ok=True)
except Exception as e:
    print(f"[train] tb copy warn: {e}")
elapsed = time.time() - t_start
writer.add_text("phase", f"done: {len(all_results)} runs, {elapsed:.1f}s", total)
writer.close()
print(f"[train] Loss: n/a (classification)"); sys.stdout.flush()
print(f"[train] all_results -> results/all_results.json  ({elapsed:.0f}s total)"); sys.stdout.flush()
print("[train] === DONE ==="); sys.stdout.flush()
