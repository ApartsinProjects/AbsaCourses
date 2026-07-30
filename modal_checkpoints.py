"""Modal A10G harness to train + persist the Zenodo best-per-target checkpoints.

Three configs, one container each, saving to the 'zenodo-checkpoints' volume:
  synthetic_transfer, pretrain_finetune_herath, top50_filtered

Run:
  modal run modal_checkpoints.py
Then download the saved checkpoints:
  modal volume get zenodo-checkpoints /ckpt paper/outputs/zenodo_checkpoints
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).parent.resolve()
PAPER = HERE / "paper"
SYNTH = PAPER / "generated_datasets" / "batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl"
HERATH = PAPER / "real_transfer" / "herath_mapped_real_reviews.jsonl"
SCORES = PAPER / "faithfulness_audit" / "at_scale_gpt-4.1-mini_per_row_scores.csv"
OUT_DIR = PAPER / "outputs" / "checkpoint_train"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel", add_python="3.11")
    .pip_install("transformers==4.46.0", "accelerate==1.1.1", "scikit-learn==1.5.2",
                 "pandas==2.2.3", "numpy==1.26.4")
    .add_local_file(str(PAPER / "absa_data_io.py"), "/app/paper/absa_data_io.py", copy=True)
    .add_local_file(str(PAPER / "absa_model_comparison.py"), "/app/paper/absa_model_comparison.py", copy=True)
    .add_local_file(str(PAPER / "evaluate_synthetic_to_real_transfer.py"), "/app/paper/evaluate_synthetic_to_real_transfer.py", copy=True)
    .add_local_file(str(PAPER / "multiseed_transfer_worker.py"), "/app/paper/multiseed_transfer_worker.py", copy=True)
    .add_local_file(str(PAPER / "checkpoint_train_worker.py"), "/app/paper/checkpoint_train_worker.py", copy=True)
    .add_local_file(str(SYNTH), "/app/data/synthetic_generated_reviews.jsonl", copy=True)
    .add_local_file(str(HERATH), "/app/data/herath_mapped_real_reviews.jsonl", copy=True)
    .add_local_file(str(SCORES), "/app/data/at_scale_per_row_scores.csv", copy=True)
)

app = modal.App("zenodo-checkpoints", image=image)
vol = modal.Volume.from_name("zenodo-checkpoints", create_if_missing=True)


@app.function(gpu="A10G", timeout=2 * 60 * 60, memory=16384, cpu=4.0, volumes={"/results": vol})
def run_config(config: str) -> dict:
    import shutil
    from pathlib import Path as P
    out = P(f"/results/ckpt/{config}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    t0 = time.time()
    cmd = ["python", "paper/checkpoint_train_worker.py", "--config", config,
           "--synthetic", "/app/data/synthetic_generated_reviews.jsonl",
           "--herath", "/app/data/herath_mapped_real_reviews.jsonl",
           "--scores", "/app/data/at_scale_per_row_scores.csv",
           "--out", str(out)]
    proc = subprocess.run(cmd, cwd="/app", capture_output=True, text=True, timeout=110 * 60)
    (out / "worker.log").write_text(f"=== stdout ===\n{proc.stdout}\n=== stderr ===\n{proc.stderr}\n")
    ok = proc.returncode == 0 and (out / "checkpoint_meta.json").exists()
    res = {"config": config, "ok": ok, "rc": proc.returncode, "elapsed_s": round(time.time() - t0, 1)}
    if ok:
        res["meta"] = json.loads((out / "checkpoint_meta.json").read_text())
    else:
        res["stderr_tail"] = proc.stderr[-4000:]
    vol.commit()
    print(f"[{config}] done ok={ok} rc={proc.returncode} in {res['elapsed_s']}s", flush=True)
    return res


@app.local_entrypoint()
def main(configs: str = "synthetic_transfer,pretrain_finetune_herath,top50_filtered") -> None:
    cfg_list = [c.strip() for c in configs.split(",") if c.strip()]
    print(f"[local] training {len(cfg_list)} checkpoints: {cfg_list}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = list(run_config.map(cfg_list))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (OUT_DIR / f"modal_summary_{ts}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
