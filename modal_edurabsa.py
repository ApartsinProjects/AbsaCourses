"""Modal A10G harness: EduRABSA real-only reference + pretrain-finetune, multi-seed.

Runs edurabsa_worker for specs like real_only:42, pretrain_finetune:42 in
parallel containers, committing result.json to the 'edurabsa-results' volume.

  modal run modal_edurabsa.py
Then:
  modal volume get edurabsa-results / paper/outputs/edurabsa
"""
from __future__ import annotations
import json, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import modal

HERE = Path(__file__).parent.resolve()
PAPER = HERE / "paper"
SYNTH = PAPER / "generated_datasets" / "batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl"
EDU_TRAIN = HERE / "external_data" / "EduRABSA_mapped" / "edurabsa_train_mapped.jsonl"
EDU_TEST = HERE / "external_data" / "EduRABSA_mapped" / "edurabsa_test_mapped.jsonl"
OUT_DIR = PAPER / "outputs" / "edurabsa"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel", add_python="3.11")
    .pip_install("transformers==4.46.0", "accelerate==1.1.1", "scikit-learn==1.5.2",
                 "pandas==2.2.3", "numpy==1.26.4")
    .add_local_file(str(PAPER / "absa_data_io.py"), "/app/paper/absa_data_io.py", copy=True)
    .add_local_file(str(PAPER / "absa_model_comparison.py"), "/app/paper/absa_model_comparison.py", copy=True)
    .add_local_file(str(PAPER / "evaluate_synthetic_to_real_transfer.py"), "/app/paper/evaluate_synthetic_to_real_transfer.py", copy=True)
    .add_local_file(str(PAPER / "multiseed_transfer_worker.py"), "/app/paper/multiseed_transfer_worker.py", copy=True)
    .add_local_file(str(PAPER / "checkpoint_train_worker.py"), "/app/paper/checkpoint_train_worker.py", copy=True)
    .add_local_file(str(PAPER / "edurabsa_worker.py"), "/app/paper/edurabsa_worker.py", copy=True)
    .add_local_file(str(SYNTH), "/app/data/synthetic_generated_reviews.jsonl", copy=True)
    .add_local_file(str(EDU_TRAIN), "/app/data/edurabsa_train_mapped.jsonl", copy=True)
    .add_local_file(str(EDU_TEST), "/app/data/edurabsa_test_mapped.jsonl", copy=True)
)

app = modal.App("edurabsa-results", image=image)
vol = modal.Volume.from_name("edurabsa-results", create_if_missing=True)


@app.function(gpu="A10G", timeout=2 * 60 * 60, memory=16384, cpu=4.0, volumes={"/results": vol})
def run_spec(spec: dict) -> dict:
    import shutil
    from pathlib import Path as P
    mode, seed = spec["mode"], int(spec["seed"])
    name = f"{mode}_seed{seed}"
    out = P(f"/results/{name}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    t0 = time.time()
    cmd = ["python", "paper/edurabsa_worker.py", "--mode", mode, "--seed", str(seed), "--out", str(out)]
    proc = subprocess.run(cmd, cwd="/app", capture_output=True, text=True, timeout=110 * 60)
    (out / "worker.log").write_text(f"=== stdout ===\n{proc.stdout}\n=== stderr ===\n{proc.stderr}\n")
    ok = proc.returncode == 0 and (out / "result.json").exists()
    res = {"name": name, "ok": ok, "rc": proc.returncode, "elapsed_s": round(time.time() - t0, 1)}
    if ok:
        res["result"] = json.loads((out / "result.json").read_text())
    else:
        res["stderr_tail"] = proc.stderr[-3000:]
    vol.commit()
    print(f"[{name}] ok={ok} rc={proc.returncode} in {res['elapsed_s']}s", flush=True)
    return res


@app.local_entrypoint()
def main(specs: str = "real_only:17,real_only:23,real_only:42,real_only:89,pretrain_finetune:17,pretrain_finetune:23,pretrain_finetune:42,pretrain_finetune:89") -> None:
    spec_list = []
    for tok in specs.split(","):
        tok = tok.strip()
        if not tok:
            continue
        mode, seed = tok.rsplit(":", 1)
        spec_list.append({"mode": mode, "seed": int(seed)})
    print(f"[local] {len(spec_list)} EduRABSA specs", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = list(run_spec.map(spec_list))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (OUT_DIR / f"modal_summary_{ts}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
