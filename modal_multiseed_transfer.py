"""Modal A10G harness for the MULTI-SEED synthetic-to-real transfer table.

Fan-out of (approach, seed) specs on A10G, one container each. Every worker runs
paper/multiseed_transfer_worker.py inside the container, commits result.json to a
Modal volume, which is pulled back to
paper/outputs/multiseed_transfer/<approach>_seed<seed>/.

Approaches: tfidf_two_step, distilbert-base-uncased, bert-base-uncased.
Seeds: 42, 17, 23, 41, 89.

Run (smoke, sanity gate at seed 42):
  modal run modal_multiseed_transfer.py --specs "bert-base-uncased:42,distilbert-base-uncased:42"
Run (full 15-cell grid):
  modal run modal_multiseed_transfer.py --specs "<all 15>"
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).parent.resolve()
PAPER_DIR = HERE / "paper"
SYNTH = PAPER_DIR / "generated_datasets" / "batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl"
HERATH_MAPPED = PAPER_DIR / "real_transfer" / "herath_mapped_real_reviews.jsonl"
OUT_DIR = PAPER_DIR / "outputs" / "multiseed_transfer"

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel", add_python="3.11"
    )
    .pip_install(
        "transformers==4.46.0",
        "accelerate==1.1.1",
        "scikit-learn==1.5.2",
        "pandas==2.2.3",
        "numpy==1.26.4",
    )
    .add_local_file(str(PAPER_DIR / "absa_data_io.py"), "/app/paper/absa_data_io.py", copy=True)
    .add_local_file(str(PAPER_DIR / "absa_model_comparison.py"), "/app/paper/absa_model_comparison.py", copy=True)
    .add_local_file(str(PAPER_DIR / "evaluate_synthetic_to_real_transfer.py"), "/app/paper/evaluate_synthetic_to_real_transfer.py", copy=True)
    .add_local_file(str(PAPER_DIR / "multiseed_transfer_worker.py"), "/app/paper/multiseed_transfer_worker.py", copy=True)
    .add_local_file(str(SYNTH), "/app/data/synthetic_generated_reviews.jsonl", copy=True)
    .add_local_file(str(HERATH_MAPPED), "/app/data/herath_mapped_real_reviews.jsonl", copy=True)
)

app = modal.App("multiseed-transfer", image=image)
results_vol = modal.Volume.from_name("multiseed-transfer-results", create_if_missing=True)


@app.function(gpu="A10G", timeout=2 * 60 * 60, memory=16384, cpu=4.0, volumes={"/results": results_vol})
def run_spec(spec: dict) -> dict:
    import shutil
    from pathlib import Path as P

    approach = spec["approach"]
    seed = int(spec["seed"])
    run_name = f"{approach.replace('/', '__')}_seed{seed}"
    out_root = P(f"/results/{run_name}")
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    t0 = time.time()
    print(f"[{run_name}] start", flush=True)

    cmd = [
        "python", "paper/multiseed_transfer_worker.py",
        "--approach", approach,
        "--seed", str(seed),
        "--out", str(out_root),
    ]
    proc = subprocess.run(cmd, cwd="/app", capture_output=True, text=True, timeout=110 * 60)
    (out_root / "worker.log").write_text(
        f"=== stdout ===\n{proc.stdout}\n=== stderr ===\n{proc.stderr}\n", encoding="utf-8"
    )
    ok = proc.returncode == 0 and (out_root / "result.json").exists()
    result = {"run_name": run_name, "approach": approach, "seed": seed, "ok": ok,
              "rc": proc.returncode, "elapsed_s": round(time.time() - t0, 1)}
    if ok:
        try:
            result["headline"] = json.loads((out_root / "result.json").read_text(encoding="utf-8"))
        except Exception as e:
            result["headline_error"] = str(e)
    else:
        result["stderr_tail"] = proc.stderr[-4000:]
        result["stdout_tail"] = proc.stdout[-2000:]
    results_vol.commit()
    print(f"[{run_name}] done ok={ok} rc={proc.returncode} in {result['elapsed_s']}s", flush=True)
    return result


@app.local_entrypoint()
def main(specs: str = "bert-base-uncased:42,distilbert-base-uncased:42") -> None:
    spec_list = []
    for tok in specs.split(","):
        tok = tok.strip()
        if not tok:
            continue
        approach, seed = tok.rsplit(":", 1)
        spec_list.append({"approach": approach.strip(), "seed": int(seed.strip())})
    print(f"[local] dispatching {len(spec_list)} specs: {spec_list}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = list(run_spec.map(spec_list))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = OUT_DIR / f"modal_summary_{ts}.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[local] summary -> {summary_path}", flush=True)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    print("Run with: modal run modal_multiseed_transfer.py --specs '...'")
