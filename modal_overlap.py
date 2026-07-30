"""Modal A10G harness: ONE consistent overlap-generalization pass.

Trains {tfidf_two_step, distilbert-base-uncased, bert-base-uncased} on the synthetic
corpus restricted to the Herath conservative 9-aspect overlap (seed 42), and evaluates
each on BOTH the held-out synthetic overlap test split (internal) and the mapped real
Herath benchmark (external, 2,829 reviews). Reuses paper/evaluate_overlap_generalization.py
verbatim via paper/overlap_worker.py, which patches the Herath loader to read the
pre-mapped jsonl instead of the XMI dir.

Run:
  modal run modal_overlap.py
Pull results:
  the local_entrypoint writes paper/outputs/overlap_consistent_modal.csv
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent.resolve()
PAPER_DIR = HERE / "paper"
SYNTH = PAPER_DIR / "generated_datasets" / "batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl"
MAPPED = PAPER_DIR / "reviewer_ab_data" / "herath_mapped_real_reviews_2829.jsonl"

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
    .add_local_file(str(PAPER_DIR / "evaluate_overlap_generalization.py"), "/app/paper/evaluate_overlap_generalization.py", copy=True)
    .add_local_file(str(PAPER_DIR / "overlap_worker.py"), "/app/paper/overlap_worker.py", copy=True)
    .add_local_file(str(SYNTH), "/app/paper/generated_datasets/batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl", copy=True)
    .add_local_file(str(MAPPED), "/app/data/herath_mapped_real_reviews_2829.jsonl", copy=True)
)

app = modal.App("overlap-consistent", image=image)
results_vol = modal.Volume.from_name("overlap-consistent-results", create_if_missing=True)


@app.function(gpu="A10G", timeout=3 * 60 * 60, memory=16384, cpu=4.0, volumes={"/results": results_vol})
def run_overlap(seed: int = 42) -> dict:
    import shutil
    from pathlib import Path as P

    out_root = P(f"/results/seed{seed}")
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    t0 = time.time()
    print(f"[run_overlap seed={seed}] start", flush=True)

    cmd = [
        "python", "paper/overlap_worker.py",
        "--out", str(out_root),
        "--seed", str(seed),
    ]
    proc = subprocess.run(cmd, cwd="/app", capture_output=True, text=True, timeout=170 * 60)
    (out_root / "worker.log").write_text(
        f"=== stdout ===\n{proc.stdout}\n=== stderr ===\n{proc.stderr}\n", encoding="utf-8"
    )
    ok = proc.returncode == 0 and (out_root / "summary.csv").exists()
    result = {"seed": seed, "ok": ok, "rc": proc.returncode, "elapsed_s": round(time.time() - t0, 1)}
    if ok:
        result["summary_csv"] = (out_root / "summary.csv").read_text(encoding="utf-8")
    else:
        result["stderr_tail"] = proc.stderr[-4000:]
        result["stdout_tail"] = proc.stdout[-2000:]
    results_vol.commit()
    print(f"[run_overlap seed={seed}] done ok={ok} rc={proc.returncode} in {result['elapsed_s']}s", flush=True)
    return result


@app.local_entrypoint()
def main(seed: int = 42) -> None:
    print(f"[local] dispatching overlap run seed={seed}", flush=True)
    result = run_overlap.remote(seed)

    out_dir = PAPER_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "overlap_consistent_modal_meta.json"
    meta = {k: v for k, v in result.items() if k != "summary_csv"}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if result.get("ok"):
        csv_path = out_dir / "overlap_consistent_modal.csv"
        csv_path.write_text(result["summary_csv"], encoding="utf-8")
        print(f"[local] WROTE {csv_path}", flush=True)
        print("=== summary.csv ===")
        print(result["summary_csv"])
    else:
        print("[local] RUN FAILED", flush=True)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    print("Run with: modal run modal_overlap.py")
