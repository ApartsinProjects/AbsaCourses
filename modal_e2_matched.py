"""Modal A10G harness for the covariate-matched faithfulness-filtering experiment (reviewer nfat N2).

Fan-out of seeds. Each worker runs paper/e2_matched_filtering_worker.py inside the
container, builds the covariate-matched retained/control training pools, trains
detection+sentiment on each arm, evaluates on the COMMON GOLD Herath set, commits
artifacts to a Modal volume pulled back to paper/outputs/e2_matched_filtering/.

CORRECT data provenance (a prior Modal run failed on the wrong Herath file):
  corpus  = paper/generated_datasets/batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl
  scores  = paper/faithfulness_audit/at_scale_gpt-4.1-mini_per_row_scores.csv
  herath  = paper/real_transfer/herath_mapped_real_reviews.jsonl   (2,829 rows, correct XMI mapping)

Run:
  modal run modal_e2_matched.py --seeds "42,17,23"
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
CORPUS = PAPER_DIR / "generated_datasets" / "batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl"
SCORES = PAPER_DIR / "faithfulness_audit" / "at_scale_gpt-4.1-mini_per_row_scores.csv"
HERATH = PAPER_DIR / "real_transfer" / "herath_mapped_real_reviews.jsonl"
PHASE_OUT = PAPER_DIR / "outputs" / "e2_matched_filtering"

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
    .add_local_file(str(PAPER_DIR / "e2_matched_filtering_worker.py"), "/app/paper/e2_matched_filtering_worker.py", copy=True)
    .add_local_file(str(CORPUS), "/app/data/generated_reviews.jsonl", copy=True)
    .add_local_file(str(SCORES), "/app/data/per_row_scores.csv", copy=True)
    .add_local_file(str(HERATH), "/app/data/herath_mapped_real_reviews.jsonl", copy=True)
)

app = modal.App("e2-matched-filtering", image=image)
results_vol = modal.Volume.from_name("e2-matched-results", create_if_missing=True)


@app.function(gpu="A10G", timeout=2 * 60 * 60, memory=16384, cpu=4.0, volumes={"/results": results_vol})
def run_seed(spec: dict) -> dict:
    import shutil
    from pathlib import Path as P

    seed = int(spec["seed"])
    do_sanity = bool(spec.get("sanity", False))
    run_name = f"seed{seed}"
    out_root = P(f"/results/{run_name}")
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    t0 = time.time()
    print(f"[{run_name}] start sanity={do_sanity}", flush=True)

    cmd = [
        "python", "paper/e2_matched_filtering_worker.py",
        "--seed", str(seed),
        "--out", str(out_root),
        "--matching-seed", "42",
    ]
    if do_sanity:
        cmd.append("--sanity")
    proc = subprocess.run(cmd, cwd="/app", capture_output=True, text=True, timeout=110 * 60)
    (out_root / "worker.log").write_text(
        f"=== stdout ===\n{proc.stdout}\n=== stderr ===\n{proc.stderr}\n", encoding="utf-8"
    )
    ok = proc.returncode == 0 and (out_root / "result.json").exists()
    result = {"run_name": run_name, "seed": seed, "ok": ok, "rc": proc.returncode,
              "elapsed_s": round(time.time() - t0, 1)}
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
def main(seeds: str = "42,17,23") -> None:
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    specs = [{"seed": s, "sanity": (s == 42)} for s in seed_list]
    print(f"[local] dispatching {len(specs)} seeds: {specs}", flush=True)
    PHASE_OUT.mkdir(parents=True, exist_ok=True)

    results = list(run_seed.map(specs))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = PHASE_OUT / f"modal_summary_{ts}.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[local] summary -> {summary_path}", flush=True)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    print("Run with: modal run modal_e2_matched.py --seeds '42,17,23'")
