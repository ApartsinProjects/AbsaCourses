"""Modal A10G harness for the two reviewer-bound experiments (A and B).

A: filtered-test detection micro-F1 (full vs high-faithfulness subset).
B: real-Herath-trained 9-aspect detection micro-F1 (upper reference for 0.4593).

Fan-out of (experiment, seed) specs on A10G. Each worker runs
paper/reviewer_ab_worker.py inside the container, commits artifacts to a Modal
volume, which is pulled back to ./paper/experiment_rounds/reviewer_AB_<TAG>/.

Run (smoke, 1 seed each):
  modal run modal_reviewer_ab.py --specs "A:42,B:42"
Run (scale):
  modal run modal_reviewer_ab.py --specs "A:42,A:17,A:23,B:42,B:17,B:23"

Watchdog: every @app.function has a hard timeout cap; worker subprocess has its
own timeout; --watchdog-minutes bounds the local dispatch wait.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).parent.resolve()
PAPER_DIR = HERE / "paper"
DATA_DIR = PAPER_DIR / "reviewer_ab_data"
TAG = "validation_v5_20260605"
PHASE_OUT = PAPER_DIR / "experiment_rounds" / TAG

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
    .add_local_file(str(PAPER_DIR / "reviewer_ab_worker.py"), "/app/paper/reviewer_ab_worker.py", copy=True)
    .add_local_file(str(DATA_DIR / "generated_reviews_10k.jsonl"), "/app/data/generated_reviews_10k.jsonl", copy=True)
    .add_local_file(str(DATA_DIR / "at_scale_per_row_scores.csv"), "/app/data/at_scale_per_row_scores.csv", copy=True)
    .add_local_file(str(DATA_DIR / "herath_mapped_real_reviews_2829.jsonl"), "/app/data/herath_mapped_real_reviews_2829.jsonl", copy=True)
)

app = modal.App("reviewer-ab", image=image)
results_vol = modal.Volume.from_name("reviewer-ab-results", create_if_missing=True)


@app.function(gpu="A10G", timeout=2 * 60 * 60, memory=16384, cpu=4.0, volumes={"/results": results_vol})
def run_spec(spec: dict) -> dict:
    import shutil
    from pathlib import Path as P

    experiment = spec["experiment"]
    seed = int(spec["seed"])
    run_name = f"{experiment}_seed{seed}"
    out_root = P(f"/results/{run_name}")
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    t0 = time.time()
    print(f"[{run_name}] start", flush=True)

    cmd = [
        "python", "paper/reviewer_ab_worker.py",
        "--experiment", experiment,
        "--seed", str(seed),
        "--out", str(out_root),
    ]
    proc = subprocess.run(cmd, cwd="/app", capture_output=True, text=True, timeout=110 * 60)
    (out_root / "worker.log").write_text(
        f"=== stdout ===\n{proc.stdout}\n=== stderr ===\n{proc.stderr}\n", encoding="utf-8"
    )
    ok = proc.returncode == 0 and (out_root / "result.json").exists()
    result = {"run_name": run_name, "experiment": experiment, "seed": seed, "ok": ok,
              "rc": proc.returncode, "elapsed_s": round(time.time() - t0, 1)}
    if ok:
        try:
            result["headline"] = json.loads((out_root / "result.json").read_text(encoding="utf-8"))
        except Exception as e:
            result["headline_error"] = str(e)
    else:
        result["stderr_tail"] = proc.stderr[-3000:]
        result["stdout_tail"] = proc.stdout[-1500:]
    results_vol.commit()
    print(f"[{run_name}] done ok={ok} rc={proc.returncode} in {result['elapsed_s']}s", flush=True)
    return result


@app.local_entrypoint()
def main(specs: str = "A:42,B:42") -> None:
    spec_list = []
    for tok in specs.split(","):
        tok = tok.strip()
        if not tok:
            continue
        exp, seed = tok.split(":")
        spec_list.append({"experiment": exp.strip(), "seed": int(seed.strip())})
    print(f"[local] dispatching {len(spec_list)} specs: {spec_list}", flush=True)
    PHASE_OUT.mkdir(parents=True, exist_ok=True)

    results = list(run_spec.map(spec_list))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = PHASE_OUT / f"modal_summary_{ts}.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[local] summary -> {summary_path}", flush=True)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    print("Run with: modal run modal_reviewer_ab.py --specs 'A:42,B:42'")
