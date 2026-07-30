"""Modal A10G harness for reviewer weaknesses W5 and W7.

W5: held-out-generator detection transfer (all-synthetic; train GPT -> eval Gemini/GLM/Llama + reverse).
W7: audit-quartile dose-response (train detector per audit-score quartile; eval internal + canonical Herath).

Fan-out of (experiment, seed) specs on A10G. Each worker runs paper/w{5,7}_worker.py
inside the container, commits artifacts to a Modal volume, pulled back locally.

Based on the working modal_reviewer_ab.py template. Uses the repo's real harness
(absa_model_comparison, evaluate_synthetic_to_real_transfer) and canonical data:
  - GPT synthetic corpus: batch_69cc15c...jsonl (10k)
  - held-out gens: paper/outputs/n3_gen_{gpt5nano,gemini_flash,glm_46,llama33_70b}.jsonl
  - per-row audit scores: paper/faithfulness_audit/at_scale_gpt-4.1-mini_per_row_scores.csv
  - Herath: regenerated from canonical XMI (external_data/.../Annotated Student Feedback Data)

Run:
  modal run modal_w5_w7.py --specs "W5:42,W5:17,W5:23,W7:42"
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
GEN_DS = PAPER_DIR / "generated_datasets"
OUTPUTS = PAPER_DIR / "outputs"
AUDIT = PAPER_DIR / "faithfulness_audit"
HERATH_ROOT = HERE / "external_data" / "Student_feedback_analysis_dataset" / "Annotated Student Feedback Data"
TAG = "w5_w7_20260730"
PHASE_OUT = PAPER_DIR / "experiment_rounds" / TAG

# 6 canonical Herath XMI files (load_herath_mapped_dataset rglobs *.xmi under the root)
HERATH_XMI = [
    "Annotator_1/Annotated_part_1/Final_Dataset_tsv.xmi",
    "Annotator_1/Annotated_part_2/ratemy_professor_data_from_sorted_list_shuffle_1.xmi",
    "Annotator_2/Annotated_part_3/Final_Dataset_tsv.xmi",
    "Annotator_2/Annotated_part_4/ratemy_professor_data_from_sorted_list_shuffle_1.xmi",
    "Annotator_2/Annotated_part_5/additional_100_of_rate_my_proffesor.xmi",
    "Annotator_3/Annotated_part_6/Final_Dataset_tsv.xmi",
]

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
    .add_local_file(str(PAPER_DIR / "w5_worker.py"), "/app/paper/w5_worker.py", copy=True)
    .add_local_file(str(PAPER_DIR / "w7_worker.py"), "/app/paper/w7_worker.py", copy=True)
    .add_local_file(str(GEN_DS / "batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl"), "/app/data/gpt_corpus_10k.jsonl", copy=True)
    .add_local_file(str(OUTPUTS / "n3_gen_gpt5nano.jsonl"), "/app/data/n3_gen_gpt5nano.jsonl", copy=True)
    .add_local_file(str(OUTPUTS / "n3_gen_gemini_flash.jsonl"), "/app/data/n3_gen_gemini_flash.jsonl", copy=True)
    .add_local_file(str(OUTPUTS / "n3_gen_glm_46.jsonl"), "/app/data/n3_gen_glm_46.jsonl", copy=True)
    .add_local_file(str(OUTPUTS / "n3_gen_llama33_70b.jsonl"), "/app/data/n3_gen_llama33_70b.jsonl", copy=True)
    .add_local_file(str(AUDIT / "at_scale_gpt-4.1-mini_per_row_scores.csv"), "/app/data/at_scale_per_row_scores.csv", copy=True)
)
for rel in HERATH_XMI:
    image = image.add_local_file(str(HERATH_ROOT / rel), f"/app/herath/{rel}", copy=True)

app = modal.App("reviewer-w5-w7", image=image)
results_vol = modal.Volume.from_name("reviewer-w5-w7-results", create_if_missing=True)


@app.function(gpu="A10G", timeout=3 * 60 * 60, memory=24576, cpu=4.0, volumes={"/results": results_vol})
def run_spec(spec: dict) -> dict:
    import shutil
    from pathlib import Path as P

    experiment = spec["experiment"]  # "W5" or "W7"
    seed = int(spec["seed"])
    run_name = f"{experiment}_seed{seed}"
    out_root = P(f"/results/{run_name}")
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    t0 = time.time()
    print(f"[{run_name}] start", flush=True)

    worker = {"W5": "paper/w5_worker.py", "W7": "paper/w7_worker.py"}[experiment]
    cmd = ["python", worker, "--seed", str(seed), "--out", str(out_root)]
    proc = subprocess.run(cmd, cwd="/app", capture_output=True, text=True, timeout=170 * 60)
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
        result["stderr_tail"] = proc.stderr[-4000:]
        result["stdout_tail"] = proc.stdout[-2000:]
    results_vol.commit()
    print(f"[{run_name}] done ok={ok} rc={proc.returncode} in {result['elapsed_s']}s", flush=True)
    return result


@app.local_entrypoint()
def main(specs: str = "W5:42,W7:42") -> None:
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
    print("Run with: modal run modal_w5_w7.py --specs 'W5:42,W5:17,W5:23,W7:42'")
