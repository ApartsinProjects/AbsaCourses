"""Phase D2 Modal harness: train BERT-base on each bucket and evaluate Herath transfer.

Parallel fan-out of 5 buckets on A10G via Modal .map().

Run:
  modal run modal_phase_d2.py
  modal run modal_phase_d2.py --buckets top25,top50,full,bot25,random_5k

Each job runs `paper/evaluate_synthetic_to_real_transfer.py` against the
per-bucket synthetic JSONL and the mapped-Herath benchmark, producing a
`summary.csv` per bucket. Results are committed to a Modal volume and
pulled back to ./paper/experiment_rounds/phase_d2_filtering_<date>/runs/<bucket>/
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
BUCKETS_DIR = HERE / "paper" / "faithfulness_audit" / "buckets"
HERATH_ROOT_LOCAL = Path(r"E:\Projects\CourseABSA\external_data\Student_feedback_analysis_dataset\Annotated Student Feedback Data")
EDURABSA_MAPPED_LOCAL = HERE / "external_data" / "EduRABSA_mapped" / "edurabsa_all_mapped.jsonl"
PHASE_OUT = HERE / "paper" / "experiment_rounds" / f"phase_d2_filtering_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
BUCKET_NAMES = ["top25", "top50", "full", "bot25", "random_5k"]

# ----- Modal image -----
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
    .add_local_dir(str(HERATH_ROOT_LOCAL), "/app/external_data/Student_feedback_analysis_dataset/Annotated Student Feedback Data", copy=True)
    .add_local_file(str(EDURABSA_MAPPED_LOCAL), "/app/external_data/edurabsa_all_mapped.jsonl", copy=True)
    .add_local_dir(str(BUCKETS_DIR), "/app/buckets", copy=True)
)

app = modal.App("phase-d2-bucket-train", image=image)
results_vol = modal.Volume.from_name("phase-d2-results", create_if_missing=True)


@app.function(
    gpu="A10G",
    timeout=2 * 60 * 60,
    memory=16384,
    cpu=4.0,
    volumes={"/results": results_vol},
)
def train_bucket(spec: dict) -> dict:
    """Run evaluate_synthetic_to_real_transfer.py for (bucket, seed, arch, target); commit to vol."""
    import shutil
    from pathlib import Path as P

    bucket = spec["bucket"]
    seed = int(spec["seed"])
    arch = spec.get("arch", "bert-base-uncased")
    target = spec.get("target", "herath")  # "herath" or "edurabsa"
    arch_tag = arch.replace("-base-uncased", "").replace("/", "_")
    t0 = time.time()
    bucket_jsonl = P(f"/app/buckets/{bucket}.jsonl")
    assert bucket_jsonl.exists(), f"missing bucket: {bucket_jsonl}"
    n_rows = sum(1 for _ in bucket_jsonl.open(encoding="utf-8"))
    print(f"[{bucket} seed={seed} arch={arch_tag} target={target}] start n_rows={n_rows}", flush=True)

    # Path scheme: legacy ../{bucket}/ and ../{bucket}_seedS/ are Herath BERT seed42 / +seed
    # For non-default arch or target, suffix accordingly.
    suffix = ""
    if arch != "bert-base-uncased": suffix += f"_{arch_tag}"
    if target != "herath": suffix += f"_{target}"
    if arch == "bert-base-uncased" and target == "herath":
        out_root = P(f"/results/{bucket}_seed{seed}") if seed != 42 else P(f"/results/{bucket}")
    else:
        out_root = P(f"/results/{bucket}_seed{seed}{suffix}")
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    cmd = [
        "python", "paper/evaluate_synthetic_to_real_transfer.py",
        "--synthetic-path", str(bucket_jsonl),
        "--herath-root", "/app/external_data/Student_feedback_analysis_dataset/Annotated Student Feedback Data",
        "--approaches", arch,
        "--epochs-detection", "3",
        "--epochs-sentiment", "3",
        "--seed", str(seed),
        "--no-write-latest",
    ]
    if target == "edurabsa":
        cmd += ["--real-mapped-jsonl", "/app/external_data/edurabsa_all_mapped.jsonl"]
    print(f"[{bucket} seed={seed} arch={arch_tag} target={target}] cmd: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd="/app", capture_output=True, text=True, timeout=110 * 60)
    log_path = out_root / "train.log"
    log_path.write_text(
        f"=== stdout ===\n{proc.stdout}\n=== stderr ===\n{proc.stderr}\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        print(f"[{bucket} seed={seed}] FAILED rc={proc.returncode}; see {log_path}", flush=True)
        results_vol.commit()
        return {"bucket": bucket, "seed": seed, "ok": False, "rc": proc.returncode,
                "stderr_tail": proc.stderr[-2000:]}

    # Find the most-recent run dir produced
    runs_root = P("/app/paper/real_transfer/runs")
    if not runs_root.exists():
        print(f"[{bucket} seed={seed}] no runs_root after success?", flush=True)
        results_vol.commit()
        return {"bucket": bucket, "seed": seed, "ok": False, "error": "no runs_root produced"}
    candidates = sorted([d for d in runs_root.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
    if not candidates:
        print(f"[{bucket} seed={seed}] no run dir under {runs_root}", flush=True)
        results_vol.commit()
        return {"bucket": bucket, "seed": seed, "ok": False, "error": "no run dir produced"}
    run_dir = candidates[0]
    print(f"[{bucket} seed={seed}] run_dir={run_dir.name}", flush=True)

    # Copy all artifacts into the volume bucket dir
    for f in run_dir.rglob("*"):
        if f.is_file():
            rel = f.relative_to(run_dir)
            dest = out_root / "run" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
    # Also copy the latest paper-facing summary / per_aspect / mapping if present
    extra = P("/app/paper/real_transfer")
    for name in ("herath_mapped_real_reviews.jsonl", "herath_overlap_summary.csv", "herath_mapping.json"):
        p = extra / name
        if p.exists():
            shutil.copy2(p, out_root / name)
    results_vol.commit()
    elapsed = time.time() - t0
    print(f"[{bucket} seed={seed}] done in {elapsed:.0f}s", flush=True)
    return {"bucket": bucket, "seed": seed, "ok": True, "elapsed_s": round(elapsed, 1),
            "n_rows": n_rows, "run_dir": run_dir.name}


@app.function(volumes={"/results": results_vol}, timeout=15 * 60)
def list_results() -> list[dict]:
    """Inventory what's in the volume."""
    from pathlib import Path as P
    out = []
    root = P("/results")
    for d in sorted(root.iterdir()):
        if d.is_dir():
            files = [str(f.relative_to(root)) for f in d.rglob("*") if f.is_file()]
            out.append({"bucket": d.name, "n_files": len(files), "files_head": files[:10]})
    return out


@app.local_entrypoint()
def main(buckets: str = ",".join(BUCKET_NAMES), seeds: str = "42",
         archs: str = "bert-base-uncased", targets: str = "herath") -> None:
    names = [b.strip() for b in buckets.split(",") if b.strip()]
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    arch_list = [a.strip() for a in archs.split(",") if a.strip()]
    target_list = [t.strip() for t in targets.split(",") if t.strip()]
    print(f"[local] dispatching buckets={names} seeds={seed_list} archs={arch_list} targets={target_list}", flush=True)
    print(f"[local] results will be pulled to: {PHASE_OUT}/runs/", flush=True)
    PHASE_OUT.mkdir(parents=True, exist_ok=True)
    (PHASE_OUT / "runs").mkdir(exist_ok=True)

    specs = [{"bucket": b, "seed": s, "arch": a, "target": t}
             for b in names for s in seed_list for a in arch_list for t in target_list]
    print(f"[local] {len(specs)} (bucket,seed,arch,target) specs", flush=True)
    # Fan out
    results = list(train_bucket.map(specs))
    seed_tag = "_s" + "_".join(str(s) for s in seed_list)
    arch_tag = "_" + "_".join(a.replace("-base-uncased","") for a in arch_list)
    target_tag = "_t" + "_".join(target_list) if target_list != ["herath"] else ""
    summary_path = PHASE_OUT / f"modal_summary{seed_tag}{arch_tag}{target_tag}.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[local] modal summary -> {summary_path}", flush=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    print("Run with: modal run modal_phase_d2.py")
