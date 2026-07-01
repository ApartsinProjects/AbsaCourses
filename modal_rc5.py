"""RC5 (reviewer Cycle 1): minimum fine-tuning-size curve on Modal A10G.

Fan out the synth-pretrain -> real-Herath fine-tune (worker experiment B5) across
real-train subset sizes {100,250,500,1000,full} x seeds {42,17}. Each worker
subsamples the real-train split to N (seeded), leaving calib/test fixed, so the
curve shows real-test micro-F1 vs the number of real fine-tuning examples.

Pulls results back to paper/experiment_rounds/rc5_finetune_curve_<ts>/.

  modal run modal_rc5.py                       # default grid
  modal run modal_rc5.py --sizes "100,500,full" --seeds "42"
"""
from __future__ import annotations
import json, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import modal

HERE = Path(__file__).parent.resolve()
PAPER_DIR = HERE / "paper"
DATA_DIR = PAPER_DIR / "reviewer_ab_data"
TAG = "rc5_finetune_curve"
PHASE_OUT = PAPER_DIR / "experiment_rounds" / TAG

image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel", add_python="3.11")
    .pip_install("transformers==4.46.0", "accelerate==1.1.1", "scikit-learn==1.5.2",
                 "pandas==2.2.3", "numpy==1.26.4")
    .add_local_file(str(PAPER_DIR / "absa_data_io.py"), "/app/paper/absa_data_io.py", copy=True)
    .add_local_file(str(PAPER_DIR / "absa_model_comparison.py"), "/app/paper/absa_model_comparison.py", copy=True)
    .add_local_file(str(PAPER_DIR / "reviewer_ab_worker.py"), "/app/paper/reviewer_ab_worker.py", copy=True)
    .add_local_file(str(DATA_DIR / "generated_reviews_10k.jsonl"), "/app/data/generated_reviews_10k.jsonl", copy=True)
    .add_local_file(str(DATA_DIR / "herath_mapped_real_reviews_2829.jsonl"), "/app/data/herath_mapped_real_reviews_2829.jsonl", copy=True)
)

app = modal.App("courseabsa-rc5", image=image)
results_vol = modal.Volume.from_name("courseabsa-rc5-results", create_if_missing=True)


@app.function(gpu="A10G", timeout=2 * 60 * 60, memory=16384, cpu=4.0, volumes={"/results": results_vol})
def run_spec(spec: dict) -> dict:
    import shutil
    from pathlib import Path as P
    seed = int(spec["seed"]); n = spec["real_train_n"]
    run_name = f"B5_n{n if n is not None else 'full'}_seed{seed}"
    out_root = P(f"/results/{run_name}")
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    t0 = time.time()
    cmd = ["python", "paper/reviewer_ab_worker.py", "--experiment", "B5",
           "--seed", str(seed), "--out", str(out_root)]
    if n is not None:
        cmd += ["--real-train-n", str(n)]
    print(f"[{run_name}] start: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd="/app", capture_output=True, text=True, timeout=110 * 60)
    (out_root / "worker.log").write_text(f"=== stdout ===\n{proc.stdout}\n=== stderr ===\n{proc.stderr}\n", encoding="utf-8")
    ok = proc.returncode == 0 and (out_root / "result.json").exists()
    res = {"run_name": run_name, "seed": seed, "real_train_n": n, "ok": ok,
           "rc": proc.returncode, "elapsed_s": round(time.time() - t0, 1)}
    if ok:
        res["headline"] = json.loads((out_root / "result.json").read_text(encoding="utf-8"))
    else:
        res["stderr_tail"] = proc.stderr[-3000:]
    results_vol.commit()
    print(f"[{run_name}] done ok={ok} rc={proc.returncode} in {res['elapsed_s']}s", flush=True)
    return res


@app.local_entrypoint()
def main(sizes: str = "100,250,500,1000,full", seeds: str = "42,17") -> None:
    size_list = [None if s.strip().lower() == "full" else int(s) for s in sizes.split(",") if s.strip()]
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    specs = [{"seed": sd, "real_train_n": n} for n in size_list for sd in seed_list]
    print(f"[local] dispatching {len(specs)} B5 runs: sizes={size_list} seeds={seed_list}", flush=True)
    PHASE_OUT.mkdir(parents=True, exist_ok=True)
    results = list(run_spec.map(specs))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (PHASE_OUT / f"modal_summary_{ts}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    # compact curve table
    curve = []
    for r in results:
        h = r.get("headline") or {}
        curve.append({"seed": r["seed"], "real_train_n": r["real_train_n"],
                      "n_real_train": h.get("n_real_train"),
                      "micro_f1": (h.get("full_test") or h.get("real_test") or {}).get("micro_f1") if isinstance(h.get("full_test") or h.get("real_test"), dict) else h.get("micro_f1"),
                      "ok": r["ok"]})
    (PHASE_OUT / f"curve_{ts}.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")
    print(json.dumps(curve, indent=2))
    print(f"[local] summary -> {PHASE_OUT}")
