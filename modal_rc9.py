"""RC9 (reviewer comment 9): sample-efficiency curve for Figure 6 on Modal A10G.

Produces BOTH curves the reviewer asked for, on the SAME fixed real-Herath test set:
  arm=real_only              : train bert-base-uncased detection head FROM SCRATCH on N
                               real Herath train reviews; eval detection micro-F1 on the
                               held-out real Herath test (9-aspect).
  arm=synth_pretrain_finetune: load/train the synthetic-pretrained detector, fine-tune on
                               the same N real reviews; eval on the same held-out test.

Grid: N in {100,250,500,1000,full(~1980)} x seed in {17,23,41,42,89} x arm in the two above.
5 x 5 x 2 = 50 short fine-tunes. Identical split logic to modal_rc5.py (the original
Figure 6 two-seed curve), same data files, so the new numbers are directly comparable.

Pulls results back to paper/experiment_rounds/rc9_sample_efficiency/.

  modal run modal_rc9.py                                              # full 50-run grid
  modal run modal_rc9.py --sizes 100 --seeds 17 --arms real_only,synth_pretrain_finetune  # smoke
"""
from __future__ import annotations
import json, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import modal

HERE = Path(__file__).parent.resolve()
PAPER_DIR = HERE / "paper"
DATA_DIR = PAPER_DIR / "reviewer_ab_data"
TAG = "rc9_sample_efficiency"
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

app = modal.App("courseabsa-rc9", image=image)
results_vol = modal.Volume.from_name("courseabsa-rc9-results", create_if_missing=True)


@app.function(gpu="A10G", timeout=2 * 60 * 60, memory=16384, cpu=4.0, volumes={"/results": results_vol})
def run_spec(spec: dict) -> dict:
    import shutil
    from pathlib import Path as P
    seed = int(spec["seed"]); n = spec["real_train_n"]; arm = spec["arm"]
    run_name = f"{arm}_n{n if n is not None else 'full'}_seed{seed}"
    out_root = P(f"/results/{run_name}")
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    t0 = time.time()
    cmd = ["python", "paper/reviewer_ab_worker.py", "--experiment", "B5",
           "--arm", arm, "--seed", str(seed), "--out", str(out_root)]
    if n is not None:
        cmd += ["--real-train-n", str(n)]
    print(f"[{run_name}] start: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd="/app", capture_output=True, text=True, timeout=110 * 60)
    (out_root / "worker.log").write_text(f"=== stdout ===\n{proc.stdout}\n=== stderr ===\n{proc.stderr}\n", encoding="utf-8")
    ok = proc.returncode == 0 and (out_root / "result.json").exists()
    res = {"run_name": run_name, "arm": arm, "seed": seed, "real_train_n": n, "ok": ok,
           "rc": proc.returncode, "elapsed_s": round(time.time() - t0, 1)}
    if ok:
        res["headline"] = json.loads((out_root / "result.json").read_text(encoding="utf-8"))
    else:
        res["stderr_tail"] = proc.stderr[-3000:]
    results_vol.commit()
    print(f"[{run_name}] done ok={ok} rc={proc.returncode} in {res['elapsed_s']}s", flush=True)
    return res


@app.local_entrypoint()
def main(sizes: str = "100,250,500,1000,full", seeds: str = "17,23,41,42,89",
         arms: str = "real_only,synth_pretrain_finetune") -> None:
    size_list = [None if s.strip().lower() == "full" else int(s) for s in sizes.split(",") if s.strip()]
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    specs = [{"seed": sd, "real_train_n": n, "arm": a}
             for a in arm_list for n in size_list for sd in seed_list]
    print(f"[local] dispatching {len(specs)} runs: arms={arm_list} sizes={size_list} seeds={seed_list}", flush=True)
    PHASE_OUT.mkdir(parents=True, exist_ok=True)
    results = list(run_spec.map(specs))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (PHASE_OUT / f"modal_summary_{ts}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    curve = []
    for r in results:
        h = r.get("headline") or {}
        curve.append({"arm": r["arm"], "seed": r["seed"], "real_train_n": r["real_train_n"],
                      "n_real_train": h.get("n_real_train"),
                      "micro_f1": h.get("synth_to_real_micro_f1"), "ok": r["ok"]})
    (PHASE_OUT / f"curve_{ts}.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")
    print(json.dumps(curve, indent=2))
    print(f"[local] summary -> {PHASE_OUT}")
