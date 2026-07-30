"""In-container worker: runs evaluate_overlap_generalization ONE consistent pass
(internal synthetic-overlap-test + external mapped-real-Herath) for the 3 approaches,
seed 42, patching the Herath loader to read the pre-mapped 2,829-review jsonl instead
of the XMI directory (which is not uploaded to Modal).

Writes /app/paper/real_transfer/overlap_internal_vs_external_summary.csv and copies it
to the --out dir as summary.csv.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app/paper")

import evaluate_synthetic_to_real_transfer as tr  # noqa: E402
import evaluate_overlap_generalization as ovl  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--synthetic-path", default="/app/paper/generated_datasets/batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl")
    p.add_argument("--mapped-jsonl", default="/app/data/herath_mapped_real_reviews_2829.jsonl")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    mapped = Path(args.mapped_jsonl)

    # Patch: overlap harness calls load_herath_mapped_dataset(root) -> XMI parse.
    # Redirect it to the pre-mapped jsonl (same 2,829-review 9-aspect benchmark).
    def _load_from_mapped(root):  # noqa: ANN001
        df = tr.load_real_from_mapped_jsonl(mapped)
        print(f"[overlap_worker] loaded mapped real jsonl rows={len(df)} from {mapped}", flush=True)
        return df

    ovl.load_herath_mapped_dataset = _load_from_mapped

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.argv = [
        "evaluate_overlap_generalization.py",
        "--synthetic-path", args.synthetic_path,
        "--herath-root", "/app/data",  # ignored by the patched loader
        "--approaches", "tfidf_two_step", "distilbert-base-uncased", "bert-base-uncased",
        "--epochs-detection", "3",
        "--epochs-sentiment", "3",
        "--batch-size", "8",
        "--max-len", "192",
        "--lr", "3e-5",
        "--seed", str(args.seed),
        "--write-latest",
    ]

    t0 = time.time()
    print(f"[overlap_worker] start seed={args.seed}", flush=True)
    ovl.main()
    elapsed = round(time.time() - t0, 1)

    latest = ovl.OUT_DIR / "overlap_internal_vs_external_summary.csv"
    if not latest.exists():
        raise SystemExit(f"expected summary CSV not found at {latest}")
    shutil.copy(str(latest), str(out_dir / "summary.csv"))
    per_aspect = ovl.OUT_DIR / "overlap_internal_vs_external_per_aspect.csv"
    if per_aspect.exists():
        shutil.copy(str(per_aspect), str(out_dir / "per_aspect.csv"))
    (out_dir / "worker_done.json").write_text(
        json.dumps({"ok": True, "elapsed_seconds": elapsed, "seed": args.seed}, indent=2),
        encoding="utf-8",
    )
    print(f"[overlap_worker] DONE in {elapsed}s -> {out_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
