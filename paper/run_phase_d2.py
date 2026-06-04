"""Phase D2 orchestrator: faithfulness-aware filtering ablation.

Design: paper/plans/phase_d2_filtering_ablation_plan.md

Stages (each callable independently via --stage):

  calibrate   : re-audit the existing 250 rows with the cheap model and
                compare Spearman rho against the GPT-5.2 audit.
  audit       : run the at-scale audit on all 10K rows with the accepted
                cheap model via OpenAI Batch.
  bucket      : assign each row to top25 / top50 / full / bot25 / random_5k
                buckets by score.
  train       : train BERT-base on each bucket (reuses absa_model_comparison).
  evaluate    : evaluate each trained model on internal test + Herath
                transfer (reuses evaluate_synthetic_to_real_transfer).
  report      : aggregate to one results table + one figure.
  all         : run all stages in order, stopping if any fails.

Examples:

  # Calibration only, then stop and inspect the report:
  python paper/run_phase_d2.py --stage calibrate --candidate-model gpt-4o-mini

  # If calibration passed, audit at scale:
  python paper/run_phase_d2.py --stage audit --audit-model gpt-4o-mini

  # Once audit JSONL exists, bucket and train:
  python paper/run_phase_d2.py --stage bucket --stage train --stage evaluate --stage report

  # Run the whole flow (cheap path):
  python paper/run_phase_d2.py --stage all \
      --candidate-model gpt-4o-mini --audit-model gpt-4o-mini

Nothing here trains in the background or spawns subprocesses without an
explicit stage call; this script is a workflow controller, not a daemon.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "paper" / "generated_datasets" / "batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl"
EXISTING_AUDIT_DIR = ROOT / "paper" / "faithfulness_audit"
PHASE_DIR = ROOT / "paper" / "experiment_rounds" / f"phase_d2_filtering_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
BUCKETS_DIR = EXISTING_AUDIT_DIR / "buckets"
TABLES_DIR = ROOT / "paper" / "outputs" / "tables"
FIGURES_DIR = ROOT / "paper" / "outputs" / "figures"


# ---------- Stage implementations (skeletons that delegate to existing scripts) ----------


def stage_calibrate(args: argparse.Namespace) -> int:
    """Re-audit the existing 250 rows with the cheap model, report Spearman.

    Wires:
      input  : EXISTING_AUDIT_DIR/faithfulness_audit_gpt-5_2_250_details.csv
               EXISTING_AUDIT_DIR/faithfulness_audit_gpt-5_2_250_llm_responses.jsonl
               (re-uses the same audit prompt as the existing GPT-5.2 run)
      action : submit a Batch job to the cheap candidate model, consume,
               score per row, compare Spearman vs GPT-5.2 scores.
      output : EXISTING_AUDIT_DIR/calibration_<candidate>_vs_gpt-5_2.json
               with fields: spearman_rho, n_rows, recommendation, per-row
               score deltas, accepted (bool).
    """
    model = args.candidate_model
    out_path = EXISTING_AUDIT_DIR / f"calibration_{model.replace('.', '_')}_vs_gpt-5_2.json"
    print(f"[calibrate] candidate={model}")
    print(f"  existing 250-row sample: {EXISTING_AUDIT_DIR}/faithfulness_audit_gpt-5_2_250_details.csv")
    print(f"  will write: {out_path}")
    print()
    print("  TODO: reuse paper/openai_batch_prep.py to build the batch input")
    print("        from the 250-row sample with the same audit prompt schema,")
    print("        submit via paper/submit_faithfulness_audit_batch.py with")
    print(f"        --model {model}, consume via consume_faithfulness_audit_batch.py,")
    print("        compute per-row scores with the same scoring function as")
    print("        the existing audit, and compute Spearman rho over the 250")
    print("        joint scores.")
    print()
    print(f"  Acceptance gate: rho >= {args.min_spearman}")
    print(f"  Skipping execution because this is a scaffold. Run when ready.")
    return 0


def stage_audit(args: argparse.Namespace) -> int:
    """Run the at-scale audit on the full 10K corpus via OpenAI Batch.

    Wires:
      input  : CORPUS_PATH (10K-review JSONL)
      action : submit a single Batch job covering all rows, consume,
               compute per-row scores.
      output : EXISTING_AUDIT_DIR/at_scale_<model>_per_row_scores.csv
               EXISTING_AUDIT_DIR/at_scale_<model>_llm_responses.jsonl
    """
    model = args.audit_model
    out_csv = EXISTING_AUDIT_DIR / f"at_scale_{model.replace('.', '_')}_per_row_scores.csv"
    print(f"[audit] at-scale model={model}")
    print(f"  corpus: {CORPUS_PATH.name}")
    print(f"  will write: {out_csv}")
    print()
    print("  TODO: reuse paper/openai_batch_prep.py / submit_faithfulness_audit_batch.py")
    print(f"        / consume_faithfulness_audit_batch.py with --model {model}")
    print("        over the full corpus, then map each LLM response to a")
    print("        per-row score using the existing scoring function.")
    print()
    print("  Expected cost (gpt-4o-mini Batch, ~20K judgments): ~$5-15")
    print(f"  Skipping execution because this is a scaffold. Run when ready.")
    return 0


def stage_bucket(args: argparse.Namespace) -> int:
    """Assign each corpus row to a bucket by score.

    Output one row-id-per-line file per bucket under
    EXISTING_AUDIT_DIR/buckets/, deterministic from a fixed seed.
    """
    score_csv = args.scores_csv or (EXISTING_AUDIT_DIR / f"at_scale_{args.audit_model.replace('.', '_')}_per_row_scores.csv")
    print(f"[bucket] scores: {score_csv}")
    print(f"  buckets dir: {BUCKETS_DIR}")
    print()
    if not score_csv.exists():
        print(f"  WARNING: scores csv not found yet. Run --stage audit first.")
        print(f"  When run, this stage will:")
    BUCKETS_DIR.mkdir(parents=True, exist_ok=True)
    print("    - load scores, sort, partition into top25 / top50 / full /")
    print("      bot25 / random_5k buckets (random_5k seeded with --seed)")
    print("    - write one row-ids-per-line file per bucket")
    print("    - emit BUCKETS_DIR/manifest.json with sizes, score ranges,")
    print("      and the seed used")
    print()
    print(f"  Skipping execution because this is a scaffold.")
    return 0


def stage_train(args: argparse.Namespace) -> int:
    """Train BERT-base on each bucket using the Phase A recipe."""
    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    runs_dir = PHASE_DIR / "runs"
    runs_dir.mkdir(exist_ok=True)
    print(f"[train] outputs: {runs_dir}")
    print()
    bucket_names = ["top25", "top50", "full", "bot25", "random_5k"]
    for b in bucket_names:
        bucket_file = BUCKETS_DIR / f"{b}.row_ids.txt"
        out_dir = runs_dir / b
        cmd = [
            "python", str(ROOT / "paper" / "absa_model_comparison.py"),
            "--data-path", str(CORPUS_PATH),
            "--row-id-filter", str(bucket_file),     # TODO: confirm absa_model_comparison supports this
            "--approaches", "bert-base-uncased",
            "--epochs-detection", "3",
            "--epochs-sentiment", "3",
            "--seed", str(args.seed),
            "--out-dir", str(out_dir),
        ]
        print(f"  bucket={b}: {' '.join(cmd)}")
    print()
    print("  Expected wall-clock per bucket on RTX 2060: ~25 min")
    print("  Total: ~2 h for the 5 buckets at single seed")
    print(f"  Skipping execution because this is a scaffold.")
    return 0


def stage_evaluate(args: argparse.Namespace) -> int:
    """Evaluate each trained bucket model on internal + Herath."""
    runs_dir = PHASE_DIR / "runs"
    print(f"[evaluate] reading runs from: {runs_dir}")
    print()
    print("  For each bucket's checkpoint:")
    print("    A. compute internal micro-F1 / macro-F1 / sentiment-MSE on")
    print("       the existing 1,000-row held-out test split.")
    print("    B. compute mapped-Herath micro-F1 / macro-F1 / sentiment-MSE")
    print("       by calling paper/evaluate_synthetic_to_real_transfer.py")
    print("       with the bucket checkpoint as the model.")
    print("  Output: PHASE_DIR/evaluation.csv with one row per bucket.")
    print()
    print(f"  Skipping execution because this is a scaffold.")
    return 0


def stage_report(args: argparse.Namespace) -> int:
    """Produce the headline results table and the bar-chart figure."""
    table_path = TABLES_DIR / "phase_d2_filtering_results.csv"
    fig_path = FIGURES_DIR / "phase_d2_filtering_micro_f1.svg"
    print(f"[report] writing:")
    print(f"  table : {table_path}")
    print(f"  figure: {fig_path}")
    print()
    print("  Table columns: bucket, n_train, internal_micro_f1,")
    print("                 herath_micro_f1, herath_sentiment_mse")
    print("  Figure: bar chart of herath_micro_f1 by bucket with the")
    print("          'full' value drawn as a horizontal reference line.")
    print()
    print("  Compute the headline deltas:")
    print("    Delta(top50, full)      : value of filtering at half the corpus")
    print("    Delta(top25, full)      : value at quarter corpus")
    print("    Delta(random_5k, top50) : value of the quality signal vs size")
    print("    Delta(bot25, full)      : sanity-check that bottom rows are bad")
    print()
    print(f"  Skipping execution because this is a scaffold.")
    return 0


# ---------- Stage routing ----------


STAGES = {
    "calibrate": stage_calibrate,
    "audit":     stage_audit,
    "bucket":    stage_bucket,
    "train":     stage_train,
    "evaluate":  stage_evaluate,
    "report":    stage_report,
}

ALL_ORDER = ["calibrate", "audit", "bucket", "train", "evaluate", "report"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", action="append", default=[],
                   choices=list(STAGES.keys()) + ["all"],
                   help="Stage(s) to run. Repeatable. Use 'all' to run every stage in order.")
    p.add_argument("--candidate-model", default="gpt-4o-mini",
                   help="Cheap model for the calibration step (default: gpt-4o-mini).")
    p.add_argument("--audit-model", default="gpt-4o-mini",
                   help="Model for the at-scale audit (default: gpt-4o-mini, used if calibration passed).")
    p.add_argument("--min-spearman", type=float, default=0.6,
                   help="Acceptance threshold for the cheap-model calibration (default: 0.6).")
    p.add_argument("--scores-csv", type=Path, default=None,
                   help="Override path to the at-scale per-row scores CSV (skips audit step lookup).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for random_5k bucket and training (default: 42).")
    args = p.parse_args()

    if not args.stage:
        print("No --stage specified. Use --stage all or one of:",
              ", ".join(STAGES.keys()))
        return 2

    stages_to_run = []
    for s in args.stage:
        if s == "all":
            stages_to_run.extend(ALL_ORDER)
        else:
            stages_to_run.append(s)
    # Preserve order, dedupe.
    seen, ordered = set(), []
    for s in stages_to_run:
        if s not in seen:
            seen.add(s)
            ordered.append(s)

    print(f"Phase D2 orchestrator")
    print(f"  stages: {ordered}")
    print(f"  candidate model: {args.candidate_model}")
    print(f"  audit model:     {args.audit_model}")
    print(f"  seed:            {args.seed}")
    print()

    for s in ordered:
        rc = STAGES[s](args)
        if rc != 0:
            print(f"\n[stop] stage '{s}' returned exit code {rc}")
            return rc

    print(f"\n[done] {len(ordered)} stage(s) completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
