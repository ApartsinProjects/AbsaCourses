from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from absa_model_comparison import (
    Config,
    calibrate_thresholds,
    close_logger,
    configure_logger,
    default_log_path,
    evaluate_models,
    load_approach_artifact_checkpoint,
    load_jsonl,
    load_resume_state,
    log_event,
    resolve_data_path,
    run_tfidf_two_step_approach,
    save_approach_artifact_checkpoint,
    set_seed,
    three_way_split,
    train_detection,
    train_sentiment,
    utc_now,
    write_resume_state,
)
from evaluate_synthetic_to_real_transfer import (
    DEFAULT_HERATH_ROOT,
    DEFAULT_SYNTHETIC_PATH,
    ensure_dirs as ensure_real_transfer_dirs,
    load_herath_mapped_dataset,
    restrict_to_overlap,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "real_transfer"
RUNS_DIR = OUT_DIR / "runs"


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_real_transfer_dirs()


def make_run_dir(prefix: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = RUNS_DIR / f"{prefix}_{timestamp}"
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = RUNS_DIR / f"{prefix}_{timestamp}_{suffix}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare overlap-matched internal synthetic performance with external real-transfer performance.")
    parser.add_argument("--synthetic-path", default=str(DEFAULT_SYNTHETIC_PATH))
    parser.add_argument("--herath-root", default=str(DEFAULT_HERATH_ROOT))
    parser.add_argument(
        "--approaches",
        nargs="+",
        default=["tfidf_two_step", "distilbert-base-uncased", "bert-base-uncased"],
    )
    parser.add_argument("--epochs-detection", type=int, default=3)
    parser.add_argument("--epochs-sentiment", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-file", default="", help="Optional detailed experiment log path.")
    parser.add_argument("--resume-file", default="", help="Optional resume checkpoint path.")
    parser.add_argument("--write-latest", dest="write_latest", action="store_true", help="Write paper-facing latest overlap outputs.")
    parser.add_argument("--no-write-latest", dest="write_latest", action="store_false", help="Archive the run bundle only without overwriting paper-facing latest overlap outputs.")
    parser.set_defaults(write_latest=None)
    args = parser.parse_args()

    ensure_dirs()
    log_path = configure_logger(args.log_file or default_log_path("overlap_generalization"))
    cfg = Config(
        max_len=args.max_len,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs_detection=args.epochs_detection,
        epochs_sentiment=args.epochs_sentiment,
        seed=args.seed,
    )
    set_seed(cfg.seed)
    resume_path = Path(args.resume_file or __import__("os").environ.get("EXPERIMENT_RESUME_FILE") or (OUT_DIR / "resume_overlap_generalization.json")).resolve()
    signature = {
        "script": "evaluate_overlap_generalization.py",
        "synthetic_path": str(resolve_data_path(args.synthetic_path)),
        "herath_root": str(Path(args.herath_root).resolve()),
        "approaches": list(args.approaches),
        "epochs_detection": int(args.epochs_detection),
        "epochs_sentiment": int(args.epochs_sentiment),
        "batch_size": int(args.batch_size),
        "max_len": int(args.max_len),
        "lr": float(args.lr),
        "seed": int(args.seed),
    }
    resume_state = load_resume_state(resume_path)
    if resume_state and resume_state.get("signature") != signature:
        log_event(f"Resume signature mismatch at {resume_path}; starting a fresh checkpoint")
        resume_state = {}
    if not resume_state:
        resume_state = {"signature": signature, "created_at_utc": utc_now(), "updated_at_utc": utc_now(), "completed": {}}
        write_resume_state(resume_path, resume_state)
    log_event(f"Detailed log file -> {log_path}")
    log_event(f"Resume checkpoint -> {resume_path}")

    synthetic_df = load_jsonl(resolve_data_path(args.synthetic_path))
    real_df = load_herath_mapped_dataset(Path(args.herath_root))
    aspects = sorted({aspect for labels in real_df["aspects"] for aspect in labels.keys()})
    synth_overlap = restrict_to_overlap(synthetic_df, aspects)
    synth_train, synth_val, synth_test = three_way_split(synth_overlap, cfg.split_calib, cfg.split_test, cfg.seed)

    rows = []
    per_aspect_frames = []
    artifact_payloads = {}

    for approach in args.approaches:
        checkpoint_key = f"approach::{approach}"
        completed_entry = resume_state["completed"].get(checkpoint_key, {})
        if completed_entry:
            summary_rows = completed_entry["summary_rows"]
            rows.extend(summary_rows)
            per_aspect_frames.append(pd.read_csv(completed_entry["synthetic_per_aspect_path"]))
            per_aspect_frames.append(pd.read_csv(completed_entry["real_per_aspect_path"]))
            artifact_paths = completed_entry.get("artifact_paths", {})
            if artifact_paths:
                artifact_payloads[approach] = load_approach_artifact_checkpoint(artifact_paths)
            log_event(f"[resume] restored {approach} overlap comparison from checkpoint")
            continue
        log_event(f"[{approach}] overlap comparison start")
        if approach == "tfidf_two_step":
            synth_per_aspect, synth_summary, synth_artifacts = run_tfidf_two_step_approach(
                approach, synth_train, synth_val, synth_test, aspects, return_artifacts=True
            )
            real_per_aspect, real_summary, real_artifacts = run_tfidf_two_step_approach(
                approach, synth_train, synth_val, real_df, aspects, return_artifacts=True
            )
        else:
            det_model, det_tokenizer = train_detection(approach, synth_train, synth_val, aspects, cfg)
            sent_model, sent_tokenizer = train_sentiment(approach, synth_train, synth_val, aspects, cfg)
            thresholds = calibrate_thresholds(det_model, synth_val, det_tokenizer, aspects, cfg)
            synth_per_aspect, synth_summary, synth_artifacts = evaluate_models(
                approach, det_model, sent_model, synth_test, det_tokenizer, sent_tokenizer, aspects, thresholds, cfg, return_artifacts=True
            )
            real_per_aspect, real_summary, real_artifacts = evaluate_models(
                approach, det_model, sent_model, real_df, det_tokenizer, sent_tokenizer, aspects, thresholds, cfg, return_artifacts=True
            )

        synth_summary["eval_split"] = "synthetic_overlap_test"
        synth_summary["n_reviews"] = int(len(synth_test))
        synth_summary["n_aspects"] = int(len(aspects))
        real_summary["eval_split"] = "real_herath_mapped"
        real_summary["n_reviews"] = int(len(real_df))
        real_summary["n_aspects"] = int(len(aspects))
        synth_summary["generalization_gap_f1"] = float(real_summary["micro_f1"] - synth_summary["micro_f1"])
        real_summary["generalization_gap_f1"] = float(real_summary["micro_f1"] - synth_summary["micro_f1"])
        rows.extend([synth_summary, real_summary])

        synth_per_aspect = synth_per_aspect.copy()
        synth_per_aspect["eval_split"] = "synthetic_overlap_test"
        real_per_aspect = real_per_aspect.copy()
        real_per_aspect["eval_split"] = "real_herath_mapped"
        per_aspect_frames.extend([synth_per_aspect, real_per_aspect])
        artifact_payloads[approach] = {
            "synthetic_sample_predictions": synth_artifacts.get("sample_predictions", []),
            "synthetic_thresholds": synth_artifacts.get("thresholds", {}),
            "real_sample_predictions": real_artifacts.get("sample_predictions", []),
            "real_thresholds": real_artifacts.get("thresholds", {}),
        }
        synth_csv = resume_path.parent / f"{resume_path.stem}.{approach.replace('/', '__')}.synthetic_per_aspect.csv"
        real_csv = resume_path.parent / f"{resume_path.stem}.{approach.replace('/', '__')}.real_per_aspect.csv"
        synth_per_aspect.to_csv(synth_csv, index=False)
        real_per_aspect.to_csv(real_csv, index=False)
        artifact_paths = save_approach_artifact_checkpoint(
            resume_path.parent,
            f"{resume_path.stem}.{approach.replace('/', '__')}",
            artifact_payloads[approach],
        )
        resume_state["completed"][checkpoint_key] = {
            "summary_rows": [synth_summary, real_summary],
            "synthetic_per_aspect_path": str(synth_csv),
            "real_per_aspect_path": str(real_csv),
            "artifact_paths": artifact_paths,
            "completed_at_utc": utc_now(),
        }
        resume_state["updated_at_utc"] = utc_now()
        write_resume_state(resume_path, resume_state)
        log_event(f"[{approach}] overlap comparison complete: synthetic_micro_f1={synth_summary['micro_f1']:.4f} real_micro_f1={real_summary['micro_f1']:.4f}")

    summary_df = pd.DataFrame(rows)
    latest_summary_path = OUT_DIR / "overlap_internal_vs_external_summary.csv"
    latest_per_aspect_path = OUT_DIR / "overlap_internal_vs_external_per_aspect.csv"
    latest_metadata_path = OUT_DIR / "overlap_internal_vs_external_metadata.json"
    write_latest = args.write_latest if args.write_latest is not None else (len(args.approaches) > 1)
    if write_latest:
        summary_df.to_csv(latest_summary_path, index=False)
    per_aspect_df = pd.concat(per_aspect_frames, ignore_index=True)
    if write_latest:
        per_aspect_df.to_csv(latest_per_aspect_path, index=False)
    run_dir = make_run_dir("overlap_internal_vs_external")
    run_summary_path = run_dir / "summary.csv"
    run_per_aspect_path = run_dir / "per_aspect.csv"
    run_metadata_path = run_dir / "metadata.json"
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_path": str(resolve_data_path(args.synthetic_path)),
        "herath_root": str(Path(args.herath_root).resolve()),
        "approaches": args.approaches,
        "epochs_detection": args.epochs_detection,
        "epochs_sentiment": args.epochs_sentiment,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "lr": args.lr,
        "seed": args.seed,
        "aspects": aspects,
        "n_synthetic_overlap": int(len(synth_overlap)),
        "n_synth_train": int(len(synth_train)),
        "n_synth_val": int(len(synth_val)),
        "n_synth_test": int(len(synth_test)),
        "n_real": int(len(real_df)),
        "artifact_policy": {
            "synthetic_sample_predictions": True,
            "real_sample_predictions": True,
            "detection_probabilities": True,
            "detection_logits": True,
            "sentiment_prediction_values": True,
            "checkpoint_artifact_mirror": True,
        },
        "files": {
            "summary": str(run_dir / "summary.csv"),
            "per_aspect": str(run_dir / "per_aspect.csv"),
            "artifact_dir": str(run_dir / "artifacts"),
        },
    }
    if write_latest:
        latest_metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
            ),
            encoding="utf-8",
        )
    summary_df.to_csv(run_summary_path, index=False)
    per_aspect_df.to_csv(run_per_aspect_path, index=False)
    for approach, payload in artifact_payloads.items():
        save_approach_artifact_checkpoint(artifact_dir, approach.replace("/", "__"), payload)
    run_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(latest_summary_path) if write_latest else "",
                "per_aspect": str(latest_per_aspect_path) if write_latest else "",
                "run_dir": str(run_dir),
                "artifact_dir": str(artifact_dir),
            },
            indent=2,
        )
    )
    resume_state["final_run_dir"] = str(run_dir)
    resume_state["finalized_at_utc"] = utc_now()
    write_resume_state(resume_path, resume_state)
    close_logger()


if __name__ == "__main__":
    main()
