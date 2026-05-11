from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

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


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "real_transfer"
RUNS_DIR = OUT_DIR / "runs"
DEFAULT_SYNTHETIC_PATH = ROOT / "paper" / "generated_datasets" / "batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl"
DEFAULT_HERATH_ROOT = ROOT / "external_data" / "Student_feedback_analysis_dataset" / "Annotated Student Feedback Data"
NS = {
    "cas": "http:///uima/cas.ecore",
    "custom": "http:///webanno/custom.ecore",
}

# Conservative mapping: only keep categories that have a defensible correspondence to the paper's 20-aspect schema.
HERATH_TO_PROJECT = {
    "Lecturer#X_1": "lecturer_quality",
    "Lecturer#X_2": "lecturer_quality",
    "Lecturer#X_3": "lecturer_quality",
    "Lecturer#X_x": "lecturer_quality",
    "Lecturer#X_1_Impli": "lecturer_quality",
    "Lecturer#X_2_Impli": "lecturer_quality",
    "Lecturer#X_3_Impli": "lecturer_quality",
    "Lecturer#X_x_Impli": "lecturer_quality",
    "Course#X_x": "overall_experience",
    "Course#X_x_Impli": "overall_experience",
    "Course_Structure#X_1": "organization",
    "Course_Structure#X_1_Impli": "organization",
    "Course_Structure#X_2": "workload",
    "Course_Structure#X_2_Impli": "workload",
    "Course_Structure#X_5": "grading_transparency",
    "Course_Structure#X_5_Impli": "grading_transparency",
    "Subject_Material#X_1": "materials",
    "Subject_Material#X_1_Impli": "materials",
    "Subject_Material#X_x": "materials",
    "Subject_Material#X_1a": "materials",
    "End_Exam#X_x": "exam_fairness",
    "End_Exam#X_x_Impli": "exam_fairness",
    "End_Exam#X_x2": "exam_fairness",
    "CA#X_x": "assessment_design",
    "CA#X_x_Impli": "assessment_design",
    "Learning_Environment#X_x": "accessibility",
    "Learning_Environment#X_x_Impli": "accessibility",
}
POLARITY_MAP = {
    "Positive": "positive",
    "Negative": "negative",
    "Neutral#Sug": "neutral",
    "Neutral#NSug": "neutral",
}


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def make_run_dir(prefix: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = RUNS_DIR / f"{prefix}_{timestamp}"
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = RUNS_DIR / f"{prefix}_{timestamp}_{suffix}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def normalize_real_text(text: str) -> str:
    return " ".join(str(text).split())


def extract_review_spans(root_el: ET.Element) -> List[Dict[str, object]]:
    spans: List[Dict[str, object]] = []
    for el in root_el.findall("custom:Document_levelopinion", NS):
        begin = el.attrib.get("begin")
        end = el.attrib.get("end")
        if begin is None or end is None:
            continue
        spans.append(
            {
                "begin": int(begin),
                "end": int(end),
                "doc_sent": el.attrib.get("Document_levelopinion", ""),
            }
        )
    spans.sort(key=lambda item: (item["begin"], item["end"]))
    return spans


def extract_aspect_annotations(root_el: ET.Element) -> List[Dict[str, object]]:
    annotations: List[Dict[str, object]] = []
    for el in root_el:
        tag = el.tag.split("}")[-1]
        if not tag.startswith("Aspect"):
            continue
        begin = el.attrib.get("begin")
        end = el.attrib.get("end")
        if begin is None or end is None:
            continue
        annotations.append(
            {
                "begin": int(begin),
                "end": int(end),
                "aspect_raw": el.attrib.get("Aspect", ""),
                "opinion_raw": el.attrib.get("Opinion", ""),
            }
        )
    annotations.sort(key=lambda item: (item["begin"], item["end"]))
    return annotations


def pair_review_aspects(annotations: List[Dict[str, object]]) -> List[Tuple[str, str]]:
    # Some annotations carry both Aspect and Opinion directly. Others separate them into neighboring spans.
    opinion_only = [
        (idx, ann)
        for idx, ann in enumerate(annotations)
        if ann["opinion_raw"] in POLARITY_MAP and not ann["aspect_raw"]
    ]
    used_opinions = set()
    pairs: List[Tuple[str, str]] = []
    for ann in annotations:
        raw_aspect = str(ann["aspect_raw"]).strip()
        raw_opinion = str(ann["opinion_raw"]).strip()
        if raw_aspect and raw_opinion in POLARITY_MAP:
            pairs.append((raw_aspect, POLARITY_MAP[raw_opinion]))
            continue
        if not raw_aspect:
            continue
        best_idx = None
        best_dist = 10**9
        for idx, opinion_ann in opinion_only:
            if idx in used_opinions:
                continue
            dist = min(abs(ann["begin"] - opinion_ann["begin"]), abs(ann["end"] - opinion_ann["end"]))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is not None and best_dist <= 80:
            used_opinions.add(best_idx)
            opinion_raw = str(annotations[best_idx]["opinion_raw"]).strip()
            if opinion_raw in POLARITY_MAP:
                pairs.append((raw_aspect, POLARITY_MAP[opinion_raw]))
    return pairs


def collapse_review_labels(pairs: List[Tuple[str, str]]) -> Dict[str, str]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for raw_aspect, sentiment in pairs:
        mapped = HERATH_TO_PROJECT.get(raw_aspect)
        if mapped:
            grouped[mapped].append(sentiment)
    collapsed: Dict[str, str] = {}
    for aspect, sentiments in grouped.items():
        counts = Counter(sentiments).most_common()
        if len(counts) > 1 and counts[0][1] == counts[1][1]:
            collapsed[aspect] = "neutral"
        else:
            collapsed[aspect] = counts[0][0]
    return collapsed


def load_herath_mapped_dataset(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for xmi_path in sorted(root.rglob("*.xmi")):
        tree = ET.parse(xmi_path)
        root_el = tree.getroot()
        sofa = root_el.find("cas:Sofa", NS)
        text = sofa.attrib.get("sofaString", "") if sofa is not None else ""
        spans = extract_review_spans(root_el)
        annotations = extract_aspect_annotations(root_el)
        for span in spans:
            begin, end = int(span["begin"]), int(span["end"])
            review_text = normalize_real_text(text[begin:end])
            if not review_text:
                continue
            review_annotations = [ann for ann in annotations if ann["begin"] >= begin and ann["end"] <= end]
            labels = collapse_review_labels(pair_review_aspects(review_annotations))
            if not labels:
                continue
            rows.append(
                {
                    "text": review_text,
                    "aspects": labels,
                    "target_attributes": labels,
                    "nuance_attributes": {},
                    "course_name": "",
                    "grade": "",
                    "style": "",
                    "source_path": str(xmi_path),
                    "doc_sent": str(span["doc_sent"]),
                }
            )
    if not rows:
        raise ValueError(f"No mappable rows found under {root}")
    return pd.DataFrame(rows)


def restrict_to_overlap(df: pd.DataFrame, overlap_aspects: List[str]) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        labels = {aspect: sentiment for aspect, sentiment in row["aspects"].items() if aspect in overlap_aspects}
        if not labels:
            continue
        updated = dict(row)
        updated["aspects"] = labels
        updated["target_attributes"] = labels
        rows.append(updated)
    return pd.DataFrame(rows).reset_index(drop=True)


def summarize_overlap(real_df: pd.DataFrame) -> pd.DataFrame:
    counts = Counter()
    sentiments = Counter()
    for labels in real_df["aspects"]:
        for aspect, sentiment in labels.items():
            counts[aspect] += 1
            sentiments[(aspect, sentiment)] += 1
    rows = []
    for aspect in sorted(counts):
        rows.append(
            {
                "aspect": aspect,
                "review_count": counts[aspect],
                "positive": sentiments[(aspect, "positive")],
                "neutral": sentiments[(aspect, "neutral")],
                "negative": sentiments[(aspect, "negative")],
            }
        )
    return pd.DataFrame(rows)


def run_transfer(
    synthetic_df: pd.DataFrame,
    real_df: pd.DataFrame,
    approaches: List[str],
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aspects = sorted({aspect for labels in real_df["aspects"] for aspect in labels.keys()})
    synthetic_df = restrict_to_overlap(synthetic_df, aspects)
    synth_train, synth_calib, _ = three_way_split(synthetic_df, cfg.split_calib, cfg.split_test, cfg.seed)
    if synth_calib.empty:
        raise ValueError("Synthetic calibration split is empty after overlap restriction.")

    summary_rows = []
    per_aspect_frames = []
    for approach in approaches:
        if approach == "tfidf_two_step":
            per_aspect_df, summary = run_tfidf_two_step_approach(approach, synth_train, synth_calib, real_df, aspects)
        else:
            det_model, det_tokenizer = train_detection(approach, synth_train, synth_calib, aspects, cfg)
            sent_model, sent_tokenizer = train_sentiment(approach, synth_train, synth_calib, aspects, cfg)
            thresholds = calibrate_thresholds(det_model, synth_calib, det_tokenizer, aspects, cfg)
            per_aspect_df, summary = evaluate_models(
                approach,
                det_model,
                sent_model,
                real_df,
                det_tokenizer,
                sent_tokenizer,
                aspects,
                thresholds,
                cfg,
            )
        summary["eval_split"] = "real_herath_mapped"
        summary["n_real_reviews"] = int(len(real_df))
        summary["n_overlap_aspects"] = int(len(aspects))
        summary_rows.append(summary)
        per_aspect_frames.append(per_aspect_df)

    return pd.DataFrame(summary_rows).sort_values("micro_f1", ascending=False).reset_index(drop=True), pd.concat(
        per_aspect_frames, ignore_index=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate synthetic-trained ABSA models on mapped real student-feedback data.")
    parser.add_argument("--synthetic-path", default=str(DEFAULT_SYNTHETIC_PATH))
    parser.add_argument("--herath-root", default=str(DEFAULT_HERATH_ROOT))
    parser.add_argument(
        "--approaches",
        nargs="+",
        default=["tfidf_two_step", "distilbert-base-uncased", "bert-base-uncased"],
        help="Models to train on synthetic data and evaluate on mapped real reviews.",
    )
    parser.add_argument("--epochs-detection", type=int, default=3)
    parser.add_argument("--epochs-sentiment", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-file", default="", help="Optional detailed experiment log path.")
    parser.add_argument("--resume-file", default="", help="Optional resume checkpoint path.")
    parser.add_argument("--write-latest", dest="write_latest", action="store_true", help="Write paper-facing latest transfer outputs.")
    parser.add_argument("--no-write-latest", dest="write_latest", action="store_false", help="Archive the run bundle only without overwriting paper-facing latest transfer outputs.")
    parser.set_defaults(write_latest=None)
    args = parser.parse_args()

    ensure_dirs()
    log_path = configure_logger(args.log_file or default_log_path("real_transfer"))
    cfg = Config(
        max_len=args.max_len,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs_detection=args.epochs_detection,
        epochs_sentiment=args.epochs_sentiment,
        seed=args.seed,
    )
    set_seed(cfg.seed)
    resume_path = Path(args.resume_file or __import__("os").environ.get("EXPERIMENT_RESUME_FILE") or (OUT_DIR / "resume_real_transfer.json")).resolve()
    signature = {
        "script": "evaluate_synthetic_to_real_transfer.py",
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
    overlap_table = summarize_overlap(real_df)

    real_jsonl = OUT_DIR / "herath_mapped_real_reviews.jsonl"
    with real_jsonl.open("w", encoding="utf-8") as handle:
        for record in real_df.to_dict(orient="records"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    overlap_table.to_csv(OUT_DIR / "herath_overlap_summary.csv", index=False)
    (OUT_DIR / "herath_mapping.json").write_text(
        json.dumps(
            {
                "source": str(Path(args.herath_root)),
                "mapping": HERATH_TO_PROJECT,
                "polarity_map": POLARITY_MAP,
                "n_reviews": int(len(real_df)),
                "overlap_aspects": sorted(overlap_table["aspect"].tolist()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    aspects = sorted({aspect for labels in real_df["aspects"] for aspect in labels.keys()})
    synthetic_overlap = restrict_to_overlap(synthetic_df, aspects)
    synth_train, synth_calib, _ = three_way_split(synthetic_overlap, cfg.split_calib, cfg.split_test, cfg.seed)
    summary_rows = []
    per_aspect_frames = []
    artifact_payloads = {}
    for approach in args.approaches:
        checkpoint_key = f"approach::{approach}"
        completed_entry = resume_state["completed"].get(checkpoint_key, {})
        if completed_entry:
            summary_rows.append(completed_entry["summary"])
            per_aspect_frames.append(pd.read_csv(completed_entry["per_aspect_path"]))
            artifact_paths = completed_entry.get("artifact_paths", {})
            if artifact_paths:
                artifact_payloads[approach] = load_approach_artifact_checkpoint(artifact_paths)
            log_event(f"[resume] restored {approach} transfer evaluation from checkpoint")
            continue
        log_event(f"[{approach}] transfer evaluation start")
        if approach == "tfidf_two_step":
            per_aspect_df, summary, artifact_payload = run_tfidf_two_step_approach(
                approach, synth_train, synth_calib, real_df, aspects, return_artifacts=True
            )
        else:
            det_model, det_tokenizer = train_detection(approach, synth_train, synth_calib, aspects, cfg)
            sent_model, sent_tokenizer = train_sentiment(approach, synth_train, synth_calib, aspects, cfg)
            thresholds = calibrate_thresholds(det_model, synth_calib, det_tokenizer, aspects, cfg)
            per_aspect_df, summary, artifact_payload = evaluate_models(
                approach,
                det_model,
                sent_model,
                real_df,
                det_tokenizer,
                sent_tokenizer,
                aspects,
                thresholds,
                cfg,
                return_artifacts=True,
            )
        summary["eval_split"] = "real_herath_mapped"
        summary["n_real_reviews"] = int(len(real_df))
        summary["n_overlap_aspects"] = int(len(aspects))
        summary_rows.append(summary)
        per_aspect_frames.append(per_aspect_df)
        artifact_payloads[approach] = artifact_payload
        checkpoint_csv = resume_path.parent / f"{resume_path.stem}.{approach.replace('/', '__')}.per_aspect.csv"
        per_aspect_df.to_csv(checkpoint_csv, index=False)
        artifact_paths = save_approach_artifact_checkpoint(
            resume_path.parent,
            f"{resume_path.stem}.{approach.replace('/', '__')}",
            artifact_payload,
        )
        resume_state["completed"][checkpoint_key] = {
            "summary": summary,
            "per_aspect_path": str(checkpoint_csv),
            "artifact_paths": artifact_paths,
            "completed_at_utc": utc_now(),
        }
        resume_state["updated_at_utc"] = utc_now()
        write_resume_state(resume_path, resume_state)
        log_event(f"[{approach}] transfer evaluation complete: micro_f1={summary['micro_f1']:.4f} sentiment_mse={summary['sentiment_mse_detected']:.4f}")
    summary_df = pd.DataFrame(summary_rows).sort_values("micro_f1", ascending=False).reset_index(drop=True)
    per_aspect_df = pd.concat(per_aspect_frames, ignore_index=True)
    write_latest = args.write_latest if args.write_latest is not None else (len(args.approaches) > 1)
    summary_path = OUT_DIR / "synthetic_to_real_transfer_summary.csv"
    per_aspect_path = OUT_DIR / "synthetic_to_real_transfer_per_aspect.csv"
    if write_latest:
        summary_df.to_csv(summary_path, index=False)
        per_aspect_df.to_csv(per_aspect_path, index=False)
    run_dir = make_run_dir("synthetic_to_real_transfer")
    run_summary_path = run_dir / "summary.csv"
    run_per_aspect_path = run_dir / "per_aspect.csv"
    run_metadata_path = run_dir / "metadata.json"
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(run_summary_path, index=False)
    per_aspect_df.to_csv(run_per_aspect_path, index=False)
    for approach, payload in artifact_payloads.items():
        save_approach_artifact_checkpoint(artifact_dir, approach.replace("/", "__"), payload)
    run_metadata_path.write_text(
        json.dumps(
            {
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
                "n_real_reviews": int(len(real_df)),
                "n_overlap_aspects": int(len(overlap_table)),
                "artifact_policy": {
                    "sample_predictions": True,
                    "detection_probabilities": True,
                    "detection_logits": True,
                    "sentiment_prediction_values": True,
                    "checkpoint_artifact_mirror": True,
                },
                "files": {
                    "summary": str(run_summary_path),
                    "per_aspect": str(run_per_aspect_path),
                    "artifact_dir": str(artifact_dir),
                    "latest_summary": str(summary_path) if write_latest else "",
                    "latest_per_aspect": str(per_aspect_path) if write_latest else "",
                },
                "write_latest": write_latest,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "real_jsonl": str(real_jsonl),
                "overlap_summary": str(OUT_DIR / "herath_overlap_summary.csv"),
                "summary": str(summary_path),
                "per_aspect": str(per_aspect_path),
                "run_dir": str(run_dir),
                "n_real_reviews": int(len(real_df)),
                "overlap_aspects": sorted(overlap_table["aspect"].tolist()),
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
