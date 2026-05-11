from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROUND_DIR = ROOT / "paper" / "experiment_rounds"
DEFAULT_OUTPUT = ROUND_DIR / "next_round_individual_20260403_plan.json"
PRIMARY_SYNTHETIC_PATH = ROOT / "paper" / "generated_datasets" / "batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl"
HERATH_ROOT = ROOT / "external_data" / "Student_feedback_analysis_dataset" / "Annotated Student Feedback Data"


def batch(
    *,
    batch_id: str,
    title: str,
    purpose: str,
    script: str,
    args: list[str],
    ordinal: int,
) -> dict[str, object]:
    run_dir = ROUND_DIR / "next_round_individual_20260403"
    return {
        "id": batch_id,
        "title": title,
        "purpose": purpose,
        "workdir": str(ROOT),
        "log_path": str(run_dir / "logs" / f"{ordinal:02d}_{batch_id}.log"),
        "command": ["python", str(ROOT / "paper" / script), *args],
    }


def build_plan() -> dict[str, object]:
    batches: list[dict[str, object]] = []
    i = 1

    local_models = [
        ("tfidf_two_step", "TF-IDF local benchmark"),
        ("distilbert-base-uncased", "DistilBERT local benchmark"),
        ("bert-base-uncased", "BERT local benchmark"),
        ("bert_joint", "BERT joint local benchmark"),
        ("distilbert_joint", "DistilBERT joint local benchmark"),
        ("roberta-base", "RoBERTa local benchmark"),
        ("albert-base-v2", "ALBERT local benchmark"),
    ]
    for model, title in local_models:
        batches.append(
            batch(
                batch_id=f"local_{model.replace('/', '__').replace('-', '_')}",
                title=title,
                purpose=f"Run the synthetic-benchmark local evaluation for {model} as an individual task.",
                script="absa_model_comparison.py",
                args=[
                    "--data-path",
                    str(PRIMARY_SYNTHETIC_PATH),
                    "--approaches",
                    model,
                    "--no-write-latest",
                    "--epochs-detection",
                    "3",
                    "--epochs-sentiment",
                    "3",
                    "--seed",
                    "42",
                ],
                ordinal=i,
            )
        )
        i += 1

    transfer_models = [
        ("tfidf_two_step", "TF-IDF mapped real transfer"),
        ("distilbert-base-uncased", "DistilBERT mapped real transfer"),
        ("bert-base-uncased", "BERT mapped real transfer"),
    ]
    for model, title in transfer_models:
        batches.append(
            batch(
                batch_id=f"real_transfer_{model.replace('/', '__').replace('-', '_')}",
                title=title,
                purpose=f"Run the mapped Herath transfer evaluation for {model} as an individual task.",
                script="evaluate_synthetic_to_real_transfer.py",
                args=[
                    "--synthetic-path",
                    str(PRIMARY_SYNTHETIC_PATH),
                    "--herath-root",
                    str(HERATH_ROOT),
                    "--approaches",
                    model,
                    "--no-write-latest",
                    "--epochs-detection",
                    "3",
                    "--epochs-sentiment",
                    "3",
                    "--seed",
                    "42",
                ],
                ordinal=i,
            )
        )
        i += 1

    overlap_models = [
        ("tfidf_two_step", "TF-IDF overlap comparison"),
        ("distilbert-base-uncased", "DistilBERT overlap comparison"),
        ("bert-base-uncased", "BERT overlap comparison"),
    ]
    for model, title in overlap_models:
        batches.append(
            batch(
                batch_id=f"overlap_{model.replace('/', '__').replace('-', '_')}",
                title=title,
                purpose=f"Run the overlap-matched synthetic-versus-real comparison for {model} as an individual task.",
                script="evaluate_overlap_generalization.py",
                args=[
                    "--synthetic-path",
                    str(PRIMARY_SYNTHETIC_PATH),
                    "--herath-root",
                    str(HERATH_ROOT),
                    "--approaches",
                    model,
                    "--no-write-latest",
                    "--epochs-detection",
                    "3",
                    "--epochs-sentiment",
                    "3",
                    "--seed",
                    "42",
                ],
                ordinal=i,
            )
        )
        i += 1

    return {
        "round_id": "next_round_individual_20260403",
        "title": "Per-model benchmark round",
        "purpose": "Run benchmark, transfer, and overlap evaluations as individual model-level tasks so progress, resume behavior, and reruns remain granular.",
        "stop_on_failure": True,
        "batches": batches,
    }


def main() -> None:
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plan = build_plan()
    DEFAULT_OUTPUT.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"plan_path": str(DEFAULT_OUTPUT), "batch_count": len(plan["batches"])}, indent=2))


if __name__ == "__main__":
    main()
