from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROUND_DIR = ROOT / "paper" / "experiment_rounds"
RUN_ID = "phase_a_benchmark_20260403"
DEFAULT_OUTPUT = ROUND_DIR / f"{RUN_ID}_plan.json"
PRIMARY_SYNTHETIC_PATH = ROOT / "paper" / "generated_datasets" / "batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl"


def batch(*, batch_id: str, title: str, purpose: str, args: list[str], ordinal: int) -> dict[str, object]:
    run_dir = ROUND_DIR / RUN_ID
    return {
        "id": batch_id,
        "title": title,
        "purpose": purpose,
        "workdir": str(ROOT),
        "log_path": str(run_dir / "logs" / f"{ordinal:02d}_{batch_id}.log"),
        "command": ["python", str(ROOT / "paper" / "absa_model_comparison.py"), *args],
    }


def build_plan() -> dict[str, object]:
    models = [
        ("tfidf_two_step", "TF-IDF calibrated rerun"),
        ("distilbert-base-uncased", "DistilBERT calibrated rerun"),
        ("bert-base-uncased", "BERT calibrated rerun"),
        ("bert_joint", "BERT joint calibrated rerun"),
        ("distilbert_joint", "DistilBERT joint calibrated rerun"),
        ("roberta-base", "RoBERTa calibrated rerun"),
        ("albert-base-v2", "ALBERT calibrated rerun"),
    ]
    batches: list[dict[str, object]] = []
    for ordinal, (model, title) in enumerate(models, start=1):
        batches.append(
            batch(
                batch_id=f"phase_a_{model.replace('/', '__').replace('-', '_')}",
                title=title,
                purpose=f"Run the Phase A trust-repair benchmark rerun for {model} on the 10K / 20-aspect corpus.",
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
                ordinal=ordinal,
            )
        )
    return {
        "round_id": RUN_ID,
        "title": "Phase A calibrated local benchmark rerun",
        "purpose": "Run the seven local benchmark approaches as individual resumable tasks to refresh the main benchmark evidence under the corrected evaluation stack.",
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
