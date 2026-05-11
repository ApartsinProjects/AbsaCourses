from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROUND_DIR = ROOT / "paper" / "experiment_rounds"
RUN_ID = "phase_b1_multiseed_20260404"
DEFAULT_OUTPUT = ROUND_DIR / f"{RUN_ID}_plan.json"
PRIMARY_SYNTHETIC_PATH = ROOT / "paper" / "generated_datasets" / "batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl"

SEEDS = [3, 13, 23]
MODELS = [
    ("tfidf_two_step", "TF-IDF"),
    ("distilbert-base-uncased", "DistilBERT"),
    ("bert-base-uncased", "BERT"),
    ("bert_joint", "BERT joint"),
    ("distilbert_joint", "DistilBERT joint"),
]


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
    batches: list[dict[str, object]] = []
    ordinal = 1
    for model, label in MODELS:
        for seed in SEEDS:
            batches.append(
                batch(
                    batch_id=f"multiseed_{model.replace('/', '__').replace('-', '_')}_seed_{seed}",
                    title=f"{label} multiseed run (seed {seed})",
                    purpose=f"Run the Phase B1 multiseed stability evaluation for {model} with seed {seed} on the 10K / 20-aspect corpus.",
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
                        str(seed),
                    ],
                    ordinal=ordinal,
                )
            )
            ordinal += 1
    return {
        "round_id": RUN_ID,
        "title": "Phase B1 multiseed local stability round",
        "purpose": "Run the leading local models across three seeds as individual resumable tasks to quantify stability and uncertainty.",
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
