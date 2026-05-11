from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "paper" / "benchmark_outputs" / "runs"
ANALYSIS_DIR = ROOT / "paper" / "analysis"

ASPECTS = [
    "accessibility",
    "assessment_design",
    "clarity",
    "difficulty",
    "exam_fairness",
    "feedback_quality",
    "grading_transparency",
    "interest",
    "lecturer_quality",
    "materials",
    "organization",
    "overall_experience",
    "pacing",
    "peer_interaction",
    "practical_application",
    "prerequisite_fit",
    "relevance",
    "support",
    "tooling_usability",
    "workload",
]

NO_GROUPING = {aspect: [aspect] for aspect in ASPECTS}

PEDAGOGICAL_GROUPS = {
    "instructional_quality": ["clarity", "lecturer_quality", "materials", "feedback_quality"],
    "assessment_course_management": [
        "assessment_design",
        "exam_fairness",
        "grading_transparency",
        "organization",
        "tooling_usability",
    ],
    "learning_demand_readiness": ["difficulty", "workload", "pacing", "prerequisite_fit"],
    "learning_environment": ["support", "accessibility", "peer_interaction"],
    "engagement_value": ["relevance", "interest", "practical_application", "overall_experience"],
}

# Regularized confusion-driven grouping. Pure agglomerative clustering on the sparse
# confusion graph produced unstable singletons for lightly confused labels, so this
# mapping keeps the strongest confusion families while preserving interpretability.
CONFUSION_GROUPS = {
    "teaching_content": ["clarity", "lecturer_quality", "materials", "relevance", "practical_application"],
    "assessment_structure": ["assessment_design", "exam_fairness", "grading_transparency", "organization"],
    "demand_readiness": ["difficulty", "workload", "pacing", "prerequisite_fit"],
    "support_friction": ["support", "accessibility", "peer_interaction", "feedback_quality", "tooling_usability"],
    "global_engagement": ["interest", "overall_experience"],
}


@dataclass
class PredictionArtifact:
    approach: str
    source_type: str
    path: Path


def invert_groups(groups: dict[str, list[str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for group_name, members in groups.items():
        for aspect in members:
            mapping[aspect] = group_name
    missing = [aspect for aspect in ASPECTS if aspect not in mapping]
    if missing:
        raise ValueError(f"Group mapping does not cover all aspects: {missing}")
    return mapping


def discover_local_artifacts() -> list[PredictionArtifact]:
    latest_by_approach: dict[str, Path] = {}
    for path in RUNS_DIR.glob("benchmark_full_*/*/*_sample_predictions.jsonl"):
        approach = path.name.replace("_sample_predictions.jsonl", "")
        current = latest_by_approach.get(approach)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            latest_by_approach[approach] = path
    return [
        PredictionArtifact(approach=approach, source_type="local", path=path)
        for approach, path in sorted(latest_by_approach.items())
    ]


def discover_llm_artifacts() -> list[PredictionArtifact]:
    artifacts: list[PredictionArtifact] = []
    for path in RUNS_DIR.glob("openai_batch_eval_*/artifacts/llm_responses.jsonl"):
        # Each file can contain multiple approaches, so we keep it as one artifact and split later.
        artifacts.append(PredictionArtifact(approach="llm_batch", source_type="llm", path=path))
    artifacts.sort(key=lambda item: item.path.stat().st_mtime)
    return artifacts


def iter_examples(artifact: PredictionArtifact) -> Iterable[tuple[str, dict[str, str], dict[str, str]]]:
    with artifact.path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if artifact.source_type == "local":
                yield artifact.approach, row.get("gold_aspects", {}), row.get("predicted_aspects", {})
            else:
                yield str(row.get("approach", "unknown")), row.get("gold_aspects", {}), row.get("predicted_aspects", {})


def collapse_to_groups(aspect_map: dict[str, str], aspect_to_group: dict[str, str]) -> set[str]:
    return {aspect_to_group[aspect] for aspect in aspect_map.keys() if aspect in aspect_to_group}


def multilabel_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "micro_precision": float(precision_score(y_true.ravel(), y_pred.ravel(), zero_division=0)),
        "micro_recall": float(recall_score(y_true.ravel(), y_pred.ravel(), zero_division=0)),
        "micro_f1": float(f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "label_accuracy": float((y_true == y_pred).mean()),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "samples_jaccard": float(jaccard_score(y_true, y_pred, average="samples", zero_division=0)),
    }


def evaluate_grouping(
    artifacts: list[PredictionArtifact],
    grouping_name: str,
    groups: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aspect_to_group = invert_groups(groups)
    group_names = list(groups.keys())
    summary_rows: list[dict] = []
    per_group_rows: list[dict] = []

    for artifact in artifacts:
        grouped_examples: dict[str, list[tuple[set[str], set[str]]]] = {}
        for approach, gold_aspects, predicted_aspects in iter_examples(artifact):
            grouped_examples.setdefault(approach, []).append(
                (
                    collapse_to_groups(gold_aspects, aspect_to_group),
                    collapse_to_groups(predicted_aspects, aspect_to_group),
                )
            )

        for approach, rows in grouped_examples.items():
            y_true = np.array(
                [[1 if group_name in gold else 0 for group_name in group_names] for gold, _ in rows],
                dtype=int,
            )
            y_pred = np.array(
                [[1 if group_name in pred else 0 for group_name in group_names] for _, pred in rows],
                dtype=int,
            )
            metrics = multilabel_summary(y_true, y_pred)
            summary_rows.append(
                {
                    "grouping": grouping_name,
                    "approach": approach,
                    **metrics,
                    "n_groups": len(group_names),
                    "n_examples": int(len(rows)),
                    "source_artifact": str(artifact.path),
                }
            )

            for idx, group_name in enumerate(group_names):
                yt = y_true[:, idx]
                yp = y_pred[:, idx]
                per_group_rows.append(
                    {
                        "grouping": grouping_name,
                        "approach": approach,
                        "group_name": group_name,
                        "precision": float(precision_score(yt, yp, zero_division=0)),
                        "recall": float(recall_score(yt, yp, zero_division=0)),
                        "f1": float(f1_score(yt, yp, zero_division=0)),
                        "support": int(yt.sum()),
                    }
                )

    summary_df = pd.DataFrame(summary_rows).sort_values(["grouping", "micro_f1"], ascending=[True, False]).reset_index(drop=True)
    per_group_df = pd.DataFrame(per_group_rows).sort_values(["grouping", "approach", "group_name"]).reset_index(drop=True)
    return summary_df, per_group_df


def write_markdown_report(
    summary_df: pd.DataFrame,
    pedagogical_groups: dict[str, list[str]],
    confusion_groups: dict[str, list[str]],
    output_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Grouped-Label Evaluation\n\n")
    lines.append("This report recomputes multilabel detection metrics after collapsing the original 20 aspects into coarser groups.\n\n")
    lines.append("## Grouping schemes\n\n")
    lines.append("### Pedagogical groups from the paper\n")
    for group_name, members in pedagogical_groups.items():
        lines.append(f"- `{group_name}`: {', '.join(members)}\n")
    lines.append("\n### Confusion-informed coarse groups\n")
    lines.append("These groups are regularized from observed confusion patterns rather than taken directly from a raw clustering output, because the fully unsupervised clustering produced unstable singleton groups for lightly confused labels.\n")
    for group_name, members in confusion_groups.items():
        lines.append(f"- `{group_name}`: {', '.join(members)}\n")
    lines.append("\n## Summary metrics\n\n")
    for grouping_name, subset in summary_df.groupby("grouping", sort=False):
        lines.append(f"### {grouping_name}\n")
        for _, row in subset.iterrows():
            lines.append(
                f"- `{row['approach']}`: micro-F1 {row['micro_f1']:.4f}, macro-F1 {row['macro_f1']:.4f}, samples-F1 {row['samples_f1']:.4f}, subset accuracy {row['subset_accuracy']:.4f}\n"
            )
        lines.append("\n")
    output_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = discover_local_artifacts() + discover_llm_artifacts()
    summary_none, per_group_none = evaluate_grouping(artifacts, "none_20_aspects", NO_GROUPING)
    summary_ped, per_group_ped = evaluate_grouping(artifacts, "pedagogical", PEDAGOGICAL_GROUPS)
    summary_conf, per_group_conf = evaluate_grouping(artifacts, "confusion_regularized", CONFUSION_GROUPS)

    summary_df = pd.concat([summary_none, summary_ped, summary_conf], ignore_index=True)
    per_group_df = pd.concat([per_group_none, per_group_ped, per_group_conf], ignore_index=True)

    summary_path = ANALYSIS_DIR / "grouped_label_metrics_20260404.csv"
    per_group_path = ANALYSIS_DIR / "grouped_label_per_group_metrics_20260404.csv"
    mapping_path = ANALYSIS_DIR / "grouped_label_mappings_20260404.json"
    report_path = ANALYSIS_DIR / "grouped_label_report_20260404.md"
    pivot_path = ANALYSIS_DIR / "grouped_label_micro_f1_pivot_20260404.csv"

    summary_df.to_csv(summary_path, index=False)
    per_group_df.to_csv(per_group_path, index=False)
    pivot_df = (
        summary_df.pivot(index="approach", columns="grouping", values="micro_f1")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    pivot_df.to_csv(pivot_path, index=False)
    mapping_path.write_text(
        json.dumps(
            {
                "none_20_aspects": NO_GROUPING,
                "pedagogical": PEDAGOGICAL_GROUPS,
                "confusion_regularized": CONFUSION_GROUPS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown_report(summary_df, PEDAGOGICAL_GROUPS, CONFUSION_GROUPS, report_path)

    print(
        json.dumps(
            {
                "summary_csv": str(summary_path),
                "per_group_csv": str(per_group_path),
                "pivot_csv": str(pivot_path),
                "mappings_json": str(mapping_path),
                "report_md": str(report_path),
                "n_artifacts": len(artifacts),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
