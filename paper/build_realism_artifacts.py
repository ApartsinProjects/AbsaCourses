from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
VALIDATION_DIR = PAPER_DIR / "validation"
GEN_PROTOCOL_DIR = PAPER_DIR / "generation_protocol"
OUTPUT_TABLE_DIR = PAPER_DIR / "outputs" / "tables"
OUTPUT_FIG_DIR = PAPER_DIR / "outputs" / "figures"


def ensure_dirs() -> None:
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def choose_cycle_summary(cycle_id: int) -> Dict[str, object] | None:
    preferred = sorted(VALIDATION_DIR.glob(f"prompt_debug_cycle{cycle_id}_n30_*_summary.json"))
    if preferred:
        payload = load_json(preferred[-1])
        payload["_source_path"] = str(preferred[-1])
        return payload
    latest = VALIDATION_DIR / f"prompt_debug_cycle_{cycle_id}_summary.json"
    if latest.exists():
        payload = load_json(latest)
        payload["_source_path"] = str(latest)
        return payload
    return None


def write_cycle_metrics() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for cycle_id in range(3):
        payload = choose_cycle_summary(cycle_id)
        if payload is None:
            continue
        mean_entropy = payload.get("mean_entropy_bits", "")
        if mean_entropy in ("", None):
            source_path = Path(str(payload.get("_source_path", "")))
            judgments_path = source_path.with_name(source_path.name.replace("_summary.json", "_judgments.csv"))
            if judgments_path.exists():
                entropy_values: List[float] = []
                with judgments_path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    for item in reader:
                        for key in ("review_a_confidence", "review_b_confidence"):
                            try:
                                confidence = float(item.get(key, 50))
                            except Exception:
                                confidence = 50.0
                            p = max(0.5, min(1.0, confidence / 100.0))
                            q = 1.0 - p
                            if p in (0.0, 1.0) or q in (0.0, 1.0):
                                entropy_values.append(0.0)
                            else:
                                entropy_values.append(-(p * math.log2(p) + q * math.log2(q)))
                if entropy_values:
                    mean_entropy = round(sum(entropy_values) / len(entropy_values), 4)
        stats = payload.get("statistical_indistinguishability", {})
        rows.append(
            {
                "cycle_id": cycle_id,
                "cycle_name": payload.get("cycle_name", f"cycle_{cycle_id}"),
                "run_id": payload.get("run_id", ""),
                "n_judge_questions": payload.get("n_judge_questions", int(payload.get("n_pairs", 0)) * 2),
                "judge_item_accuracy": payload.get("judge_item_accuracy", ""),
                "mean_judge_confidence": payload.get("mean_judge_confidence", ""),
                "mean_confusion": payload.get("mean_confusion", ""),
                "mean_entropy_bits": mean_entropy,
                "chance_confusion_pct": payload.get("chance_confusion_pct", ""),
                "binomial_p_value_vs_chance": stats.get("binomial_p_value_vs_chance", ""),
                "wilson_95ci_low": (stats.get("wilson_95ci") or ["", ""])[0],
                "wilson_95ci_high": (stats.get("wilson_95ci") or ["", ""])[1],
                "equivalent_to_chance_within_margin": stats.get("equivalent_to_chance_within_margin", ""),
                "correctly_detected_synthetic_count": payload.get("correctly_detected_synthetic_count", ""),
                "editor_triggered": payload.get("editor_triggered", ""),
                "source_path": payload.get("_source_path", ""),
            }
        )
    out_path = OUTPUT_TABLE_DIR / "realism_cycle_metrics.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["cycle_id"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_curve_svg(rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    width = 900
    height = 420
    margin = 60
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    def x_pos(idx: int) -> float:
        if len(rows) == 1:
            return margin + plot_w / 2
        return margin + idx * (plot_w / (len(rows) - 1))

    def y_pos(value: float) -> float:
        return margin + (1.0 - value) * plot_h

    accuracy_points = " ".join(f"{x_pos(i):.1f},{y_pos(float(row['judge_item_accuracy'])):.1f}" for i, row in enumerate(rows))
    confusion_points = " ".join(f"{x_pos(i):.1f},{y_pos(float(row['chance_confusion_pct']) / 100.0):.1f}" for i, row in enumerate(rows))
    entropy_values = [float(row["mean_entropy_bits"] or 0.0) for row in rows]
    entropy_points = " ".join(
        f"{x_pos(i):.1f},{y_pos((entropy_values[i] / 1.0)):.1f}" for i in range(len(rows))
    )
    labels = "\n".join(
        f'<text x="{x_pos(i):.1f}" y="{height - 20}" text-anchor="middle" font-size="14">Cycle {row["cycle_id"]}</text>'
        for i, row in enumerate(rows)
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fffdf8"/>
  <text x="{margin}" y="30" font-size="24" font-family="Georgia, serif" fill="#1f2937">Realism Improvement Curve</text>
  <text x="{margin}" y="52" font-size="13" font-family="Georgia, serif" fill="#4b5563">Judge accuracy, chance confusion, and entropy across realism-debug cycles</text>
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#334155" stroke-width="2"/>
  <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#334155" stroke-width="2"/>
  <line x1="{margin}" y1="{margin}" x2="{width - margin}" y2="{margin}" stroke="#e5e7eb" stroke-dasharray="4 4"/>
  <line x1="{margin}" y1="{margin + plot_h/2:.1f}" x2="{width - margin}" y2="{margin + plot_h/2:.1f}" stroke="#e5e7eb" stroke-dasharray="4 4"/>
  <polyline fill="none" stroke="#b91c1c" stroke-width="3" points="{accuracy_points}"/>
  <polyline fill="none" stroke="#0f766e" stroke-width="3" points="{confusion_points}"/>
  <polyline fill="none" stroke="#4338ca" stroke-width="3" points="{entropy_points}"/>
  <text x="{width - 220}" y="{margin + 10}" font-size="13" fill="#b91c1c">Judge accuracy</text>
  <text x="{width - 220}" y="{margin + 30}" font-size="13" fill="#0f766e">Chance confusion</text>
  <text x="{width - 220}" y="{margin + 50}" font-size="13" fill="#4338ca">Entropy</text>
  {labels}
  <text x="20" y="{margin + 5}" font-size="12" fill="#475569">1.0</text>
  <text x="20" y="{margin + plot_h/2 + 5:.1f}" font-size="12" fill="#475569">0.5</text>
  <text x="20" y="{height - margin + 5}" font-size="12" fill="#475569">0.0</text>
</svg>
"""
    (OUTPUT_FIG_DIR / "realism_improvement_curve.svg").write_text(svg, encoding="utf-8")


def write_aspect_inventory_table() -> None:
    seed = load_json(GEN_PROTOCOL_DIR / "seed_attribute_schema.json")
    aspects = seed["proposed_extended_aspects"]
    rows = [{"aspect": aspect, "description": description} for aspect, description in aspects.items()]
    csv_path = OUTPUT_TABLE_DIR / "aspect_inventory_20.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["aspect", "description"])
        writer.writeheader()
        writer.writerows(rows)
    md_lines = [
        "# Aspect Inventory (20 Aspects)",
        "",
        "| Aspect | Description |",
        "|---|---|",
    ]
    md_lines.extend(f"| `{row['aspect']}` | {row['description']} |" for row in rows)
    (OUTPUT_TABLE_DIR / "aspect_inventory_20.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def choose_latest_n30_run() -> Tuple[Path, Path] | None:
    summaries = sorted(VALIDATION_DIR.glob("prompt_debug_cycle2_n30_*_summary.json"))
    if not summaries:
        return None
    summary_path = summaries[-1]
    run_id = summary_path.stem.replace("_summary", "")
    generations_path = VALIDATION_DIR / f"{run_id}_generations.csv"
    judgments_path = VALIDATION_DIR / f"{run_id}_judgments.csv"
    if not generations_path.exists() or not judgments_path.exists():
        return None
    return generations_path, judgments_path


def write_appendix_examples() -> None:
    latest = choose_latest_n30_run()
    if latest is None:
        return
    generations_path, judgments_path = latest
    import pandas as pd

    generations_df = pd.read_csv(generations_path)
    judgments_df = pd.read_csv(judgments_path)
    merged = generations_df.merge(judgments_df, on=["pair_id", "cycle_id", "course_code"], how="inner")
    if merged.empty:
        return
    selected = merged.head(3)
    csv_rows = []
    html_rows = []
    for _, row in selected.iterrows():
        csv_rows.append(
            {
                "pair_id": int(row["pair_id"]),
                "course_code": row["course_code"],
                "real_review_text": row["real_review_text"],
                "synthetic_review_text": row["synthetic_review_text"],
                "synthetic_aspect_labels": row["aspect_labels"],
                "synthetic_nuance_attributes": row["attributes"],
            }
        )
        html_rows.append(
            f"""
            <tr>
              <td>{int(row['pair_id'])}</td>
              <td><pre>{row['real_review_text']}</pre></td>
              <td><pre>{row['synthetic_review_text']}</pre></td>
              <td><pre>{row['aspect_labels']}</pre></td>
              <td><pre>{row['attributes']}</pre></td>
            </tr>
            """
        )
    csv_path = OUTPUT_TABLE_DIR / "appendix_real_vs_synthetic_examples.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    html = """<html><head><meta charset="utf-8"><style>
body{font-family:Georgia,serif;margin:32px}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #d1d5db;padding:10px;vertical-align:top;text-align:left}
pre{white-space:pre-wrap;font-family:inherit;margin:0}
</style></head><body>
<h2>Appendix: Real and Synthetic Review Examples</h2>
<p>Each synthetic review is shown with its sampled aspect labels and nuance attributes.</p>
<table>
<thead><tr><th>ID</th><th>Real review</th><th>Synthetic review</th><th>Synthetic aspects</th><th>Synthetic nuance attributes</th></tr></thead>
<tbody>
""" + "\n".join(html_rows) + """
</tbody></table></body></html>"""
    (OUTPUT_TABLE_DIR / "appendix_real_vs_synthetic_examples.html").write_text(html, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    cycle_rows = write_cycle_metrics()
    write_curve_svg(cycle_rows)
    write_aspect_inventory_table()
    write_appendix_examples()
    print("Saved realism and aspect artifacts.")


if __name__ == "__main__":
    main()
