"""Reviewer-response EVIDENCE ITEM 3: multi-judge convergence.

For each cost-matched judge, reports per-aspect SUPPORT agreement and per-aspect
MATCH agreement against the gpt-5.2 reference audit on the same 250-row sample.

- Haiku stores per_aspect_support_agreement / per_aspect_match_agreement
  directly in its calibration JSON; we re-derive them from the raw responses to
  confirm the methodology, and report both.
- gpt-4.1-mini and gpt-4o-mini store only aggregate rates + spearman; we COMPUTE
  per-aspect agreement from their raw response jsonl vs the gpt-5.2 reference
  responses (same metric definition as Haiku).

Per-aspect agreement is computed over the (row_id, aspect) pairs that BOTH the
candidate and gpt-5.2 judged:
  support_agreement = mean[ candidate.supported == gpt5.supported ]
  match_agreement   = mean[ candidate.sentiment_match == gpt5.sentiment_match ]

Reads only on-disk audit response files. No training, no API.
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(r"E:\Claude\CourseABSA\hopeful-kowalevski-04ee10")
FA = REPO / "paper" / "faithfulness_audit"
REF = Path(r"E:\Projects\CourseABSA\paper\faithfulness_audit\faithfulness_audit_gpt-5_2_250_llm_responses.jsonl")
OUT_JSON = REPO / "paper" / "outputs" / "tables" / "reviewer_response_item3_judge_agreement.json"
OUT_CSV = REPO / "paper" / "outputs" / "tables" / "reviewer_response_item3_judge_agreement.csv"

CANDIDATES = [
    ("gpt-4.1-mini", FA / "calibration_gpt-4.1-mini_responses.jsonl", FA / "calibration_gpt-4.1-mini_vs_gpt-5_2.json"),
    ("gpt-4o-mini", FA / "calibration_gpt-4o-mini_responses.jsonl", FA / "calibration_gpt-4o-mini_vs_gpt-5_2.json"),
    ("claude-3.5-haiku", FA / "calibration_claude-3_5-haiku_responses.jsonl", FA / "calibration_claude-3_5-haiku_vs_gpt-5_2.json"),
]


def load_ref():
    """gpt-5.2 reference: row_id(int) -> {aspect: (supported, sentiment_match)}."""
    out = {}
    for line in REF.open(encoding="utf-8"):
        r = json.loads(line)
        rid = int(r["row_id"])
        d = {}
        for a in r.get("parsed_response", {}).get("aspects", []):
            d[a["aspect"]] = (bool(a["supported"]), bool(a["sentiment_match"]))
        out[rid] = d
    return out


def load_candidate(path):
    """candidate: row_id(int) -> {aspect: (supported, sentiment_match)}."""
    out = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if not r.get("ok", True):
            continue
        rid = int(r["row_id"])
        d = {}
        for a in r.get("judged", {}).get("aspects", []):
            asp = a.get("aspect")
            if asp == "__row_summary__":
                continue
            d[asp] = (bool(a["supported"]), bool(a["sentiment_match"]))
        out[rid] = d
    return out


def per_aspect_agreement(cand, ref):
    n_common = 0
    sup_agree = 0
    mat_agree = 0
    for rid, casp in cand.items():
        rasp = ref.get(rid, {})
        for asp, (csup, cmat) in casp.items():
            if asp in rasp:
                rsup, rmat = rasp[asp]
                n_common += 1
                if csup == rsup:
                    sup_agree += 1
                if cmat == rmat:
                    mat_agree += 1
    return {
        "n_aspects_common": n_common,
        "per_aspect_support_agreement": round(sup_agree / n_common, 4) if n_common else None,
        "per_aspect_match_agreement": round(mat_agree / n_common, 4) if n_common else None,
    }


def main():
    ref = load_ref()
    n_ref_aspects = sum(len(v) for v in ref.values())
    rows = []
    out = {
        "evidence_item": "3 - multi-judge convergence vs gpt-5.2",
        "reference": str(REF),
        "reference_n_rows": len(ref),
        "reference_n_aspects": n_ref_aspects,
        "n_sample_rows": 250,
        "judges": [],
        "note_behavioral_validation": (
            "The strongest validity argument is behavioral, not human: the bot25 "
            "negative-control bucket (rows the judge scored LOWEST) collapses "
            "downstream Herath/EduRABSA sentiment transfer on ALL (architecture, "
            "target) conditions (Item 2: bot25 MSE >> full MSE in every cell), "
            "showing low-audit rows really are worse training data independent of "
            "any human label."
        ),
        "FLAG_manuscript_label_swap": (
            "Manuscript line 1018 cites the AT-SCALE judge gpt-4.1-mini at 'per-aspect "
            "support agreement 0.845 and sentiment-match agreement 0.715'. Those values "
            "are actually gpt-4o-mini's (recomputed 0.8454 / 0.7149). The TRUE gpt-4.1-mini "
            "agreement vs gpt-5.2, recomputed here from the raw response files, is 0.8743 "
            "support / 0.7904 match (HIGHER, more favorable). The at-scale audit "
            "(_d2_atscale.py, MODEL='gpt-4.1-mini') was run with gpt-4.1-mini, so the "
            "manuscript should cite 0.874 / 0.790, not 0.845 / 0.715. Methodology is "
            "verified: recomputing Haiku from raw responses reproduces its stored "
            "0.8169 / 0.6781 exactly."
        ),
        "note_human_study": (
            "A human study is DESIGNED at human/ (codebook.md, "
            "tasks/task_1_realism_and_faithfulness, tasks/task_3_llm_judge_agreement) "
            "but has NO collected responses for the faithfulness/judge-agreement "
            "tasks (human/responses/task_1, task_3 hold only empty .gitkeep). A "
            "separate real-vs-synthetic discrimination study (human/responses/task_9/"
            "rater_A_complete.csv) does have one rater file, but it is a different "
            "study (realism discrimination, not faithfulness or judge agreement). "
            "Faithfulness/judge-agreement human validation is specified-but-not-executed; "
            "no human faithfulness results exist."
        ),
    }

    for name, resp_path, json_path in CANDIDATES:
        cand = load_candidate(resp_path)
        recomputed = per_aspect_agreement(cand, ref)
        stored = json.loads(json_path.read_text(encoding="utf-8"))
        entry = {
            "judge": name,
            "recomputed_vs_gpt5_2": recomputed,
            "stored_in_calibration_json": {
                k: stored.get(k) for k in (
                    "per_aspect_support_agreement", "per_aspect_match_agreement",
                    "n_aspects_common", "spearman_rho", "pearson_r",
                    "candidate_aggregate_support_rate", "candidate_aggregate_match_rate",
                    "candidate_support_rate", "candidate_match_rate",
                ) if k in stored
            },
        }
        out["judges"].append(entry)
        rows.append({
            "judge": name,
            "n_aspects_common": recomputed["n_aspects_common"],
            "support_agreement": recomputed["per_aspect_support_agreement"],
            "match_agreement": recomputed["per_aspect_match_agreement"],
            "spearman_rho": stored.get("spearman_rho"),
        })

    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    import csv
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["judge", "n_aspects_common", "support_agreement",
                                          "match_agreement", "spearman_rho"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("=== EVIDENCE ITEM 3: per-judge per-aspect agreement vs gpt-5.2 ===")
    print(f"reference: {len(ref)} rows, {n_ref_aspects} aspects\n")
    print(f"{'judge':18s} {'n_common':>9s} {'support_agr':>12s} {'match_agr':>11s} {'spearman':>9s}")
    for r in rows:
        sp = f"{r['spearman_rho']:.4f}" if r['spearman_rho'] is not None else "   -"
        print(f"{r['judge']:18s} {r['n_aspects_common']:>9d} "
              f"{r['support_agreement']:>12.4f} {r['match_agreement']:>11.4f} {sp:>9s}")
    print("\n--- methodology cross-check (recomputed vs stored Haiku) ---")
    h = [j for j in out["judges"] if j["judge"] == "claude-3.5-haiku"][0]
    print(f"  recomputed: support={h['recomputed_vs_gpt5_2']['per_aspect_support_agreement']}  "
          f"match={h['recomputed_vs_gpt5_2']['per_aspect_match_agreement']}  "
          f"n={h['recomputed_vs_gpt5_2']['n_aspects_common']}")
    print(f"  stored    : support={h['stored_in_calibration_json'].get('per_aspect_support_agreement')}  "
          f"match={h['stored_in_calibration_json'].get('per_aspect_match_agreement')}  "
          f"n={h['stored_in_calibration_json'].get('n_aspects_common')}")
    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
