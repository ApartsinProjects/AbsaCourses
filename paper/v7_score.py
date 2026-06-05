"""Score the V7 audit batch: does the gpt-4.1-mini faithfulness audit agree with
HUMAN labels on real data? Retrieves the batch, and if complete, computes:

  - per-variant rates: on FAITHFUL (gold) requests, the audit's support/match
    CONFIRM rate (should be high); on FLIP requests, the sentiment-mismatch
    DETECT rate (should be high); on INJECT requests, the absent-aspect
    DETECT rate (should be high).
  - audit-as-classifier vs human ground truth (gold=faithful, flip/inject=
    unfaithful): accuracy, precision, recall, Cohen's kappa.

Usage: python v7_score.py            # auto-reads the submitted batch id
       python v7_score.py <batch_id>
"""
import json
import os
import sys

import pandas as pd
from openai import OpenAI

from label_faithfulness_audit import BATCH_DIR, extract_output_text

HERE = os.path.dirname(os.path.abspath(__file__))


def client():
    for p in (os.path.join(HERE, "..", ".opeai.key"), os.path.join(HERE, "..", "..", "..", "Projects", "CourseABSA", ".opeai.key")):
        if os.path.exists(p):
            return OpenAI(api_key=open(p).read().strip())
    return OpenAI()  # falls back to OPENAI_API_KEY


def kappa(a, b):
    """Cohen's kappa for two binary label lists."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n; pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def main():
    bid = sys.argv[1] if len(sys.argv) > 1 else None
    if not bid:
        sub = json.load(open(BATCH_DIR / "v7_audit_submitted.json")) if (BATCH_DIR / "v7_audit_submitted.json").exists() else None
        bid = sub["batch_id"] if sub else None
    assert bid, "no batch id"
    c = client()
    b = c.batches.retrieve(bid)
    print(f"batch {bid} status={b.status} ({getattr(b.request_counts,'completed',0)}/{getattr(b.request_counts,'total',0)})")
    if b.status != "completed":
        print("not complete yet; rerun later.")
        return
    out = c.files.content(b.output_file_id).text
    # parse results -> custom_id -> per-aspect verdicts
    verdict = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cid = row.get("custom_id")
        txt = extract_output_text(row)
        try:
            data = json.loads(txt)
            items = data.get("aspects", data) if isinstance(data, dict) else data
        except Exception:
            items = []
        verdict[cid] = {it.get("aspect"): (bool(it.get("supported")), bool(it.get("sentiment_match")))
                        for it in (items or []) if isinstance(it, dict)}
    man = pd.read_csv(BATCH_DIR / "v7_audit_manifest.csv")
    rows = []
    for _, m in man.iterrows():
        labels = json.loads(m["labels"]); v = verdict.get(m["custom_id"], {})
        sup = [v.get(a, (False, False))[0] for a in labels]
        mat = [v.get(a, (False, False))[1] for a in labels]
        # audit "calls it faithful" iff every aspect supported AND matched
        audit_faithful = bool(sup) and all(sup) and all(mat)
        rows.append({**m.to_dict(), "support_rate": sum(sup) / len(sup) if sup else 0,
                     "match_rate": sum(mat) / len(mat) if mat else 0,
                     "audit_faithful": audit_faithful})
    df = pd.DataFrame(rows)
    print("\n=== per-variant audit behavior (mean) ===")
    print(df.groupby(["dataset", "variant"])[["support_rate", "match_rate", "audit_faithful"]].mean().round(3))
    print("\n=== audit-as-classifier vs human ground truth ===")
    for ds in ["herath", "edurabsa", "ALL"]:
        d = df if ds == "ALL" else df[df.dataset == ds]
        gt = d["gt_faithful"].astype(bool).tolist()
        pr = d["audit_faithful"].astype(bool).tolist()
        tp = sum(1 for g, p in zip(gt, pr) if g and p); tn = sum(1 for g, p in zip(gt, pr) if not g and not p)
        fp = sum(1 for g, p in zip(gt, pr) if not g and p); fn = sum(1 for g, p in zip(gt, pr) if g and not p)
        acc = (tp + tn) / len(gt) if gt else 0
        prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0
        print(f"  {ds:9} n={len(gt):4} acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} kappa={kappa([int(x) for x in gt],[int(x) for x in pr]):.3f}")
    out_path = BATCH_DIR / "v7_audit_scored.csv"
    df.to_csv(out_path, index=False)
    print("\nwrote", out_path)


if __name__ == "__main__":
    main()
