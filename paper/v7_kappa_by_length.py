"""Audit-vs-human agreement (kappa) as a function of real-review length.

Bins the 2,482 per-aspect faithfulness decisions by the word count of the real
review, and computes Cohen's kappa per bin for BOTH auditors (cost-matched
gpt-4.1-mini and independent gemini-2.5-flash). Tests whether longer, more
complete reviews yield higher audit-human agreement.
"""
import json, re
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BR = ROOT / "paper/batch_requests"


def load_results(fn):
    return {json.loads(l)["custom_id"]: json.loads(l)["output_text"]
            for l in open(BR / fn, encoding="utf-8")}


def review_wc():
    wc = {}
    for l in open(BR / "v7_audit_requests.jsonl", encoding="utf-8"):
        r = json.loads(l)
        m = re.search(r"Review:\s*(.*?)\s*Declared labels:", r["body"]["input"], re.S)
        wc[r["custom_id"]] = len(m.group(1).split()) if m else 0
    return wc


def verd(txt):
    try:
        d = json.loads(txt)
        items = d.get("aspects", d) if isinstance(d, dict) else d
    except Exception:
        items = []
    return {it.get("aspect"): (bool(it.get("supported")) and bool(it.get("sentiment_match")))
            for it in (items or []) if isinstance(it, dict)}


def kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def build_rows(results, wc):
    man = pd.read_csv(BR / "v7_audit_manifest.csv")
    gold = {(str(m.dataset), int(m.row_idx)): set(json.loads(m["labels"]).keys())
            for _, m in man[man.variant == "faithful"].iterrows()}
    rows = []
    for _, m in man.iterrows():
        cid = str(m["custom_id"])
        if cid not in results:
            continue
        v = verd(results[cid])
        orig = gold.get((str(m.dataset), int(m.row_idx)), set())
        w = wc.get(cid, 0)
        for a in json.loads(m["labels"]):
            pred = 1 if v.get(a, False) else 0
            gt = 1 if m.variant == "faithful" else (0 if m.variant == "flip" else (1 if a in orig else 0))
            rows.append({"dataset": str(m.dataset), "variant": str(m.variant), "wc": w, "gt": gt, "pred": pred})
    return pd.DataFrame(rows)


def main():
    wc = review_wc()
    gpt = build_rows(load_results("v7_audit_realtime_results.jsonl"), wc)
    gem = build_rows(load_results("v7_gemini_results.jsonl"), wc)
    bins = [(0, 15), (16, 25), (26, 40), (41, 70), (71, 10000)]
    labels = ["<=15", "16-25", "26-40", "41-70", "71+"]
    out = {"bins": []}
    print(f"{'length':8} {'n_dec':>6} {'meanW':>6} {'kappa_gpt':>10} {'kappa_gemini':>13}")
    for (lo, hi), lab in zip(bins, labels):
        g = gpt[(gpt["wc"] >= lo) & (gpt.wc <= hi)]
        e = gem[(gem["wc"] >= lo) & (gem["wc"] <= hi)]
        kg = kappa(g["gt"].tolist(), g["pred"].tolist())
        ke = kappa(e["gt"].tolist(), e["pred"].tolist())
        mw = float(g["wc"].mean()) if len(g) else 0
        out["bins"].append({"bin": lab, "n_decisions": int(len(g)), "mean_words": round(mw, 1),
                            "kappa_gpt": round(kg, 4), "kappa_gemini": round(ke, 4)})
        print(f"{lab:8} {len(g):6d} {mw:6.1f} {kg:10.3f} {ke:13.3f}")
    # spearman correlation kappa vs length using per-review mean
    from scipy.stats import spearmanr
    # per-review kappa is noisy; report bin-level monotonicity via correlation of bin midpoint vs kappa
    mids = [8, 20, 33, 55, 90]
    kgs = [b["kappa_gpt"] for b in out["bins"]]
    kes = [b["kappa_gemini"] for b in out["bins"]]
    out["spearman_len_vs_kappa_gpt"] = round(float(spearmanr(mids, kgs).correlation), 3)
    out["spearman_len_vs_kappa_gemini"] = round(float(spearmanr(mids, kes).correlation), 3)
    print("spearman(len,kappa) gpt=", out["spearman_len_vs_kappa_gpt"], "gemini=", out["spearman_len_vs_kappa_gemini"])
    json.dump(out, open(ROOT / "paper/outputs/kappa_by_length.json", "w"), indent=2)


if __name__ == "__main__":
    main()
