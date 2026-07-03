"""Disentangle audit reliability by review LENGTH vs NUMBER OF ASPECTS.
Longer reviews carry more aspects; this checks whether the long-review
over-acceptance is driven by aspect density rather than length per se.
"""
import json, re
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BR = ROOT / "paper/batch_requests"


def load_results(fn):
    return {json.loads(l)["custom_id"]: json.loads(l)["output_text"] for l in open(BR / fn, encoding="utf-8")}


def review_wc():
    wc = {}
    for l in open(BR / "v7_audit_requests.jsonl", encoding="utf-8"):
        r = json.loads(l)
        m = re.search(r"Review:\s*(.*?)\s*Declared labels:", r["body"]["input"], re.S)
        wc[r["custom_id"]] = len(m.group(1).split()) if m else 0
    return wc


def verd(txt):
    try:
        d = json.loads(txt); items = d.get("aspects", d) if isinstance(d, dict) else d
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


def build(results, wc):
    man = pd.read_csv(BR / "v7_audit_manifest.csv")
    goldset = {(str(m.dataset), int(m.row_idx)): set(json.loads(m["labels"]).keys())
               for _, m in man[man.variant == "faithful"].iterrows()}
    rows = []
    for _, m in man.iterrows():
        cid = str(m["custom_id"])
        if cid not in results:
            continue
        key = (str(m.dataset), int(m.row_idx))
        gold = goldset.get(key, set())
        v = verd(results[cid]); w = wc.get(cid, 0); nasp = len(gold)
        for a in json.loads(m["labels"]):
            pred = 1 if v.get(a, False) else 0
            gt = 1 if m.variant == "faithful" else (0 if m.variant == "flip" else (1 if a in gold else 0))
            rows.append({"variant": str(m.variant), "wc": w, "nasp": nasp, "gt": gt, "pred": pred})
    return pd.DataFrame(rows)


def rates(b):
    fa = b[b["variant"] == "faithful"]; fl = b[b["variant"] == "flip"]; ij = b[(b["variant"] == "inject") & (b["gt"] == 0)]
    fk = fa["pred"].mean() if len(fa) else float("nan")
    fr = 1 - fl["pred"].mean() if len(fl) else float("nan")
    ir = 1 - ij["pred"].mean() if len(ij) else float("nan")
    return fk, fr, ir


def main():
    wc = review_wc()
    df = build(load_results("v7_audit_realtime_results.jsonl"), wc)  # gpt-4.1-mini
    print("corr(length, n_aspects) =", round(df[["wc", "nasp"]].drop_duplicates().corr().iloc[0, 1], 3))

    print("\n=== by NUMBER OF ASPECTS (gpt-4.1-mini) ===")
    print(f"{'nasp':6}{'n':>6}{'meanW':>7}{'kappa':>8}{'faith_keep':>12}{'flip_rej':>10}{'inj_rej':>9}")
    for k in [1, 2, 3]:
        b = df[df["nasp"] == k]
        fk, fr, ir = rates(b)
        print(f"{k:<6}{len(b):6d}{b['wc'].mean():7.1f}{kappa(b['gt'].tolist(), b['pred'].tolist()):8.3f}{fk:12.3f}{fr:10.3f}{ir:9.3f}")
    b = df[df["nasp"] >= 4]
    if len(b):
        fk, fr, ir = rates(b)
        print(f"{'4+':<6}{len(b):6d}{b['wc'].mean():7.1f}{kappa(b['gt'].tolist(), b['pred'].tolist()):8.3f}{fk:12.3f}{fr:10.3f}{ir:9.3f}")

    print("\n=== 2D kappa: length bin x n_aspects (controls length within aspect count) ===")
    lbins = [(0, 20), (21, 40), (41, 200)]; llab = ["<=20w", "21-40w", "41+w"]
    print(f"{'nasp':6}" + "".join(f"{l:>10}" for l in llab))
    for k in [1, 2, 3]:
        cells = []
        for lo, hi in lbins:
            b = df[(df["nasp"] == k) & (df["wc"] >= lo) & (df["wc"] <= hi)]
            cells.append(f"{kappa(b['gt'].tolist(), b['pred'].tolist()):.3f}({len(b)})" if len(b) >= 20 else "  -  ")
        print(f"{k:<6}" + "".join(f"{c:>10}" for c in cells))


if __name__ == "__main__":
    main()
