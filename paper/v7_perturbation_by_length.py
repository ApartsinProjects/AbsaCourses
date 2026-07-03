"""Mechanism: decompose audit errors by review length.
faithful_keep = audit accepts a genuinely faithful aspect (recall);
flip_reject   = audit rejects a polarity-flipped aspect (specificity);
inject_reject = audit rejects an injected absent aspect.
Reveals whether short reviews fail by under-confirming (low keep) and long
reviews fail by over-accepting (low reject).
"""
import json, re
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


def build(results, wc):
    man = pd.read_csv(BR / "v7_audit_manifest.csv")
    gold = {(str(m.dataset), int(m.row_idx)): set(json.loads(m["labels"]).keys())
            for _, m in man[man.variant == "faithful"].iterrows()}
    rows = []
    for _, m in man.iterrows():
        cid = str(m["custom_id"])
        if cid not in results:
            continue
        v = verd(results[cid]); orig = gold.get((str(m.dataset), int(m.row_idx)), set()); w = wc.get(cid, 0)
        for a in json.loads(m["labels"]):
            pred = 1 if v.get(a, False) else 0
            gt = 1 if m.variant == "faithful" else (0 if m.variant == "flip" else (1 if a in orig else 0))
            rows.append({"variant": str(m.variant), "wc": w, "gt": gt, "pred": pred})
    return pd.DataFrame(rows)


def main():
    wc = review_wc()
    bins = [(0, 15), (16, 25), (26, 40), (41, 70), (71, 10000)]
    labs = ["<=15", "16-25", "26-40", "41-70", "71+"]
    for name, fn in [("gpt-4.1-mini", "v7_audit_realtime_results.jsonl"),
                     ("gemini-2.5-flash", "v7_gemini_results.jsonl")]:
        df = build(load_results(fn), wc)
        print(f"\n=== {name}: per-perturbation accuracy by review length ===")
        print(f"{'len':7}{'n':>5}{'faithful_keep':>15}{'flip_reject':>13}{'inject_reject':>15}")
        for (lo, hi), lab in zip(bins, labs):
            b = df[(df["wc"] >= lo) & (df["wc"] <= hi)]
            fa = b[b["variant"] == "faithful"]; fl = b[b["variant"] == "flip"]
            ij = b[(b["variant"] == "inject") & (b["gt"] == 0)]
            fk = fa["pred"].mean() if len(fa) else float("nan")
            fr = 1 - fl["pred"].mean() if len(fl) else float("nan")
            ir = 1 - ij["pred"].mean() if len(ij) else float("nan")
            print(f"{lab:7}{len(b):5d}{fk:15.3f}{fr:13.3f}{ir:15.3f}")


if __name__ == "__main__":
    main()
