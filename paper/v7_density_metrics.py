"""Audit agreement by ASPECT DENSITY (aspects per word), with metrics beyond kappa.
Density = n_aspects / n_words = inverse of textual evidence per aspect. Hypothesis:
agreement decreases monotonically with density. Reports kappa, MCC, balanced
accuracy, recall, specificity per density quintile.
"""
import json, re, math
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


def metrics(gt, pred):
    tp = sum(1 for g, p in zip(gt, pred) if g and p); tn = sum(1 for g, p in zip(gt, pred) if not g and not p)
    fp = sum(1 for g, p in zip(gt, pred) if not g and p); fn = sum(1 for g, p in zip(gt, pred) if g and not p)
    n = len(gt); po = (tp + tn) / n
    pa, pb = sum(gt) / n, sum(pred) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = (po - pe) / (1 - pe) if pe != 1 else 1.0
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {"n": n, "kappa": round(kappa, 3), "mcc": round(mcc, 3),
            "bal_acc": round((rec + spec) / 2, 3), "recall": round(rec, 3), "specificity": round(spec, 3)}


def build(results, wc):
    man = pd.read_csv(BR / "v7_audit_manifest.csv")
    goldset = {(str(m.dataset), int(m.row_idx)): set(json.loads(m["labels"]).keys())
               for _, m in man[man.variant == "faithful"].iterrows()}
    rows = []
    for _, m in man.iterrows():
        cid = str(m["custom_id"])
        if cid not in results:
            continue
        key = (str(m.dataset), int(m.row_idx)); gold = goldset.get(key, set())
        v = verd(results[cid]); w = wc.get(cid, 0); nasp = len(gold)
        dens = nasp / w if w else 0
        for a in json.loads(m["labels"]):
            pred = 1 if v.get(a, False) else 0
            gt = 1 if m.variant == "faithful" else (0 if m.variant == "flip" else (1 if a in gold else 0))
            rows.append({"density": dens, "gt": gt, "pred": pred})
    return pd.DataFrame(rows)


def main():
    wc = review_wc()
    out = {}
    for name, fn in [("gpt-4.1-mini", "v7_audit_realtime_results.jsonl"),
                     ("gemini-2.5-flash", "v7_gemini_results.jsonl")]:
        df = build(load_results(fn), wc)
        df = df[df["density"] > 0]
        q = df["density"].quantile([0, .2, .4, .6, .8, 1.0]).values
        print(f"\n=== {name}: agreement by aspect density (aspects/word) ===")
        print(f"{'density_bin':>16}{'n':>6}{'kappa':>7}{'MCC':>7}{'balAcc':>8}{'recall':>8}{'spec':>7}")
        rows = []
        for i in range(5):
            lo, hi = q[i], q[i + 1]
            b = df[(df["density"] >= lo) & (df["density"] <= hi)] if i == 4 else df[(df["density"] >= lo) & (df["density"] < hi)]
            mt = metrics(b["gt"].tolist(), b["pred"].tolist())
            print(f"{lo:.4f}-{hi:.4f}".rjust(16) + f"{mt['n']:6d}{mt['kappa']:7.3f}{mt['mcc']:7.3f}{mt['bal_acc']:8.3f}{mt['recall']:8.3f}{mt['specificity']:7.3f}")
            rows.append({"density_lo": round(lo, 4), "density_hi": round(hi, 4), **mt})
        # spearman density vs kappa (bin level)
        from scipy.stats import spearmanr
        mids = [(r["density_lo"] + r["density_hi"]) / 2 for r in rows]
        ks = [r["kappa"] for r in rows]
        sp = round(float(spearmanr(mids, ks).correlation), 3)
        print(f"spearman(density, kappa) = {sp}")
        out[name] = {"bins": rows, "spearman_density_kappa": sp}
    json.dump(out, open(ROOT / "paper/outputs/audit_by_density.json", "w"), indent=2)


if __name__ == "__main__":
    main()
