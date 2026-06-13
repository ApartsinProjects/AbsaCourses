"""Filter-aligned per-aspect scoring of the V7 audit vs human gold:
the filter audits EACH aspect (supported AND sentiment_match), so the right
human-agreement metric is per-aspect, not the all-or-nothing per-review set.
Ground truth per aspect: faithful->1 for all; flip->0 for all; inject->1 for the
original gold aspects, 0 for the injected absent aspect.
"""
import json
import pandas as pd
from label_faithfulness_audit import BATCH_DIR

res = {json.loads(l)["custom_id"]: json.loads(l)["output_text"]
       for l in open(BATCH_DIR / "v7_audit_realtime_results.jsonl", encoding="utf-8")}
man = pd.read_csv(BATCH_DIR / "v7_audit_manifest.csv")

gold = {}
for _, m in man[man.variant == "faithful"].iterrows():
    gold[(str(m.dataset), int(m.row_idx))] = set(json.loads(m["labels"]).keys())


def verd(txt):
    try:
        d = json.loads(txt)
        items = d.get("aspects", d) if isinstance(d, dict) else d
    except Exception:
        items = []
    return {it.get("aspect"): (bool(it.get("supported")) and bool(it.get("sentiment_match")))
            for it in (items or []) if isinstance(it, dict)}


rows = []
for _, m in man.iterrows():
    cid = str(m["custom_id"])
    if cid not in res:
        continue
    labels = json.loads(m["labels"])
    v = verd(res[cid])
    orig = gold.get((str(m.dataset), int(m.row_idx)), set())
    for a in labels:
        pred = 1 if v.get(a, False) else 0
        if m.variant == "faithful":
            gt = 1
        elif m.variant == "flip":
            gt = 0
        else:
            gt = 1 if a in orig else 0
        rows.append({"dataset": str(m.dataset), "variant": str(m.variant),
                     "gt": gt, "pred": pred})

d = pd.DataFrame(rows)


def kappa(a, b):
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n; pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


print("=== PER-ASPECT audit-vs-human (filter-aligned) ===")
for ds in ["herath", "edurabsa", "ALL"]:
    dd = d if ds == "ALL" else d[d.dataset == ds]
    gt = dd["gt"].tolist(); pr = dd["pred"].tolist()
    tp = sum(1 for g, p in zip(gt, pr) if g and p); tn = sum(1 for g, p in zip(gt, pr) if not g and not p)
    fp = sum(1 for g, p in zip(gt, pr) if not g and p); fn = sum(1 for g, p in zip(gt, pr) if g and not p)
    acc = (tp + tn) / len(gt); prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    print(f"  {ds:9} n={len(gt):5} acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} kappa={kappa(gt, pr):.3f}")

print("\n=== per-perturbation per-aspect behavior ===")
faith = d[d.variant == "faithful"]
print(f"  faithful  keep rate (true-faithful kept)   = {faith.pred.mean():.3f} (n={len(faith)})")
flip = d[d.variant == "flip"]
print(f"  flip      reject rate (flipped flagged)     = {1 - flip.pred.mean():.3f} (n={len(flip)})")
inj = d[(d["variant"] == "inject") & (d["gt"] == 0)]
print(f"  inject    reject rate (injected flagged)    = {1 - inj.pred.mean():.3f} (n={len(inj)})")
