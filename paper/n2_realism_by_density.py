"""Realism (judge synthetic-detection) as a function of aspect density.

Joins the cached realism-judge verdicts (n2_judge.jsonl: cycle0_orig + four clean
generator families) to each synthetic review's declared aspect count (from the
cycle-0 label conditioning) and its own word count. Tests whether detection rate
rises monotonically with aspect density (aspects/word), i.e. whether label-dense
reviews read more checklist-like, and separates density from raw aspect count.
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTD = ROOT / "paper/outputs"
CYC0 = ROOT / "paper/validation/batch_realism/runs/realism_synth_cycle0_20260404T131844Z/generated_reviews.jsonl"

FAMS = ["gpt5nano_clean", "gemini_flash", "glm_46", "llama33_70b"]


def main():
    # labels per custom_id from cycle0 conditioning
    nasp, cyc0_text = {}, {}
    for l in open(CYC0, encoding="utf-8"):
        r = json.loads(l)
        labs = r.get("aspect_labels")
        labs = json.loads(labs) if isinstance(labs, str) else (labs or {})
        nasp[r["custom_id"]] = len(labs)
        cyc0_text[r["custom_id"]] = r.get("generated_review_text", "")
    # texts per (source, id)
    texts = {}
    for fam in FAMS:
        p = OUTD / f"n2_gen_{fam}.jsonl"
        if p.exists():
            for l in open(p, encoding="utf-8"):
                r = json.loads(l)
                texts[(fam, r["custom_id"])] = r["text"]
    for cid, t in cyc0_text.items():
        texts[("cycle0_orig", cid)] = t
    # judge verdicts
    rows = []
    for l in open(OUTD / "n2_judge.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["source"] == "real" or r["label"] not in ("real", "synthetic"):
            continue
        t = texts.get((r["source"], r["id"]), "")
        w = len(str(t).split())
        k = nasp.get(r["id"], 0)
        if w < 3 or k == 0:
            continue
        rows.append({"src": r["source"], "wc": w, "nasp": k, "dens": k / w,
                     "det": 1 if r["label"] == "synthetic" else 0})
    print(f"synthetic judged items with labels: {len(rows)}")

    def rate(sel):
        return (sum(x["det"] for x in sel) / len(sel), len(sel)) if sel else (float("nan"), 0)

    # 1) detection by aspect count
    print("\n=== detection rate by n_aspects ===")
    for k in [1, 2, 3]:
        r_, n_ = rate([x for x in rows if x["nasp"] == k])
        print(f"  {k} aspects: det={r_:.3f} (n={n_})")

    # 2) detection by density quintile
    dens = np.array([x["dens"] for x in rows])
    qs = np.quantile(dens, [0, .2, .4, .6, .8, 1.0])
    print("\n=== detection rate by density quintile (aspects/word) ===")
    binstats = []
    for i in range(5):
        lo, hi = qs[i], qs[i + 1]
        sel = [x for x in rows if (x["dens"] >= lo and (x["dens"] < hi or i == 4))]
        r_, n_ = rate(sel)
        mw = np.mean([x["wc"] for x in sel])
        print(f"  {lo:.4f}-{hi:.4f}: det={r_:.3f} (n={n_}, mean_words={mw:.0f})")
        binstats.append({"lo": round(float(lo), 4), "hi": round(float(hi), 4),
                         "det": round(r_, 4), "n": n_, "mean_words": round(float(mw), 1)})
    from scipy.stats import spearmanr, pointbiserialr
    sp = spearmanr([x["dens"] for x in rows], [x["det"] for x in rows])
    print(f"spearman(density, detected) = {sp.correlation:.3f} (p={sp.pvalue:.3g})")

    # 3) density controlled for n_aspects (within each aspect count, split by word count)
    print("\n=== detection by n_aspects x length (density controlled within row) ===")
    print(f"{'nasp':6}{'shorter half':>15}{'longer half':>14}")
    ctrl = {}
    for k in [1, 2, 3]:
        sel = [x for x in rows if x["nasp"] == k]
        if len(sel) < 30:
            continue
        med = np.median([x["wc"] for x in sel])
        lo_ = [x for x in sel if x["wc"] <= med]
        hi_ = [x for x in sel if x["wc"] > med]
        rl, nl = rate(lo_); rh, nh = rate(hi_)
        print(f"{k:<6}{rl:>10.3f}({nl:3d}){rh:>9.3f}({nh:3d})")
        ctrl[k] = {"shorter_det": round(rl, 4), "longer_det": round(rh, 4)}

    json.dump({"n": len(rows), "by_density": binstats,
               "spearman_density_det": round(float(sp.correlation), 4),
               "spearman_p": float(f"{sp.pvalue:.3g}"),
               "by_nasp_length": ctrl},
              open(OUTD / "realism_by_density.json", "w"), indent=2)


if __name__ == "__main__":
    main()
