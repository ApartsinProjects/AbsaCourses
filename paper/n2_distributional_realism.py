"""N2 (statistical realism): distributional equivalence of synthetic vs real.

A frontier LLM judge can detect LLM text by style, an adversarial ceiling no
synthetic corpus passes and the wrong construct for "realism". The task-relevant
statistical question is whether synthetic reviews match real student feedback on
the measurable properties that govern ABSA training. This script co-computes, in
ONE pass on ONE config, a battery of per-review features for:

  REAL  = 200 OMSCS baseline reviews + de-duplicated RateMyProfessor comments
          (Kaggle student-feedback set), broadening the real pool well beyond 32.
  SYNTH = the CLEANED corpus: 9,159 complete rows + the 841 N1-regenerations.

For each axis it reports a two-sample test (Mann-Whitney U + Kolmogorov-Smirnov)
and a standardized effect size (Cliff's delta / Cohen's d). Axes where the effect
is negligible are statistically indistinguishable. Outputs paper/outputs/.
"""
import json, math, random, re
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "paper/generated_datasets/batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl"
REGEN = ROOT / "paper/outputs/n1_regenerated_841.jsonl"
IDS = ROOT / "paper/outputs/rc2_incomplete_row_ids.json"
REAL_CSV = ROOT / "paper/validation/batch_realism/runs/realism_real_baseline_200_20260404T131844Z/real_reviews.csv"
KAGGLE = ROOT / "external_data/Student_feedback_analysis_dataset/Annotated Student Feedback Data"
OUT = ROOT / "paper/outputs/n2_distributional_realism.json"

VOWELS = re.compile(r"[aeiouy]+", re.I)


def syllables(word):
    return max(1, len(VOWELS.findall(word)))


def features(text):
    t = " ".join(str(text).split())
    words = re.findall(r"[A-Za-z']+", t)
    n = len(words)
    if n < 3:
        return None
    sents = max(1, len(re.findall(r"[.!?]+", t)))
    syl = sum(syllables(w) for w in words)
    types = len(set(w.lower() for w in words))
    return {
        "word_count": n,
        "ttr": types / n,
        "mean_word_len": sum(len(w) for w in words) / n,
        "words_per_sentence": n / sents,
        "flesch": 206.835 - 1.015 * (n / sents) - 84.6 * (syl / n),
        "exclaim_per_100w": 100 * t.count("!") / n,
        "comma_per_100w": 100 * t.count(",") / n,
        "upper_ratio": sum(c.isupper() for c in t) / max(len(t), 1),
    }


def load_synth(n=1200, seed=42):
    ids = set(json.load(open(IDS))["incomplete_sample_ids"])
    complete = []
    with open(GEN, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if str(r.get("sample_id")) not in ids:
                tx = r.get("text", "")
                if tx and tx.lower() != "nan":
                    complete.append(tx)
    regen = [r["new_text"] for r in map(json.loads, open(REGEN, encoding="utf-8"))
             if r.get("new_text")]
    rng = random.Random(seed)
    rng.shuffle(complete)
    pool = complete[: n - len(regen)] + regen  # cleaned corpus proxy
    return pool


def load_real():
    import csv
    base = []
    with open(REAL_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = (row.get("review_text") or "").strip()
            if t and t.lower() != "nan":
                base.append(t)
    # Kaggle RateMyProfessor comments: col 2 of the TOWE tsv files, de-duplicated
    kag = set()
    for tsv in KAGGLE.rglob("*.tsv"):
        try:
            for line in tsv.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split("\t")
                if len(parts) >= 2 and len(parts[1].split()) >= 4:
                    kag.add(parts[1].strip())
        except Exception:
            pass
    return base, sorted(kag)


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    # P(a>b) - P(a<b) via rank method
    n, m = len(a), len(b)
    allv = np.concatenate([a, b])
    ranks = stats.rankdata(allv)
    ra = ranks[:n].sum()
    u = ra - n * (n + 1) / 2
    return round(2 * u / (n * m) - 1, 4)


def compare(real, synth, axes):
    rows = {}
    for ax in axes:
        r = np.array([f[ax] for f in real])
        s = np.array([f[ax] for f in synth])
        mw = stats.mannwhitneyu(r, s, alternative="two-sided")
        ks = stats.ks_2samp(r, s)
        d = cliffs_delta(s, r)
        mag = "negligible" if abs(d) < 0.147 else "small" if abs(d) < 0.33 else "medium" if abs(d) < 0.474 else "large"
        rows[ax] = {"real_mean": round(float(r.mean()), 3), "synth_mean": round(float(s.mean()), 3),
                    "mwu_p": float(f"{mw.pvalue:.3g}"), "ks_stat": round(float(ks.statistic), 4),
                    "cliffs_delta": d, "effect": mag}
    return rows


def main():
    synth_texts = load_synth()
    base, kaggle = load_real()
    real_texts = base + kaggle
    synth = [f for f in map(features, synth_texts) if f]
    real = [f for f in map(features, real_texts) if f]
    real_base = [f for f in map(features, base) if f]
    axes = ["word_count", "ttr", "mean_word_len", "words_per_sentence", "flesch",
            "exclaim_per_100w", "comma_per_100w", "upper_ratio"]
    combined = compare(real, synth, axes)
    base_only = compare(real_base, synth, axes)
    indistinguishable = [a for a, v in combined.items() if v["effect"] == "negligible"]
    out = {
        "n_synth": len(synth), "n_real_total": len(real),
        "n_real_omscs_baseline": len(real_base), "n_real_kaggle_rmp": len(kaggle),
        "axes_vs_real_pool": combined,
        "axes_vs_omscs_baseline_only": base_only,
        "indistinguishable_axes_negligible_effect": indistinguishable,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"synth n={len(synth)}  real n={len(real)} (omscs {len(real_base)} + kaggle {len(kaggle)})")
    for a in axes:
        v = combined[a]
        print(f"  {a:20s} real={v['real_mean']:8} synth={v['synth_mean']:8} "
              f"delta={v['cliffs_delta']:+.3f} ({v['effect']})  mwu_p={v['mwu_p']}")
    print("indistinguishable (negligible effect):", indistinguishable)


if __name__ == "__main__":
    main()
