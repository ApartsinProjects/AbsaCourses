"""(A) Judge ERROR analysis: which items does the realism judge get wrong,
and does that depend on length / aspects / density?
  - real side: false-synthetic rate by review length (n=200, rate ~0.15)
  - synthetic side: the few misses (fooled the judge), listed with shape stats
(B) Sentence-level MAUVE: apples-to-apples at sentence granularity, with
enough points (>=1000/side) to satisfy MAUVE's sample-size preference.
"""
import csv, json, random, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTD = ROOT / "paper/outputs"
CYC0 = ROOT / "paper/validation/batch_realism/runs/realism_synth_cycle0_20260404T131844Z/generated_reviews.jsonl"
REAL = ROOT / "paper/validation/batch_realism/runs/realism_real_baseline_200_20260404T131844Z/real_reviews.csv"
GEN = ROOT / "paper/generated_datasets/batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl"
IDS = ROOT / "paper/outputs/rc2_incomplete_row_ids.json"
HERATH = ROOT / "paper/real_transfer/herath_mapped_real_reviews.jsonl"
FAMS = ["gpt5nano_clean", "gemini_flash", "glm_46", "llama33_70b"]


def load_real_texts():
    out = []
    for row in csv.DictReader(open(REAL, encoding="utf-8")):
        t = (row.get("review_text") or "").strip()
        if t and t.lower() != "nan":
            out.append(t)
    return out[:200]


def part_a():
    reals = load_real_texts()
    # judge verdicts
    verdict = {}
    for l in open(OUTD / "n2_judge.jsonl", encoding="utf-8"):
        r = json.loads(l)
        verdict[(r["source"], r["id"])] = r["label"]
    # real-side FP by length
    rows = []
    for i, t in enumerate(reals):
        lab = verdict.get(("real", f"real_{i}"))
        if lab in ("real", "synthetic"):
            rows.append({"wc": len(t.split()), "fp": 1 if lab == "synthetic" else 0})
    wcs = np.array([r["wc"] for r in rows])
    qs = np.quantile(wcs, [0, .25, .5, .75, 1.0])
    print(f"=== (A) REAL reviews falsely judged synthetic, by length (n={len(rows)}, overall FP={np.mean([r['fp'] for r in rows]):.3f}) ===")
    binstats = []
    for i in range(4):
        lo, hi = qs[i], qs[i + 1]
        sel = [r for r in rows if (r["wc"] >= lo and (r["wc"] < hi or i == 3))]
        fp = np.mean([r["fp"] for r in sel])
        print(f"  {int(lo):4d}-{int(hi):4d} words: FP={fp:.3f} (n={len(sel)})")
        binstats.append({"lo": int(lo), "hi": int(hi), "fp": round(float(fp), 3), "n": len(sel)})
    from scipy.stats import spearmanr
    sp = spearmanr([r["wc"] for r in rows], [r["fp"] for r in rows])
    print(f"  spearman(length, false-synthetic) = {sp.correlation:.3f} (p={sp.pvalue:.3g})")
    # synthetic-side misses
    nasp = {}
    for l in open(CYC0, encoding="utf-8"):
        r = json.loads(l)
        labs = r.get("aspect_labels")
        labs = json.loads(labs) if isinstance(labs, str) else (labs or {})
        nasp[r["custom_id"]] = len(labs)
    texts = {}
    for fam in FAMS:
        p = OUTD / f"n2_gen_{fam}.jsonl"
        if p.exists():
            for l in open(p, encoding="utf-8"):
                r = json.loads(l)
                texts[(fam, r["custom_id"])] = r["text"]
    misses = []
    tot = 0
    for (src, iid), lab in verdict.items():
        if src == "real" or lab not in ("real", "synthetic"):
            continue
        tot += 1
        if lab == "real":
            t = texts.get((src, iid), "")
            misses.append({"src": src, "wc": len(str(t).split()), "nasp": nasp.get(iid, 0)})
    print(f"\n  synthetic items that FOOLED the judge: {len(misses)}/{tot}")
    for m in misses:
        print(f"    {m['src']:16s} words={m['wc']:4d} aspects={m['nasp']}")
    return {"real_fp_by_length": binstats,
            "spearman_len_fp": round(float(sp.correlation), 3), "p": float(f"{sp.pvalue:.3g}"),
            "synthetic_misses": misses, "n_synth_judged": tot}


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentences(texts, min_w=4):
    out = []
    for t in texts:
        for s in SENT_SPLIT.split(" ".join(str(t).split())):
            if len(s.split()) >= min_w:
                out.append(s.strip())
    return out


def part_b():
    import mauve
    rng = random.Random(42)
    ids = set(json.load(open(IDS))["incomplete_sample_ids"])
    synth = []
    for l in open(GEN, encoding="utf-8"):
        r = json.loads(l)
        if str(r.get("sample_id")) not in ids:
            t = str(r.get("text", ""))
            if t and t.lower() != "nan":
                synth.append(t)
    rng.shuffle(synth)
    syn_sents = sentences(synth[:800])
    oms_sents = sentences(load_real_texts())
    her_sents = [str(json.loads(l).get("text", "")) for l in open(HERATH, encoding="utf-8")]
    her_sents = [s for s in her_sents if len(s.split()) >= 4]
    rng.shuffle(syn_sents); rng.shuffle(oms_sents); rng.shuffle(her_sents)
    N = min(1000, len(oms_sents), len(syn_sents), len(her_sents))
    print(f"\n=== (B) sentence-level MAUVE (N={N}/side) ===")

    def mv(p, q):
        return round(float(mauve.compute_mauve(p_text=p, q_text=q, device_id=-1, max_text_length=96,
                                               verbose=False, featurize_model_name="gpt2",
                                               batch_size=32).mauve), 4)
    half = N // 2
    res = {
        "n_per_side": N,
        "oms_vs_synth_sent": mv(oms_sents[:N], syn_sents[:N]),
        "oms_vs_oms_upper": mv(oms_sents[:half], oms_sents[half:2 * half]),
        "oms_vs_herath_crossreal": mv(oms_sents[:N], her_sents[:N]),
        "herath_vs_synth_sent": mv(her_sents[:N], syn_sents[:N]),
    }
    for k, v in res.items():
        print(f"  {k}: {v}")
    return res


if __name__ == "__main__":
    a = part_a()
    b = part_b()
    json.dump({"judge_errors": a, "sentence_mauve": b},
              open(OUTD / "judge_errors_sentence_mauve.json", "w"), indent=2)
