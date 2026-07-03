"""Fairness control for realism: is the synthetic-vs-real MAUVE gap driven by
ENTITY/TOPIC identity (course codes, instructor names, platform terms) or by
genuine writing STYLE? Scrub high-signal entities symmetrically from both pools
and recompute sentence-level MAUVE; compare to unscrubbed.
"""
import csv, json, random, re
from pathlib import Path
import mauve

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "paper/generated_datasets/batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl"
IDS = ROOT / "paper/outputs/rc2_incomplete_row_ids.json"
REAL = ROOT / "paper/validation/batch_realism/runs/realism_real_baseline_200_20260404T131844Z/real_reviews.csv"
HERATH = ROOT / "paper/real_transfer/herath_mapped_real_reviews.jsonl"

CODE = re.compile(r"\b[A-Za-z]{2,4}[-\s]?\d{3,4}\b")
TITLE = re.compile(r"\b(Dr|Prof|Professor|Mr|Ms|Mrs|Instructor|Lecturer)\.?\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?")
PLATFORM = re.compile(r"\b(OMSCS|OMSCA|Georgia\s*Tech|GaTech|GA\s*Tech|Georgia\s*Institute\s*of\s*Technology|Udacity|Coursera|edX|Piazza|Canvas|Ed\s*Discussion|Gradescope|Bonnie|Slack|Udemy)\b", re.I)
# course titles seen in the two sides
COURSE_TITLES = re.compile(r"\b(Graduate Introduction to Operating Systems|Computer Networks|Database Systems Concepts and Design|Machine Learning|Operating Systems|Digital Marketing)\b", re.I)


def scrub(t):
    t = CODE.sub("[COURSE]", t)
    t = COURSE_TITLES.sub("[COURSE]", t)
    t = TITLE.sub("[INSTRUCTOR]", t)
    t = PLATFORM.sub("[PLATFORM]", t)
    return t


SENT = re.compile(r"(?<=[.!?])\s+")


def sents(texts, mw=4):
    out = []
    for t in texts:
        for s in SENT.split(" ".join(str(t).split())):
            if len(s.split()) >= mw:
                out.append(s.strip())
    return out


def load_real():
    return [r["review_text"].strip() for r in csv.DictReader(open(REAL, encoding="utf-8"))
            if (r.get("review_text") or "").strip() and r["review_text"].strip().lower() != "nan"][:200]


def load_synth(n, seed=42):
    ids = set(json.load(open(IDS))["incomplete_sample_ids"])
    out = []
    for l in open(GEN, encoding="utf-8"):
        r = json.loads(l)
        if str(r.get("sample_id")) not in ids:
            t = str(r.get("text", ""))
            if t and t.lower() != "nan":
                out.append(t)
    random.Random(seed).shuffle(out)
    return out[:n]


def load_herath():
    return [str(json.loads(l).get("text", "")) for l in open(HERATH, encoding="utf-8")]


def mv(p, q):
    return round(float(mauve.compute_mauve(p_text=p, q_text=q, device_id=-1, max_text_length=96,
                                           verbose=False, featurize_model_name="gpt2", batch_size=32).mauve), 4)


def main():
    rng = random.Random(1)
    oms = sents(load_real()); syn = sents(load_synth(800)); her = [s for s in load_herath() if len(s.split()) >= 4]
    for x in (oms, syn, her):
        rng.shuffle(x)
    N = min(1000, len(oms), len(syn), len(her)); half = N // 2
    print(f"N={N}/side")
    res = {}
    for tag, fn in [("raw", lambda x: x), ("scrubbed", scrub)]:
        o = [fn(s) for s in oms]; s_ = [fn(s) for s in syn]; h = [fn(s) for s in her]
        res[tag] = {
            "oms_vs_synth": mv(o[:N], s_[:N]),
            "oms_vs_oms_ceiling": mv(o[:half], o[half:2 * half]),
            "oms_vs_herath_realreal": mv(o[:N], h[:N]),
        }
        print(f"[{tag:9}] synth={res[tag]['oms_vs_synth']}  ceiling={res[tag]['oms_vs_oms_ceiling']}  real-real={res[tag]['oms_vs_herath_realreal']}")
    # how much did scrubbing move the synth gap?
    d = res["scrubbed"]["oms_vs_synth"] - res["raw"]["oms_vs_synth"]
    print(f"scrub effect on synth MAUVE: {d:+.4f} (positive = gap was partly entity/topic, not style)")
    res["scrub_effect_synth"] = round(d, 4)
    json.dump(res, open(ROOT / "paper/outputs/mauve_scrubbed.json", "w"), indent=2)


if __name__ == "__main__":
    main()
