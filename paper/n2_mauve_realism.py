"""N2 realism (distributional): MAUVE between real and synthetic reviews.

MAUVE (Pillutla et al., NeurIPS 2021) measures how close a generated-text
distribution is to a human-text distribution via quantized language-model
features; it is the standard single-number realism metric and is high for
faithful synthetic text even when a surface classifier can still separate.

We report MAUVE(real course reviews, synthetic) against an upper-bound
reference MAUVE(real split A, real split B), plus a cross-domain reference
MAUVE(real course reviews, RateMyProfessor) to contextualize the scale.
"""
import json, random
from pathlib import Path
import mauve

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "paper/generated_datasets/batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl"
REGEN = ROOT / "paper/outputs/n1_regenerated_841.jsonl"
IDS = ROOT / "paper/outputs/rc2_incomplete_row_ids.json"
OMSCS = ROOT / "paper/validation/batch_realism/runs/realism_real_baseline_200_20260404T131844Z/real_reviews.csv"
HERATH = ROOT / "paper/real_transfer/herath_mapped_real_reviews.jsonl"
KAGGLE = ROOT / "external_data/Student_feedback_analysis_dataset/Annotated Student Feedback Data"
OUT = ROOT / "paper/outputs/n2_mauve_realism.json"
SEED = 42


def clean(t):
    t = " ".join(str(t).split())
    return t if len(t.split()) >= 5 and t.lower() != "nan" else None


def synth_pool(n):
    ids = set(json.load(open(IDS))["incomplete_sample_ids"])
    comp = []
    for line in open(GEN, encoding="utf-8"):
        r = json.loads(line)
        if str(r.get("sample_id")) not in ids:
            c = clean(r.get("text", ""))
            if c: comp.append(c)
    regen = [c for c in (clean(r.get("new_text", "")) for r in map(json.loads, open(REGEN, encoding="utf-8"))) if c]
    rng = random.Random(SEED); rng.shuffle(comp)
    return (comp[: max(0, n - len(regen))] + regen)[:n]


def real_course(n):
    import csv
    out = []
    for row in csv.DictReader(open(OMSCS, encoding="utf-8")):
        c = clean(row.get("review_text", ""))
        if c: out.append(c)
    her = [c for c in (clean(json.loads(l).get("text", "")) for l in open(HERATH, encoding="utf-8")) if c]
    rng = random.Random(SEED); rng.shuffle(her)
    return out + her[: max(0, n - len(out))]


def kaggle_pool(n):
    s = set()
    for tsv in KAGGLE.rglob("*.tsv"):
        for line in tsv.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = line.split("\t")
            if len(p) >= 2:
                c = clean(p[1])
                if c: s.add(c)
    out = sorted(s); random.Random(SEED).shuffle(out)
    return out[:n]


def mv(p, q):
    r = mauve.compute_mauve(p_text=p, q_text=q, device_id=-1,
                            max_text_length=256, verbose=False,
                            featurize_model_name="gpt2", batch_size=16)
    return round(float(r.mauve), 4)


def main():
    N = 400
    synth = synth_pool(N)
    real = real_course(N)
    kag = kaggle_pool(N)
    half = min(len(real) // 2, 200)
    ra, rb = real[:half], real[half:2 * half]
    res = {
        "n_per_side": N,
        "mauve_real_vs_synth": mv(real, synth),
        "mauve_real_vs_real_upperbound": mv(ra, rb),
        "mauve_real_vs_kaggle_rmp_crossdomain": mv(real, kag),
        "n_synth": len(synth), "n_real": len(real), "n_kaggle": len(kag),
    }
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
