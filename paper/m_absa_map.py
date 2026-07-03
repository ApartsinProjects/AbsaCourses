"""Map M-ABSA (Coursera, English) sentence-level triplets to the 20-aspect
project schema and write a mapped JSONL in the transfer-eval format
{text, aspects:{<project_aspect>: <polarity>}}.

Conservative category mapping (analogous to HERATH_TO_PROJECT / EduRABSA).
"""
import ast, glob, json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "external_data/M-ABSA_coursera"
OUT = ROOT / "external_data/M-ABSA_coursera/m_absa_mapped.jsonl"

MAP = {
    "course general": "overall_experience", "course quality": "overall_experience",
    "course comprehensiveness": "difficulty", "course workload": "workload",
    "course relatability": "relevance", "course value": "relevance",
    "material quality": "materials", "material quantity": "materials",
    "material comprehensiveness": "materials", "material relatability": "materials",
    "material workload": "workload",
    "presentation quality": "clarity", "presentation comprehensiveness": "clarity",
    "presentation quantity": "organization", "presentation relatability": "clarity",
    "faculty general": "lecturer_quality", "faculty comprehensiveness": "lecturer_quality",
    "faculty response": "support", "faculty relatability": "lecturer_quality",
    "faculty value": "lecturer_quality",
    "assignments quality": "assessment_design", "assignments quantity": "workload",
    "assignments comprehensiveness": "assessment_design",
    "assignments relatability": "assessment_design", "assignments workload": "workload",
    "grades general": "grading_transparency",
}
POL = {"positive": "positive", "negative": "negative", "neutral": "neutral"}


def main():
    n_sent = n_kept = 0
    asp_ct, pol_ct = Counter(), Counter()
    rows = []
    for f in sorted(glob.glob(str(SRC / "*.txt"))):
        for line in open(f, encoding="utf-8"):
            line = line.rstrip("\n")
            if "####" not in line:
                continue
            sent, labs = line.split("####", 1)
            sent = sent.strip()
            n_sent += 1
            try:
                trips = ast.literal_eval(labs)
            except Exception:
                continue
            # collapse to {aspect: [polarities]}
            acc = {}
            for t in trips:
                if len(t) < 3:
                    continue
                a = MAP.get(str(t[1]).strip().lower())
                p = POL.get(str(t[2]).strip().lower())
                if a and p:
                    acc.setdefault(a, []).append(p)
            # majority polarity per aspect; drop ties
            aspects = {}
            for a, ps in acc.items():
                c = Counter(ps).most_common()
                if len(c) == 1 or c[0][1] > c[1][1]:
                    aspects[a] = c[0][0]
            if not aspects or len(sent.split()) < 3:
                continue
            n_kept += 1
            for a, p in aspects.items():
                asp_ct[a] += 1; pol_ct[p] += 1
            rows.append({"text": sent, "aspects": aspects})
    with open(OUT, "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"sentences={n_sent} kept(with >=1 mapped aspect)={n_kept} -> {OUT}")
    print(f"distinct project aspects covered: {len(asp_ct)}")
    print("aspect coverage:", dict(asp_ct.most_common()))
    print("polarity:", dict(pol_ct))


if __name__ == "__main__":
    main()
