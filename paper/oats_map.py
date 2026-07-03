"""Map OATS-ABSA Coursera review-level tuples to the 20-aspect project schema."""
import ast, glob, json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from m_absa_map import MAP  # same category scheme

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external_data/OATS_coursera/oats_mapped.jsonl"
POL = {"positive": "positive", "negative": "negative", "neutral": "neutral"}  # drop 'conflict'

rows = []; asp_ct = Counter(); n = 0
for f in sorted(glob.glob(str(ROOT / "external_data/OATS_coursera/tuples_*.txt"))):
    for line in open(f, encoding="utf-8"):
        line = line.rstrip("\n")
        if "####" not in line: continue
        txt, labs = line.split("####", 1); n += 1
        try: tups = ast.literal_eval(labs)
        except Exception: continue
        acc = {}
        for t in tups:
            if len(t) < 2: continue
            a = MAP.get(str(t[0]).strip().lower()); p = POL.get(str(t[1]).strip().lower())
            if a and p: acc.setdefault(a, []).append(p)
        aspects = {}
        for a, ps in acc.items():
            c = Counter(ps).most_common()
            if len(c) == 1 or c[0][1] > c[1][1]: aspects[a] = c[0][0]
        if not aspects or len(txt.split()) < 5: continue
        for a in aspects: asp_ct[a] += 1
        rows.append({"text": " ".join(txt.split()), "aspects": aspects})
with open(OUT, "w", encoding="utf-8") as fo:
    for r in rows: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"reviews={n} kept={len(rows)} aspects_covered={len(asp_ct)}")
print(dict(asp_ct.most_common()))
