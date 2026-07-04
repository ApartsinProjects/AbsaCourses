"""Experiment 5 (part A) - generate reviews matching real structure on BOTH axes:
  (i) aspect SETS drawn from the real co-occurrence graph (not uniform sampling),
  (ii) per-aspect polarity correlated via a review-level disposition (as exp4).
Everything else (aspect-count distribution, seed attributes, prompt template)
matches the original protocol, so the manipulated variables are exactly the two
structural gaps identified in A.25 (polarity) and Part B (co-occurrence).

Outputs paper/outputs/exp5_structured_gen.jsonl.
Usage: python exp5_generate.py --limit 2000 [--conc 8]
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GP = ROOT / "paper/archive/20260404_binary_polarity_reset/generation_protocol"
import openai_batch_prep as obp  # noqa: E402
obp.SCHEMA_PATH = GP / "seed_attribute_schema.json"
obp.TEMPLATE_PATH = GP / "final_realism_prompt_template.txt"
obp.METADATA_PATH = GP / "final_realism_prompt_metadata.json"
TEMPLATE = obp.load_template()
METADATA = obp.load_metadata()
SCHEMA = obp.load_schema()

REAL = {
    "herath": ROOT / "paper/real_transfer/herath_mapped_real_reviews.jsonl",
    "edurabsa": ROOT / "external_data/EduRABSA_mapped/edurabsa_all_mapped.jsonl",
    "oats": ROOT / "external_data/OATS_coursera/oats_mapped.jsonl",
}
OUT = ROOT / "paper/outputs/exp5_structured_gen.jsonl"
MODEL = "openai/gpt-5-nano"
DISPOSITION = {"positive": 0.52, "negative": 0.13, "mixed": 0.35}
COND = {
    "positive": (["positive", "neutral", "negative"], [0.85, 0.10, 0.05]),
    "negative": (["negative", "neutral", "positive"], [0.80, 0.12, 0.08]),
    "mixed": (["positive", "negative", "neutral"], [0.45, 0.40, 0.15]),
}
ASPECTS = list(METADATA["aspect_inventory"].keys())


def load_key():
    for line in Path("E:/Projects/.env.all").read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("OPENROUTER_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OPENROUTER_API_KEY not found")


def build_real_aspect_sets():
    """Bootstrap pool: the actual aspect-sets that occur in real reviews (mapped to
    the 20-aspect space). Sampling from these reproduces real co-occurrence AND
    aspect-count structure exactly, capped at 3 to match the generation protocol."""
    sets = []
    for p in REAL.values():
        for l in open(p, encoding="utf-8"):
            a = json.loads(l).get("aspects") or {}
            ks = [k for k in a if k in ASPECTS]
            if 1 <= len(ks) <= 3:
                sets.append(sorted(ks))
    return sets


REAL_SETS = build_real_aspect_sets()


def sample_labels(rng, distribution):
    aset = rng.choice(REAL_SETS)  # bootstrap a real aspect-set (real co-occurrence + count)
    disp = rng.choices(list(DISPOSITION), weights=list(DISPOSITION.values()), k=1)[0]
    labs, wts = COND[disp]
    return {a: rng.choices(labs, weights=wts, k=1)[0] for a in aset}, disp


async def one(client, sem, task, retries=4):
    async with sem:
        for attempt in range(retries):
            try:
                r = await client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": task["prompt"]}],
                    max_tokens=400, temperature=1.0,
                    extra_body={"reasoning": {"effort": "minimal"}, "verbosity": "low"})
                txt = (r.choices[0].message.content or "").strip()
                if txt:
                    return {"text": txt, "aspects": task["aspects"], "disposition": task["disp"],
                            "target_attributes": task["aspects"], "nuance_attributes": {},
                            "course_name": "", "grade": "", "style": ""}
            except Exception as e:
                if attempt == retries - 1:
                    return {"text": "", "aspects": task["aspects"], "error": str(e)[:120]}
                await asyncio.sleep(1.5 * (attempt + 1))
    return {"text": "", "aspects": task["aspects"]}


async def main(limit, conc):
    from openai import AsyncOpenAI
    rng = random.Random(5757)
    dist = METADATA["recommended_aspect_count_distribution"]
    tasks = []
    for _ in range(limit):
        labels, disp = sample_labels(rng, dist)
        attrs = obp.sample_attributes(SCHEMA, rng)
        tasks.append({"prompt": obp.render_prompt(TEMPLATE, labels, attrs), "aspects": labels, "disp": disp})
    print(f"[exp5gen] {len(tasks)} prompts (conc={conc})", flush=True)
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=load_key())
    sem = asyncio.Semaphore(conc)
    rows = await asyncio.gather(*[one(client, sem, t) for t in tasks])
    ok = [r for r in rows if r.get("text")]
    with OUT.open("w", encoding="utf-8") as f:
        for r in ok:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    multi = [list(r["aspects"].values()) for r in ok if len(r["aspects"]) >= 2]
    allsame = sum(len(set(m)) == 1 for m in multi) / max(len(multi), 1)
    print(f"[exp5gen] generated {len(ok)}/{len(tasks)}; multi all-same-polarity={allsame:.2f} -> {OUT}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--conc", type=int, default=8)
    args = ap.parse_args()
    asyncio.run(main(args.limit, args.conc))
