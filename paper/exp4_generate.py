"""Experiment 4 (part A) - regenerate synthetic reviews with CORRELATED sentiment.

The original generator drew each aspect's polarity independently
(sample_aspect_labels -> rng.choice(SENTIMENTS)), producing unnaturally balanced
reviews (A.25: only 22% all-same vs 50-72% in real). Here we first draw a
review-level disposition (positive / negative / mixed) at the real frequency,
then condition per-aspect polarities on it. Everything else (aspect-count
distribution, seed attributes, prompt template, length bands) is unchanged, so
the only manipulated variable is sentiment structure.

Outputs paper/outputs/exp4_correlated_gen.jsonl with {text, aspects, ...}.
Usage: python exp4_generate.py --limit 1500 [--conc 8]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GP = ROOT / "paper/archive/20260404_binary_polarity_reset/generation_protocol"
# Reuse the original prep module's loaders/samplers, but point its paths at the archive.
import openai_batch_prep as obp  # noqa: E402
obp.SCHEMA_PATH = GP / "seed_attribute_schema.json"
obp.TEMPLATE_PATH = GP / "final_realism_prompt_template.txt"
obp.METADATA_PATH = GP / "final_realism_prompt_metadata.json"
TEMPLATE = obp.load_template()
METADATA = obp.load_metadata()
SCHEMA = obp.load_schema()
OUT = ROOT / "paper/outputs/exp4_correlated_gen.jsonl"
MODEL = "openai/gpt-5-nano"

# Disposition frequencies chosen to match real reviews (A.25: all-same 0.50-0.72,
# mixed 0.13-0.24). Averages to ~0.6 consistent / ~0.2 mixed.
DISPOSITION = {"positive": 0.52, "negative": 0.13, "mixed": 0.35}
# per-aspect polarity given disposition
COND = {
    "positive": (["positive", "neutral", "negative"], [0.85, 0.10, 0.05]),
    "negative": (["negative", "neutral", "positive"], [0.80, 0.12, 0.08]),
    "mixed": (["positive", "negative", "neutral"], [0.45, 0.40, 0.15]),
}


def load_key():
    for line in Path("E:/Projects/.env.all").read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("OPENROUTER_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OPENROUTER_API_KEY not found")


def sample_aspect_labels_correlated(rng, aspects, distribution):
    options = [int(k) for k in distribution]
    weights = [distribution[str(k)] for k in options]
    n = rng.choices(options, weights=weights, k=1)[0]
    selected = rng.sample(aspects, n)
    disp = rng.choices(list(DISPOSITION), weights=list(DISPOSITION.values()), k=1)[0]
    labels, wts = COND[disp]
    return {a: rng.choices(labels, weights=wts, k=1)[0] for a in selected}, disp


def sample_attributes(rng):
    return obp.sample_attributes(SCHEMA, rng)


def render_prompt(aspect_labels, attributes):
    return obp.render_prompt(TEMPLATE, aspect_labels, attributes)


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
                    return {"text": "", "aspects": task["aspects"], "disposition": task["disp"], "error": str(e)[:120]}
                await asyncio.sleep(1.5 * (attempt + 1))
    return {"text": "", "aspects": task["aspects"], "disposition": task["disp"]}


async def main(limit, conc):
    from openai import AsyncOpenAI
    rng = random.Random(4242)
    aspects = list(METADATA["aspect_inventory"].keys())
    dist = METADATA["recommended_aspect_count_distribution"]
    tasks = []
    for i in range(limit):
        labels, disp = sample_aspect_labels_correlated(rng, aspects, dist)
        attrs = sample_attributes(rng)
        tasks.append({"prompt": render_prompt(labels, attrs), "aspects": labels, "disp": disp})
    print(f"[exp4gen] {len(tasks)} prompts (conc={conc}); disposition mix="
          f"{ {d: round(sum(1 for t in tasks if t['disp']==d)/len(tasks),2) for d in DISPOSITION} }", flush=True)
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=load_key())
    sem = asyncio.Semaphore(conc)
    rows = await asyncio.gather(*[one(client, sem, t) for t in tasks])
    ok = [r for r in rows if r.get("text")]
    with OUT.open("w", encoding="utf-8") as f:
        for r in ok:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # verify the manipulation took: within-review consistency of the LABELS
    multi = [list(r["aspects"].values()) for r in ok if len(r["aspects"]) >= 2]
    allsame = sum(len(set(m)) == 1 for m in multi) / max(len(multi), 1)
    print(f"[exp4gen] generated {len(ok)}/{len(tasks)}; multi-aspect all-same-polarity={allsame:.2f} "
          f"(target ~0.6; original synthetic 0.22) -> {OUT}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--conc", type=int, default=8)
    args = ap.parse_args()
    asyncio.run(main(args.limit, args.conc))
