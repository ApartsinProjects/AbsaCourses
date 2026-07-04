"""Matched INDEPENDENT-structure baseline for Exp4/Exp5.

Same pipeline as exp4_generate.py / exp5_generate.py (same OpenRouter gpt-5-nano,
same template, same seed attributes, same aspect-count distribution), but using the
ORIGINAL independent sampler: aspect sets drawn uniformly and per-aspect polarity
drawn independently. This is the correct control for the structure experiments -
it differs from the correlated corpus ONLY in polarity independence and from the
structured corpus ONLY in polarity independence + uniform co-occurrence, with the
generation model/API/date/cleanliness all held fixed.

Outputs paper/outputs/exp_indep_gen.jsonl.  Usage: python exp_indep_generate.py --limit 2000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
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
OUT = ROOT / "paper/outputs/exp_indep_gen.jsonl"
MODEL = "openai/gpt-5-nano"
ASPECTS = list(METADATA["aspect_inventory"].keys())
DIST = METADATA["recommended_aspect_count_distribution"]


def load_key():
    for line in Path("E:/Projects/.env.all").read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("OPENROUTER_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OPENROUTER_API_KEY not found")


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
                    return {"text": txt, "aspects": task["aspects"], "target_attributes": task["aspects"],
                            "nuance_attributes": {}, "course_name": "", "grade": "", "style": ""}
            except Exception as e:
                if attempt == retries - 1:
                    return {"text": "", "aspects": task["aspects"], "error": str(e)[:120]}
                await asyncio.sleep(1.5 * (attempt + 1))
    return {"text": "", "aspects": task["aspects"]}


async def main(limit, conc):
    from openai import AsyncOpenAI
    rng = random.Random(3131)  # distinct stream from corr(4242)/struct(5757)
    tasks = []
    for _ in range(limit):
        labels = obp.sample_aspect_labels(rng, ASPECTS, DIST)  # ORIGINAL: uniform aspects + independent polarity
        attrs = obp.sample_attributes(SCHEMA, rng)
        tasks.append({"prompt": obp.render_prompt(TEMPLATE, labels, attrs), "aspects": labels})
    print(f"[indepgen] {len(tasks)} prompts (conc={conc})", flush=True)
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=load_key())
    sem = asyncio.Semaphore(conc)
    rows = await asyncio.gather(*[one(client, sem, t) for t in tasks])
    ok = [r for r in rows if r.get("text")]
    with OUT.open("w", encoding="utf-8") as f:
        for r in ok:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    multi = [list(r["aspects"].values()) for r in ok if len(r["aspects"]) >= 2]
    allsame = sum(len(set(m)) == 1 for m in multi) / max(len(multi), 1)
    print(f"[indepgen] generated {len(ok)}/{len(tasks)}; multi all-same-polarity={allsame:.2f} "
          f"(should be ~0.22 like original) -> {OUT}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--conc", type=int, default=8)
    args = ap.parse_args()
    asyncio.run(main(args.limit, args.conc))
