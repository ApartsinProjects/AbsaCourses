"""N3: the generate->audit->filter pipeline is generator-agnostic.

Reviewer h7LN: generation is confined to a single provider's GPT family. We
regenerate the same 150 label-conditioned prompts with four generator families
spanning four providers via OpenRouter (OpenAI gpt-5-nano, Google gemini-2.5-flash,
Zhipu glm-4.6, Meta llama-3.3-70b) and run the identical label-fidelity audit
(gpt-5.2) on each. Comparable support / sentiment-match rates show the audit and
the pipeline are not specific to the GPT family.

Cache-based / resumable. Outputs paper/outputs/n3_generator_fidelity_summary.json .
"""
import argparse, asyncio, json, random, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CYC0 = ROOT / "paper/validation/batch_realism/runs/realism_synth_cycle0_20260404T131844Z/generated_reviews.jsonl"
OUTD = ROOT / "paper/outputs"

FAMILIES = {
    "gpt5nano": "openai/gpt-5-nano",
    "gemini_flash": "google/gemini-2.5-flash",
    "glm_46": "z-ai/glm-4.6",
    "llama33_70b": "meta-llama/llama-3.3-70b-instruct",
}
AUDITOR = "openai/gpt-5.2"
AUDIT_PROMPT = """You are auditing whether aspect-sentiment labels are faithful to a student course review.

Review:
{text}

Declared labels:
{labels}

Return JSON only with this schema:
{{
  "aspects": [
    {{"aspect": "aspect_name", "supported": true, "sentiment_match": true, "note": "short explanation"}}
  ],
  "overall_note": "one short sentence"
}}

Rules:
- "supported" means the review text clearly expresses that aspect.
- "sentiment_match" means the declared sentiment matches the review text for that aspect.
- If an aspect is not supported, set sentiment_match to false.
- Be strict and conservative.
"""


def load_key():
    for line in Path("E:/Projects/.env.all").read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("OPENROUTER_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")


async def call(client, sem, slug, prompt, cap, judge=False, retries=4):
    async with sem:
        for a in range(retries):
            try:
                await asyncio.sleep(random.uniform(0, 0.4) * (a + 1))
                kw = dict(model=slug, messages=[{"role": "user", "content": prompt}],
                          max_tokens=cap, temperature=1.0)
                if slug.startswith("openai/"):
                    extra = {"reasoning": {"effort": "minimal"}}
                    if not judge:
                        extra["verbosity"] = "low"
                    kw["extra_body"] = extra
                elif slug.startswith("z-ai/"):
                    kw["extra_body"] = {"reasoning": {"enabled": False}}
                r = await client.chat.completions.create(**kw)
                t = (r.choices[0].message.content or "").strip()
                if t:
                    return t
            except Exception as e:
                if a == retries - 1:
                    return ""
        return ""


def load_rows(n):
    out = []
    for line in open(CYC0, encoding="utf-8"):
        r = json.loads(line)
        labs = r.get("aspect_labels")
        labs = json.loads(labs) if isinstance(labs, str) else (labs or {})
        if labs:
            out.append({"cid": r["custom_id"], "prompt": r["generation_prompt"], "labels": labs})
    return out[:n]


def labels_block(labs):
    return "\n".join(f"- {a}: {s}" for a, s in labs.items())


def parse_audit(raw, labs):
    try:
        m = re.search(r"\{.*\}", raw or "", flags=re.S)
        obj = json.loads(m.group(0)) if m else {}
        got = {str(a.get("aspect", "")).lower(): a for a in obj.get("aspects", [])}
    except Exception:
        got = {}
    sup = sm = 0
    for a in labs:
        e = got.get(a.lower())
        if e:
            sup += 1 if e.get("supported") else 0
            sm += 1 if e.get("sentiment_match") else 0
    return sup, sm, len(labs)


async def gen_phase(client, sem, rows, cap):
    for fam, slug in FAMILIES.items():
        cache = OUTD / f"n3_gen_{fam}.jsonl"
        if cache.exists() and sum(1 for _ in open(cache)) >= len(rows):
            print(f"[gen] {fam}: cached"); continue
        print(f"[gen] {fam} x{len(rows)}")
        outs = await asyncio.gather(*[call(client, sem, slug, r["prompt"], cap) for r in rows])
        with open(cache, "w", encoding="utf-8") as f:
            for r, t in zip(rows, outs):
                f.write(json.dumps({"cid": r["cid"], "labels": r["labels"], "text": t}, ensure_ascii=False) + "\n")


async def audit_phase(client, sem, fam):
    gen = [json.loads(l) for l in open(OUTD / f"n3_gen_{fam}.jsonl", encoding="utf-8")]
    gen = [g for g in gen if g["text"]]
    cache = OUTD / f"n3_audit_{fam}.jsonl"
    done = {}
    if cache.exists():
        for l in open(cache, encoding="utf-8"):
            r = json.loads(l); done[r["cid"]] = (r["sup"], r["sm"], r["tot"])
    todo = [g for g in gen if g["cid"] not in done]
    if todo:
        prompts = [AUDIT_PROMPT.format(text=g["text"], labels=labels_block(g["labels"])) for g in todo]
        outs = await asyncio.gather(*[call(client, sem, AUDITOR, p, 600, judge=True) for p in prompts])
        with open(cache, "a", encoding="utf-8") as f:
            for g, raw in zip(todo, outs):
                sup, sm, tot = parse_audit(raw, g["labels"])
                done[g["cid"]] = (sup, sm, tot)
                f.write(json.dumps({"cid": g["cid"], "sup": sup, "sm": sm, "tot": tot}) + "\n")
    tot = sum(v[2] for v in done.values())
    sup = sum(v[0] for v in done.values())
    sm = sum(v[1] for v in done.values())
    return {"n_reviews": len(done), "n_aspects": tot,
            "support_rate": round(sup / max(tot, 1), 4),
            "sentiment_match_rate": round(sm / max(tot, 1), 4)}


async def main(n, conc, cap):
    from openai import AsyncOpenAI
    OUTD.mkdir(parents=True, exist_ok=True)
    rows = load_rows(n)
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=load_key())
    sem = asyncio.Semaphore(conc)
    await gen_phase(client, sem, rows, cap)
    rep = {}
    for fam in FAMILIES:
        rep[fam] = {"slug": FAMILIES[fam], **await audit_phase(client, sem, fam)}
        print(f"[audit] {fam:14s} support={rep[fam]['support_rate']}  "
              f"sentiment_match={rep[fam]['sentiment_match_rate']}  (n_aspects={rep[fam]['n_aspects']})")
    out = {"auditor": AUDITOR, "n_prompts": n, "per_family": rep}
    json.dump(out, open(OUTD / "n3_generator_fidelity_summary.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--cap", type=int, default=900)
    args = ap.parse_args()
    asyncio.run(main(args.n, args.conc, args.cap))
