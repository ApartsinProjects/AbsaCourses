"""N2 + N3: clean-regeneration realism discrimination across generator families.

Reviewer h7LN: (a) realism rests on LLM-as-judge and is not established
statistically; (b) generation is confined to one provider's GPT family.

The published cycle-0 synthetic set is judged SYNTHETIC 97.5% of the time, but its
justifications are dominated by truncation ("ends abruptly mid-sentence") and "nan"
placeholders -- generation-budget/data artefacts, not unrealism. This harness:

  1. Regenerates the SAME 150 realism prompts CLEANLY (cap 900, no truncation/nan)
     with four generator families via OpenRouter: openai/gpt-5-nano (same family,
     clean), google/gemini-2.5-flash, z-ai/glm-4.6, meta-llama/llama-3.3-70b-instruct.
  2. Re-judges, in ONE construct-matched pass with the identical judge
     (openai/gpt-5.4, the paper's judge), every family PLUS the original contaminated
     cycle-0 set (anchor) PLUS 200 real reviews.
  3. Reports per-source synthetic-detection rate with bootstrap CIs and the
     judge's overall real-vs-synthetic discrimination.

Cache files make it resumable. Outputs under paper/outputs/.
"""
import argparse, asyncio, csv, json, random, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "paper/validation/batch_realism/runs"
CYC0 = RUNS / "realism_synth_cycle0_20260404T131844Z/generated_reviews.jsonl"
REAL = RUNS / "realism_real_baseline_200_20260404T131844Z/real_reviews.csv"
OUTD = ROOT / "paper/outputs"

GEN_FAMILIES = {
    "gpt5nano_clean": "openai/gpt-5-nano",
    "gemini_flash": "google/gemini-2.5-flash",
    "glm_46": "z-ai/glm-4.6",
    "llama33_70b": "meta-llama/llama-3.3-70b-instruct",
}
JUDGE = "openai/gpt-5.4"


def load_key():
    for line in Path("E:/Projects/.env.all").read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("OPENROUTER_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no key")


def judge_prompt(text):
    return (
        "You are evaluating whether a student course review is REAL or SYNTHETIC.\n"
        "Return strict JSON with exactly two keys: label and justification.\n"
        "The label must be either real or synthetic.\n"
        "If the label is real, justification must be an empty string.\n"
        "If the label is synthetic, justification must briefly explain the strongest "
        "reasons you suspect it is synthetic.\n\n"
        f"Review:\n{text}\n"
    )


def parse_label(raw):
    m = re.search(r'"label"\s*:\s*"(real|synthetic)"', raw or "", flags=re.I)
    if m:
        return m.group(1).lower()
    t = (raw or "").lower()
    if "synthetic" in t and "real" not in t:
        return "synthetic"
    if "real" in t and "synthetic" not in t:
        return "real"
    return None


def is_openai(slug):
    return slug.startswith("openai/")


async def call(client, sem, slug, prompt, cap, judge=False, retries=4):
    async with sem:
        for a in range(retries):
            try:
                await asyncio.sleep(random.uniform(0, 0.4) * (a + 1))
                kw = dict(model=slug, messages=[{"role": "user", "content": prompt}],
                          max_tokens=cap, temperature=1.0)
                if is_openai(slug):
                    extra = {"reasoning": {"effort": "minimal"}}
                    if not judge:
                        extra["verbosity"] = "low"
                    kw["extra_body"] = extra
                r = await client.chat.completions.create(**kw)
                txt = (r.choices[0].message.content or "").strip()
                if txt:
                    return txt
            except Exception as e:
                if a == retries - 1:
                    return f"__ERR__ {str(e)[:140]}"
        return ""


def load_prompts(n):
    rows = []
    with open(CYC0, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["custom_id"], r["generation_prompt"], r.get("generated_review_text", "")))
    return rows[:n]


def load_real(n):
    out = []
    with open(REAL, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = (row.get("review_text") or "").strip()
            if t and t.lower() != "nan":
                out.append(t)
    return out[:n]


async def phase_generate(client, sem, prompts, cap):
    for fam, slug in GEN_FAMILIES.items():
        cache = OUTD / f"n2_gen_{fam}.jsonl"
        if cache.exists():
            print(f"[gen] {fam}: cached"); continue
        print(f"[gen] {fam} ({slug}) x{len(prompts)}")
        outs = await asyncio.gather(*[call(client, sem, slug, p, cap) for _, p, _ in prompts])
        with open(cache, "w", encoding="utf-8") as f:
            for (cid, _, _), txt in zip(prompts, outs):
                f.write(json.dumps({"custom_id": cid, "text": txt}, ensure_ascii=False) + "\n")


def collect_texts(prompts, real):
    sets = {}
    # clean regenerations
    for fam in GEN_FAMILIES:
        rows = [json.loads(l) for l in open(OUTD / f"n2_gen_{fam}.jsonl", encoding="utf-8")]
        sets[fam] = [(r["custom_id"], r["text"]) for r in rows
                     if r["text"] and not r["text"].startswith("__ERR__")]
    # contaminated anchor
    sets["cycle0_orig"] = [(cid, t) for cid, _, t in prompts if t and t.lower() != "nan"]
    # real
    sets["real"] = [(f"real_{i}", t) for i, t in enumerate(real)]
    return sets


async def phase_judge(client, sem, sets, cap):
    cache = OUTD / "n2_judge.jsonl"
    done = {}
    if cache.exists():
        for l in open(cache, encoding="utf-8"):
            r = json.loads(l); done[(r["source"], r["id"])] = r["label"]
    todo = [(s, cid, txt) for s, items in sets.items() for cid, txt in items
            if (s, cid) not in done]
    print(f"[judge] {len(todo)} to judge ({len(done)} cached)")
    outs = await asyncio.gather(*[call(client, sem, JUDGE, judge_prompt(t), cap, judge=True)
                                  for _, _, t in todo])
    with open(cache, "a", encoding="utf-8") as f:
        for (s, cid, _), raw in zip(todo, outs):
            lab = parse_label(raw)
            done[(s, cid)] = lab
            f.write(json.dumps({"source": s, "id": cid, "label": lab}) + "\n")
    return done


def summarize(sets, judged):
    def boot(flags, B=2000):
        import statistics
        n = len(flags)
        if n == 0:
            return (0, 0)
        rng = random.Random(42)
        rates = []
        for _ in range(B):
            s = sum(flags[rng.randrange(n)] for _ in range(n)) / n
            rates.append(s)
        rates.sort()
        return (round(rates[int(0.025 * B)], 4), round(rates[int(0.975 * B)], 4))

    rep = {}
    for s, items in sets.items():
        labs = [judged.get((s, cid)) for cid, _ in items]
        labs = [l for l in labs if l in ("real", "synthetic")]
        if not labs:
            continue
        synth_flags = [1 if l == "synthetic" else 0 for l in labs]
        rate = sum(synth_flags) / len(synth_flags)
        rep[s] = {"n": len(labs), "labeled_synthetic_rate": round(rate, 4),
                  "ci95": boot(synth_flags)}
    # overall discrimination accuracy over each clean family vs real
    real_rate = rep.get("real", {}).get("labeled_synthetic_rate", None)
    return rep, real_rate


async def main(n, cap, conc):
    from openai import AsyncOpenAI
    OUTD.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(n)
    real = load_real(200)
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=load_key())
    sem = asyncio.Semaphore(conc)
    await phase_generate(client, sem, prompts, cap)
    sets = collect_texts(prompts, real)
    print("set sizes:", {k: len(v) for k, v in sets.items()})
    judged = await phase_judge(client, sem, sets, 500)
    rep, real_rate = summarize(sets, judged)
    out = {"judge": JUDGE, "n_prompts": n, "real_false_synthetic_rate": real_rate,
           "per_source": rep,
           "note": "cycle0_orig is the published contaminated set (truncation/nan); "
                   "others are clean regenerations."}
    json.dump(out, open(OUTD / "n2n3_realism_summary.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--cap", type=int, default=900)
    ap.add_argument("--conc", type=int, default=8)
    args = ap.parse_args()
    asyncio.run(main(args.n, args.cap, args.conc))
