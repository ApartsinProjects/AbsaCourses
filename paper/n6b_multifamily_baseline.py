"""#6b: multi-family zero-shot ABSA baseline (construct-matched).

Reviewer h7LN: the prompted baseline is confined to one provider (GPT) and one
prompting method. We co-compute the SAME zero-shot-glossary ABSA baseline the
paper reports for gpt-5.4, on the SAME synthetic test split (seed 42), for four
families across four providers via OpenRouter, in one scoring pass:
OpenAI gpt-5.4, Google gemini-2.5-flash, Zhipu glm-4.6, Meta llama-3.3-70b.

Reuses absa_model_comparison.build_openai_prompt (the paper's exact prompt) and
the same per-aspect detection / sentiment scoring. Outputs paper/outputs/.
"""
import argparse, asyncio, json, random, re
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from absa_model_comparison import (load_jsonl, three_way_split, discover_aspects,
                                   build_openai_prompt, SENT2VAL)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper/generated_datasets/batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl"
OUTD = ROOT / "paper/outputs"
FAMILIES = {
    "gpt5.4": "openai/gpt-5.4",
    "gemini_flash": "google/gemini-2.5-flash",
    "glm_46": "z-ai/glm-4.6",
    "llama33_70b": "meta-llama/llama-3.3-70b-instruct",
}
JSON_DIRECTIVE = ("\n\nReturn ONLY a JSON object mapping each aspect you find to one of "
                  '"negative", "neutral", or "positive". Omit aspects not expressed. '
                  'Example: {"workload": "negative", "clarity": "positive"}.')


def load_key():
    for line in Path("E:/Projects/.env.all").read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("OPENROUTER_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")


def parse_map(raw, aspects):
    aset = {a.lower(): a for a in aspects}
    out = {}
    ok = ("negative", "neutral", "positive")
    m = re.search(r"\{.*\}", raw or "", flags=re.S)
    if not m:
        return out
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return out
    items = obj.get("aspects") if isinstance(obj, dict) else None
    if isinstance(items, list):  # {"aspects":[{"aspect":..,"sentiment":..}]}
        for e in items:
            if isinstance(e, dict):
                ak = aset.get(str(e.get("aspect", "")).strip().lower())
                vs = str(e.get("sentiment", "")).strip().lower()
                if ak and vs in ok:
                    out[ak] = vs
    elif isinstance(obj, dict):  # flat {"aspect":"sentiment"}
        for k, v in obj.items():
            ak = aset.get(str(k).strip().lower())
            vs = str(v).strip().lower()
            if ak and vs in ok:
                out[ak] = vs
    return out


async def call(client, sem, slug, prompt, retries=4):
    async with sem:
        for a in range(retries):
            try:
                await asyncio.sleep(random.uniform(0, 0.4) * (a + 1))
                kw = dict(model=slug, messages=[{"role": "user", "content": prompt}],
                          max_tokens=1400, temperature=0.0)
                if slug.startswith("openai/"):
                    kw["extra_body"] = {"reasoning": {"effort": "minimal"}}
                elif slug.startswith("z-ai/"):
                    kw["extra_body"] = {"reasoning": {"enabled": False}}
                r = await client.chat.completions.create(**kw)
                t = (r.choices[0].message.content or "").strip()
                if t:
                    return t
            except Exception:
                if a == retries - 1:
                    return ""
        return ""


def score(rows, preds, aspects):
    a2i = {a: i for i, a in enumerate(aspects)}
    Yt = np.zeros((len(rows), len(aspects)), dtype=int)
    Yp = np.zeros((len(rows), len(aspects)), dtype=int)
    st, sp = [], []
    for r, (gold, pred) in enumerate(zip(rows, preds)):
        for a in gold:
            if a in a2i: Yt[r, a2i[a]] = 1
        for a in pred:
            if a in a2i: Yp[r, a2i[a]] = 1
        for a in pred:
            if a in gold:
                st.append(SENT2VAL[gold[a]]); sp.append(SENT2VAL[pred[a]])
    yt, yp = Yt.flatten(), Yp.flatten()
    st, sp = np.array(st), np.array(sp)
    return {
        "detect_micro_f1": round(float(f1_score(yt, yp, zero_division=0)), 4),
        "detect_precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
        "detect_recall": round(float(recall_score(yt, yp, zero_division=0)), 4),
        "sentiment_accuracy_on_matched": round(float((st == sp).mean()), 4) if len(st) else 0.0,
        "sentiment_mse_on_matched": round(float(((st - sp) ** 2).mean()), 4) if len(st) else 0.0,
        "n_matched_aspects": int(len(st)),
    }


async def main(n, conc):
    from openai import AsyncOpenAI
    OUTD.mkdir(parents=True, exist_ok=True)
    df = load_jsonl(DATA)
    _, _, te = three_way_split(df, 0.10, 0.10, 42)
    aspects = discover_aspects(df)
    te = te.head(n).reset_index(drop=True)
    golds = [dict(te.iloc[i]["aspects"]) for i in range(len(te))]
    prompts = [build_openai_prompt(str(te.iloc[i]["text"]), aspects, "zero-shot-glossary", []) + JSON_DIRECTIVE
               for i in range(len(te))]
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=load_key())
    sem = asyncio.Semaphore(conc)
    report = {}
    for fam, slug in FAMILIES.items():
        cache = OUTD / f"n6b_pred_{fam}.jsonl"
        if cache.exists() and sum(1 for _ in open(cache)) >= len(prompts):
            raws = [json.loads(l)["raw"] for l in open(cache, encoding="utf-8")]
        else:
            raws = await asyncio.gather(*[call(client, sem, slug, p) for p in prompts])
            with open(cache, "w", encoding="utf-8") as f:
                for rr in raws:
                    f.write(json.dumps({"raw": rr}, ensure_ascii=False) + "\n")
        preds = [parse_map(rr, aspects) for rr in raws]
        n_parsed = sum(1 for p in preds if p)
        report[fam] = {"slug": slug, "n_test": len(te), "n_parsed": n_parsed, **score(golds, preds, aspects)}
        print(f"[{fam:13s}] micro-F1={report[fam]['detect_micro_f1']} "
              f"sent_acc={report[fam]['sentiment_accuracy_on_matched']} parsed={n_parsed}/{len(te)}")
    out = {"variant": "zero-shot-glossary", "seed": 42, "n_test_rows": len(te),
           "n_aspects": len(aspects), "per_family": report}
    json.dump(out, open(OUTD / "n6b_multifamily_baseline_summary.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--conc", type=int, default=8)
    args = ap.parse_args()
    asyncio.run(main(args.n, args.conc))
