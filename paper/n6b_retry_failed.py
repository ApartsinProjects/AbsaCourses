"""Patch only the FAILED calls in the n6b multi-provider caches.

A failed/empty call is stored as {"raw": ""} (call() returns "" after its
retries). This re-calls only those prompts per provider, merges the results
back into the same cache line-for-line, and recomputes the summary. Cheap:
touches only the drops, not the whole 1000-review split.

  python n6b_retry_failed.py            # patch all providers' empty rows
"""
import argparse, asyncio, json
from pathlib import Path

import n6b_multifamily_baseline as base
from n6b_multifamily_baseline import (FAMILIES, OUTD, DATA, JSON_DIRECTIVE, call,
                                      parse_map, score, build_openai_prompt)
from absa_model_comparison import load_jsonl, three_way_split, discover_aspects


async def main(n, conc):
    from openai import AsyncOpenAI
    df = load_jsonl(DATA)
    _, _, te = three_way_split(df, 0.10, 0.10, 42)
    aspects = discover_aspects(df)
    te = te.head(n).reset_index(drop=True)
    golds = [dict(te.iloc[i]["aspects"]) for i in range(len(te))]
    prompts = [build_openai_prompt(str(te.iloc[i]["text"]), aspects, "zero-shot-glossary", []) + JSON_DIRECTIVE
               for i in range(len(te))]
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=base.load_key())

    async def patch_model(fam, slug):
        cache = OUTD / f"n6b_pred_{fam}.jsonl"
        if not cache.exists():
            print(f"[{fam}] no cache, skipping"); return None
        raws = [json.loads(l)["raw"] for l in open(cache, encoding="utf-8")]
        raws = (raws + [""] * len(prompts))[:len(prompts)]
        failed = [i for i, r in enumerate(raws) if not (r or "").strip()]
        if failed:
            sem = asyncio.Semaphore(conc)
            new = await asyncio.gather(*[call(client, sem, slug, prompts[i]) for i in failed])
            fixed = sum(1 for x in new if (x or "").strip())
            for i, x in zip(failed, new):
                raws[i] = x
            with open(cache, "w", encoding="utf-8") as f:
                for rr in raws:
                    f.write(json.dumps({"raw": rr}, ensure_ascii=False) + "\n")
            print(f"[{fam:13s}] retried {len(failed)} empties, recovered {fixed}", flush=True)
        preds = [parse_map(rr, aspects) for rr in raws]
        n_parsed = sum(1 for p in preds if p)
        entry = {"slug": slug, "n_test": len(te), "n_parsed": n_parsed, **score(golds, preds, aspects)}
        print(f"[{fam:13s}] micro-F1={entry['detect_micro_f1']} parsed={n_parsed}/{len(te)}", flush=True)
        return fam, entry

    pairs = [p for p in await asyncio.gather(*[patch_model(f, s) for f, s in FAMILIES.items()]) if p]
    report = {fam: entry for fam, entry in pairs}
    out = {"variant": "zero-shot-glossary", "seed": 42, "n_test_rows": len(te),
           "n_aspects": len(aspects), "per_family": report}
    json.dump(out, open(OUTD / "n6b_multifamily_baseline_summary.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--conc", type=int, default=8)
    a = ap.parse_args()
    asyncio.run(main(a.n, a.conc))
