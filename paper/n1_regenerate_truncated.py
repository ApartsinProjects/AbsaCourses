"""N1: regenerate the 841 output-token-truncated rows at a proper cap.

Reviewer h7LN framed the full-corpus length-band adherence (0.6819) and the 841
`max_output_tokens` truncations as a failure of the "controlled" corpus. This
script re-runs the *identical* generation prompts for exactly those 841 rows via
OpenRouter (openai/gpt-5-nano, same family) with the output cap raised from 300
to 1200 tokens, then recomputes length-band adherence on the regenerated rows.

It changes nothing in the released corpus; it is a standalone recoverability
demonstration (the shortfall is a generation-budget knob, not a validity defect).

Outputs:
  paper/outputs/n1_regenerated_841.jsonl      per-row regenerated text + band check
  paper/outputs/n1_adherence_summary.json     before/after adherence
"""
import argparse, asyncio, json, os, random, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "paper/batch_requests/dataset_generation_10k_v2_requests.jsonl"
GEN = ROOT / "paper/generated_datasets/batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl"
IDS = ROOT / "paper/outputs/rc2_incomplete_row_ids.json"
OUT = ROOT / "paper/outputs/n1_regenerated_841.jsonl"
SUM = ROOT / "paper/outputs/n1_adherence_summary.json"

LENGTH_BANDS = {
    "very short comment": (20, 45),
    "compact but informative review": (45, 85),
    "mid-length reflective review": (85, 140),
    "detailed review with one dominant complaint": (140, 220),
}
MODEL = "openai/gpt-5-nano"


def load_key():
    for line in (Path("E:/Projects/.env.all")).read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("OPENROUTER_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OPENROUTER_API_KEY not found")


def band_for(nuance):
    if not isinstance(nuance, dict):
        return None
    b = str(nuance.get("review_length_band", "")).strip().lower()
    return b if b in LENGTH_BANDS else None


def wc(text):
    return len(" ".join(str(text).split()).split())


def build_tasks():
    ids = sorted(json.load(open(IDS))["incomplete_sample_ids"], key=int)
    cid_set = {f"gen_{i}" for i in ids}
    prompts = {}
    with open(REQ, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["custom_id"] in cid_set:
                prompts[r["custom_id"].split("_", 1)[1]] = r["body"]["input"]
    bands, old_wc = {}, {}
    with open(GEN, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            sid = str(r.get("sample_id"))
            if sid in ids:
                bands[sid] = band_for(r.get("nuance_attributes"))
                old_wc[sid] = wc(r.get("text", ""))
    tasks = []
    for sid in ids:
        if sid in prompts and bands.get(sid):
            tasks.append({"sample_id": sid, "prompt": prompts[sid],
                          "band": bands[sid], "old_wc": old_wc.get(sid, 0)})
    return tasks


VERBOSITY = "low"  # overridden by --verbosity


async def one(client, sem, t, cap, retries=4):
    async with sem:
        for a in range(retries):
            try:
                await asyncio.sleep(random.uniform(0, 0.4) * (a + 1))
                extra = {"reasoning": {"effort": "minimal"}}
                if VERBOSITY != "none":
                    extra["verbosity"] = VERBOSITY
                r = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": t["prompt"]}],
                    max_tokens=cap, temperature=1.0,
                    extra_body=extra,
                )
                txt = (r.choices[0].message.content or "").strip()
                if txt:
                    return {**t, "new_text": txt, "new_wc": wc(txt)}
            except Exception as e:
                if a == retries - 1:
                    return {**t, "new_text": "", "new_wc": 0, "error": str(e)[:160]}
        return {**t, "new_text": "", "new_wc": 0, "error": "empty"}


async def main(limit, cap, conc):
    from openai import AsyncOpenAI
    tasks = build_tasks()
    if limit:
        tasks = tasks[:limit]
    print(f"tasks with resolvable band+prompt: {len(tasks)} (cap={cap}, conc={conc})")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=load_key())
    sem = asyncio.Semaphore(conc)
    rows = await asyncio.gather(*[one(client, sem, t, cap) for t in tasks])

    def in_band(wc_, band):
        lo, hi = LENGTH_BANDS[band]
        return lo <= wc_ <= hi

    ok = [r for r in rows if r["new_text"]]
    before = sum(in_band(r["old_wc"], r["band"]) for r in rows) / max(len(rows), 1)
    after = sum(in_band(r["new_wc"], r["band"]) for r in ok) / max(len(ok), 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            r = dict(r); r.pop("prompt", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "model": MODEL, "cap_tokens": cap, "n_targeted": len(rows),
        "n_regenerated_ok": len(ok), "n_errors": len(rows) - len(ok),
        "adherence_841_before": round(before, 4),
        "adherence_841_after": round(after, 4),
        "mean_wc_before": round(sum(r["old_wc"] for r in rows) / max(len(rows), 1), 1),
        "mean_wc_after": round(sum(r["new_wc"] for r in ok) / max(len(ok), 1), 1),
    }
    json.dump(summary, open(SUM, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cap", type=int, default=700)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--verbosity", default="low", choices=["low", "medium", "high", "none"])
    args = ap.parse_args()
    VERBOSITY = args.verbosity
    asyncio.run(main(args.limit, args.cap, args.conc))
