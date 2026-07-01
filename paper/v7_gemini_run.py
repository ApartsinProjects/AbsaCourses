"""RC1: cross-FAMILY faithfulness audit with Google Gemini (via OpenRouter),
replaying the exact V7 perturbation set (1,200 faithful/flip/inject variants over
Herath + EduRABSA). Scores Gemini-vs-HUMAN per-aspect agreement the same way as the
GPT auditor (v7_peraspect.py), so kappa is directly comparable to the GPT 0.56.

If a genuinely independent model family (Gemini) matches the human labels at ~the
same kappa, the audit is not an artifact of the generator's own (GPT) family.

Reads OPENROUTER_API_KEY from E:\\Projects\\.env.all. CPU/network only.
"""
import asyncio, json, os, collections, time
import pandas as pd
from openai import AsyncOpenAI
from label_faithfulness_audit import BATCH_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
REQS = os.path.join(BATCH_DIR, "v7_audit_requests.jsonl")
MANIFEST = os.path.join(BATCH_DIR, "v7_audit_manifest.csv")
RAW = os.path.join(BATCH_DIR, "v7_gemini_results.jsonl")
OUT = os.path.join(HERE, "outputs", "rc1_gemini_vs_human.json")
MODEL = "google/gemini-2.5-flash"
CONC = 12
MAX_RETRY = 5


def key():
    for l in open(r"E:\Projects\.env.all", encoding="utf-8"):
        if l.startswith("OPENROUTER_API_KEY="):
            return l.split("=", 1)[1].strip()
    raise SystemExit("OPENROUTER_API_KEY not found in E:\\Projects\\.env.all")


async def one(client, sem, cid, prompt, t0):
    async with sem:
        for a in range(MAX_RETRY):
            try:
                r = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_tokens=900, temperature=0)
                return cid, r.choices[0].message.content, None
            except Exception as e:  # noqa: BLE001
                s = f"{type(e).__name__}:{e}".lower()
                if any(t in s for t in ("400", "401", "403", "invalid", "unsupported")) and a == 0:
                    # try without strict json response_format (some models reject it)
                    try:
                        r = await client.chat.completions.create(
                            model=MODEL, messages=[{"role": "user", "content": prompt}],
                            max_tokens=900, temperature=0)
                        return cid, r.choices[0].message.content, None
                    except Exception:
                        pass
                if a < MAX_RETRY - 1:
                    await asyncio.sleep(min(2 ** a, 20)); continue
                return cid, None, f"{type(e).__name__}:{e}"[:160]
    return cid, None, "unreached"


async def run():
    reqs = [json.loads(l) for l in open(REQS, encoding="utf-8")]
    client = AsyncOpenAI(api_key=key(), base_url="https://openrouter.ai/api/v1",
                         default_headers={"HTTP-Referer": "https://github.com/ApartsinProjects/AbsaCourses",
                                          "X-Title": "CourseABSA RC1 Gemini audit"})
    sem = asyncio.Semaphore(CONC)
    t0 = time.time()
    tasks = [asyncio.create_task(one(client, sem, r["custom_id"], r["body"]["input"], t0)) for r in reqs]
    results, errors, done = {}, {}, 0
    for fut in asyncio.as_completed(tasks):
        cid, txt, err = await fut
        done += 1
        (errors if err else results).__setitem__(cid, err or txt)
        if done % 100 == 0 or done == len(reqs):
            print(f"[gemini] {done}/{len(reqs)} ok={len(results)} err={len(errors)} ({time.time()-t0:.0f}s)", flush=True)
    with open(RAW, "w", encoding="utf-8") as fh:
        for cid, txt in results.items():
            fh.write(json.dumps({"custom_id": cid, "output_text": txt}, ensure_ascii=False) + "\n")
    if errors:
        print("[gemini] sample errors:", dict(list(errors.items())[:3]), flush=True)
    return results


def parse(txt):
    try:
        d = json.loads(txt)
    except Exception:
        try:
            d = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
        except Exception:
            return {}
    items = d.get("aspects", d) if isinstance(d, dict) else d
    return {it.get("aspect"): (bool(it.get("supported")) and bool(it.get("sentiment_match")))
            for it in (items or []) if isinstance(it, dict)}


def kappa(a, b):
    n = len(a)
    if not n:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def score(results):
    man = pd.read_csv(MANIFEST)
    gold = {(str(m.dataset), int(m.row_idx)): set(json.loads(m["labels"]).keys())
            for _, m in man[man.variant == "faithful"].iterrows()}
    verd = {cid: parse(txt) for cid, txt in results.items()}
    rows = []
    for _, m in man.iterrows():
        cid = str(m["custom_id"])
        if cid not in verd:
            continue
        labels = json.loads(m["labels"]); v = verd[cid]
        orig = gold.get((str(m.dataset), int(m.row_idx)), set())
        for a in labels:
            pred = 1 if v.get(a, False) else 0
            gt = 1 if m.variant == "faithful" else (0 if m.variant == "flip" else (1 if a in orig else 0))
            rows.append({"dataset": str(m.dataset), "variant": str(m.variant), "gt": gt, "pred": pred})
    d = pd.DataFrame(rows)
    res = {"model": MODEL, "n_scored_requests": len(verd), "n_aspect_decisions": len(d), "per_aspect": {}}
    for ds in ["ALL", "herath", "edurabsa"]:
        dd = d if ds == "ALL" else d[d.dataset == ds]
        gt = dd["gt"].tolist(); pr = dd["pred"].tolist()
        tp = sum(1 for g, p in zip(gt, pr) if g and p); tn = sum(1 for g, p in zip(gt, pr) if not g and not p)
        fp = sum(1 for g, p in zip(gt, pr) if not g and p); fn = sum(1 for g, p in zip(gt, pr) if g and not p)
        acc = (tp + tn) / len(gt) if gt else 0
        prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0
        res["per_aspect"][ds] = {"n": len(gt), "kappa": round(kappa(gt, pr), 3), "precision": round(prec, 3),
                                 "recall": round(rec, 3), "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0,
                                 "accuracy": round(acc, 3)}
    faith = d[d.variant == "faithful"]; flip = d[d.variant == "flip"]; inj = d[(d.variant == "inject") & (d["gt"] == 0)]
    res["per_perturbation"] = {"faithful_keep": round(faith["pred"].mean(), 3),
                               "flip_reject": round(1 - flip["pred"].mean(), 3),
                               "inject_reject": round(1 - inj["pred"].mean(), 3) if len(inj) else None}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print("\n=== RC1 Gemini-vs-human ===")
    for ds, v in res["per_aspect"].items():
        print(f"  {ds:9} kappa={v['kappa']} prec={v['precision']} rec={v['recall']} f1={v['f1']} (n={v['n']})")
    print("  per-perturbation:", res["per_perturbation"])
    print("wrote", OUT)


def main():
    results = asyncio.run(run())
    if not results:
        raise SystemExit("no Gemini results")
    score(results)
    print("[gemini] === DONE ===", flush=True)


if __name__ == "__main__":
    main()
