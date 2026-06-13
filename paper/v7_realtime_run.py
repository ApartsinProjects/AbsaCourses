"""Real-time hedge for the V7 audit: replay the EXACT 1200 batch request bodies
against the Responses API concurrently (bypasses the stuck batch queue), then run
the same scoring as v7_score.py. Race partner to the live batch; kill the loser.

Writes:
  batch_requests/v7_audit_realtime_results.jsonl   (raw per-request output text)
  batch_requests/v7_audit_realtime_scored.csv      (scored rows)
and prints the per-variant + audit-as-classifier tables.

Local, CPU-only (network-bound). Usage: python v7_realtime_run.py
"""
import asyncio
import json
import os
import sys
import time

import pandas as pd
from openai import AsyncOpenAI

from label_faithfulness_audit import BATCH_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
REQS = os.path.join(BATCH_DIR, "v7_audit_requests.jsonl")
MANIFEST = os.path.join(BATCH_DIR, "v7_audit_manifest.csv")
RAW_OUT = os.path.join(BATCH_DIR, "v7_audit_realtime_results.jsonl")
SCORED = os.path.join(BATCH_DIR, "v7_audit_realtime_scored.csv")
CONCURRENCY = 16
MAX_RETRY = 5
WALL_CAP_S = 30 * 60


def key():
    for p in (os.path.join(HERE, "..", ".opeai.key"),
              os.path.join(HERE, "..", "..", "..", "Projects", "CourseABSA", ".opeai.key")):
        if os.path.exists(p):
            return open(p).read().strip()
    return os.environ.get("OPENAI_API_KEY")


def classify(exc):
    s = f"{type(exc).__name__}: {exc}".lower()
    if isinstance(exc, (TypeError, AttributeError)):
        return "config"
    if any(t in s for t in ("400", "401", "403", "invalid", "unsupported", "bad request")):
        return "config"
    if any(t in s for t in ("429", "500", "502", "503", "504", "timeout", "timed out",
                            "connection", "temporarily", "overloaded", "rate limit")):
        return "transient"
    return "fatal"


async def one(client, sem, req, t0):
    cid = req["custom_id"]
    body = req["body"]
    async with sem:
        for attempt in range(MAX_RETRY):
            if time.time() - t0 > WALL_CAP_S:
                return cid, None, "walltimeout"
            try:
                resp = await client.responses.create(**body)
                return cid, resp.output_text, None
            except Exception as e:  # noqa: BLE001
                kind = classify(e)
                if kind == "config":
                    return cid, None, f"config:{e}"
                if kind == "transient" and attempt < MAX_RETRY - 1:
                    await asyncio.sleep(min(2 ** attempt, 20))
                    continue
                if attempt < MAX_RETRY - 1:
                    await asyncio.sleep(min(2 ** attempt, 20))
                    continue
                return cid, None, f"{kind}:{e}"
    return cid, None, "unreached"


async def run():
    reqs = [json.loads(l) for l in open(REQS, encoding="utf-8")]
    client = AsyncOpenAI(api_key=key())
    sem = asyncio.Semaphore(CONCURRENCY)
    t0 = time.time()
    results = {}
    errors = {}
    done = 0
    tasks = [asyncio.create_task(one(client, sem, r, t0)) for r in reqs]
    for fut in asyncio.as_completed(tasks):
        cid, txt, err = await fut
        done += 1
        if err:
            errors[cid] = err
            if classify and err.startswith("config"):
                print(f"[realtime] CONFIG ERROR on {cid}: {err} -- aborting (no retry fixes this)", flush=True)
        else:
            results[cid] = txt
        if done % 100 == 0 or done == len(reqs):
            print(f"[realtime] {done}/{len(reqs)} ok={len(results)} err={len(errors)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    with open(RAW_OUT, "w", encoding="utf-8") as fh:
        for cid, txt in results.items():
            fh.write(json.dumps({"custom_id": cid, "output_text": txt}, ensure_ascii=False) + "\n")
    if errors:
        print(f"[realtime] {len(errors)} errors; sample:",
              dict(list(errors.items())[:5]), flush=True)
    return results, errors


def kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n; pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def score(results):
    verdict = {}
    for cid, txt in results.items():
        try:
            data = json.loads(txt)
            items = data.get("aspects", data) if isinstance(data, dict) else data
        except Exception:
            items = []
        verdict[cid] = {it.get("aspect"): (bool(it.get("supported")), bool(it.get("sentiment_match")))
                        for it in (items or []) if isinstance(it, dict)}
    man = pd.read_csv(MANIFEST)
    rows = []
    for _, m in man.iterrows():
        labels = json.loads(m["labels"]); v = verdict.get(m["custom_id"], {})
        if m["custom_id"] not in verdict:
            continue  # not (yet) answered real-time
        sup = [v.get(a, (False, False))[0] for a in labels]
        mat = [v.get(a, (False, False))[1] for a in labels]
        audit_faithful = bool(sup) and all(sup) and all(mat)
        rows.append({**m.to_dict(), "support_rate": sum(sup) / len(sup) if sup else 0,
                     "match_rate": sum(mat) / len(mat) if mat else 0,
                     "audit_faithful": audit_faithful})
    df = pd.DataFrame(rows)
    print(f"\n=== scored {len(df)} / {len(man)} manifest rows ===")
    print("\n=== per-variant audit behavior (mean) ===")
    print(df.groupby(["dataset", "variant"])[["support_rate", "match_rate", "audit_faithful"]].mean().round(3))
    print("\n=== audit-as-classifier vs human ground truth ===")
    for ds in ["herath", "edurabsa", "ALL"]:
        d = df if ds == "ALL" else df[df.dataset == ds]
        gt = d["gt_faithful"].astype(bool).tolist()
        pr = d["audit_faithful"].astype(bool).tolist()
        tp = sum(1 for g, p in zip(gt, pr) if g and p); tn = sum(1 for g, p in zip(gt, pr) if not g and not p)
        fp = sum(1 for g, p in zip(gt, pr) if not g and p); fn = sum(1 for g, p in zip(gt, pr) if g and not p)
        acc = (tp + tn) / len(gt) if gt else 0
        prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0
        print(f"  {ds:9} n={len(gt):4} acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} "
              f"kappa={kappa([int(x) for x in gt],[int(x) for x in pr]):.3f}")
    df.to_csv(SCORED, index=False)
    print("\nwrote", SCORED, flush=True)


def main():
    results, errors = asyncio.run(run())
    if not results:
        print("[realtime] NO results; aborting scoring", flush=True)
        sys.exit(1)
    score(results)
    print("[realtime] === DONE ===", flush=True)


if __name__ == "__main__":
    main()
