"""Poll the V7 batch every ~30 min until terminal, then exit so the agent folds
the result in. Safety valve: if the batch is STILL EMPTY (completed==0) at 12h
past the ORIGINAL submission, cancel it and resubmit ONCE, then keep polling the
new batch. Run in background; the harness re-invokes the agent on exit.
"""
import json
import os
import subprocess
import sys
import time
import datetime

from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
BID = "batch_6a2c6af5b974819084367b21536c7545"
PREFIX = "v7_audit"
INTERVAL = 1800          # 30 min
ORIG_CREATED = 1781295861  # 2026-06-12T20:24:21Z (epoch) — original submit time
RESUBMIT_AFTER = 12 * 3600
MAX_HOURS = 26
BATCH_DIR = os.path.join(HERE, "batch_requests")


def client():
    for p in (os.path.join(HERE, "..", ".opeai.key"),
              os.path.join(HERE, "..", "..", "..", "Projects", "CourseABSA", ".opeai.key")):
        if os.path.exists(p):
            return OpenAI(api_key=open(p).read().strip())
    return OpenAI()


def sync_scorer_id(new_bid):
    """Point BOTH submitted-json files at new_bid so v7_score.py reads it."""
    for fn in ("v7_audit_submitted.json", "v7_audit_submitted_batch.json"):
        p = os.path.join(BATCH_DIR, fn)
        try:
            d = json.load(open(p)) if os.path.exists(p) else {}
        except Exception:
            d = {}
        d["batch_id"] = new_bid
        json.dump(d, open(p, "w"), indent=2)


def resubmit():
    print("[resubmit] launching submit_faithfulness_audit_batch.py", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, "submit_faithfulness_audit_batch.py"),
                        "--prefix", PREFIX],
                       capture_output=True, text=True)
    print(r.stdout[-2000:], flush=True)
    if r.returncode != 0:
        print("[resubmit] FAILED:\n" + r.stderr[-2000:], flush=True)
        return None
    rec = json.load(open(os.path.join(BATCH_DIR, f"{PREFIX}_submitted_batch.json")))
    new_bid = rec["batch_id"]
    sync_scorer_id(new_bid)
    print(f"[resubmit] new batch_id={new_bid}", flush=True)
    return new_bid


def main():
    global BID
    c = client()
    t0 = time.time()
    n = 0
    resubmitted = False
    while True:
        n += 1
        b = c.batches.retrieve(BID)
        rc = b.request_counts
        comp = getattr(rc, "completed", 0)
        now = datetime.datetime.now(datetime.UTC).strftime("%H:%M:%SZ")
        print(f"[poll {n} {now}] bid={BID[-8:]} status={b.status} "
              f"completed={comp}/{getattr(rc,'total',0)} failed={getattr(rc,'failed',0)}",
              flush=True)
        if b.status not in ("validating", "in_progress", "finalizing"):
            print(f"=== TERMINAL: {b.status} bid={BID} "
                  f"(completed={comp}/{getattr(rc,'total',0)}, "
                  f"failed={getattr(rc,'failed',0)}, "
                  f"output_file={b.output_file_id}, error_file={b.error_file_id}) ===",
                  flush=True)
            return 0
        # 12h-empty safety valve (once)
        if (not resubmitted and comp == 0
                and (time.time() - ORIG_CREATED) > RESUBMIT_AFTER):
            print("[safety] still 0/1200 at 12h past original submit; cancelling + resubmitting",
                  flush=True)
            try:
                c.batches.cancel(BID)
            except Exception as e:
                print(f"[safety] cancel raised (continuing): {e}", flush=True)
            new = resubmit()
            resubmitted = True
            if new:
                BID = new
                t0 = time.time()
                time.sleep(INTERVAL)
                continue
            else:
                print("=== RESUBMIT_FAILED: agent must intervene ===", flush=True)
                return 3
        if (time.time() - t0) > MAX_HOURS * 3600:
            print(f"=== TIMEOUT after {MAX_HOURS}h, last status={b.status} bid={BID} ===", flush=True)
            return 2
        time.sleep(INTERVAL)


if __name__ == "__main__":
    sys.exit(main())
