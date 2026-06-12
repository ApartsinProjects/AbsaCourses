"""V7 — validate the faithfulness audit against EXISTING HUMAN labels (Herath,
EduRABSA), as the free substitute for a new human study.

Discriminative design: for each sampled REAL review, build three label variants
and audit each with the SAME gpt-4.1-mini audit used at scale:
  - faithful : the human gold labels            -> audit SHOULD confirm  (supported & match)
  - flip     : human aspects, polarity flipped  -> audit SHOULD reject   (sentiment_match False)
  - inject   : add one aspect NOT in the gold   -> audit SHOULD reject   (supported False)

If the audit confirms gold and rejects perturbations at high rates, it matches
human judgment on ground-truth data -> the filter is human-validated.

Writes a batch requests.jsonl + a manifest with the ground-truth flags. Submit
with submit_faithfulness_audit_batch.py --prefix v7_audit ; score with v7_score.py.
"""
import json
import os
import random

import pandas as pd

from label_faithfulness_audit import build_prompt, build_text_format, BATCH_DIR, ensure_dirs

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "gpt-4.1-mini"
SEED = 42
N_PER = {"herath": 250, "edurabsa": 150}
DATA = {
    "herath": os.path.join(HERE, "reviewer_ab_data", "herath_mapped_real_reviews_2829.jsonl"),
    "edurabsa": os.path.join(HERE, "..", "external_data", "EduRABSA_mapped", "edurabsa_test_mapped.jsonl"),
}

POS, NEG, NEU = "positive", "negative", "neutral"


def flip(pol: str) -> str:
    p = str(pol).lower()
    if p.startswith("pos"):
        return NEG
    if p.startswith("neg"):
        return POS
    return NEG  # neutral -> negative (a real change the audit should flag)


def main():
    ensure_dirs(); BATCH_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    req_path = BATCH_DIR / "v7_audit_requests.jsonl"
    man_rows = []
    text_format = build_text_format()
    # gpt-4.1-mini-batch rejects text.verbosity='low' (only 'medium' supported)
    if isinstance(text_format, dict) and text_format.get("verbosity") == "low":
        text_format["verbosity"] = "medium"
    schema_aspects = set()
    # first pass to collect the global aspect vocabulary (for injection)
    loaded = {}
    for ds, path in DATA.items():
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        loaded[ds] = rows
        for r in rows:
            schema_aspects.update((r.get("aspects") or {}).keys())
    schema_aspects = sorted(schema_aspects)

    with req_path.open("w", encoding="utf-8") as fh:
        for ds, rows in loaded.items():
            idx = list(range(len(rows)))
            rng.shuffle(idx)
            kept = 0
            for i in idx:
                if kept >= N_PER[ds]:
                    break
                r = rows[i]
                gold = {a: str(p) for a, p in (r.get("aspects") or {}).items() if p}
                if not gold:
                    continue
                text = r.get("text", "")
                if len(text.split()) < 5:
                    continue
                kept += 1
                # variant 1: faithful (gold)
                variants = [("faithful", gold, True)]
                # variant 2: polarity-flipped (only if any non-neutral exists)
                flipped = {a: flip(p) for a, p in gold.items()}
                if flipped != gold:
                    variants.append(("flip", flipped, False))
                # variant 3: inject one absent aspect (neutral->forces a presence claim)
                absent = [a for a in schema_aspects if a not in gold]
                if absent:
                    inj = dict(gold)
                    inj[rng.choice(absent)] = rng.choice([POS, NEG])
                    variants.append(("inject", inj, False))
                for vname, labels, gt_faithful in variants:
                    cid = f"v7_{ds}_{i}_{vname}"
                    req = {
                        "custom_id": cid, "method": "POST", "url": "/v1/responses",
                        "body": {"model": MODEL, "input": build_prompt(text, labels),
                                 "max_output_tokens": 800, "text": text_format},
                    }
                    fh.write(json.dumps(req, ensure_ascii=False) + "\n")
                    man_rows.append({"custom_id": cid, "dataset": ds, "row_idx": i,
                                     "variant": vname, "gt_faithful": gt_faithful,
                                     "labels": json.dumps(labels, ensure_ascii=False),
                                     "n_aspects": len(labels)})
    man = pd.DataFrame(man_rows)
    man_path = BATCH_DIR / "v7_audit_manifest.csv"
    man.to_csv(man_path, index=False)
    print(f"requests: {req_path} ({len(man_rows)} requests)")
    print(f"manifest: {man_path}")
    print("variant counts:\n", man.groupby(["dataset", "variant"]).size())


if __name__ == "__main__":
    main()
