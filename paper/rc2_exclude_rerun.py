"""RC2 part 2: does excluding the 841 output-token-capped rows change the benchmark?
Retrain the BERT-base two-step DETECTION on the same 20-aspect corpus for two
conditions -- FULL (10,000) and COMPLETE-ONLY (9,159, dropping rc2_incomplete_row_ids)
-- across 3 seeds, and compare held-out micro-F1 and macro balanced accuracy.

Local, low-priority GPU. Saves every per-seed result.
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("C1_BASE", os.path.dirname(HERE))
sys.path.insert(0, HERE)
CORPUS = os.path.join(HERE, "reviewer_ab_data", "generated_reviews_10k.jsonl")
INC = json.load(open(os.path.join(HERE, "outputs", "rc2_incomplete_row_ids.json")))["incomplete_sample_ids"]
INC = set(int(x) for x in INC)
OUT = os.path.join(HERE, "outputs")
SEEDS = [42, 17, 23]


def load_rows():
    return [json.loads(l) for l in open(CORPUS, encoding="utf-8")]


def to_df(rows):
    return pd.DataFrame([{"text": r["text"], "aspects": r.get("aspects", {}),
                          "target_attributes": r.get("aspects", {})} for r in rows])


def run_one(df, aspects, seed, eng):
    cfg = eng.Config(seed=seed)
    if hasattr(cfg, "epochs_sentiment"):
        cfg.epochs_sentiment = 0  # detection-only; skip sentiment head to save time
    tr, ca, te = eng.three_way_split(df, cfg.split_calib, cfg.split_test, seed)
    eng.set_seed(seed)
    det, tok = eng.train_detection("bert-base-uncased", tr, ca, aspects, cfg)
    thr = eng.calibrate_thresholds(det, ca, tok, aspects, cfg)
    from torch.utils.data import DataLoader
    dl = DataLoader(eng.DetectionDataset(te, tok, aspects, cfg.max_len), batch_size=cfg.batch_size, shuffle=False)
    probs, true = eng.collect_detection(det, dl, cfg.device)
    true = np.asarray(true).astype(int)  # collect_detection returns float targets; bitwise ops need int
    thr_vec = np.array([thr[a] for a in aspects], dtype=np.float32)
    preds = (probs >= thr_vec).astype(int)
    tp = int((preds & true).sum()); fp = int((preds & (1 - true)).sum()); fn = int(((1 - preds) & true).sum())
    micro_f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    # macro balanced accuracy
    bals = []
    for j in range(true.shape[1]):
        P = true[:, j].sum(); N = len(true) - P
        if 0 < P < len(true):
            tpr = (preds[:, j] & true[:, j]).sum() / P
            tnr = ((1 - preds[:, j]) & (1 - true[:, j])).sum() / N
            bals.append(0.5 * (tpr + tnr))
    return {"micro_f1": round(float(micro_f1), 4),
            "macro_balanced_accuracy": round(float(np.mean(bals)), 4),
            "n_test": int(len(te))}


def main():
    import absa_model_comparison as eng
    import torch
    assert torch.cuda.is_available(), "no CUDA"
    eng.configure_console_encoding(); eng.ensure_dirs()
    rows = load_rows()
    aspects = sorted({a for r in rows for a in (r.get("aspects") or {})})
    full_df = to_df(rows)
    comp_rows = [r for i, r in enumerate(rows) if i not in INC]
    comp_df = to_df(comp_rows)
    print(f"[rc2] full={len(full_df)} complete_only={len(comp_df)} (excluded {len(INC)}) aspects={len(aspects)}", flush=True)

    records = []
    for cond, df in (("full", full_df), ("complete_only", comp_df)):
        for seed in SEEDS:
            print(f"[rc2] === {cond} seed={seed} ===", flush=True)
            m = run_one(df, aspects, seed, eng)
            m.update(condition=cond, seed=seed, n_train_corpus=len(df))
            records.append(m)
            pd.DataFrame(records).to_csv(os.path.join(OUT, "rc2_exclude_rerun_per_seed.csv"), index=False)
            print(f"[rc2] {cond} seed={seed}: micro_f1={m['micro_f1']} bal_acc={m['macro_balanced_accuracy']}", flush=True)

    d = pd.DataFrame(records)
    summ = {}
    for cond in ("full", "complete_only"):
        s = d[d.condition == cond]
        summ[cond] = {"micro_f1_mean": round(s.micro_f1.mean(), 4), "micro_f1_std": round(s.micro_f1.std(ddof=0), 4),
                      "bal_acc_mean": round(s.macro_balanced_accuracy.mean(), 4),
                      "bal_acc_std": round(s.macro_balanced_accuracy.std(ddof=0), 4), "seeds": SEEDS}
    summ["delta_complete_minus_full"] = {
        "micro_f1": round(summ["complete_only"]["micro_f1_mean"] - summ["full"]["micro_f1_mean"], 4),
        "bal_acc": round(summ["complete_only"]["bal_acc_mean"] - summ["full"]["bal_acc_mean"], 4)}
    json.dump(summ, open(os.path.join(OUT, "rc2_exclude_rerun_summary.json"), "w"), indent=2)
    print("[rc2] SUMMARY:", json.dumps(summ), flush=True)
    print("[rc2] === DONE ===", flush=True)


if __name__ == "__main__":
    main()
