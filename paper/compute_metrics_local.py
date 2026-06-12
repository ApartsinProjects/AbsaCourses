"""Co-compute detection metrics on ONE consistent model+split for the internal
20-aspect benchmark: micro-F1, macro balanced accuracy (thresholded) AND
threshold-free macro/micro AUROC + macro average-precision. Answers "F1 is low
but does the model actually discriminate?" with a metric decoupled from the
operating point and the noisy gold threshold.

Local, low-priority GPU. CPU dry-run first: `python compute_metrics_local.py --dry`.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("C1_BASE", os.path.dirname(HERE))
sys.path.insert(0, HERE)

CORPUS = os.path.join(HERE, "reviewer_ab_data", "generated_reviews_10k.jsonl")
OUT = os.path.join(HERE, "outputs", "internal_auroc_local.json")


def load_df():
    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
    df = pd.DataFrame([{"text": r["text"], "aspects": r.get("aspects", {}),
                        "target_attributes": r.get("aspects", {})} for r in rows])
    aspects = sorted({a for r in rows for a in (r.get("aspects") or {})})
    return df, aspects


def auroc_block(sample_predictions, aspects):
    def asd(x):
        return x if isinstance(x, dict) else json.loads(x)
    P = np.array([[asd(r["detection_probabilities"])[a] for a in aspects] for r in sample_predictions])
    Y = np.array([[asd(r["detection_targets"])[a] for a in aspects] for r in sample_predictions])
    from sklearn.metrics import roc_auc_score, average_precision_score
    per = {}
    aucs, aps = [], []
    for j, a in enumerate(aspects):
        if 0 < Y[:, j].sum() < len(Y):
            au = roc_auc_score(Y[:, j], P[:, j]); per[a] = round(au, 3); aucs.append(au)
        if Y[:, j].sum() > 0:
            aps.append(average_precision_score(Y[:, j], P[:, j]))
    return {"n_test": len(Y), "micro_auroc": round(roc_auc_score(Y.ravel(), P.ravel()), 4),
            "macro_auroc": round(float(np.mean(aucs)), 4), "macro_ap": round(float(np.mean(aps)), 4),
            "per_aspect_auroc": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    df, aspects = load_df()
    import absa_model_comparison as eng
    cfg = eng.Config(seed=42)
    if hasattr(cfg, "epochs_sentiment"):
        cfg.epochs_sentiment = 1  # sentiment irrelevant to detection AUROC/F1; keep GPU time low
    train, calib, test = eng.three_way_split(df, cfg.split_calib, cfg.split_test, 42)
    print(f"[metrics] aspects={len(aspects)} split train={len(train)} calib={len(calib)} test={len(test)}", flush=True)
    if args.dry:
        print("[metrics] DRY OK (no GPU used)"); return
    import torch
    assert torch.cuda.is_available()
    print(f"[metrics] GPU {torch.cuda.get_device_name(0)}", flush=True)
    eng.configure_console_encoding(); eng.ensure_dirs(); eng.set_seed(42)
    det_model, det_tok = eng.train_detection("bert-base-uncased", train, calib, aspects, cfg)
    sent_model, sent_tok = eng.train_sentiment("bert-base-uncased", train, calib, aspects, cfg)
    thr = eng.calibrate_thresholds(det_model, calib, det_tok, aspects, cfg)
    per_aspect_df, summary, artifact = eng.evaluate_models(
        "bert-base-uncased", det_model, sent_model, test, det_tok, sent_tok,
        aspects, thr, cfg, return_artifacts=True)
    auroc = auroc_block(artifact["sample_predictions"], aspects)
    result = {"experiment": "internal_20aspect_auroc_local", "seed": 42,
              "n_aspects": len(aspects),
              "thresholded": {"micro_f1": round(float(summary["micro_f1"]), 4),
                              "macro_balanced_accuracy": round(float(summary["macro_balanced_accuracy"]), 4),
                              "micro_recall": round(float(summary.get("micro_recall", 0)), 4),
                              "micro_precision": round(float(summary.get("micro_precision", 0)), 4)},
              "threshold_free": auroc}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(result, open(OUT, "w"), indent=2)
    print("[metrics] RESULT:", json.dumps(result["thresholded"]), json.dumps({k: auroc[k] for k in ["micro_auroc", "macro_auroc", "macro_ap"]}), flush=True)
    print("[metrics] === DONE ===", flush=True)


if __name__ == "__main__":
    main()
