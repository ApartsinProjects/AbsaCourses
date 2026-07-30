"""EduRABSA real-only reference + synthetic-pretrain-then-finetune (per seed).

Complements the already-existing multi-seed synthetic-only EduRABSA transfer
(phase_d2 full_seed*_edurabsa). Two modes:
  real_only          : train BERT det+sent on the real EduRABSA train split,
                       evaluate on the held-out EduRABSA test split.
  pretrain_finetune  : pretrain on the synthetic overlap, continue-train on the
                       real EduRABSA train split, evaluate on EduRABSA test.
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import absa_model_comparison as eng  # noqa: E402
from evaluate_synthetic_to_real_transfer import load_real_from_mapped_jsonl, restrict_to_overlap  # noqa: E402
from checkpoint_train_worker import _finetune_detection, _finetune_sentiment  # noqa: E402

MODEL = "bert-base-uncased"


def run(mode, seed, synthetic_path, edu_train, edu_test, out_dir):
    cfg = eng.Config(max_len=192, batch_size=8, lr=3e-5,
                     epochs_detection=3, epochs_sentiment=3, seed=seed)
    eng.set_seed(seed)
    train_real = load_real_from_mapped_jsonl(edu_train)
    test_real = load_real_from_mapped_jsonl(edu_test)
    aspects = sorted({a for labels in train_real["aspects"] for a in labels.keys()})
    eng.log_event(f"[edu {mode} seed={seed}] aspects({len(aspects)}) train={len(train_real)} test={len(test_real)}")

    calib = train_real.sample(frac=0.12, random_state=seed)
    rtrain = train_real.drop(calib.index)

    if mode == "real_only":
        eng.set_seed(seed); det, dt = eng.train_detection(MODEL, rtrain, calib, aspects, cfg)
        eng.set_seed(seed); sent, st = eng.train_sentiment(MODEL, rtrain, calib, aspects, cfg)
    else:  # pretrain_finetune
        synth = eng.load_jsonl(synthetic_path)
        synth_ov = restrict_to_overlap(synth, aspects)
        strain, scalib, _ = eng.three_way_split(synth_ov, cfg.split_calib, cfg.split_test, seed)
        eng.set_seed(seed); det, dt = eng.train_detection(MODEL, strain, scalib, aspects, cfg)
        eng.set_seed(seed); sent, st = eng.train_sentiment(MODEL, strain, scalib, aspects, cfg)
        eng.set_seed(seed); det = _finetune_detection(det, dt, rtrain, aspects, cfg)
        sent = _finetune_sentiment(sent, st, rtrain, aspects, cfg)

    thr = eng.calibrate_thresholds(det, calib, dt, aspects, cfg)
    _, summ = eng.evaluate_models(MODEL, det, sent, test_real, dt, st, aspects, thr, cfg)
    res = {"experiment": "edurabsa_" + mode, "mode": mode, "seed": seed,
           "n_aspects": len(aspects), "aspects": aspects,
           "n_real_train": int(len(rtrain)), "n_real_test": int(len(test_real)),
           "micro_f1": float(summ["micro_f1"]), "macro_f1": float(summ["macro_f1"]),
           "micro_recall": float(summ["micro_recall"]),
           "sentiment_mse": float(summ["sentiment_mse_detected"])}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(res, indent=2))
    eng.log_event(f"[edu {mode} seed={seed}] micro_f1={res['micro_f1']:.4f}")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["real_only", "pretrain_finetune"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic", default="/app/data/synthetic_generated_reviews.jsonl")
    p.add_argument("--edu-train", default="/app/data/edurabsa_train_mapped.jsonl")
    p.add_argument("--edu-test", default="/app/data/edurabsa_test_mapped.jsonl")
    p.add_argument("--out", default="/results/edu")
    a = p.parse_args()
    eng.ensure_dirs()
    run(a.mode, a.seed, Path(a.synthetic), Path(a.edu_train), Path(a.edu_test), Path(a.out))


if __name__ == "__main__":
    main()
