"""Experiment 2 - Sentence-level training via weak aspect localization.

The transfer mismatch is granularity: we TRAIN on full multi-aspect reviews but
EVALUATE on real sentences (Herath, M-ABSA). Here we localize each review's
declared aspects to the sentence(s) that mention them (lexicon weak supervision),
producing sentence-level training rows, then train the same BERT detection+
sentiment stack and evaluate transfer. Compares against the review-trained
baseline (exp1 whole_doc numbers) on the same targets.

Usage: python exp2_sentence_train.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from absa_model_comparison import (
    Config,
    calibrate_thresholds,
    evaluate_models,
    set_seed,
    three_way_split,
    train_detection,
    train_sentiment,
)
from evaluate_synthetic_to_real_transfer import (
    DEFAULT_HERATH_ROOT,
    DEFAULT_SYNTHETIC_PATH,
    load_herath_mapped_dataset,
    load_jsonl,
    load_real_from_mapped_jsonl,
    restrict_to_overlap,
)

ROOT = Path(__file__).resolve().parents[1]
OATS = ROOT / "external_data/OATS_coursera/oats_mapped.jsonl"
MABSA = ROOT / "external_data/M-ABSA_coursera/m_absa_mapped.jsonl"
OUT = ROOT / "paper/outputs/exp2_sentence_train.json"
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Weak-supervision lexicon: aspect -> characteristic terms (substring match, lowercased).
LEX = {
    "exam_fairness": ["exam", "fair", "unfair", "test ", "cheat", "proctor", "midterm", "final "],
    "accessibility": ["access", "caption", "disab", "accommodat", "available anytime", "self-paced", "self paced"],
    "organization": ["organiz", "structur", "syllabus", "layout", "well-organ", "disorgan", "laid out"],
    "workload": ["workload", "hours a week", "time-consum", "time consum", "heavy", "busywork", "amount of work", "hours per"],
    "overall_experience": ["overall", "experience", "enjoyed", "recommend", "worth it", "worth taking", "great course", "loved"],
    "grading_transparency": ["grading", "graded", "rubric", "transparent", "how grades", "point breakdown", "curve"],
    "pacing": ["pace", "pacing", "too fast", "too slow", "rushed", "moved quickly", "moved slowly", "speed"],
    "tooling_usability": ["tool", "platform", "software", "interface", "autograd", "buggy", "ide", "website", "portal"],
    "lecturer_quality": ["lecturer", "instructor", "professor", "teacher", "teaching", "taught", "lectures were"],
    "prerequisite_fit": ["prerequisit", "prereq", "background", "prepared", "prior knowledge", "coming in", "expected to know"],
    "support": ["support", "office hours", "the ta", " tas ", "assistance", "responsive", "helped me", "help when"],
    "materials": ["material", "reading", "textbook", "slides", "resources", "lecture notes", "content was"],
    "difficulty": ["difficult", "hard ", "easy", "challeng", "tough", "rigor", "demanding"],
    "clarity": ["clear", "clarity", "confus", "understand", "explained", "vague", "made sense"],
    "assessment_design": ["assignment", "project", "quiz", "assessment", "homework", "problem set", "coding assign"],
    "interest": ["interest", "boring", "engaging", "fascinat", "dull", "fun ", "captivat"],
    "peer_interaction": ["peer", "classmate", "group project", "collaborat", "discussion forum", "teammate", "cohort"],
    "practical_application": ["practical", "real-world", "real world", "hands-on", "hands on", "apply", "applicable to", "useful skill"],
    "feedback_quality": ["feedback", "comments on", "critique", "graded feedback", "detailed response", "no feedback"],
    "relevance": ["relevant", "relevance", "pertinent", "applicable", "useful for", "up to date", "outdated"],
}


def localize(text: str, declared: dict) -> list[tuple[str, dict]]:
    """Split a review into sentences; assign each DECLARED aspect+polarity to the
    sentence(s) whose text matches the aspect lexicon. Aspects with no matching
    sentence fall back to the longest sentence so no label is dropped."""
    sents = [s.strip() for s in SENT_SPLIT.split(" ".join(str(text).split())) if s.strip()]
    if not sents:
        return []
    low = [s.lower() for s in sents]
    assign: list[dict] = [dict() for _ in sents]
    for asp, pol in declared.items():
        terms = LEX.get(asp, [asp.replace("_", " ")])
        hit = [i for i, s in enumerate(low) if any(t in s for t in terms)]
        if not hit:
            hit = [max(range(len(sents)), key=lambda i: len(low[i]))]  # fallback: longest sentence
        for i in hit:
            assign[i][asp] = pol
    return [(sents[i], assign[i]) for i in range(len(sents))]


def build_sentence_df(synth_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in synth_df.iterrows():
        for sent, asp in localize(r["text"], r["aspects"] if isinstance(r["aspects"], dict) else {}):
            if len(sent.split()) < 3:
                continue
            rows.append({"text": sent, "aspects": asp, "target_attributes": asp,
                         "nuance_attributes": {}, "course_name": "", "grade": "", "style": ""})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", default="bert-base-uncased")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = Config()
    if args.smoke:
        cfg.epochs_detection = 1
        cfg.epochs_sentiment = 1
    set_seed(cfg.seed)

    synth = load_jsonl(DEFAULT_SYNTHETIC_PATH)
    if args.smoke:
        synth = synth.sample(300, random_state=42).reset_index(drop=True)
    sent_synth = build_sentence_df(synth)
    print(f"[exp2] reviews={len(synth)} -> sentences={len(sent_synth)} "
          f"(mean aspects/sent={sent_synth['aspects'].map(len).mean():.2f}, "
          f"empty={ (sent_synth['aspects'].map(len)==0).mean():.2f})", flush=True)

    targets = {"herath": load_herath_mapped_dataset(DEFAULT_HERATH_ROOT),
               "m_absa": load_real_from_mapped_jsonl(MABSA),
               "oats": load_real_from_mapped_jsonl(OATS)}
    if args.smoke:
        targets = {k: v.sample(min(120, len(v)), random_state=42).reset_index(drop=True) for k, v in targets.items()}

    results = {}
    for name, real in targets.items():
        aspects = sorted({a for labs in real["aspects"] for a in labs.keys()})
        train_pool = restrict_to_overlap(sent_synth, aspects)
        tr, cal, _ = three_way_split(train_pool, cfg.split_calib, cfg.split_test, cfg.seed)
        print(f"[exp2] target={name} aspects={len(aspects)} sent_train={len(tr)} real={len(real)}", flush=True)
        det, dtok = train_detection(args.approach, tr, cal, aspects, cfg)
        sen, stok = train_sentiment(args.approach, tr, cal, aspects, cfg)
        thr = calibrate_thresholds(det, cal, dtok, aspects, cfg)
        _, summ = evaluate_models(args.approach, det, sen, real, dtok, stok, aspects, thr, cfg)
        results[name] = {k: summ[k] for k in ("micro_precision", "micro_recall", "micro_f1", "sentiment_mse_detected")}
        results[name]["n_real"] = int(len(real))
        print(f"[exp2] {name}: sentence-trained microF1={summ['micro_f1']:.4f} "
              f"sentMSE={summ['sentiment_mse_detected']:.3f}", flush=True)
        import gc, torch
        del det, sen, dtok, stok
        gc.collect(); torch.cuda.empty_cache()

    print(json.dumps(results, indent=2), flush=True)
    if not args.smoke:
        OUT.write_text(json.dumps(results, indent=2))
        print(f"[exp2] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
