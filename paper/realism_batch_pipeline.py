from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

from realism_validation_experiment import (
    KEY_FILE,
    COURSE_CONTEXT,
    PROMPT_CYCLES,
    RANDOM_SEED,
    STABLE_PROMPT_STATE_PATH,
    base_cycle_instruction,
    build_generator_prompt,
    load_generation_schema,
    normalize_space,
    parse_real_reviews,
    sample_aspect_count,
    sample_aspect_labels,
    sample_rich_attributes,
)


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = ROOT / "paper" / "validation"
BATCH_REALISM_DIR = VALIDATION_DIR / "batch_realism"
RUNS_DIR = BATCH_REALISM_DIR / "runs"
CURRENT_PROMPT_PATH = BATCH_REALISM_DIR / "current_generation_prompt.json"

GENERATOR_MODEL = "gpt-5-nano"
JUDGE_MODEL = "gpt-5.4"
REWRITE_MODEL = "gpt-5.4"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    BATCH_REALISM_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def make_run_dir(prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"{prefix}_{stamp}"
    suffix = 1
    while path.exists():
        suffix += 1
        path = RUNS_DIR / f"{prefix}_{stamp}_{suffix}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_status(run_dir: Path, stage: str, status: str, **extra: Any) -> None:
    payload = {
        "updated_at_utc": now_utc(),
        "stage": stage,
        "status": status,
        **extra,
    }
    write_json(run_dir / "status.json", payload)
    append_jsonl(run_dir / "log.jsonl", payload)


def load_client() -> OpenAI:
    api_key = KEY_FILE.read_text(encoding="utf-8").strip()
    return OpenAI(api_key=api_key)


def extract_json_block(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{", text)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:400]}")
    start = match.start()
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])
    raise ValueError(f"Incomplete JSON object in response: {text[:400]}")


def extract_output_text(row: dict[str, Any]) -> str:
    body = row.get("response", {}).get("body", {})
    for item in body.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return str(content.get("text", "")).strip()
    return ""


def safe_parse_judge_response(raw_text: str) -> tuple[dict[str, Any], str | None]:
    try:
        return extract_json_block(raw_text), None
    except Exception as exc:
        label_match = re.search(r'"label"\s*:\s*"(real|synthetic)"', raw_text, flags=re.IGNORECASE)
        justification_match = re.search(r'"justification"\s*:\s*"(.*)', raw_text, flags=re.IGNORECASE | re.DOTALL)
        parsed: dict[str, Any] = {}
        if label_match:
            parsed["label"] = label_match.group(1).lower()
        if justification_match:
            partial = justification_match.group(1).replace("\\n", "\n").replace('\\"', '"').strip()
            parsed["justification"] = partial.rstrip('}", ')
        return parsed, str(exc)


def infer_model(request_file: Path) -> str:
    first_line = request_file.read_text(encoding="utf-8").splitlines()[0]
    payload = json.loads(first_line)
    return str(payload["body"]["model"])


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def chunked(seq: list[Any], size: int) -> list[list[Any]]:
    return [seq[idx : idx + size] for idx in range(0, len(seq), size)]


def batch_status_payload(batch: Any) -> dict[str, Any]:
    counts = getattr(batch, "request_counts", None)
    return {
        "polled_at_utc": now_utc(),
        "batch_id": batch.id,
        "status": batch.status,
        "input_file_id": getattr(batch, "input_file_id", None),
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "request_counts": {
            "completed": getattr(counts, "completed", None) if counts else None,
            "failed": getattr(counts, "failed", None) if counts else None,
            "total": getattr(counts, "total", None) if counts else None,
        },
    }


def download_output_if_ready(client: OpenAI, output_file_id: str | None, out_path: Path) -> bool:
    if not output_file_id:
        return False
    if not out_path.exists():
        content = client.files.content(output_file_id).read()
        out_path.write_bytes(content)
    return True


def load_current_prompt_instruction(source: str = "current") -> dict[str, Any]:
    if source == "initial":
        return {
            "source": "initial_cycle_0",
            "instruction_text": base_cycle_instruction(0),
            "updated_at_utc": now_utc(),
        }
    if source == "stable" and STABLE_PROMPT_STATE_PATH.exists():
        return json.loads(STABLE_PROMPT_STATE_PATH.read_text(encoding="utf-8"))
    if CURRENT_PROMPT_PATH.exists():
        return json.loads(CURRENT_PROMPT_PATH.read_text(encoding="utf-8"))
    return {
        "source": "initial_cycle_0",
        "instruction_text": base_cycle_instruction(0),
        "updated_at_utc": now_utc(),
    }


def set_current_prompt(payload: dict[str, Any]) -> None:
    ensure_dirs()
    write_json(CURRENT_PROMPT_PATH, payload)


def length_band(word_count: int) -> str:
    if word_count < 80:
        return "short"
    if word_count < 150:
        return "medium"
    return "long"


def sample_diverse_real_reviews(n: int, seed: int) -> pd.DataFrame:
    df = parse_real_reviews()
    df = df.copy()
    df["length_band"] = df["word_count"].apply(length_band)
    rng = random.Random(seed)
    df["rand"] = [rng.random() for _ in range(len(df))]
    strata = []
    for (_, _), group in df.groupby(["course_code", "length_band"], dropna=False):
        group = group.sort_values("rand").reset_index().rename(columns={"index": "orig_idx"})
        strata.append(group)
    sampled_indices: list[int] = []
    made_progress = True
    round_idx = 0
    while len(sampled_indices) < min(n, len(df)) and made_progress:
        made_progress = False
        for group in strata:
            if round_idx < len(group):
                sampled_indices.append(int(group.iloc[round_idx]["orig_idx"]))
                made_progress = True
                if len(sampled_indices) >= min(n, len(df)):
                    break
        round_idx += 1
    selected = df.iloc[sampled_indices].drop(columns=["rand"]).reset_index(drop=True)
    return selected


def build_realism_judge_prompt(review_text: str) -> str:
    return (
        "You are evaluating whether a student course review is REAL or SYNTHETIC.\n"
        "Return strict JSON with exactly two keys: label and justification.\n"
        "The label must be either real or synthetic.\n"
        "If the label is real, justification must be an empty string.\n"
        "If the label is synthetic, justification must briefly explain the strongest reasons you suspect it is synthetic.\n\n"
        f"Review:\n{review_text}\n"
    )


def write_batch_request(handle, custom_id: str, model: str, prompt: str, max_output_tokens: int, schema_name: str, schema: dict[str, Any]) -> None:
    payload = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": "low"},
            "text": {
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                },
            },
        },
    }
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def judge_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label": {"type": "string", "enum": ["real", "synthetic"]},
            "justification": {"type": "string"},
        },
        "required": ["label", "justification"],
    }


def prepare_real_baseline(args: argparse.Namespace) -> None:
    ensure_dirs()
    run_dir = make_run_dir(args.prefix)
    write_status(run_dir, "prepare_real_baseline", "started", n=args.n, seed=args.seed)
    df = sample_diverse_real_reviews(args.n, args.seed)
    reviews_path = run_dir / "real_reviews.csv"
    manifest_path = run_dir / "manifest.csv"
    requests_path = run_dir / "requests.jsonl"
    df.to_csv(reviews_path, index=False)
    manifest_rows = []
    with requests_path.open("w", encoding="utf-8") as handle:
        for idx, row in df.reset_index(drop=True).iterrows():
            custom_id = f"real_baseline__{idx}"
            prompt = build_realism_judge_prompt(str(row["review_text"]))
            write_batch_request(handle, custom_id, JUDGE_MODEL, prompt, 220, "realism_judge", judge_schema())
            manifest_rows.append(
                {
                    "custom_id": custom_id,
                    "course_code": row["course_code"],
                    "length_band": row["length_band"],
                    "word_count": row["word_count"],
                    "review_text": row["review_text"],
                    "source_url": row["source_url"],
                }
            )
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    write_status(
        run_dir,
        "prepare_real_baseline",
        "completed",
        n_rows=int(len(df)),
        request_file=str(requests_path),
        manifest_file=str(manifest_path),
        reviews_file=str(reviews_path),
    )
    print(json.dumps({"run_dir": str(run_dir), "n_rows": int(len(df))}, indent=2))


def prepare_synthetic_cycle(args: argparse.Namespace) -> None:
    ensure_dirs()
    run_dir = make_run_dir(args.prefix)
    prompt_state = load_current_prompt_instruction(args.prompt_source)
    schema_payload = load_generation_schema()
    aspect_distribution = schema_payload["recommended_sampling_rule"]["practical_override"]
    real_df = sample_diverse_real_reviews(args.n, args.seed)
    rng = random.Random(args.seed)
    write_status(
        run_dir,
        "prepare_synthetic_cycle",
        "started",
        n=args.n,
        seed=args.seed,
        prompt_source=prompt_state.get("source", args.prompt_source),
    )
    manifest_rows = []
    requests_path = run_dir / "requests.jsonl"
    with requests_path.open("w", encoding="utf-8") as handle:
        for idx, row in real_df.reset_index(drop=True).iterrows():
            course_code = str(row["course_code"])
            n_aspects = sample_aspect_count(rng, aspect_distribution)
            aspect_labels = sample_aspect_labels(rng, n_aspects)
            attributes = sample_rich_attributes(rng, course_code, schema_payload)
            attributes["__aspect_lines__"] = "\n".join(f"- {k}: {v}" for k, v in aspect_labels.items())
            prompt = build_generator_prompt(attributes, str(prompt_state["instruction_text"]))
            custom_id = f"synthetic_cycle_{args.cycle_id}__{idx}"
            payload = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": GENERATOR_MODEL,
                    "input": prompt,
                    "max_output_tokens": 500,
                    "reasoning": {"effort": "low"},
                    "text": {"verbosity": "low"},
                },
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            manifest_rows.append(
                {
                    "custom_id": custom_id,
                    "cycle_id": args.cycle_id,
                    "course_code": course_code,
                    "aspect_labels": json.dumps(aspect_labels, ensure_ascii=False),
                    "attributes": json.dumps(attributes, ensure_ascii=False),
                    "generation_prompt": prompt,
                    "prompt_source": prompt_state.get("source", args.prompt_source),
                }
            )
    pd.DataFrame(manifest_rows).to_csv(run_dir / "manifest.csv", index=False)
    write_json(run_dir / "prompt_input.json", prompt_state)
    write_status(run_dir, "prepare_synthetic_cycle", "completed", n_rows=len(manifest_rows), cycle_id=args.cycle_id)
    print(json.dumps({"run_dir": str(run_dir), "n_rows": len(manifest_rows), "cycle_id": args.cycle_id}, indent=2))


def submit_batch(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    requests_path = run_dir / "requests.jsonl"
    manifest_path = run_dir / "manifest.csv"
    if not requests_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("requests.jsonl or manifest.csv is missing in run_dir")
    client = load_client()
    write_status(run_dir, "submit_batch", "started")
    request_lines = requests_path.read_text(encoding="utf-8").splitlines()
    manifest_df = pd.read_csv(manifest_path)
    if args.max_requests_per_batch and len(request_lines) > args.max_requests_per_batch:
        shards_dir = run_dir / "shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        submitted = []
        request_chunks = chunked(request_lines, args.max_requests_per_batch)
        manifest_chunks = [manifest_df.iloc[idx : idx + args.max_requests_per_batch].copy() for idx in range(0, len(manifest_df), args.max_requests_per_batch)]
        for shard_idx, (request_chunk, manifest_chunk) in enumerate(zip(request_chunks, manifest_chunks), start=1):
            shard_dir = shards_dir / f"shard_{shard_idx:03d}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_requests = shard_dir / "requests.jsonl"
            shard_manifest = shard_dir / "manifest.csv"
            shard_requests.write_text("\n".join(request_chunk) + "\n", encoding="utf-8")
            manifest_chunk.to_csv(shard_manifest, index=False)
            with shard_requests.open("rb") as handle:
                uploaded = client.files.create(file=handle, purpose="batch")
            batch = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/responses", completion_window=args.completion_window)
            shard_payload = {
                "submitted_at_utc": now_utc(),
                "batch_id": batch.id,
                "status": batch.status,
                "endpoint": "/v1/responses",
                "completion_window": args.completion_window,
                "input_file_id": uploaded.id,
                "output_file_id": getattr(batch, "output_file_id", None),
                "error_file_id": getattr(batch, "error_file_id", None),
                "request_file": str(shard_requests),
                "manifest_file": str(shard_manifest),
                "model": infer_model(shard_requests),
                "request_count": len(request_chunk),
                "shard_index": shard_idx,
            }
            write_json(shard_dir / "submitted_batch.json", shard_payload)
            submitted.append(shard_payload)
        payload = {
            "submitted_at_utc": now_utc(),
            "sharded": True,
            "endpoint": "/v1/responses",
            "completion_window": args.completion_window,
            "request_file": str(requests_path),
            "manifest_file": str(manifest_path),
            "model": infer_model(requests_path),
            "request_count": len(request_lines),
            "max_requests_per_batch": args.max_requests_per_batch,
            "shards": submitted,
        }
        write_json(run_dir / "submitted_batch.json", payload)
        write_status(run_dir, "submit_batch", "completed", sharded=True, shard_count=len(submitted), model=payload["model"])
        print(json.dumps(payload, indent=2))
        return

    with requests_path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/responses", completion_window=args.completion_window)
    payload = {
        "submitted_at_utc": now_utc(),
        "batch_id": batch.id,
        "status": batch.status,
        "endpoint": "/v1/responses",
        "completion_window": args.completion_window,
        "input_file_id": uploaded.id,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "request_file": str(requests_path),
        "manifest_file": str(manifest_path),
        "model": infer_model(requests_path),
        "request_count": len(request_lines),
    }
    write_json(run_dir / "submitted_batch.json", payload)
    write_status(run_dir, "submit_batch", "completed", batch_id=batch.id, model=payload["model"], sharded=False)
    print(json.dumps(payload, indent=2))


def poll_batch(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    submitted = json.loads((run_dir / "submitted_batch.json").read_text(encoding="utf-8"))
    client = load_client()
    if submitted.get("sharded"):
        shard_statuses = []
        total_completed = 0
        total_failed = 0
        total_requests = 0
        terminal_statuses = {"completed", "failed", "cancelled", "expired"}
        aggregate_status = "completed"
        merge_ready = True
        for shard in submitted.get("shards", []):
            shard_request_file = Path(shard["request_file"])
            shard_dir = shard_request_file.parent
            shard_submitted = load_json(shard_dir / "submitted_batch.json")
            batch = client.batches.retrieve(shard_submitted["batch_id"])
            shard_payload = batch_status_payload(batch)
            shard_payload["shard_index"] = shard_submitted.get("shard_index")
            shard_payload["request_count"] = shard_submitted.get("request_count")
            write_json(shard_dir / "latest_status.json", shard_payload)
            append_jsonl(shard_dir / "log.jsonl", {"updated_at_utc": now_utc(), "stage": "poll_batch", **shard_payload})
            shard_output_path = shard_dir / "output.jsonl"
            if batch.status == "completed":
                merge_ready = download_output_if_ready(client, shard_payload.get("output_file_id"), shard_output_path) and merge_ready
            else:
                merge_ready = False
            counts = shard_payload["request_counts"]
            total_completed += int(counts.get("completed") or 0)
            total_failed += int(counts.get("failed") or 0)
            total_requests += int(counts.get("total") or shard_submitted.get("request_count") or 0)
            if batch.status not in terminal_statuses:
                aggregate_status = "in_progress"
            elif aggregate_status == "completed" and batch.status != "completed":
                aggregate_status = batch.status
            shard_statuses.append(shard_payload)
        payload = {
            "polled_at_utc": now_utc(),
            "sharded": True,
            "status": aggregate_status,
            "request_counts": {
                "completed": total_completed,
                "failed": total_failed,
                "total": total_requests,
            },
            "shard_count": len(shard_statuses),
            "shards": shard_statuses,
        }
        if merge_ready and shard_statuses:
            merged_output = run_dir / "output.jsonl"
            if not merged_output.exists():
                with merged_output.open("wb") as out_handle:
                    for shard in sorted(submitted["shards"], key=lambda item: item.get("shard_index", 0)):
                        shard_output = Path(shard["request_file"]).parent / "output.jsonl"
                        out_handle.write(shard_output.read_bytes())
        write_json(run_dir / "latest_status.json", payload)
        append_jsonl(run_dir / "log.jsonl", {"updated_at_utc": now_utc(), "stage": "poll_batch", **payload})
        print(json.dumps(payload, indent=2))
        return

    batch = client.batches.retrieve(submitted["batch_id"])
    payload = batch_status_payload(batch)
    write_json(run_dir / "latest_status.json", payload)
    append_jsonl(run_dir / "log.jsonl", {"updated_at_utc": now_utc(), "stage": "poll_batch", **payload})
    if batch.status == "completed":
        download_output_if_ready(client, payload.get("output_file_id"), run_dir / "output.jsonl")
    print(json.dumps(payload, indent=2))


def consume_real_baseline(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest = pd.read_csv(run_dir / "manifest.csv")
    results_by_id = {}
    with (run_dir / "output.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            results_by_id[row["custom_id"]] = row
    rows = []
    parse_failures = 0
    for _, item in manifest.iterrows():
        raw_text = extract_output_text(results_by_id[item["custom_id"]])
        parsed, parse_error = safe_parse_judge_response(raw_text) if raw_text else ({}, "missing_raw_response")
        label = str(parsed.get("label", "")).strip().lower()
        justification = str(parsed.get("justification", "")).strip()
        if parse_error:
            parse_failures += 1
        rows.append(
            {
                **item.to_dict(),
                "predicted_label": label,
                "justification": justification,
                "raw_response_text": raw_text,
                "parse_error": parse_error,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "judgments.csv", index=False)
    by_course = (
        df.assign(flagged_synthetic=(df["predicted_label"] == "synthetic").astype(int))
        .groupby("course_code", as_index=False)["flagged_synthetic"]
        .mean()
        .rename(columns={"flagged_synthetic": "synthetic_rate"})
    )
    by_course.to_csv(run_dir / "synthetic_rate_by_course.csv", index=False)
    by_length = (
        df.assign(flagged_synthetic=(df["predicted_label"] == "synthetic").astype(int))
        .groupby("length_band", as_index=False)["flagged_synthetic"]
        .mean()
        .rename(columns={"flagged_synthetic": "synthetic_rate"})
    )
    by_length.to_csv(run_dir / "synthetic_rate_by_length_band.csv", index=False)
    synthetic_rows = df[df["predicted_label"] == "synthetic"]
    summary = {
        "run_dir": str(run_dir),
        "n_reviews": int(len(df)),
        "n_labeled_synthetic": int((df["predicted_label"] == "synthetic").sum()),
        "real_false_synthetic_rate": round(float((df["predicted_label"] == "synthetic").mean()), 4),
        "parse_failures": int(parse_failures),
        "top_real_suspicion_justifications": synthetic_rows["justification"].head(20).tolist(),
    }
    write_json(run_dir / "summary.json", summary)
    write_status(run_dir, "consume_real_baseline", "completed", n_reviews=int(len(df)))
    print(json.dumps(summary, indent=2))


def consume_synthetic_generation(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest = pd.read_csv(run_dir / "manifest.csv")
    results_by_id = {}
    with (run_dir / "output.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            results_by_id[row["custom_id"]] = row
    rows = []
    for _, item in manifest.iterrows():
        raw_text = extract_output_text(results_by_id[item["custom_id"]])
        rows.append({**item.to_dict(), "generated_review_text": normalize_space(raw_text)})
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "generated_reviews.csv", index=False)
    with (run_dir / "generated_reviews.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_status(run_dir, "consume_synthetic_generation", "completed", n_reviews=len(rows))
    print(json.dumps({"run_dir": str(run_dir), "n_reviews": len(rows)}, indent=2))


def prepare_synthetic_judge(args: argparse.Namespace) -> None:
    source_dir = Path(args.generation_run_dir)
    run_dir = make_run_dir(args.prefix)
    df = pd.read_csv(source_dir / "generated_reviews.csv")
    manifest_rows = []
    with (run_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for idx, row in df.reset_index(drop=True).iterrows():
            custom_id = f"synthetic_judge__{idx}"
            prompt = build_realism_judge_prompt(str(row["generated_review_text"]))
            write_batch_request(handle, custom_id, JUDGE_MODEL, prompt, 220, "realism_judge", judge_schema())
            manifest_rows.append(
                {
                    "custom_id": custom_id,
                    "source_generation_run_dir": str(source_dir),
                    "source_custom_id": row["custom_id"],
                    "cycle_id": row.get("cycle_id", ""),
                    "course_code": row["course_code"],
                    "generated_review_text": row["generated_review_text"],
                    "aspect_labels": row["aspect_labels"],
                    "attributes": row["attributes"],
                    "generation_prompt": row["generation_prompt"],
                }
            )
    pd.DataFrame(manifest_rows).to_csv(run_dir / "manifest.csv", index=False)
    write_status(run_dir, "prepare_synthetic_judge", "completed", n_reviews=len(manifest_rows), source_generation_run_dir=str(source_dir))
    print(json.dumps({"run_dir": str(run_dir), "n_reviews": len(manifest_rows)}, indent=2))


def binary_entropy(prob: float) -> float:
    p = max(0.0, min(1.0, float(prob)))
    q = 1.0 - p
    if p in (0.0, 1.0) or q in (0.0, 1.0):
        return 0.0
    return round(-(p * math.log2(p) + q * math.log2(q)), 4)


def consume_synthetic_judge(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest = pd.read_csv(run_dir / "manifest.csv")
    results_by_id = {}
    with (run_dir / "output.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            results_by_id[row["custom_id"]] = row
    rows = []
    justifications = []
    parse_failures = 0
    for _, item in manifest.iterrows():
        raw_text = extract_output_text(results_by_id[item["custom_id"]])
        parsed, parse_error = safe_parse_judge_response(raw_text) if raw_text else ({}, "missing_raw_response")
        label = str(parsed.get("label", "")).strip().lower()
        justification = str(parsed.get("justification", "")).strip()
        if parse_error:
            parse_failures += 1
        row = {
            **item.to_dict(),
            "predicted_label": label,
            "justification": justification,
            "raw_response_text": raw_text,
            "parse_error": parse_error,
        }
        rows.append(row)
        if label == "synthetic":
            justifications.append(
                {
                    "custom_id": item["custom_id"],
                    "source_custom_id": item["source_custom_id"],
                    "course_code": item["course_code"],
                    "justification": justification,
                    "generated_review_text": item["generated_review_text"],
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "judgments.csv", index=False)
    with (run_dir / "synthetic_justifications.jsonl").open("w", encoding="utf-8") as handle:
        for item in justifications:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    synthetic_rate = float((df["predicted_label"] == "synthetic").mean()) if len(df) else 0.0
    counts = Counter(df["predicted_label"].astype(str).tolist())
    summary = {
        "run_dir": str(run_dir),
        "n_reviews": int(len(df)),
        "n_labeled_synthetic": int(counts.get("synthetic", 0)),
        "synthetic_labeled_synthetic_rate": round(synthetic_rate, 4),
        "synthetic_labeled_real_rate": round(float(counts.get("real", 0) / len(df)) if len(df) else 0.0, 4),
        "prediction_label_entropy_bits": binary_entropy(float(counts.get("real", 0) / len(df)) if len(df) else 0.0),
        "parse_failures": int(parse_failures),
        "top_synthetic_detection_justifications": [item["justification"] for item in justifications[:20]],
    }
    write_json(run_dir / "summary.json", summary)
    write_status(run_dir, "consume_synthetic_judge", "completed", n_reviews=int(len(df)))
    print(json.dumps(summary, indent=2))


def summarize_justifications(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    justifications = []
    with (run_dir / "synthetic_justifications.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            justifications.append(json.loads(line))
    prompt = (
        "You are summarizing why a strong judge suspected synthetic course reviews.\n"
        "You will receive a list of justifications from reviews labeled synthetic.\n"
        "Return strict JSON with keys: major_reasons, recurring_artifacts, realism_mismatches, prompt_edit_suggestions, summary.\n"
        "Each list key should contain short strings. Keep the result concise and evidence-focused.\n\n"
        f"Justifications:\n{json.dumps(justifications, ensure_ascii=False)}"
    )
    client = load_client()
    response = client.responses.create(model=REWRITE_MODEL, input=prompt, max_output_tokens=800)
    raw_text = (response.output_text or "").strip()
    parsed = extract_json_block(raw_text)
    write_json(run_dir / "justification_summary.json", parsed)
    write_json(run_dir / "justification_summary_raw.json", {"raw_response_text": raw_text})
    write_status(run_dir, "summarize_justifications", "completed", n_justifications=len(justifications))
    print(json.dumps(parsed, indent=2, ensure_ascii=False))


def rewrite_prompt(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    summary = json.loads((run_dir / "justification_summary.json").read_text(encoding="utf-8"))
    current_prompt = load_current_prompt_instruction(args.prompt_source)
    rewrite_input = {
        "current_prompt": current_prompt,
        "summary": summary,
    }
    write_json(run_dir / "rewrite_input.json", rewrite_input)
    prompt = (
        "You are rewriting a low-cost synthetic course-review generation prompt.\n"
        "The task is binary-polarity educational ABSA data generation.\n"
        "Keep the prompt concise, operational, and generator-facing.\n"
        "Return strict JSON with keys: revised_instruction, change_summary, rationale.\n"
        "The revised instruction should directly address the summarized synthetic cues while preserving binary aspect-polarity recoverability.\n\n"
        f"Current prompt state:\n{json.dumps(current_prompt, ensure_ascii=False)}\n\n"
        f"Failure summary:\n{json.dumps(summary, ensure_ascii=False)}"
    )
    client = load_client()
    response = client.responses.create(model=REWRITE_MODEL, input=prompt, max_output_tokens=1000)
    raw_text = (response.output_text or "").strip()
    parsed = extract_json_block(raw_text)
    payload = {
        "updated_at_utc": now_utc(),
        "source": f"rewrite_from_{run_dir.name}",
        "instruction_text": str(parsed["revised_instruction"]).strip(),
        "change_summary": parsed.get("change_summary", []),
        "rationale": parsed.get("rationale", ""),
    }
    write_json(run_dir / "rewrite_output.json", parsed)
    if args.set_current:
        set_current_prompt(payload)
    write_status(run_dir, "rewrite_prompt", "completed", set_current=bool(args.set_current))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def set_prompt(args: argparse.Namespace) -> None:
    source_payload = load_current_prompt_instruction(args.source)
    payload = {
        "updated_at_utc": now_utc(),
        "source": source_payload.get("source", args.source),
        "instruction_text": source_payload["instruction_text"],
    }
    set_current_prompt(payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-first realism pipeline for binary-polarity educational review generation.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("set-current-prompt")
    p.add_argument("--source", choices=["initial", "stable", "current"], default="initial")

    p = sub.add_parser("prepare-real-baseline")
    p.add_argument("--prefix", default="realism_real_baseline_200")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)

    p = sub.add_parser("prepare-synthetic-cycle")
    p.add_argument("--prefix", required=True)
    p.add_argument("--cycle-id", type=int, required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--prompt-source", choices=["initial", "stable", "current"], default="current")

    p = sub.add_parser("submit-batch")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--completion-window", default="24h")
    p.add_argument("--max-requests-per-batch", type=int, default=None)

    p = sub.add_parser("poll-batch")
    p.add_argument("--run-dir", required=True)

    p = sub.add_parser("consume-real-baseline")
    p.add_argument("--run-dir", required=True)

    p = sub.add_parser("consume-synthetic-generation")
    p.add_argument("--run-dir", required=True)

    p = sub.add_parser("prepare-synthetic-judge")
    p.add_argument("--generation-run-dir", required=True)
    p.add_argument("--prefix", required=True)

    p = sub.add_parser("consume-synthetic-judge")
    p.add_argument("--run-dir", required=True)

    p = sub.add_parser("summarize-justifications")
    p.add_argument("--run-dir", required=True)

    p = sub.add_parser("rewrite-prompt")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--prompt-source", choices=["initial", "stable", "current"], default="current")
    p.add_argument("--set-current", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "set-current-prompt":
        set_prompt(args)
    elif args.cmd == "prepare-real-baseline":
        prepare_real_baseline(args)
    elif args.cmd == "prepare-synthetic-cycle":
        prepare_synthetic_cycle(args)
    elif args.cmd == "submit-batch":
        submit_batch(args)
    elif args.cmd == "poll-batch":
        poll_batch(args)
    elif args.cmd == "consume-real-baseline":
        consume_real_baseline(args)
    elif args.cmd == "consume-synthetic-generation":
        consume_synthetic_generation(args)
    elif args.cmd == "prepare-synthetic-judge":
        prepare_synthetic_judge(args)
    elif args.cmd == "consume-synthetic-judge":
        consume_synthetic_judge(args)
    elif args.cmd == "summarize-justifications":
        summarize_justifications(args)
    elif args.cmd == "rewrite-prompt":
        rewrite_prompt(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
