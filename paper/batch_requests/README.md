# OpenAI Batch Preparation

This folder contains batch-ready request files only. Nothing here executes API jobs.

Files:
- `schema_review_requests.jsonl`
  Small batch for protocol review.
- `dataset_generation_10k_v2_requests.jsonl`
  Generation requests using the current stable realism prompt package.
- `dataset_generation_10k_v2_manifest.csv`
  Manifest containing target aspects, nuance attributes, model, and prompt lineage.
- `judge_pair_template.csv`
  Template that pairs real reviews with placeholders for generated synthetic outputs.

Batch boundary:
- generation model: `gpt-5-nano`
- judge/protocol-review model: `gpt-5.4`
- prepared generation size: `10000`
- generation max output tokens: band-dependent, default upper bound `420`
- generation reasoning effort: `minimal`
- generation text verbosity: `low`
- prompt source: `paper/generation_protocol/final_realism_prompt_template.txt`
- aspect inventory: `20` forward-protocol aspects
