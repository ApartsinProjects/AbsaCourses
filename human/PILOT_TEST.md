# Pilot test: 10-item single-labeler realism check via Argilla

This is the smallest possible end-to-end exercise of the `human-labeling`
workflow against this project's actual data. One labeler (you) classifies
10 reviews as real or synthetic, then we score against hidden ground
truth.

## What's already prepared

| file | role |
|------|------|
| `tasks/task_9_pilot/manifest.csv` | 10 items: 5 real Herath reviews + 5 synthetic, shuffled |
| `tasks/task_9_pilot/_truth.json` | hidden source labels, used after labeling for scoring |
| `tasks/task_9_pilot/rater_A.csv` | per-rater file with empty answer columns |
| `argilla_settings/task_9.py` | Argilla settings module (pattern 4, single labeler) |
| `scripts/push_to_argilla.py` | provisioner |
| `scripts/pull_from_argilla.py` | response collector |
| `scripts/score_task_1.py` | already-built scorer; works on the pilot too with `--responses_dir responses/task_9` |

## Step 1: stand up a free Argilla instance on Hugging Face

1. Open https://huggingface.co/new-space?template=argilla/argilla-template-space.
2. Pick any Space name. Choose CPU Basic (free). Choose Private if you do
   not want others to see the reviews.
3. Click "Duplicate Space". Wait ~2 minutes for the container to build.
4. Visit the Space URL (`https://<your-handle>-<space-name>.hf.space`),
   sign in with the default admin account if prompted.
5. In the Argilla UI, click your avatar (top right) > My Settings > API
   key. Copy it.

## Step 2: configure the client

Create `~/.argilla.json` (or `C:\Users\<you>\.argilla.json` on Windows):

```json
{
  "api_url": "https://<your-handle>-<your-space-name>.hf.space",
  "api_key": "<key-from-the-argilla-ui>",
  "workspace": "default",
  "rater_emails": {
    "A": "<your-argilla-username-or-email>"
  }
}
```

On Unix, `chmod 600 ~/.argilla.json` after creating it.

The `rater_emails` value for `A` should match the Argilla username you see
in the UI (often your email).

## Step 3: push the pilot

From the repository root (`E:\Claude\CourseABSA\hopeful-kowalevski-04ee10\`):

```bash
"C:/Users/apart/AppData/Local/Programs/Python/Python311/python.exe" \
    human/scripts/push_to_argilla.py \
    --study-id absa-pilot \
    --task 9
```

Note that the script expects `~/.argilla.json` and the `argilla_settings/`
folder under the same study root the script lives in (`human/`).
Either run with `python` from `human/` so its parent-of-script lookup
finds the right `tasks/` and `argilla_settings/` folders, or invoke as
above and the script's own `Path(__file__).resolve().parents[1]` will
already point at `human/`.

You should see:

```
Pushing tasks: [9]
  task_9: created dataset 'absa-pilot__task_9'
  task_9: logged 10 records
Done. Summary at .../human/tasks/argilla_state.json
```

## Step 4: label the 10 items

Open your Argilla Space URL. You should see a dataset named
`absa-pilot__task_9` in the `default` workspace. Click it. The UI will
present each review with a real/synthetic choice, a 1-5 confidence
slider, and an optional notes field. Submit each one.

Expected time: ~3 to 5 minutes for the whole batch.

## Step 5: pull responses

Back at the repo root:

```bash
"C:/Users/apart/AppData/Local/Programs/Python/Python311/python.exe" \
    human/scripts/pull_from_argilla.py \
    --task 9
```

This writes `human/responses/task_9/rater_A_complete.csv` matching the
column schema of `human/tasks/task_9_pilot/rater_A.csv`.

## Step 6: score

The existing `score_task_1.py` works on any pattern-4 task, so reuse it:

```bash
cd human
"C:/Users/apart/AppData/Local/Programs/Python/Python311/python.exe" \
    scripts/score_task_1.py --responses_dir responses/task_9
```

(or write a tiny `score_task_9.py` mirror if you prefer; the scoring
logic is identical for any single-labeler pattern-4 task.)

This prints accuracy and confidence-weighted accuracy. With 10 items and
one labeler, no kappa is computed (kappa needs at least two raters); the
output is just the per-rater accuracy block.

## Step 7: tear down (optional)

If you do not want to keep the Argilla Space running:

1. Visit the Space's Settings tab on Hugging Face.
2. Click "Pause" (free option, returns within seconds when needed) or
   "Delete this Space" to remove it entirely.

The `tasks/task_9_pilot/` data on disk is untouched by tear-down.

## Why this is the right first pilot

- Smallest unit: one labeler, one task pattern, one platform.
- Uses real project data (5 Herath reals + 5 synthetic from the 10K
  corpus).
- Exercises every script the full workflow needs (push, pull, score).
- Cheap to iterate: if the Argilla settings need tweaking, edit
  `argilla_settings/task_9.py`, rerun push with `--recreate`.

## Scaling up after the pilot works

- Add raters: edit `rater_emails` in `~/.argilla.json`, change
  `MIN_SUBMITTED` in `argilla_settings/task_9.py` to the new count, push
  with `--recreate`.
- Move to the full Task 1 (80 items + Part 2 faithfulness): rebuild
  `tasks/task_1_realism_and_faithfulness/` with the same workflow,
  create `argilla_settings/task_1.py` from
  `assets/argilla_settings/pattern_4_binary_with_confidence.py`, push.
