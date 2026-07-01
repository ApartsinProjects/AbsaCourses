"""RC2: identify the 841 output-token-capped ('incomplete') rows in the final 10K
corpus by joining the generation batch output (per-row API status) to the corpus,
then test whether those rows differ systematically from complete rows in aspect
distribution, sentiment mix, aspect count, and length. Also writes the exact
incomplete-row id set so the benchmark can be re-run excluding them.
"""
import json, collections, os

BATCH = r'E:\Projects\Submitted\CourseABSA\paper\batch_results\batch_69cc15c483488190941478aa4e3a976d_output.jsonl'
CORP = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reviewer_ab_data', 'generated_reviews_10k.jsonl')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)

# --- 1) batch status per custom_id gen_<i> ---
status = {}
for l in open(BATCH, encoding='utf-8'):
    if not l.strip():
        continue
    o = json.loads(l)
    body = (o.get('response') or {}).get('body') or {}
    status[o['custom_id']] = body.get('status')

rows = [json.loads(l) for l in open(CORP, encoding='utf-8')]
for r in rows:
    r['_status'] = status.get(f"gen_{r['sample_id']}")

matched = sum(1 for r in rows if r['_status'] is not None)
inc = [r for r in rows if r['_status'] == 'incomplete']
com = [r for r in rows if r['_status'] == 'completed']
print(f"matched {matched}/{len(rows)}  | incomplete={len(inc)} complete={len(com)}")

# --- 2) distributions: incomplete vs complete ---
def words(r): return len(r['text'].split())
def n_aspects(r): return len(r.get('aspects') or {})

def summary(group):
    n = len(group)
    wl = sorted(words(r) for r in group)
    asp = collections.Counter()
    sent = collections.Counter()
    ncount = collections.Counter()
    for r in group:
        a = r.get('aspects') or {}
        ncount[len(a)] += 1
        for k, v in a.items():
            asp[k] += 1
            sent[str(v).lower()] += 1
    tot_asp = sum(asp.values())
    return {
        'n': n,
        'mean_words': round(sum(wl) / n, 1), 'median_words': wl[n // 2],
        'p90_words': wl[int(0.9 * n)],
        'aspects_per_review': {k: round(ncount[k] / n, 3) for k in (1, 2, 3)},
        'sentiment_share': {k: round(sent[k] / tot_asp, 3) for k in ('positive', 'neutral', 'negative')},
        'aspect_share': {k: round(v / tot_asp, 4) for k, v in asp.items()},
    }

S_inc, S_com = summary(inc), summary(com)
print("\n=== length (words) ===")
print(f"  incomplete: mean {S_inc['mean_words']} median {S_inc['median_words']} p90 {S_inc['p90_words']}")
print(f"  complete  : mean {S_com['mean_words']} median {S_com['median_words']} p90 {S_com['p90_words']}")
print("\n=== aspects per review (share) ===")
print("  incomplete:", S_inc['aspects_per_review'])
print("  complete  :", S_com['aspects_per_review'])
print("\n=== sentiment share ===")
print("  incomplete:", S_inc['sentiment_share'])
print("  complete  :", S_com['sentiment_share'])

# aspect over/under-representation in incomplete vs complete (ratio of shares)
print("\n=== aspect share ratio incomplete/complete (top divergences) ===")
allasp = set(S_inc['aspect_share']) | set(S_com['aspect_share'])
div = []
for a in allasp:
    pi = S_inc['aspect_share'].get(a, 0); pc = S_com['aspect_share'].get(a, 1e-9)
    div.append((pi / pc if pc else float('inf'), a, round(pi, 4), round(pc, 4)))
for ratio, a, pi, pc in sorted(div)[:4] + sorted(div)[-4:]:
    print(f"  {a:22} inc={pi:.4f} com={pc:.4f} ratio={ratio:.2f}")

# --- 3) save artifacts ---
inc_ids = sorted((r['sample_id'] for r in inc), key=lambda x: int(x))
json.dump({'incomplete_sample_ids': inc_ids, 'n_incomplete': len(inc_ids),
           'n_total': len(rows), 'reason': 'max_output_tokens',
           'source_batch': os.path.basename(BATCH)},
          open(os.path.join(OUT, 'rc2_incomplete_row_ids.json'), 'w'), indent=2)
json.dump({'incomplete': S_inc, 'complete': S_com},
          open(os.path.join(OUT, 'rc2_incomplete_analysis.json'), 'w'), indent=2)
print("\nwrote outputs/rc2_incomplete_row_ids.json and outputs/rc2_incomplete_analysis.json")
