"""Within-review sentiment correlation: real reviews are opinion-consistent (halo),
synthetic samples aspect sentiments independently (unnaturally balanced)."""
import json
ROOT_IDS='paper/outputs/rc2_incomplete_row_ids.json'
def maps(path, syn=False):
    ids=set(json.load(open(ROOT_IDS))['incomplete_sample_ids']) if syn else set()
    out=[]
    for l in open(path,encoding='utf-8'):
        r=json.loads(l)
        if syn and str(r.get('sample_id')) in ids: continue
        a=r.get('aspects') or {}
        if len(a)>=2: out.append(list(a.values()))
    return out
def stat(m):
    n=len(m); return {'n':n,'all_same':round(sum(len(set(x))==1 for x in m)/n,3),
        'all_pos':round(sum(all(y=='positive' for y in x) for x in m)/n,3),
        'mixed_posneg':round(sum(('positive' in x and 'negative' in x) for x in m)/n,3)}
res={'SYNTHETIC':stat(maps('paper/generated_datasets/batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl',syn=True)),
     'Herath':stat(maps('paper/real_transfer/herath_mapped_real_reviews.jsonl')),
     'EduRABSA':stat(maps('external_data/EduRABSA_mapped/edurabsa_all_mapped.jsonl')),
     'OATS':stat(maps('external_data/OATS_coursera/oats_mapped.jsonl'))}
json.dump(res,open('paper/outputs/within_review_sentiment.json','w'),indent=2); print(json.dumps(res,indent=2))
