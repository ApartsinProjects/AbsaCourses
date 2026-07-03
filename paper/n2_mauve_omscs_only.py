"""Apples-to-apples realism: MAUVE(OMSCS full reviews, synthetic full reviews).
The earlier MAUVE mixed short Herath into 'real', which is not length-matched.
This compares full-review real (OMSCS) to full-review synthetic only.
"""
import json, csv, random
from pathlib import Path
import mauve
ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT/"paper/generated_datasets/batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl"
IDS = ROOT/"paper/outputs/rc2_incomplete_row_ids.json"
OMSCS = ROOT/"paper/validation/batch_realism/runs/realism_real_baseline_200_20260404T131844Z/real_reviews.csv"
def clean(t): 
    t=" ".join(str(t).split()); return t if len(t.split())>=5 and t.lower()!="nan" else None
def synth(n,seed=42):
    ids=set(json.load(open(IDS))["incomplete_sample_ids"]); comp=[]
    for l in open(GEN,encoding="utf-8"):
        r=json.loads(l)
        if str(r.get("sample_id")) not in ids:
            c=clean(r.get("text","")); 
            if c: comp.append(c)
    random.Random(seed).shuffle(comp); return comp[:n]
def omscs():
    out=[]
    for r in csv.DictReader(open(OMSCS,encoding="utf-8")):
        c=clean(r.get("review_text",""))
        if c: out.append(c)
    return out
def mv(p,q): return round(float(mauve.compute_mauve(p_text=p,q_text=q,device_id=-1,max_text_length=256,verbose=False,featurize_model_name="gpt2",batch_size=16).mauve),4)
o=omscs(); random.Random(1).shuffle(o); half=len(o)//2
s=synth(len(o))
res={"n_omscs":len(o),"n_synth":len(s),
     "mauve_omscs_vs_synth_FULLREVIEW":mv(o,s),
     "mauve_omscs_vs_omscs_upperbound":mv(o[:half],o[half:2*half])}
json.dump(res,open(ROOT/"paper/outputs/n2_mauve_omscs_only.json","w"),indent=2)
print(json.dumps(res,indent=2))
