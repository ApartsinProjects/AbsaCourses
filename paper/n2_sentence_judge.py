"""Analysis gate: judge INDIVIDUAL sentences (not whole reviews) as real/synthetic.
Tests whether review-level detectability (~93%) is a document-structure artifact
that collapses at sentence granularity.
"""
import asyncio, csv, json, random, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
GEN=ROOT/"paper/generated_datasets/batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl"
IDS=ROOT/"paper/outputs/rc2_incomplete_row_ids.json"
REAL=ROOT/"paper/validation/batch_realism/runs/realism_real_baseline_200_20260404T131844Z/real_reviews.csv"
JUDGE="openai/gpt-5.4"
SENT=re.compile(r"(?<=[.!?])\s+")
def key():
    for l in Path("E:/Projects/.env.all").read_text(encoding="utf-8",errors="ignore").splitlines():
        if l.startswith("OPENROUTER_API_KEY"): return l.split("=",1)[1].strip().strip('"').strip("'")
def sents(texts,mw=5,mx=40):
    out=[]
    for t in texts:
        for s in SENT.split(" ".join(str(t).split())):
            w=len(s.split())
            if mw<=w<=mx: out.append(s.strip())
    return out
def prompt(s):
    return ("Below is a SINGLE sentence taken from a student course review. Decide whether it "
            "comes from a REAL student-written review or a SYNTHETIC (AI-generated) one. "
            'Return strict JSON: {"label":"real"} or {"label":"synthetic"}.\n\nSentence:\n'+s)
async def call(c,sem,s,retries=4):
    async with sem:
        for a in range(retries):
            try:
                await asyncio.sleep(random.uniform(0,.3)*(a+1))
                r=await c.chat.completions.create(model=JUDGE,messages=[{"role":"user","content":prompt(s)}],
                    max_tokens=300,temperature=0.0,extra_body={"reasoning":{"effort":"minimal"}})
                t=(r.choices[0].message.content or "")
                m=re.search(r'"label"\s*:\s*"(real|synthetic)"',t,re.I)
                if m: return m.group(1).lower()
            except Exception: pass
        return None
async def main(n=150):
    from openai import AsyncOpenAI
    ids=set(json.load(open(IDS))["incomplete_sample_ids"])
    syn=[]
    for l in open(GEN,encoding="utf-8"):
        r=json.loads(l)
        if str(r.get("sample_id")) not in ids:
            t=str(r.get("text",""))
            if t and t.lower()!="nan": syn.append(t)
    rng=random.Random(42); rng.shuffle(syn)
    reals=[r["review_text"] for r in csv.DictReader(open(REAL,encoding="utf-8")) if (r.get("review_text") or "").strip()]
    ssent=sents(syn[:400]); rsent=sents(reals)
    rng.shuffle(ssent); rng.shuffle(rsent)
    ssent,rsent=ssent[:n],rsent[:n]
    c=AsyncOpenAI(base_url="https://openrouter.ai/api/v1",api_key=key()); sem=asyncio.Semaphore(8)
    sl=await asyncio.gather(*[call(c,sem,s) for s in ssent])
    rl=await asyncio.gather(*[call(c,sem,s) for s in rsent])
    sl=[x for x in sl if x]; rl=[x for x in rl if x]
    syn_det=sum(x=="synthetic" for x in sl)/len(sl)
    real_fp=sum(x=="synthetic" for x in rl)/len(rl)
    acc=(sum(x=="synthetic" for x in sl)+sum(x=="real" for x in rl))/(len(sl)+len(rl))
    res={"n_synth":len(sl),"n_real":len(rl),"synth_detection_rate":round(syn_det,3),
         "real_false_synth_rate":round(real_fp,3),"overall_accuracy":round(acc,3),
         "review_level_reference":{"synth_detection":0.975,"real_fp":0.15,"accuracy":0.93}}
    json.dump(res,open(ROOT/"paper/outputs/sentence_judge.json","w"),indent=2)
    print(json.dumps(res,indent=2))
asyncio.run(main())
