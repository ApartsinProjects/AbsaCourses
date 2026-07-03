"""Causal control for A.23: do REAL sentences, assembled into multi-aspect
documents, trigger the judge's 'synthetic' tell? If yes, the tell is document
assembly (uniform coverage), not synthetic origin. Uses same-course sentences to
limit incoherence.
"""
import asyncio, csv, json, random, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REAL=ROOT/"paper/validation/batch_realism/runs/realism_real_baseline_200_20260404T131844Z/real_reviews.csv"
JUDGE="openai/gpt-5.4"
SENT=re.compile(r"(?<=[.!?])\s+")
def key():
    for l in Path("E:/Projects/.env.all").read_text(encoding="utf-8",errors="ignore").splitlines():
        if l.startswith("OPENROUTER_API_KEY"): return l.split("=",1)[1].strip().strip('"').strip("'")
def jprompt(t):
    return ("You are evaluating whether a student course review is REAL or SYNTHETIC.\n"
            'Return strict JSON with exactly two keys: label and justification.\n'
            "The label must be either real or synthetic.\n\nReview:\n"+t+"\n")
async def judge(c,sem,t,retries=4):
    async with sem:
        for a in range(retries):
            try:
                await asyncio.sleep(random.uniform(0,.3)*(a+1))
                r=await c.chat.completions.create(model=JUDGE,messages=[{"role":"user","content":jprompt(t)}],
                    max_tokens=400,temperature=0.0,extra_body={"reasoning":{"effort":"minimal"}})
                m=re.search(r'"label"\s*:\s*"(real|synthetic)"',r.choices[0].message.content or "",re.I)
                if m: return m.group(1).lower()
            except Exception: pass
        return None
async def main(ndoc=120):
    from openai import AsyncOpenAI
    rng=random.Random(42)
    bycourse={}
    for r in csv.DictReader(open(REAL,encoding="utf-8")):
        t=(r.get("review_text") or "").strip(); cc=r.get("course_code","")
        if not t or t.lower()=="nan": continue
        for s in SENT.split(" ".join(t.split())):
            if 6<=len(s.split())<=35: bycourse.setdefault(cc,[]).append(s.strip())
    courses=[c for c,v in bycourse.items() if len(v)>=8]
    docs=[]
    for _ in range(ndoc):
        c=rng.choice(courses)
        k=rng.choice([3,4,5])
        docs.append(" ".join(rng.sample(bycourse[c],k)))
    cl=AsyncOpenAI(base_url="https://openrouter.ai/api/v1",api_key=key()); sem=asyncio.Semaphore(8)
    labs=[x for x in await asyncio.gather(*[judge(cl,sem,d) for d in docs]) if x]
    rate=sum(x=="synthetic" for x in labs)/len(labs)
    res={"n_assembled_real_docs":len(labs),"assembled_synthetic_rate":round(rate,3),
         "reference_real_full_reviews_synthetic_rate":0.15,
         "reference_synthetic_reviews_synthetic_rate":0.975,
         "mean_words":round(sum(len(d.split()) for d in docs)/len(docs),1)}
    json.dump(res,open(ROOT/"paper/outputs/assembled_real.json","w"),indent=2)
    print(json.dumps(res,indent=2))
asyncio.run(main())
